"""File-based custom slash commands.

Users drop .yaml or .md files into ``~/.hermes/commands/`` and they become
available as ``/command-name`` slash commands in the CLI (and gateway).

File formats
------------

**YAML** (``~/.hermes/commands/deploy.yaml``)::

    name: deploy
    description: Deploy the current project to production
    prompt: |
      Deploy the current project. Follow these steps:
      1. Run tests
      2. Build the project
      3. Deploy to {args}
    # Optional:
    aliases: [dp]
    category: Custom

**Markdown** (``~/.hermes/commands/commit.md``)::

    Write a good commit message for the current staged changes,
    then commit them. {args}

The filename (without extension) becomes the command name.  Markdown files
have no frontmatter — the entire content is the prompt template.

Template variables:
    ``{args}`` — replaced with everything after the command name.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_commands_dir: Path | None = None
_custom_commands: Dict[str, Dict[str, Any]] = {}
_loaded = False


def get_commands_dir() -> Path:
    """Return the custom commands directory, creating it if necessary."""
    global _commands_dir
    if _commands_dir is None:
        from hermes_constants import get_hermes_home
        _commands_dir = get_hermes_home() / "commands"
    return _commands_dir


def load_custom_commands() -> Dict[str, Dict[str, Any]]:
    """Scan ``~/.hermes/commands/`` and return a dict of command definitions.

    Returns ``{name: {"description": ..., "prompt": ..., "aliases": [...], ...}}``.
    """
    global _custom_commands, _loaded
    commands: Dict[str, Dict[str, Any]] = {}
    cmd_dir = get_commands_dir()

    if not cmd_dir.is_dir():
        _custom_commands = commands
        _loaded = True
        return commands

    for path in sorted(cmd_dir.iterdir()):
        if path.name.startswith(".") or path.is_dir():
            continue

        stem = path.stem.lower()
        # Sanitize to valid command name: lowercase, hyphens, underscores
        stem = re.sub(r"[^a-z0-9_-]", "", stem)
        if not stem:
            continue

        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            _load_yaml_command(path, stem, commands)
        elif ext == ".md":
            _load_md_command(path, stem, commands)
        # Silently skip other extensions

    _custom_commands = commands
    _loaded = True
    logger.debug("Loaded %d custom command(s) from %s", len(commands), cmd_dir)
    return commands


def get_custom_command(name: str) -> Optional[Dict[str, Any]]:
    """Look up a custom command by name or alias."""
    if not _loaded:
        load_custom_commands()

    # Direct name lookup
    if name in _custom_commands:
        return _custom_commands[name]

    # Alias lookup
    for cmd in _custom_commands.values():
        if name in cmd.get("aliases", []):
            return cmd

    return None


def get_all_custom_commands() -> Dict[str, Dict[str, Any]]:
    """Return all loaded custom commands."""
    if not _loaded:
        load_custom_commands()
    return dict(_custom_commands)


def resolve_custom_command(name: str) -> Optional[str]:
    """Resolve a name/alias to the canonical command name, or None."""
    cmd = get_custom_command(name)
    if cmd:
        # Return the canonical name stored in the definition
        return cmd.get("_canonical_name", name)
    return None


def expand_prompt(cmd: Dict[str, Any], args: str = "") -> str:
    """Expand the prompt template with user arguments."""
    template = cmd.get("prompt", "")
    return template.replace("{args}", args)


def custom_command_defs() -> List[dict]:
    """Return a list of command definitions suitable for COMMAND_REGISTRY.

    Each entry is a dict with keys matching ``CommandDef`` fields.
    """
    from hermes_cli.commands import CommandDef

    if not _loaded:
        load_custom_commands()

    defs: List[CommandDef] = []
    for name, cmd in _custom_commands.items():
        try:
            defs.append(CommandDef(
                name=name,
                description=cmd.get("description", f"Custom command: /{name}"),
                category=cmd.get("category", "Custom"),
                aliases=tuple(cmd.get("aliases", [])),
                args_hint="[arguments]",
            ))
        except Exception:
            logger.warning("Failed to create CommandDef for custom command %r", name)
    return defs


def register_custom_commands() -> int:
    """Register all custom commands into ``COMMAND_REGISTRY``.

    Returns the number of commands registered.
    """
    from hermes_cli.commands import COMMAND_REGISTRY, rebuild_lookups

    if not _loaded:
        load_custom_commands()

    # Avoid duplicate registration
    existing_custom = {c.name for c in COMMAND_REGISTRY if c.category == "Custom"}
    count = 0
    for name, cmd in _custom_commands.items():
        if name in existing_custom:
            continue
        try:
            from hermes_cli.commands import CommandDef
            COMMAND_REGISTRY.append(CommandDef(
                name=name,
                description=cmd.get("description", f"Custom command: /{name}"),
                category="Custom",
                aliases=tuple(cmd.get("aliases", [])),
                args_hint="[arguments]",
            ))
            count += 1
        except Exception:
            logger.warning("Failed to register custom command %r", name)

    if count:
        rebuild_lookups()

    return count


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_yaml_command(path: Path, stem: str, commands: Dict[str, Dict[str, Any]]) -> None:
    """Load a YAML command definition."""
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, skipping YAML command %s", path)
        return

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse custom command %s: %s", path, exc)
        return

    if not isinstance(data, dict):
        logger.warning("Custom command %s: expected YAML dict, got %s", path, type(data).__name__)
        return

    prompt = data.get("prompt", "")
    if not prompt:
        logger.warning("Custom command %s: empty or missing 'prompt'", path)
        return

    name = str(data.get("name") or stem).lower().strip()
    name = re.sub(r"[^a-z0-9_-]", "", name)
    if not name:
        name = stem

    aliases = data.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",")]
    aliases = [re.sub(r"[^a-z0-9_-]", "", str(a).lower().strip()) for a in aliases]
    aliases = [a for a in aliases if a and a != name]

    commands[name] = {
        "_canonical_name": name,
        "description": str(data.get("description", f"Custom command: /{name}")),
        "prompt": prompt,
        "aliases": aliases,
        "category": str(data.get("category", "Custom")),
    }


def _load_md_command(path: Path, stem: str, commands: Dict[str, Dict[str, Any]]) -> None:
    """Load a Markdown command definition (entire file is the prompt)."""
    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Failed to read custom command %s: %s", path, exc)
        return

    if not content:
        return

    commands[stem] = {
        "_canonical_name": stem,
        "description": f"Custom command: /{stem}",
        "prompt": content,
        "aliases": [],
        "category": "Custom",
    }
