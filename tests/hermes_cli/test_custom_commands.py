"""Tests for hermes_cli/custom_commands.py — file-based custom slash commands."""

import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Each test uses a fresh module import to avoid cached state


@pytest.fixture
def commands_dir(tmp_path, monkeypatch):
    """Create a temporary commands directory and reset module state."""
    cmd_dir = tmp_path / "commands"
    cmd_dir.mkdir()
    # Patch HERMES_HOME env var
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Reset module state
    import hermes_cli.custom_commands as _mod
    monkeypatch.setattr(_mod, "_commands_dir", cmd_dir)
    monkeypatch.setattr(_mod, "_loaded", False)
    monkeypatch.setattr(_mod, "_custom_commands", {})
    return cmd_dir


class TestLoadYAMLCommands:
    def test_load_single_yaml(self, commands_dir):
        yaml_content = textwrap.dedent("""\
            name: deploy
            description: Deploy the project
            prompt: |
              Deploy the current project to {args}
            aliases: [dp]
        """)
        (commands_dir / "deploy.yaml").write_text(yaml_content)

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "deploy" in cmds
        assert cmds["deploy"]["description"] == "Deploy the project"
        assert "{args}" in cmds["deploy"]["prompt"]
        assert "dp" in cmds["deploy"]["aliases"]

    def test_yaml_with_comma_separated_aliases(self, commands_dir):
        yaml_content = textwrap.dedent("""\
            name: test
            description: Run tests
            prompt: Run tests {args}
            aliases: "t, tst"
        """)
        (commands_dir / "test.yaml").write_text(yaml_content)

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "test" in cmds
        assert "t" in cmds["test"]["aliases"]
        assert "tst" in cmds["test"]["aliases"]

    def test_yaml_missing_prompt_skipped(self, commands_dir):
        yaml_content = textwrap.dedent("""\
            name: broken
            description: No prompt
        """)
        (commands_dir / "broken.yaml").write_text(yaml_content)

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "broken" not in cmds

    def test_yaml_invalid_content_skipped(self, commands_dir):
        (commands_dir / "bad.yaml").write_text("not: a\n: valid: yaml: [")

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        # Should not crash, just skip
        assert "bad" not in cmds


class TestLoadMarkdownCommands:
    def test_load_single_md(self, commands_dir):
        content = "Write a good commit message for the staged changes, then commit.\n{args}"
        (commands_dir / "commit.md").write_text(content)

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "commit" in cmds
        assert "{args}" in cmds["commit"]["prompt"]
        assert "Custom command" in cmds["commit"]["description"]

    def test_md_name_sanitized(self, commands_dir):
        content = "Do something useful"
        (commands_dir / "My Cool Command.md").write_text(content)

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        # "My Cool Command" -> "mycoolcommand" (only a-z0-9_- kept)
        assert "mycoolcommand" in cmds

    def test_empty_md_skipped(self, commands_dir):
        (commands_dir / "empty.md").write_text("")

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "empty" not in cmds


class TestGetCustomCommand:
    def test_get_by_name(self, commands_dir):
        (commands_dir / "hello.md").write_text("Say hello to {args}")

        from hermes_cli.custom_commands import load_custom_commands, get_custom_command
        load_custom_commands()
        cmd = get_custom_command("hello")
        assert cmd is not None
        assert "hello" in cmd["prompt"]

    def test_get_by_alias(self, commands_dir):
        yaml_content = textwrap.dedent("""\
            name: deploy
            description: Deploy
            prompt: Deploy {args}
            aliases: [dp]
        """)
        (commands_dir / "deploy.yaml").write_text(yaml_content)

        from hermes_cli.custom_commands import load_custom_commands, get_custom_command
        load_custom_commands()
        cmd = get_custom_command("dp")
        assert cmd is not None
        assert cmd["_canonical_name"] == "deploy"

    def test_get_nonexistent(self, commands_dir):
        from hermes_cli.custom_commands import load_custom_commands, get_custom_command
        load_custom_commands()
        assert get_custom_command("nonexistent") is None


class TestResolveCustomCommand:
    def test_resolve_name(self, commands_dir):
        (commands_dir / "hello.md").write_text("Hello {args}")
        from hermes_cli.custom_commands import load_custom_commands, resolve_custom_command
        load_custom_commands()
        assert resolve_custom_command("hello") == "hello"

    def test_resolve_alias(self, commands_dir):
        yaml_content = textwrap.dedent("""\
            name: deploy
            description: Deploy
            prompt: Deploy {args}
            aliases: [dp]
        """)
        (commands_dir / "deploy.yaml").write_text(yaml_content)
        from hermes_cli.custom_commands import load_custom_commands, resolve_custom_command
        load_custom_commands()
        assert resolve_custom_command("dp") == "deploy"

    def test_resolve_nonexistent(self, commands_dir):
        from hermes_cli.custom_commands import load_custom_commands, resolve_custom_command
        load_custom_commands()
        assert resolve_custom_command("nope") is None


class TestExpandPrompt:
    def test_with_args(self):
        from hermes_cli.custom_commands import expand_prompt
        cmd = {"prompt": "Deploy to {args} with full config"}
        result = expand_prompt(cmd, "production")
        assert result == "Deploy to production with full config"

    def test_without_args(self):
        from hermes_cli.custom_commands import expand_prompt
        cmd = {"prompt": "Run all tests"}
        result = expand_prompt(cmd, "")
        assert result == "Run all tests"

    def test_no_template(self):
        from hermes_cli.custom_commands import expand_prompt
        cmd = {"prompt": "No placeholder here"}
        result = expand_prompt(cmd, "ignored")
        assert result == "No placeholder here"


class TestRegisterCustomCommands:
    def test_registers_into_command_registry(self, commands_dir, monkeypatch):
        (commands_dir / "hello.md").write_text("Hello world")
        from hermes_cli.custom_commands import load_custom_commands, register_custom_commands
        load_custom_commands()

        from hermes_cli.commands import COMMAND_REGISTRY
        count = register_custom_commands()
        assert count == 1
        assert "hello" in {c.name for c in COMMAND_REGISTRY}

        # Clean up: remove the custom command from registry
        from hermes_cli.commands import rebuild_lookups
        monkeypatch.setattr(
            "hermes_cli.commands.COMMAND_REGISTRY",
            [c for c in COMMAND_REGISTRY if c.name != "hello"],
        )
        rebuild_lookups()

    def test_no_duplicate_registration(self, commands_dir):
        (commands_dir / "hello.md").write_text("Hello world")
        from hermes_cli.custom_commands import load_custom_commands, register_custom_commands
        load_custom_commands()

        count1 = register_custom_commands()
        count2 = register_custom_commands()
        assert count1 == 1
        assert count2 == 0  # Already registered


class TestGetCommandsDir:
    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.custom_commands as _mod
        monkeypatch.setattr(_mod, "_commands_dir", None)

        from hermes_cli.custom_commands import get_commands_dir
        result = get_commands_dir()
        assert result == tmp_path / "commands"


class TestEdgeCases:
    def test_hidden_files_skipped(self, commands_dir):
        (commands_dir / ".hidden.md").write_text("Should be skipped")
        (commands_dir / "visible.md").write_text("Should be loaded")

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "visible" in cmds
        assert "hidden" not in cmds

    def test_directories_skipped(self, commands_dir):
        (commands_dir / "subdir").mkdir()
        (commands_dir / "subdir" / "nested.md").write_text("Nested")

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert "nested" not in cmds

    def test_non_yaml_md_files_skipped(self, commands_dir):
        (commands_dir / "script.sh").write_text("echo hello")
        (commands_dir / "data.json").write_text('{"key": "value"}')

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert len(cmds) == 0

    def test_empty_commands_dir(self, commands_dir):
        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert len(cmds) == 0

    def test_nonexistent_commands_dir(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "nope"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.custom_commands as _mod
        monkeypatch.setattr(_mod, "_commands_dir", nonexistent)
        monkeypatch.setattr(_mod, "_loaded", False)
        monkeypatch.setattr(_mod, "_custom_commands", {})

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert len(cmds) == 0

    def test_yaml_with_custom_category(self, commands_dir):
        yaml_content = textwrap.dedent("""\
            name: mycmd
            description: Custom category test
            prompt: Do stuff
            category: Deployment
        """)
        (commands_dir / "mycmd.yaml").write_text(yaml_content)

        from hermes_cli.custom_commands import load_custom_commands
        cmds = load_custom_commands()
        assert cmds["mycmd"]["category"] == "Deployment"
