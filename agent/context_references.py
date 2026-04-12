from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.model_metadata import estimate_tokens_rough

_QUOTED_REFERENCE_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|grep|map|url):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+))"
)
TRAILING_PUNCTUATION = ",.;!?"
_SENSITIVE_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh")
_SENSITIVE_HERMES_DIRS = (Path("skills") / ".hub",)
_SENSITIVE_HOME_FILES = (
    Path(".ssh") / "authorized_keys",
    Path(".ssh") / "id_rsa",
    Path(".ssh") / "id_ed25519",
    Path(".ssh") / "config",
    Path(".bashrc"),
    Path(".zshrc"),
    Path(".profile"),
    Path(".bash_profile"),
    Path(".zprofile"),
    Path(".netrc"),
    Path(".pgpass"),
    Path(".npmrc"),
    Path(".pypirc"),
)


@dataclass(frozen=True)
class ContextReference:
    raw: str
    kind: str
    target: str
    start: int
    end: int
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class ContextReferenceResult:
    message: str
    original_message: str
    references: list[ContextReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    expanded: bool = False
    blocked: bool = False


def parse_context_references(message: str) -> list[ContextReference]:
    refs: list[ContextReference] = []
    if not message:
        return refs

    for match in REFERENCE_PATTERN.finditer(message):
        simple = match.group("simple")
        if simple:
            refs.append(
                ContextReference(
                    raw=match.group(0),
                    kind=simple,
                    target="",
                    start=match.start(),
                    end=match.end(),
                )
            )
            continue

        kind = match.group("kind")
        value = _strip_trailing_punctuation(match.group("value") or "")
        line_start = None
        line_end = None
        target = _strip_reference_wrappers(value)

        if kind == "file":
            target, line_start, line_end = _parse_file_reference_value(value)

        refs.append(
            ContextReference(
                raw=match.group(0),
                kind=kind,
                target=target,
                start=match.start(),
                end=match.end(),
                line_start=line_start,
                line_end=line_end,
            )
        )

    return refs


def preprocess_context_references(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    coro = preprocess_context_references_async(
        message,
        cwd=cwd,
        context_length=context_length,
        url_fetcher=url_fetcher,
        allowed_root=allowed_root,
    )
    # Safe for both CLI (no loop) and gateway (loop already running).
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def preprocess_context_references_async(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)

    cwd_path = Path(cwd).expanduser().resolve()
    # Default to the current working directory so @ references cannot escape
    # the active workspace unless a caller explicitly widens the root.
    allowed_root_path = (
        Path(allowed_root).expanduser().resolve() if allowed_root is not None else cwd_path
    )
    warnings: list[str] = []
    blocks: list[str] = []
    injected_tokens = 0

    for ref in refs:
        warning, block = await _expand_reference(
            ref,
            cwd_path,
            url_fetcher=url_fetcher,
            allowed_root=allowed_root_path,
        )
        if warning:
            warnings.append(warning)
        if block:
            blocks.append(block)
            injected_tokens += estimate_tokens_rough(block)

    hard_limit = max(1, int(context_length * 0.50))
    soft_limit = max(1, int(context_length * 0.25))
    if injected_tokens > hard_limit:
        warnings.append(
            f"@ context injection refused: {injected_tokens} tokens exceeds the 50% hard limit ({hard_limit})."
        )
        return ContextReferenceResult(
            message=message,
            original_message=message,
            references=refs,
            warnings=warnings,
            injected_tokens=injected_tokens,
            expanded=False,
            blocked=True,
        )

    if injected_tokens > soft_limit:
        warnings.append(
            f"@ context injection warning: {injected_tokens} tokens exceeds the 25% soft limit ({soft_limit})."
        )

    stripped = _remove_reference_tokens(message, refs)
    final = stripped
    if warnings:
        final = f"{final}\n\n--- Context Warnings ---\n" + "\n".join(f"- {warning}" for warning in warnings)
    if blocks:
        final = f"{final}\n\n--- Attached Context ---\n\n" + "\n\n".join(blocks)

    return ContextReferenceResult(
        message=final.strip(),
        original_message=message,
        references=refs,
        warnings=warnings,
        injected_tokens=injected_tokens,
        expanded=bool(blocks or warnings),
        blocked=False,
    )


async def _expand_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    try:
        if ref.kind == "file":
            return _expand_file_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "folder":
            return _expand_folder_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "diff":
            return _expand_git_reference(ref, cwd, ["diff"], "git diff")
        if ref.kind == "staged":
            return _expand_git_reference(ref, cwd, ["diff", "--staged"], "git diff --staged")
        if ref.kind == "git":
            count = max(1, min(int(ref.target or "1"), 10))
            return _expand_git_reference(ref, cwd, ["log", f"-{count}", "-p"], f"git log -{count} -p")
        if ref.kind == "url":
            content = await _fetch_url_content(ref.target, url_fetcher=url_fetcher)
            if not content:
                return f"{ref.raw}: no content extracted", None
            return None, f"🌐 {ref.raw} ({estimate_tokens_rough(content)} tokens)\n{content}"
        if ref.kind == "map":
            return _expand_map_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "grep":
            return _expand_grep_reference(ref, cwd, allowed_root=allowed_root)
    except Exception as exc:
        return f"{ref.raw}: {exc}", None

    return f"{ref.raw}: unsupported reference type", None


def _expand_file_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: file not found", None
    if not path.is_file():
        return f"{ref.raw}: path is not a file", None
    if _is_binary_file(path):
        return f"{ref.raw}: binary files are not supported", None

    text = path.read_text(encoding="utf-8")
    if ref.line_start is not None:
        lines = text.splitlines()
        start_idx = max(ref.line_start - 1, 0)
        end_idx = min(ref.line_end or ref.line_start, len(lines))
        text = "\n".join(lines[start_idx:end_idx])

    lang = _code_fence_language(path)
    label = ref.raw
    return None, f"📄 {label} ({estimate_tokens_rough(text)} tokens)\n```{lang}\n{text}\n```"


def _expand_folder_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: folder not found", None
    if not path.is_dir():
        return f"{ref.raw}: path is not a folder", None

    listing = _build_folder_listing(path, cwd)
    return None, f"📁 {ref.raw} ({estimate_tokens_rough(listing)} tokens)\n{listing}"


def _format_func_sig(node: "ast.FunctionDef | ast.AsyncFunctionDef") -> str:
    """Format a function signature string from an AST node."""
    import ast as _ast
    args_parts: list[str] = []
    pos_only = getattr(node.args, "posonlyargs", [])
    all_args = list(pos_only) + node.args.args
    defaults = node.args.defaults
    # defaults align to the end of args
    n_defaults = len(defaults)
    n_args = len(all_args)
    for i, arg in enumerate(all_args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        s = arg.arg
        ann = arg.annotation
        if ann:
            s += f": {_ast.unparse(ann)}"
        default_idx = i - (n_args - n_defaults)
        if default_idx >= 0 and default_idx < n_defaults:
            s += f" = {_ast.unparse(defaults[default_idx])}"
        args_parts.append(s)
    # *args
    if node.args.vararg:
        s = f"*{node.args.vararg.arg}"
        ann = node.args.vararg.annotation
        if ann:
            s += f": {_ast.unparse(ann)}"
        args_parts.append(s)
    elif defaults and not any(
        a.arg in ("self", "cls") for a in all_args
    ):
        if not any(p.startswith("*") for p in args_parts):
            args_parts.append("*")
    # **kwargs
    if node.args.kwarg:
        s = f"**{node.args.kwarg.arg}"
        ann = node.args.kwarg.annotation
        if ann:
            s += f": {_ast.unparse(ann)}"
        args_parts.append(s)
    ret = ""
    if node.returns:
        ret = f" -> {_ast.unparse(node.returns)}"
    return f"({', '.join(args_parts)}){ret}"


def _format_class_sig(node: "ast.ClassDef") -> str:
    """Format class bases for display."""
    import ast as _ast
    if not node.bases:
        return ""
    bases = [_ast.unparse(b) for b in node.bases]
    return f"({', '.join(bases)})"


# ---------------------------------------------------------------------------
# Regex-based symbol extraction for non-Python languages
# ---------------------------------------------------------------------------

_JS_TS_PATTERNS = [
    # class Name extends Base
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"),
    # function name(args)
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"),
    # Shorthand method: name(args) { (inside class body)
    re.compile(r"^\s+(?:async\s+)?(?!if|for|while|switch|catch)(\w+)\s*\(([^)]*)\)\s*{"),
    # const/let/var name = (args) => ...
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*[^=]+?)?\s*=>"),
    # const/let/var name = function(args)
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(([^)]*)\)"),
    # interface Name
    re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)"),
    # type Name = ...
    re.compile(r"^\s*(?:export\s+)?type\s+(\w+)"),
]

_GO_PATTERNS = [
    # func Name(args) or func (recv) Name(args)
    re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)"),
    # type Name struct/interface
    re.compile(r"^\s*type\s+(\w+)\s+(struct|interface)"),
]

_RUST_PATTERNS = [
    # fn name(args) -> RetType
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)"),
    # struct/enum/trait Name
    re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)"),
    # impl Name
    re.compile(r"^\s*impl\s+(?:<[^>]*>\s+)?(\w+)"),
]

_LANG_CONFIG = {
    ".js": ("JavaScript", _JS_TS_PATTERNS),
    ".jsx": ("JavaScript", _JS_TS_PATTERNS),
    ".mjs": ("JavaScript", _JS_TS_PATTERNS),
    ".cjs": ("JavaScript", _JS_TS_PATTERNS),
    ".ts": ("TypeScript", _JS_TS_PATTERNS),
    ".tsx": ("TypeScript", _JS_TS_PATTERNS),
    ".go": ("Go", _GO_PATTERNS),
    ".rs": ("Rust", _RUST_PATTERNS),
}


def _extract_regex_symbols(source: str, lang: str, patterns: list) -> list[str]:
    """Extract symbol names from non-Python source using regex patterns."""
    entries: list[str] = []
    brace_depth = 0
    in_class = False
    class_indent = 0

    for line in source.splitlines():
        stripped = line.rstrip()
        indent = len(stripped) - len(stripped.lstrip())

        # Track brace depth for JS/TS class body detection
        if lang in ("JavaScript", "TypeScript"):
            brace_depth += stripped.count("{") - stripped.count("}")

        matched = False
        for pat in patterns:
            m = pat.match(stripped)
            if not m:
                continue
            name = m.group(1)
            args = m.group(2).strip() if len(m.groups()) > 1 else ""

            if lang in ("JavaScript", "TypeScript"):
                # Check for class keyword anywhere before the name
                if re.search(r"\bclass\s+" + re.escape(name), stripped):
                    entries.append(f"  class {name}")
                    in_class = True
                    class_indent = indent
                elif "interface " in stripped and re.search(r"\binterface\s+" + re.escape(name), stripped):
                    entries.append(f"  interface {name}")
                elif "type " in stripped and re.search(r"\btype\s+" + re.escape(name), stripped):
                    entries.append(f"  type {name}")
                else:
                    # Function or method
                    is_method = in_class and indent > class_indent
                    prefix = "    " if is_method else "  "
                    entries.append(f"{prefix}fn {name}({args})")
            elif lang == "Go":
                if "type " in stripped:
                    kind = m.group(2) if len(m.groups()) > 1 else ""
                    entries.append(f"  type {name} {kind}".rstrip())
                else:
                    entries.append(f"  fn {name}({args})")
            elif lang == "Rust":
                if "impl " in stripped:
                    entries.append(f"  impl {name}")
                elif "trait " in stripped:
                    entries.append(f"  trait {name}")
                elif "enum " in stripped:
                    entries.append(f"  enum {name}")
                elif re.search(r"\bstruct\s+" + re.escape(name), stripped):
                    entries.append(f"  struct {name}")
                else:
                    entries.append(f"  fn {name}({args})")

            matched = True
            break

        # Reset class tracking when brace depth returns to 0
        if lang in ("JavaScript", "TypeScript") and brace_depth <= 0:
            in_class = False
            brace_depth = 0

    return entries



def _expand_map_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """Generate a compact code-structure map of a directory.

    Scans Python files via AST (classes, functions, signatures) and
    JavaScript/TypeScript, Go, Rust files via regex-based extraction.
    Produces a compact repo-map similar to Aider's ``--map-tokens`` feature.
    """
    import ast as _ast

    # Resolve the directory to scan
    raw_target = ref.target or "."
    path = _resolve_path(cwd, raw_target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: path not found", None
    if path.is_file():
        path = path.parent

    # Collect source files (skip venv, __pycache__, .git, node_modules)
    _SKIP_DIRS = {
        "venv", ".venv", "__pycache__", ".git", "node_modules",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", "egg-info",
    }
    _PYTHON_EXT = ".py"
    _ALL_EXTS = {_PYTHON_EXT} | set(_LANG_CONFIG.keys())

    source_files: list[tuple[Path, str]] = []  # (path, ext)
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in sorted(files):
            ext = os.path.splitext(f)[1]
            if ext in _ALL_EXTS:
                source_files.append((Path(root) / f, ext))
        if len(source_files) >= 80:
            break

    if not source_files:
        return f"{ref.raw}: no source files found", None

    lines: list[str] = []
    total_symbols = 0

    for fpath, ext in source_files:
        rel = fpath.relative_to(cwd) if fpath.is_relative_to(cwd) else fpath.relative_to(path)
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        file_entries: list[str] = []

        if ext == _PYTHON_EXT:
            # Use AST for Python files
            try:
                tree = _ast.parse(source, filename=str(fpath))
            except SyntaxError:
                continue

            for node in _ast.iter_child_nodes(tree):
                if isinstance(node, _ast.ClassDef):
                    sig = _format_class_sig(node)
                    file_entries.append(f"  class {node.name}{sig}")
                    total_symbols += 1
                    for item in node.body:
                        if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                            mname = item.name
                            if mname.startswith("_") and mname != "__init__":
                                continue
                            msig = _format_func_sig(item)
                            file_entries.append(f"    def {mname}{msig}")
                            total_symbols += 1
                elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    sig = _format_func_sig(node)
                    file_entries.append(f"  def {node.name}{sig}")
                    total_symbols += 1
        else:
            # Use regex-based extraction for other languages
            lang, patterns = _LANG_CONFIG[ext]
            file_entries = _extract_regex_symbols(source, lang, patterns)
            total_symbols += len(file_entries)

        if file_entries:
            lines.append(f"{rel}")
            lines.extend(file_entries)

    if not lines:
        return f"{ref.raw}: no symbols found in {len(source_files)} files", None

    content = "\n".join(lines)
    label = f"@map:{raw_target}" if raw_target != "." else "@map:."
    n_py = sum(1 for _, e in source_files if e == _PYTHON_EXT)
    n_other = len(source_files) - n_py
    if n_other:
        lang_note = f"{n_py} Python + {n_other} other files"
    else:
        lang_note = f"{len(source_files)} files"
    return None, f"\U0001f5fa  {label} \u2014 {total_symbols} symbols from {lang_note} ({estimate_tokens_rough(content)} tokens)\n{content}"
def _expand_grep_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """Search files for a regex pattern using ``grep`` and return matching lines.

    The target is a pattern (required) optionally followed by ``:`` and a path
    prefix, e.g. ``@grep:"TODO|FIXME"`` or ``@grep:"class Foo:src/"``.
    Results are capped at 200 matches across a maximum of 50 files.
    """
    # Parse target: "pattern" or "pattern:path_prefix"
    raw_target = ref.target or ""
    if not raw_target:
        return f"{ref.raw}: no search pattern provided", None

    # Split on the last colon that's followed by a path-like string
    # Heuristic: if there's a colon and the part after looks like a path, use it
    path_prefix = "."
    pattern = raw_target
    if ":" in raw_target:
        parts = raw_target.rsplit(":", 1)
        candidate_path = parts[1].strip()
        candidate_pattern = parts[0].strip()
        if candidate_path and candidate_pattern:
            test_path = Path(os.path.expanduser(candidate_path))
            if not test_path.is_absolute():
                test_path = cwd / test_path
            if test_path.exists():
                pattern = candidate_pattern
                path_prefix = candidate_path

    # Resolve and validate the search path
    search_path = _resolve_path(cwd, path_prefix, allowed_root=allowed_root)
    _ensure_reference_path_allowed(search_path)
    if not search_path.exists():
        return f"{ref.raw}: path not found: {path_prefix}", None

    # Build grep command with extended regex
    skip_dirs = [
        "venv", ".venv", "__pycache__", ".git", "node_modules",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        ".hg", ".svn", "vendor", ".cargo", "target",
    ]
    grep_args = [
        "grep", "-rn", "-E", "--binary-files=without-match",
    ]
    for d in skip_dirs:
        grep_args.append("--exclude-dir=" + d)
    grep_args.extend([
        "--max-count=50",
        "--color=never",
        "--",
        pattern,
        str(search_path),
    ])

    try:
        result = subprocess.run(
            grep_args,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return f"{ref.raw}: grep timed out (15s)", None
    except FileNotFoundError:
        return f"{ref.raw}: grep not available on this system", None

    # grep returns 1 when no matches found
    if result.returncode not in (0, 1):
        stderr = (result.stderr or "").strip()
        if "unrecognized option" in stderr or "invalid" in stderr.lower():
            # Fall back to basic grep without --exclude-dir (e.g. BusyBox grep)
            grep_args_fallback = [
                "grep", "-rn", "-E", "--binary-files=without-match",
                "--max-count=50", "--color=never",
                "--", pattern, str(search_path),
            ]
            try:
                result = subprocess.run(
                    grep_args_fallback,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except subprocess.TimeoutExpired:
                return f"{ref.raw}: grep timed out (15s)", None
            except FileNotFoundError:
                return f"{ref.raw}: grep not available on this system", None
        else:
            return f"{ref.raw}: {stderr or 'grep failed'}", None

    output = (result.stdout or "").strip()
    if not output:
        return f"{ref.raw}: no matches for pattern", None

    # Limit output to 200 lines max
    lines_found = output.splitlines()
    truncated = False
    if len(lines_found) > 200:
        lines_found = lines_found[:200]
        truncated = True

    content_text = "\n".join(lines_found)
    if truncated:
        content_text += "\n... (truncated, showing first 200 of more matches)"

    label = "@grep:" + raw_target
    n_matches = len(lines_found)
    tokens = estimate_tokens_rough(content_text)
    return None, "\U0001f50d " + label + " (" + str(n_matches) + " matches, " + str(tokens) + " tokens)\n" + content_text


def _expand_git_reference(
    ref: ContextReference,
    cwd: Path,
    args: list[str],
    label: str,
) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"{ref.raw}: git command timed out (30s)", None
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "git command failed"
        return f"{ref.raw}: {stderr}", None
    content = result.stdout.strip()
    if not content:
        content = "(no output)"
    return None, f"🧾 {label} ({estimate_tokens_rough(content)} tokens)\n```diff\n{content}\n```"


async def _fetch_url_content(
    url: str,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> str:
    fetcher = url_fetcher or _default_url_fetcher
    content = fetcher(url)
    if inspect.isawaitable(content):
        content = await content
    return str(content or "").strip()


async def _default_url_fetcher(url: str) -> str:
    from tools.web_tools import web_extract_tool

    raw = await web_extract_tool([url], format="markdown", use_llm_processing=True)
    payload = json.loads(raw)
    docs = payload.get("data", {}).get("documents", [])
    if not docs:
        return ""
    doc = docs[0]
    return str(doc.get("content") or doc.get("raw_content") or "").strip()


def _resolve_path(cwd: Path, target: str, *, allowed_root: Path | None = None) -> Path:
    path = Path(os.path.expanduser(target))
    if not path.is_absolute():
        path = cwd / path
    resolved = path.resolve()
    if allowed_root is not None:
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("path is outside the allowed workspace") from exc
    return resolved


def _ensure_reference_path_allowed(path: Path) -> None:
    from hermes_constants import get_hermes_home
    home = Path(os.path.expanduser("~")).resolve()
    hermes_home = get_hermes_home().resolve()

    blocked_exact = {home / rel for rel in _SENSITIVE_HOME_FILES}
    blocked_exact.add(hermes_home / ".env")
    blocked_dirs = [home / rel for rel in _SENSITIVE_HOME_DIRS]
    blocked_dirs.extend(hermes_home / rel for rel in _SENSITIVE_HERMES_DIRS)

    if path in blocked_exact:
        raise ValueError("path is a sensitive credential file and cannot be attached")

    for blocked_dir in blocked_dirs:
        try:
            path.relative_to(blocked_dir)
        except ValueError:
            continue
        raise ValueError("path is a sensitive credential or internal Hermes path and cannot be attached")


def _strip_trailing_punctuation(value: str) -> str:
    stripped = value.rstrip(TRAILING_PUNCTUATION)
    while stripped.endswith((")", "]", "}")):
        closer = stripped[-1]
        opener = {")": "(", "]": "[", "}": "{"}[closer]
        if stripped.count(closer) > stripped.count(opener):
            stripped = stripped[:-1]
            continue
        break
    return stripped


def _strip_reference_wrappers(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
        return value[1:-1]
    return value


def _parse_file_reference_value(value: str) -> tuple[str, int | None, int | None]:
    quoted_match = re.match(
        r'^(?P<quote>`|"|\')(?P<path>.+?)(?P=quote)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$',
        value,
    )
    if quoted_match:
        line_start = quoted_match.group("start")
        line_end = quoted_match.group("end")
        return (
            quoted_match.group("path"),
            int(line_start) if line_start is not None else None,
            int(line_end or line_start) if line_start is not None else None,
        )

    range_match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if range_match:
        line_start = int(range_match.group("start"))
        return (
            range_match.group("path"),
            line_start,
            int(range_match.group("end") or range_match.group("start")),
        )

    return _strip_reference_wrappers(value), None, None


def _remove_reference_tokens(message: str, refs: list[ContextReference]) -> str:
    pieces: list[str] = []
    cursor = 0
    for ref in refs:
        pieces.append(message[cursor:ref.start])
        cursor = ref.end
    pieces.append(message[cursor:])
    text = "".join(pieces)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _is_binary_file(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    if mime and not mime.startswith("text/") and not any(
        path.name.endswith(ext) for ext in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts")
    ):
        return True
    chunk = path.read_bytes()[:4096]
    return b"\x00" in chunk


def _build_folder_listing(path: Path, cwd: Path, limit: int = 200) -> str:
    lines = [f"{path.relative_to(cwd)}/"]
    entries = _iter_visible_entries(path, cwd, limit=limit)
    for entry in entries:
        rel = entry.relative_to(cwd)
        indent = "  " * max(len(rel.parts) - len(path.relative_to(cwd).parts) - 1, 0)
        if entry.is_dir():
            lines.append(f"{indent}- {entry.name}/")
        else:
            meta = _file_metadata(entry)
            lines.append(f"{indent}- {entry.name} ({meta})")
    if len(entries) >= limit:
        lines.append("- ...")
    return "\n".join(lines)


def _iter_visible_entries(path: Path, cwd: Path, limit: int) -> list[Path]:
    rg_entries = _rg_files(path, cwd, limit=limit)
    if rg_entries is not None:
        output: list[Path] = []
        seen_dirs: set[Path] = set()
        for rel in rg_entries:
            full = cwd / rel
            for parent in full.parents:
                if parent == cwd or parent in seen_dirs or path not in {parent, *parent.parents}:
                    continue
                seen_dirs.add(parent)
                output.append(parent)
            output.append(full)
        return sorted({p for p in output if p.exists()}, key=lambda p: (not p.is_dir(), str(p)))

    output = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        files = sorted(f for f in files if not f.startswith("."))
        root_path = Path(root)
        for d in dirs:
            output.append(root_path / d)
            if len(output) >= limit:
                return output
        for f in files:
            output.append(root_path / f)
            if len(output) >= limit:
                return output
    return output


def _rg_files(path: Path, cwd: Path, limit: int) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["rg", "--files", str(path.relative_to(cwd))],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return files[:limit]


def _file_metadata(path: Path) -> str:
    if _is_binary_file(path):
        return f"{path.stat().st_size} bytes"
    try:
        line_count = path.read_text(encoding="utf-8").count("\n") + 1
    except Exception:
        return f"{path.stat().st_size} bytes"
    return f"{line_count} lines"


def _code_fence_language(path: Path) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
    }
    return mapping.get(path.suffix.lower(), "")
