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

REFERENCE_PATTERN = re.compile(
    r"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url|gh):(?P<value>\S+))"
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
        target = value

        if kind == "file":
            range_match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
            if range_match:
                target = range_match.group("path")
                line_start = int(range_match.group("start"))
                line_end = int(range_match.group("end") or range_match.group("start"))

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
        if ref.kind == "gh":
            return _expand_gh_reference(ref, cwd)
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


# ── @gh: GitHub context reference ─────────────────────────────────────

def _gh_run(args: list[str], cwd: Path, timeout: int = 15) -> tuple[int, str]:
    """Run a gh CLI command and return (exit_code, output)."""
    import subprocess
    try:
        r = subprocess.run(
            ["gh"] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd), stdin=subprocess.DEVNULL,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return -1, "gh CLI not installed (https://cli.github.com)"
    except subprocess.TimeoutExpired:
        return -2, f"gh {' '.join(args)} timed out after {timeout}s"


def _expand_gh_reference(
    ref: ContextReference,
    cwd: Path,
) -> tuple[str | None, str | None]:
    """Expand @gh: — inject GitHub information via the ``gh`` CLI.

    Sub-commands:

    ==============  ================================================
    ``@gh:status``  Current PR/issue status for the branch
    ``@gh:pr``      Open PRs for the current repo
    ``@gh:pr:N``    Details of PR #N (title, state, labels, mergeable)
    ``@gh:issue``   Open issues (first 15)
    ``@gh:issue:N`` Details of issue #N
    ``@gh:repo``    Repository metadata (stars, forks, description)
    ``@gh:ci``      Recent CI/CD workflow runs
    ``@gh:release`` Latest release tag and name
    ``@gh:auth``    Current ``gh auth status``
    ``@gh:N``       Short for ``@gh:pr:N``
    ==============  ================================================

    Requires the ``gh`` CLI to be installed and authenticated.
    """
    target = ref.target

    if target in ("status", "st"):
        return _gh_status(cwd, ref)
    if target == "auth":
        return _gh_auth(ref)
    if target == "repo":
        return _gh_repo(cwd, ref)
    if target in ("pr", "prs"):
        return _gh_pr_list(cwd, ref)
    if target.startswith("pr:"):
        return _gh_pr_detail(cwd, ref, target[3:])
    if target in ("issue", "issues"):
        return _gh_issue_list(cwd, ref)
    if target.startswith("issue:"):
        return _gh_issue_detail(cwd, ref, target[6:])
    if target in ("ci", "actions", "runs"):
        return _gh_ci(cwd, ref)
    if target in ("release", "releases"):
        return _gh_release(cwd, ref)
    # Bare number -> treat as PR
    if target.isdigit():
        return _gh_pr_detail(cwd, ref, target)

    return (
        f"{ref.raw}: unknown sub-command {target!r}. "
        f"Try: status, pr, pr:N, issue, issue:N, repo, ci, release, auth",
        None,
    )


def _gh_status(cwd: Path, ref: ContextReference) -> tuple[str | None, str | None]:
    """Show PR/issue status for the current branch."""
    code, out = _gh_run(["repo", "view", "--json", "nameWithOwner"], cwd, timeout=10)
    if code != 0:
        return f"{ref.raw}: not in a GitHub repo (or gh not authenticated)", None

    import json
    try:
        repo_data = json.loads(out.strip())
        repo_name = repo_data.get("nameWithOwner", "unknown")
    except (json.JSONDecodeError, KeyError):
        repo_name = "unknown"

    parts = [f"\U0001f4e6 Repo: {repo_name}"]

    # Current branch
    import subprocess
    branch = None
    try:
        br = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(cwd),
        )
        if br.returncode == 0:
            branch = br.stdout.strip()
            parts.append(f"\U0001f33f Branch: {branch}")
    except Exception:
        pass

    # Associated PR for current branch
    if branch:
        pr_code, pr_out = _gh_run(
            ["pr", "list", "--head", branch, "--json", "number,title,state,url", "--limit", "1"],
            cwd,
        )
        if pr_code == 0 and pr_out.strip() and pr_out.strip() != "[]":
            try:
                prs = json.loads(pr_out.strip())
                if prs:
                    p = prs[0]
                    parts.append(
                        f"\U0001f517 PR #{p.get('number', '?')}: "
                        f"{p.get('title', '')} [{p.get('state', '')}]"
                    )
                    parts.append(f"   {p.get('url', '')}")
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            parts.append("\U0001f517 No associated PR for this branch")

    output = "\n".join(parts)
    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} ({tok} tokens)\n{output}"


def _gh_auth(ref: ContextReference) -> tuple[str | None, str | None]:
    """Show gh auth status."""
    code, out = _gh_run(["auth", "status"], Path("."), timeout=10)
    if code == -1:
        return f"{ref.raw}: gh CLI not installed", None
    output = out.strip() if out.strip() else "(no output)"
    tok = estimate_tokens_rough(output)
    return None, f"\U0001f511 {ref.raw} ({tok} tokens)\n```\n{output}\n```"


def _gh_repo(cwd: Path, ref: ContextReference) -> tuple[str | None, str | None]:
    """Show repo metadata."""
    code, out = _gh_run(
        ["repo", "view", "--json",
         "nameWithOwner,description,stargazerCount,forkCount,isPrivate,defaultBranchRef"],
        cwd,
    )
    if code != 0:
        return f"{ref.raw}: failed to fetch repo info — {out.strip()}", None
    import json
    try:
        d = json.loads(out.strip())
        vis = "\U0001f512 Private" if d.get("isPrivate") else "\U0001f310 Public"
        lines = [
            f"\U0001f4e6 {d.get('nameWithOwner', '?')}",
            f"   {vis}",
        ]
        if d.get("description"):
            lines.append(f"   {d['description']}")
        lines.append(f"   \u2b50 {d.get('stargazerCount', 0)}  \U0001f374 {d.get('forkCount', 0)}")
        brf = d.get("defaultBranchRef")
        if brf and isinstance(brf, dict) and brf.get("name"):
            lines.append(f"   Default branch: {brf['name']}")
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} ({tok} tokens)\n{output}"


def _gh_pr_list(cwd: Path, ref: ContextReference) -> tuple[str | None, str | None]:
    """List open PRs."""
    code, out = _gh_run(
        ["pr", "list", "--json", "number,title,author,state", "--limit", "15"], cwd,
    )
    if code != 0:
        return f"{ref.raw}: failed to list PRs — {out.strip()}", None
    import json
    prs: list = []
    try:
        prs = json.loads(out.strip())
        if not prs:
            return None, f"\U0001f418 {ref.raw}: no open PRs"
        lines = []
        for p in prs:
            author = p.get("author", {}).get("login", "?")
            lines.append(f"  #{p['number']} {p['title']} (@{author}) [{p.get('state', '')}]")
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} — {len(prs)} PRs ({tok} tokens)\n{output}"


def _gh_pr_detail(cwd: Path, ref: ContextReference, num: str) -> tuple[str | None, str | None]:
    """Show details of a specific PR."""
    code, out = _gh_run(
        ["pr", "view", num, "--json",
         "number,title,state,author,labels,mergeable,additions,deletions,changedFiles,url,body"],
        cwd,
    )
    if code != 0:
        return f"{ref.raw}: PR #{num} not found — {out.strip()}", None
    import json
    try:
        d = json.loads(out.strip())
        author = d.get("author", {}).get("login", "?")
        labels = ", ".join(l.get("name", "") for l in d.get("labels", []))
        lines = [
            f"Pull Request #{d.get('number', num)}",
            f"Title: {d.get('title', '')}",
            f"State: {d.get('state', '')}  Author: @{author}",
        ]
        if labels:
            lines.append(f"Labels: {labels}")
        if d.get("mergeable") is not None:
            lines.append(f"Mergeable: {d['mergeable']}")
        lines.append(
            f"Changes: +{d.get('additions', 0)}/-{d.get('deletions', 0)} "
            f"across {d.get('changedFiles', 0)} files"
        )
        lines.append(f"URL: {d.get('url', '')}")
        body = (d.get("body") or "").strip()
        if body:
            body_lines = body.split("\n")
            if len(body_lines) > 20:
                body = "\n".join(body_lines[:20]) + f"\n... ({len(body_lines) - 20} more lines)"
            lines.append(f"\n{body}")
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} ({tok} tokens)\n{output}"


def _gh_issue_list(cwd: Path, ref: ContextReference) -> tuple[str | None, str | None]:
    """List open issues."""
    code, out = _gh_run(
        ["issue", "list", "--json", "number,title,author,labels,state", "--limit", "15"],
        cwd,
    )
    if code != 0:
        return f"{ref.raw}: failed to list issues — {out.strip()}", None
    import json
    issues: list = []
    try:
        issues = json.loads(out.strip())
        if not issues:
            return None, f"\U0001f418 {ref.raw}: no open issues"
        lines = []
        for i in issues:
            author = i.get("author", {}).get("login", "?")
            lbls = ", ".join(l.get("name", "") for l in i.get("labels", []))
            label_str = f" [{lbls}]" if lbls else ""
            lines.append(f"  #{i['number']} {i['title']} (@{author}){label_str}")
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} — {len(issues)} issues ({tok} tokens)\n{output}"


def _gh_issue_detail(cwd: Path, ref: ContextReference, num: str) -> tuple[str | None, str | None]:
    """Show details of a specific issue."""
    code, out = _gh_run(
        ["issue", "view", num, "--json",
         "number,title,state,author,labels,assignees,url,body"],
        cwd,
    )
    if code != 0:
        return f"{ref.raw}: issue #{num} not found — {out.strip()}", None
    import json
    try:
        d = json.loads(out.strip())
        author = d.get("author", {}).get("login", "?")
        labels = ", ".join(l.get("name", "") for l in d.get("labels", []))
        assignees = ", ".join(a.get("login", "") for a in d.get("assignees", []))
        lines = [
            f"Issue #{d.get('number', num)}",
            f"Title: {d.get('title', '')}",
            f"State: {d.get('state', '')}  Author: @{author}",
        ]
        if labels:
            lines.append(f"Labels: {labels}")
        if assignees:
            lines.append(f"Assignees: {assignees}")
        lines.append(f"URL: {d.get('url', '')}")
        body = (d.get("body") or "").strip()
        if body:
            body_lines = body.split("\n")
            if len(body_lines) > 20:
                body = "\n".join(body_lines[:20]) + f"\n... ({len(body_lines) - 20} more lines)"
            lines.append(f"\n{body}")
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} ({tok} tokens)\n{output}"


def _gh_ci(cwd: Path, ref: ContextReference) -> tuple[str | None, str | None]:
    """Show recent CI/CD runs."""
    code, out = _gh_run(
        ["run", "list", "--json", "name,status,conclusion,headBranch,databaseId", "--limit", "10"],
        cwd,
    )
    if code != 0:
        return f"{ref.raw}: failed to list CI runs — {out.strip()}", None
    import json
    runs: list = []
    try:
        runs = json.loads(out.strip())
        if not runs:
            return None, f"\U0001f418 {ref.raw}: no recent CI runs"
        _ICONS = {
            "success": "\u2705", "failure": "\u274c",
            "cancelled": "\u26d4", "timed_out": "\u23f1\ufe0f",
        }
        lines = []
        for r in runs:
            conclusion = r.get("conclusion") or r.get("status", "in_progress")
            icon = _ICONS.get(conclusion, "\U0001f504" if conclusion == "in_progress" else "\u2753")
            lines.append(
                f"  {icon} {r.get('name', '?')} \u2192 {conclusion} "
                f"({r.get('headBranch', '?')}) [{r.get('databaseId', '')}]"
            )
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} — recent CI runs ({tok} tokens)\n{output}"


def _gh_release(cwd: Path, ref: ContextReference) -> tuple[str | None, str | None]:
    """Show latest releases."""
    code, out = _gh_run(
        ["release", "list", "--json", "tagName,name,isLatest", "--limit", "5"],
        cwd,
    )
    if code != 0:
        return f"{ref.raw}: failed to list releases — {out.strip()}", None
    import json
    try:
        releases = json.loads(out.strip())
        if not releases:
            return None, f"\U0001f418 {ref.raw}: no releases found"
        lines = []
        for r in releases:
            latest = " (latest)" if r.get("isLatest") else ""
            lines.append(f"  \U0001f3f7\ufe0f {r.get('tagName', '?')} — {r.get('name', '')}{latest}")
        output = "\n".join(lines)
    except (json.JSONDecodeError, KeyError):
        output = out.strip()

    tok = estimate_tokens_rough(output)
    return None, f"\U0001f418 {ref.raw} ({tok} tokens)\n{output}"
