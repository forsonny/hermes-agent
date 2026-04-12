from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Hermes Tests")
    _git(repo, "config", "user.email", "tests@example.com")

    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text(
        "def alpha():\n"
        "    return 'a'\n\n"
        "def beta():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary")

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    (repo / "src" / "main.py").write_text(
        "def alpha():\n"
        "    return 'changed'\n\n"
        "def beta():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/helper.py")
    return repo


def test_parse_typed_references_ignores_emails_and_handles():
    from agent.context_references import parse_context_references

    message = (
        "email me at user@example.com and ping @teammate "
        "but include @file:src/main.py:1-2 plus @diff and @git:2 "
        "and @url:https://example.com/docs"
    )

    refs = parse_context_references(message)

    assert [ref.kind for ref in refs] == ["file", "diff", "git", "url"]
    assert refs[0].target == "src/main.py"
    assert refs[0].line_start == 1
    assert refs[0].line_end == 2
    assert refs[2].target == "2"


def test_parse_references_strips_trailing_punctuation():
    from agent.context_references import parse_context_references

    refs = parse_context_references(
        "review @file:README.md, then see (@url:https://example.com/docs)."
    )

    assert [ref.kind for ref in refs] == ["file", "url"]
    assert refs[0].target == "README.md"
    assert refs[1].target == "https://example.com/docs"


def test_parse_quoted_references_with_spaces_and_preserve_unquoted_ranges():
    from agent.context_references import parse_context_references

    refs = parse_context_references(
        'review @file:"C:\\Users\\Simba\\My Project\\main.py":7-9 '
        'and @folder:"docs and specs" plus @file:src/main.py:1-2'
    )

    assert [ref.kind for ref in refs] == ["file", "folder", "file"]
    assert refs[0].target == r"C:\Users\Simba\My Project\main.py"
    assert refs[0].line_start == 7
    assert refs[0].line_end == 9
    assert refs[1].target == "docs and specs"
    assert refs[2].target == "src/main.py"
    assert refs[2].line_start == 1
    assert refs[2].line_end == 2


def test_expand_file_range_and_folder_listing(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Review @file:src/main.py:1-2 and @folder:src/",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "Review and" in result.message
    assert "Review @file:src/main.py:1-2" not in result.message
    assert "--- Attached Context ---" in result.message
    assert "def alpha():" in result.message
    assert "return 'changed'" in result.message
    assert "def beta():" not in result.message
    assert "src/" in result.message
    assert "main.py" in result.message
    assert "helper.py" in result.message
    assert result.injected_tokens > 0
    assert not result.warnings


def test_expand_quoted_file_reference_with_spaces(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    workspace = tmp_path / "repo"
    folder = workspace / "docs and specs"
    folder.mkdir(parents=True)
    file_path = folder / "release notes.txt"
    file_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    result = preprocess_context_references(
        'Review @file:"docs and specs/release notes.txt":2-3',
        cwd=workspace,
        context_length=100_000,
    )

    assert result.expanded
    assert result.message.startswith("Review")
    assert "line 1" not in result.message
    assert "line 2" in result.message
    assert "line 3" in result.message
    assert "release notes.txt" in result.message
    assert not result.warnings


def test_expand_git_diff_staged_and_log(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Inspect @diff and @staged and @git:1",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "git diff" in result.message
    assert "git diff --staged" in result.message
    assert "git log -1 -p" in result.message
    assert "initial" in result.message
    assert "return 'changed'" in result.message
    assert "VALUE = 2" in result.message


def test_binary_and_missing_files_become_warnings(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Check @file:blob.bin and @file:nope.txt",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert len(result.warnings) == 2
    assert "binary" in result.message.lower()
    assert "not found" in result.message.lower()


def test_soft_budget_warns_and_hard_budget_refuses(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    soft = preprocess_context_references(
        "Check @file:src/main.py",
        cwd=sample_repo,
        context_length=100,
    )
    assert soft.expanded
    assert any("25%" in warning for warning in soft.warnings)

    hard = preprocess_context_references(
        "Check @file:src/main.py and @file:README.md",
        cwd=sample_repo,
        context_length=20,
    )
    assert not hard.expanded
    assert hard.blocked
    assert "@file:src/main.py" in hard.message
    assert any("50%" in warning for warning in hard.warnings)


@pytest.mark.asyncio
async def test_async_url_expansion_uses_fetcher(sample_repo: Path):
    from agent.context_references import preprocess_context_references_async

    async def fake_fetch(url: str) -> str:
        assert url == "https://example.com/spec"
        return "# Spec\n\nImportant details."

    result = await preprocess_context_references_async(
        "Use @url:https://example.com/spec",
        cwd=sample_repo,
        context_length=100_000,
        url_fetcher=fake_fetch,
    )

    assert result.expanded
    assert "Important details." in result.message
    assert result.injected_tokens > 0


def test_sync_url_expansion_uses_async_fetcher(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    async def fake_fetch(url: str) -> str:
        await asyncio.sleep(0)
        return f"Content for {url}"

    result = preprocess_context_references(
        "Use @url:https://example.com/spec",
        cwd=sample_repo,
        context_length=100_000,
        url_fetcher=fake_fetch,
    )

    assert result.expanded
    assert "Content for https://example.com/spec" in result.message


def test_restricts_paths_to_allowed_root(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("inside\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside\n", encoding="utf-8")

    result = preprocess_context_references(
        "read @file:../secret.txt and @file:notes.txt",
        cwd=workspace,
        context_length=100_000,
        allowed_root=workspace,
    )

    assert result.expanded
    assert "```\noutside\n```" not in result.message
    assert "inside" in result.message
    assert any("outside the allowed workspace" in warning for warning in result.warnings)


def test_defaults_allowed_root_to_cwd(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("outside\n", encoding="utf-8")

    result = preprocess_context_references(
        f"read @file:{secret}",
        cwd=workspace,
        context_length=100_000,
    )

    assert result.expanded
    assert "```\noutside\n```" not in result.message
    assert any("outside the allowed workspace" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_blocks_sensitive_home_and_hermes_paths(tmp_path: Path, monkeypatch):
    from agent.context_references import preprocess_context_references_async

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    hermes_env = tmp_path / ".hermes" / ".env"
    hermes_env.parent.mkdir(parents=True)
    hermes_env.write_text("API_KEY=super-secret\n", encoding="utf-8")

    ssh_key = tmp_path / ".ssh" / "id_rsa"
    ssh_key.parent.mkdir(parents=True)
    ssh_key.write_text("PRIVATE-KEY\n", encoding="utf-8")

    result = await preprocess_context_references_async(
        "read @file:.hermes/.env and @file:.ssh/id_rsa",
        cwd=tmp_path,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert "API_KEY=super-secret" not in result.message
    assert "PRIVATE-KEY" not in result.message
    assert any("sensitive credential" in warning for warning in result.warnings)

def test_parse_map_reference():
    from agent.context_references import parse_context_references

    refs = parse_context_references("show @map:. and @map:src/")
    assert len(refs) == 2
    assert refs[0].kind == "map"
    assert refs[0].target == ""
    assert refs[1].kind == "map"
    assert refs[1].target == "src/"


def test_expand_map_reference_shows_symbols(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    src = "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self, volume: str = 'normal') -> str:\n        return 'hi'\n\ndef helper(x: int, y: int = 0) -> int:\n    return x + y\n"
    (repo / "example.py").write_text(src, encoding="utf-8")

    result = preprocess_context_references(
        "Review @map:.",
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "class Animal" in result.message
    assert "def __init__" in result.message
    assert "def speak" in result.message
    assert "def helper" in result.message
    assert "volume" in result.message
    assert not result.warnings


def test_expand_map_skips_private_methods(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    src = "class Service:\n    def __init__(self):\n        pass\n    def _internal(self):\n        pass\n    def public(self):\n        pass\n"
    (repo / "priv.py").write_text(src, encoding="utf-8")

    result = preprocess_context_references(
        "Review @map:.",
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "class Service" in result.message
    assert "def __init__" in result.message
    assert "def public" in result.message
    assert "_internal" not in result.message


def test_expand_map_no_python_files(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "README.md").write_text("# Hello", encoding="utf-8")

    result = preprocess_context_references(
        "Review @map:.",
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "no Python files found" in result.message


def test_expand_map_on_file_uses_parent(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "code.py").write_text(
        "def foo() -> int:\n    return 42\n",
        encoding="utf-8",
    )

    # Point @map at a file, not a directory -- should use parent
    result = preprocess_context_references(
        "Review @map:code.py",
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "def foo" in result.message


def test_expand_map_nonexistent_path(tmp_path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Review @map:nope/",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert "not found" in result.message


def test_parse_grep_reference():
    from agent.context_references import parse_context_references

    refs = parse_context_references('find @grep:"TODO|FIXME" in the code')
    assert len(refs) == 1
    assert refs[0].kind == "grep"
    assert "TODO" in refs[0].target


def test_parse_grep_reference_unquoted():
    from agent.context_references import parse_context_references

    refs = parse_context_references("show @grep:TODO")
    assert len(refs) == 1
    assert refs[0].kind == "grep"
    assert "TODO" in refs[0].target


def test_parse_grep_reference_with_path():
    from agent.context_references import parse_context_references

    refs = parse_context_references('find @grep:"class Foo:src/" in code')
    assert len(refs) == 1
    assert refs[0].kind == "grep"
    assert "class Foo" in refs[0].target or "Foo" in refs[0].target


def test_expand_grep_finds_matches(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "code.py").write_text(
        "def hello():\n    # TODO: fix this\n    pass\n\ndef world():\n    # FIXME: also broken\n    pass\n",
        encoding="utf-8",
    )

    result = preprocess_context_references(
        'Find @grep:"TODO|FIXME"',
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "TODO" in result.message
    assert "FIXME" in result.message
    assert not result.warnings


def test_expand_grep_no_matches(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "code.py").write_text("print('hello')\n", encoding="utf-8")

    result = preprocess_context_references(
        'Find @grep:"NONEXISTENT_PATTERN_XYZ"',
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "no matches" in result.message


def test_expand_grep_with_path_prefix(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    src_dir = repo / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text(
        "TARGET_MARKER_UNIQUE = True\n",
        encoding="utf-8",
    )
    (repo / "other.py").write_text(
        "TARGET_MARKER_UNIQUE = False\n",
        encoding="utf-8",
    )

    result = preprocess_context_references(
        'Find @grep:"TARGET_MARKER_UNIQUE:src"',
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "TARGET_MARKER_UNIQUE" in result.message


def test_expand_grep_empty_pattern(tmp_path):
    from agent.context_references import preprocess_context_references

    # @grep: without a target does not parse as a reference (regex requires
    # at least one char after the colon), so it stays as literal text.
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")

    result = preprocess_context_references(
        "Find @grep:",
        cwd=repo,
        context_length=100_000,
    )

    # Not expanded -- stays as literal "@grep:" in the message
    assert not result.expanded
    assert "@grep:" in result.message


def test_expand_grep_skips_venv_dirs(tmp_path):
    from agent.context_references import preprocess_context_references

    repo = tmp_path / "project"
    repo.mkdir()
    venv = repo / "venv"
    venv.mkdir()
    (venv / "lib.py").write_text("SKIP_MARKER_UNIQUE = True\n", encoding="utf-8")
    (repo / "main.py").write_text("FIND_MARKER_UNIQUE = True\n", encoding="utf-8")

    result = preprocess_context_references(
        'Find @grep:"_MARKER_UNIQUE"',
        cwd=repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "FIND_MARKER_UNIQUE" in result.message
    assert "SKIP_MARKER_UNIQUE" not in result.message
