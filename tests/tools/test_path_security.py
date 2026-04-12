"""Tests for the path_security module (validate_within_dir, has_traversal_component).

Tests verify path traversal protection, symlink following, edge cases for
relative/absolute paths, and boundary conditions for the security helpers.
"""

import os
import tempfile
from pathlib import Path

import pytest

from tools.path_security import has_traversal_component, validate_within_dir


# ---------------------------------------------------------------------------
# has_traversal_component
# ---------------------------------------------------------------------------

class TestHasTraversalComponent:
    """Tests for the quick '..' component check."""

    def test_plain_filename(self):
        assert has_traversal_component("file.txt") is False

    def test_absolute_path_no_dots(self):
        assert has_traversal_component("/home/user/file.txt") is False

    def test_relative_path_no_dots(self):
        assert has_traversal_component("subdir/file.txt") is False

    def test_single_dot_component(self):
        """Single '.' is not a traversal component."""
        assert has_traversal_component("./file.txt") is False

    def test_double_dot_component(self):
        """'..' IS a traversal component."""
        assert has_traversal_component("../file.txt") is True

    def test_nested_double_dot(self):
        assert has_traversal_component("a/../b/file.txt") is True

    def test_double_dot_at_end(self):
        assert has_traversal_component("subdir/..") is True

    def test_multiple_double_dots(self):
        assert has_traversal_component("../../etc/passwd") is True

    def test_double_dot_in_filename_not_traversal(self):
        """'..' embedded in a filename (not a path component) should not match."""
        # Path("file..txt").parts == ("file..txt",) — no ".." part
        assert has_traversal_component("file..txt") is False

    def test_empty_string(self):
        assert has_traversal_component("") is False

    def test_dot_only(self):
        assert has_traversal_component(".") is False

    def test_double_dot_only(self):
        assert has_traversal_component("..") is True

    def test_trailing_slash(self):
        """Path with trailing slash — '..' is still a component."""
        assert has_traversal_component("../") is True


# ---------------------------------------------------------------------------
# validate_within_dir
# ---------------------------------------------------------------------------

class TestValidateWithinDir:
    """Tests for the resolve-and-check path validator."""

    def test_path_within_root(self, tmp_path):
        """File inside root — should return None (safe)."""
        f = tmp_path / "safe.txt"
        f.write_text("ok")
        assert validate_within_dir(f, tmp_path) is None

    def test_subdirectory_within_root(self, tmp_path):
        """File in a subdirectory of root — should return None (safe)."""
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "file.txt"
        f.write_text("ok")
        assert validate_within_dir(f, tmp_path) is None

    def test_path_escapes_root(self, tmp_path):
        """Path pointing outside root — should return an error string."""
        outside = tmp_path.parent / "outside.txt"
        result = validate_within_dir(outside, tmp_path)
        assert result is not None
        assert "escapes" in result.lower()

    def test_dotdot_traversal(self, tmp_path):
        """Using '..' to escape root — should return an error string."""
        sub = tmp_path / "sub"
        sub.mkdir()
        escape = sub / ".." / ".." / "etc" / "passwd"
        result = validate_within_dir(escape, tmp_path)
        assert result is not None
        assert "escapes" in result.lower()

    def test_dotdot_stays_within_root(self, tmp_path):
        """'..' that stays within root — should return None (safe)."""
        sub = tmp_path / "sub"
        sub.mkdir()
        # sub/../tmp_path is still within tmp_path
        inside = sub / ".."
        assert validate_within_dir(inside, tmp_path) is None

    def test_symlink_pointing_outside(self, tmp_path):
        """Symlink inside root pointing outside — should be caught after resolve."""
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("secret")
        link = tmp_path / "link"
        link.symlink_to(outside)
        result = validate_within_dir(link, tmp_path)
        assert result is not None
        assert "escapes" in result.lower()

    def test_symlink_pointing_inside(self, tmp_path):
        """Symlink inside root pointing to another file inside root — safe."""
        target = tmp_path / "real.txt"
        target.write_text("data")
        link = tmp_path / "link"
        link.symlink_to(target)
        assert validate_within_dir(link, tmp_path) is None

    def test_root_as_path(self, tmp_path):
        """The root directory itself — should return None (safe)."""
        assert validate_within_dir(tmp_path, tmp_path) is None

    def test_nonexistent_path_within_root(self, tmp_path):
        """Non-existent path that would be inside root — safe (resolve still works)."""
        nonexistent = tmp_path / "does_not_exist.txt"
        assert validate_within_dir(nonexistent, tmp_path) is None

    def test_nonexistent_path_outside_root(self, tmp_path):
        """Non-existent path that would be outside root — caught."""
        nonexistent = tmp_path.parent / "does_not_exist.txt"
        result = validate_within_dir(nonexistent, tmp_path)
        assert result is not None

    def test_absolute_path_within_root(self, tmp_path):
        """Absolute path inside root — safe."""
        f = tmp_path / "abs.txt"
        f.write_text("ok")
        assert validate_within_dir(f.resolve(), tmp_path.resolve()) is None

    def test_root_is_file_path_still_within(self, tmp_path):
        """validate_within_dir is a pure path-containment check — it does not
        verify that root is a directory.  A path like rootfile/child resolves
        as a valid relative_of rootfile even though rootfile is not a dir.
        This documents the current (intentional) behavior."""
        f = tmp_path / "rootfile"
        f.write_text("root")
        child = tmp_path / "rootfile" / "child"
        # resolve() normalizes both paths; relative_to succeeds as string prefix
        result = validate_within_dir(child, f)
        assert result is None  # path containment check only, not fs-type check

    def test_deeply_nested_path(self, tmp_path):
        """Deeply nested path still within root — safe."""
        deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "file.txt"
        deep.parent.mkdir(parents=True)
        deep.write_text("deep")
        assert validate_within_dir(deep, tmp_path) is None

    def test_error_message_includes_details(self, tmp_path):
        """Error message should be informative."""
        outside = tmp_path.parent / "outside.txt"
        result = validate_within_dir(outside, tmp_path)
        assert isinstance(result, str)
        assert len(result) > 10  # not just "error" or empty
