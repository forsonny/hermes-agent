"""Tests for hermes_cli/uninstall.py."""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import hermes_cli.uninstall as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeColors:
    """Minimal colors stub used by log_* helpers."""
    CYAN = "cyan"
    GREEN = "green"
    YELLOW = "yellow"
    MAGENTA = "magenta"
    RED = "red"
    BOLD = "bold"
    DIM = "dim"


# ---------------------------------------------------------------------------
# log_info / log_success / log_warn
# ---------------------------------------------------------------------------

class TestLogHelpers:
    """Simple print helpers should not raise."""

    def test_log_info(self, capsys):
        with patch.object(mod, "color", side_effect=lambda t, *a: t):
            mod.log_info("hello")
        out = capsys.readouterr().out
        assert "hello" in out

    def test_log_success(self, capsys):
        with patch.object(mod, "color", side_effect=lambda t, *a: t):
            mod.log_success("done")
        out = capsys.readouterr().out
        assert "done" in out

    def test_log_warn(self, capsys):
        with patch.object(mod, "color", side_effect=lambda t, *a: t):
            mod.log_warn("careful")
        out = capsys.readouterr().out
        assert "careful" in out


# ---------------------------------------------------------------------------
# get_project_root
# ---------------------------------------------------------------------------

class TestGetProjectRoot:
    def test_returns_path(self):
        result = mod.get_project_root()
        assert isinstance(result, Path)

    def test_is_resolved(self):
        result = mod.get_project_root()
        assert result == result.resolve()

    def test_parent_is_hermes_cli_parent(self):
        result = mod.get_project_root()
        # uninstall.py lives in hermes_cli/; parent.parent = project root
        assert result == Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# find_shell_configs
# ---------------------------------------------------------------------------

class TestFindShellConfigs:
    def test_returns_list(self):
        with patch.object(Path, "home", return_value=Path("/fake")):
            result = mod.find_shell_configs()
        assert isinstance(result, list)

    def test_includes_existing_files(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("old")
        with patch.object(Path, "home", return_value=tmp_path):
            result = mod.find_shell_configs()
        assert bashrc in result

    def test_skips_missing_files(self, tmp_path):
        # No shell config files exist
        with patch.object(Path, "home", return_value=tmp_path):
            result = mod.find_shell_configs()
        assert result == []

    def test_all_candidates_if_present(self, tmp_path):
        for name in (".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile"):
            (tmp_path / name).write_text("x")
        with patch.object(Path, "home", return_value=tmp_path):
            result = mod.find_shell_configs()
        assert len(result) == 5


# ---------------------------------------------------------------------------
# remove_path_from_shell_configs
# ---------------------------------------------------------------------------

class TestRemovePathFromShellConfigs:
    def test_removes_hermes_path_line(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text(textwrap.dedent("""\
            export PATH=/usr/bin:$PATH
            export PATH=~/.hermes/hermes-agent:$PATH
            echo hello
        """))
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_path_from_shell_configs()
        assert bashrc in removed
        content = bashrc.read_text()
        assert "hermes" not in content
        assert "echo hello" in content

    def test_removes_hermes_comment_and_next_line(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text(textwrap.dedent("""\
            export PATH=/usr/bin:$PATH
            # Hermes Agent
            export PATH=~/.hermes/hermes-agent:$PATH
            echo hello
        """))
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_path_from_shell_configs()
        assert bashrc in removed
        content = bashrc.read_text()
        assert "Hermes Agent" not in content
        assert "echo hello" in content

    def test_no_change_when_no_hermes(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        original = "export PATH=/usr/bin:$PATH\nalias ll=ls\n"
        bashrc.write_text(original)
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_path_from_shell_configs()
        assert removed == []
        assert bashrc.read_text() == original

    def test_handles_multiple_files(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        zshrc = tmp_path / ".zshrc"
        bashrc.write_text("export PATH=~/.hermes/hermes-agent:$PATH\n")
        zshrc.write_text("export PATH=/usr/bin:$PATH\n")
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_path_from_shell_configs()
        assert bashrc in removed
        assert zshrc not in removed

    def test_exception_in_single_file_continues(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("export PATH=~/.hermes/hermes-agent:$PATH\n")
        zshrc = tmp_path / ".zshrc"
        zshrc.write_text("export PATH=~/.hermes/hermes-agent:$PATH\n")
        # Make zshrc unreadable
        os.chmod(str(zshrc), 0o000)
        try:
            with patch.object(Path, "home", return_value=tmp_path):
                removed = mod.remove_path_from_shell_configs()
            assert bashrc in removed
            # zshrc raised exception, but bashrc still processed
        finally:
            os.chmod(str(zshrc), 0o644)

    def test_cleans_up_triple_blank_lines(self, tmp_path):
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("export PATH=~/.hermes/hermes-agent:$PATH\n\n\n\necho hi\n")
        with patch.object(Path, "home", return_value=tmp_path):
            mod.remove_path_from_shell_configs()
        content = bashrc.read_text()
        assert "\n\n\n" not in content


# ---------------------------------------------------------------------------
# remove_wrapper_script
# ---------------------------------------------------------------------------

class TestRemoveWrapperScript:
    def test_removes_hermes_wrapper(self, tmp_path):
        wrapper = tmp_path / ".local" / "bin" / "hermes"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\npython -m hermes_cli\n")
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_wrapper_script()
        assert wrapper in removed
        assert not wrapper.exists()

    def test_skips_non_hermes_wrapper(self, tmp_path):
        wrapper = tmp_path / ".local" / "bin" / "hermes"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\necho unrelated\n")
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_wrapper_script()
        assert removed == []
        assert wrapper.exists()

    def test_returns_empty_when_no_wrappers(self, tmp_path):
        with patch.object(Path, "home", return_value=tmp_path):
            removed = mod.remove_wrapper_script()
        assert removed == []

    def test_skips_permission_denied(self, tmp_path):
        wrapper = tmp_path / ".local" / "bin" / "hermes"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\n# hermes_cli\n")
        os.chmod(str(wrapper), 0o000)
        try:
            with patch.object(Path, "home", return_value=tmp_path):
                removed = mod.remove_wrapper_script()
            # Should not crash
            assert isinstance(removed, list)
        finally:
            os.chmod(str(wrapper), 0o644)


# ---------------------------------------------------------------------------
# uninstall_gateway_service
# ---------------------------------------------------------------------------

class TestUninstallGatewayService:
    def test_returns_false_on_non_linux(self):
        with patch("platform.system", return_value="Darwin"):
            result = mod.uninstall_gateway_service()
        assert result is False

    def test_returns_false_on_termux(self):
        with patch("platform.system", return_value="Linux"), \
             patch.dict(os.environ, {"TERMUX_VERSION": "1"}):
            result = mod.uninstall_gateway_service()
        assert result is False

    def test_returns_false_when_no_service_file(self, tmp_path):
        config_dir = tmp_path / ".config" / "systemd" / "user"
        config_dir.mkdir(parents=True)
        with patch("platform.system", return_value="Linux"), \
             patch.object(Path, "home", return_value=tmp_path), \
             patch.dict(os.environ, {}, clear=False):
            result = mod.uninstall_gateway_service()
        assert result is False

    def test_removes_existing_service(self, tmp_path):
        config_dir = tmp_path / ".config" / "systemd" / "user"
        config_dir.mkdir(parents=True)
        svc_file = config_dir / "hermes-gateway.service"
        svc_file.write_text("[Unit]\n")
        mock_run = MagicMock()
        with patch("platform.system", return_value="Linux"), \
             patch.object(Path, "home", return_value=tmp_path), \
             patch.object(mod.subprocess, "run", mock_run), \
             patch.dict(os.environ, {}, clear=False):
            result = mod.uninstall_gateway_service()
        assert result is True
        assert not svc_file.exists()
        assert mock_run.call_count == 3  # stop, disable, daemon-reload

    def test_custom_service_name(self, tmp_path):
        config_dir = tmp_path / ".config" / "systemd" / "user"
        config_dir.mkdir(parents=True)
        svc_file = config_dir / "custom-hermes.service"
        svc_file.write_text("[Unit]\n")
        mock_gateway = MagicMock()
        mock_gateway.get_service_name = MagicMock(return_value="custom-hermes")
        mock_run = MagicMock()
        with patch("platform.system", return_value="Linux"), \
             patch.object(Path, "home", return_value=tmp_path), \
             patch.object(mod.subprocess, "run", mock_run), \
             patch.dict(os.environ, {}, clear=False), \
             patch.dict("sys.modules", {"hermes_cli.gateway": mock_gateway}):
            result = mod.uninstall_gateway_service()
        assert result is True

    def test_exception_during_service_removal(self, tmp_path):
        config_dir = tmp_path / ".config" / "systemd" / "user"
        config_dir.mkdir(parents=True)
        svc_file = config_dir / "hermes-gateway.service"
        svc_file.write_text("[Unit]\n")
        mock_run = MagicMock(side_effect=OSError("fail"))
        with patch("platform.system", return_value="Linux"), \
             patch.object(Path, "home", return_value=tmp_path), \
             patch.object(mod.subprocess, "run", mock_run), \
             patch.dict(os.environ, {}, clear=False):
            result = mod.uninstall_gateway_service()
        assert result is False


# ---------------------------------------------------------------------------
# run_uninstall
# ---------------------------------------------------------------------------

class TestRunUninstall:
    def test_cancel_option_3(self, capsys):
        with patch("builtins.input", return_value="3"):
            mod.run_uninstall(None)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_cancel_quit(self, capsys):
        with patch("builtins.input", side_effect=["q"]):
            mod.run_uninstall(None)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_cancel_no_confirmation(self, capsys):
        # First input = option 2 (full uninstall), second = "nope" (not "yes")
        with patch("builtins.input", side_effect=["2", "nope"]):
            mod.run_uninstall(None)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_keyboard_interrupt_cancels(self, capsys):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            mod.run_uninstall(None)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_eof_error_cancels(self, capsys):
        with patch("builtins.input", side_effect=EOFError):
            mod.run_uninstall(None)
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_keep_data_uninstall(self, tmp_path, capsys):
        """Option 1 + "yes" should trigger uninstall without removing hermes_home."""
        fake_root = tmp_path / "code"
        fake_root.mkdir()
        fake_home = tmp_path / "hermes_home"
        fake_home.mkdir()

        with patch.object(mod, "get_project_root", return_value=fake_root), \
             patch.object(mod, "get_hermes_home", return_value=fake_home), \
             patch.object(mod, "uninstall_gateway_service", return_value=False), \
             patch.object(mod, "remove_path_from_shell_configs", return_value=[]), \
             patch.object(mod, "remove_wrapper_script", return_value=[]), \
             patch("builtins.input", side_effect=["1", "yes"]):
            mod.run_uninstall(None)

        out = capsys.readouterr().out
        assert "complete" in out.lower()
        # hermes_home should still exist (keep data)
        assert fake_home.exists()

    def test_full_uninstall_removes_home(self, tmp_path, capsys):
        fake_root = tmp_path / "code"
        fake_root.mkdir()
        fake_home = tmp_path / "hermes_home"
        fake_home.mkdir()
        (fake_home / "config.yaml").write_text("test: true")

        with patch.object(mod, "get_project_root", return_value=fake_root), \
             patch.object(mod, "get_hermes_home", return_value=fake_home), \
             patch.object(mod, "uninstall_gateway_service", return_value=False), \
             patch.object(mod, "remove_path_from_shell_configs", return_value=[]), \
             patch.object(mod, "remove_wrapper_script", return_value=[]), \
             patch("builtins.input", side_effect=["2", "yes"]):
            mod.run_uninstall(None)

        out = capsys.readouterr().out
        assert "complete" in out.lower()
        assert not fake_home.exists()

    def test_reports_removed_items(self, tmp_path, capsys):
        fake_root = tmp_path / "code"
        fake_root.mkdir()
        fake_home = tmp_path / "hermes_home"
        fake_home.mkdir()
        fake_bashrc = tmp_path / ".bashrc"

        with patch.object(mod, "get_project_root", return_value=fake_root), \
             patch.object(mod, "get_hermes_home", return_value=fake_home), \
             patch.object(mod, "uninstall_gateway_service", return_value=True), \
             patch.object(mod, "remove_path_from_shell_configs", return_value=[fake_bashrc]), \
             patch.object(mod, "remove_wrapper_script", return_value=[Path("/usr/local/bin/hermes")]), \
             patch("builtins.input", side_effect=["1", "yes"]):
            mod.run_uninstall(None)

        out = capsys.readouterr().out
        assert "gateway" in out.lower()
        assert "complete" in out.lower()

    def test_install_dir_removal_failure_handled(self, tmp_path, capsys):
        fake_root = tmp_path / "code"
        fake_root.mkdir()
        fake_home = tmp_path / "hermes_home"
        fake_home.mkdir()

        with patch.object(mod, "get_project_root", return_value=fake_root), \
             patch.object(mod, "get_hermes_home", return_value=fake_home), \
             patch.object(mod, "uninstall_gateway_service", return_value=False), \
             patch.object(mod, "remove_path_from_shell_configs", return_value=[]), \
             patch.object(mod, "remove_wrapper_script", return_value=[]), \
             patch("shutil.rmtree", side_effect=OSError("permission denied")), \
             patch("builtins.input", side_effect=["1", "yes"]):
            mod.run_uninstall(None)

        out = capsys.readouterr().out
        # Should still complete (gracefully handles the error)
        assert "could not" in out.lower() or "may need" in out.lower()
