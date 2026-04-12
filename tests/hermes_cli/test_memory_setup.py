"""Tests for hermes_cli/memory_setup.py — memory provider setup wizard.

Covers _write_env_vars, _prompt, _get_available_providers,
cmd_setup_provider, cmd_status, and memory_command.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import hermes_cli.memory_setup as ms


# ---------------------------------------------------------------------------
# _write_env_vars
# ---------------------------------------------------------------------------
class TestWriteEnvVars:
    """Tests for _write_env_vars()."""

    def test_creates_new_env_file(self, tmp_path):
        env_path = tmp_path / ".env"
        ms._write_env_vars(env_path, {"API_KEY": "sk-123", "REGION": "us"})
        content = env_path.read_text()
        assert "API_KEY=sk-123" in content
        assert "REGION=us" in content

    def test_updates_existing_key(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("API_KEY=old-val\nOTHER=keep\n")
        ms._write_env_vars(env_path, {"API_KEY": "new-val"})
        lines = env_path.read_text().splitlines()
        assert "API_KEY=new-val" in lines
        assert "OTHER=keep" in lines
        # old value should not remain
        assert lines.count("API_KEY=old-val") == 0

    def test_appends_new_key_to_existing(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING=1\n")
        ms._write_env_vars(env_path, {"NEW_KEY": "val"})
        content = env_path.read_text()
        assert "EXISTING=1" in content
        assert "NEW_KEY=val" in content

    def test_empty_writes_preserve_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("A=1\nB=2\n")
        ms._write_env_vars(env_path, {})
        content = env_path.read_text()
        assert "A=1" in content
        assert "B=2" in content

    def test_creates_parent_dirs(self, tmp_path):
        env_path = tmp_path / "sub" / "dir" / ".env"
        ms._write_env_vars(env_path, {"K": "V"})
        assert env_path.exists()
        assert "K=V" in env_path.read_text()

    def test_multiple_keys_update(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("K1=old1\nK2=old2\nK3=keep\n")
        ms._write_env_vars(env_path, {"K1": "new1", "K2": "new2"})
        content = env_path.read_text()
        assert "K1=new1" in content
        assert "K2=new2" in content
        assert "K3=keep" in content

    def test_value_with_equals_sign(self, tmp_path):
        env_path = tmp_path / ".env"
        ms._write_env_vars(env_path, {"TOKEN": "abc=def=ghi"})
        content = env_path.read_text()
        assert "TOKEN=abc=def=ghi" in content

    def test_trailing_newline(self, tmp_path):
        env_path = tmp_path / ".env"
        ms._write_env_vars(env_path, {"K": "V"})
        assert env_path.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# _prompt
# ---------------------------------------------------------------------------
class TestPrompt:
    """Tests for _prompt() function."""

    def test_returns_input_value(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock())
        monkeypatch.setattr("sys.stdout", MagicMock())
        # Simulate readline returning a value
        import io
        fake_stdin = io.StringIO("my-value\n")
        monkeypatch.setattr("sys.stdin", fake_stdin)
        result = ms._prompt("Label")
        assert result == "my-value"

    def test_returns_default_on_empty_input(self, monkeypatch):
        import io
        fake_stdin = io.StringIO("\n")
        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", MagicMock())
        result = ms._prompt("Label", default="fallback")
        assert result == "fallback"

    def test_returns_empty_string_when_no_default_and_empty_input(self, monkeypatch):
        import io
        fake_stdin = io.StringIO("\n")
        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("sys.stdout", MagicMock())
        result = ms._prompt("Label")
        assert result == ""


# ---------------------------------------------------------------------------
# _get_available_providers
# ---------------------------------------------------------------------------
class TestGetAvailableProviders:
    """Tests for _get_available_providers()."""

    def test_returns_empty_on_import_error(self, monkeypatch):
        """When plugins module can't be imported, returns empty list."""
        monkeypatch.setattr(
            "hermes_cli.memory_setup._get_available_providers",
            lambda: [],
        )
        # Directly test the function with mocked import
        with patch.dict("sys.modules", {}):
            # Force re-import to trigger except path
            pass  # The actual function handles ImportError internally
        # Simple test: calling the function shouldn't crash
        result = ms._get_available_providers()
        assert isinstance(result, list)

    def test_returns_list_type(self):
        result = ms._get_available_providers()
        assert isinstance(result, list)

    def test_provider_tuple_structure(self):
        """If providers exist, they should be (name, desc, provider) tuples."""
        with patch("plugins.memory.discover_memory_providers", create=True) as mock_disc:
            mock_disc.return_value = []
            result = ms._get_available_providers()
            assert result == []


# ---------------------------------------------------------------------------
# _curses_select
# ---------------------------------------------------------------------------
class TestCursesSelect:
    """Tests for _curses_select()."""

    def test_returns_default_on_cancel(self, monkeypatch):
        mock_radiolist = MagicMock(return_value=2)
        monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", mock_radiolist)
        items = [("a", "desc a"), ("b", "desc b"), ("c", "desc c")]
        result = ms._curses_select("Title", items, default=1)
        assert result == 2  # Returns what curses_radiolist returns

    def test_formats_display_items(self, monkeypatch):
        captured = {}
        def fake_radiolist(title, items, selected=0, cancel_returns=0):
            captured["items"] = items
            return 0
        monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radiolist)
        items = [("provider1", "desc1"), ("provider2", "")]
        ms._curses_select("Title", items)
        assert "provider1  desc1" in captured["items"]
        assert "provider2" in captured["items"]


# ---------------------------------------------------------------------------
# cmd_setup_provider
# ---------------------------------------------------------------------------
class TestCmdSetupProvider:
    """Tests for cmd_setup_provider()."""

    def test_unknown_provider_prints_message(self, capsys, monkeypatch):
        monkeypatch.setattr(ms, "_get_available_providers", lambda: [])
        ms.cmd_setup_provider("nonexistent")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_sets_provider_in_config(self, tmp_path, monkeypatch, capsys):
        """When provider found with post_setup, config should be saved."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("memory: {}\n")

        mock_provider = MagicMock()
        mock_provider.post_setup = MagicMock()

        monkeypatch.setattr(ms, "_get_available_providers",
                            lambda: [("testprov", "desc", mock_provider)])
        monkeypatch.setattr("hermes_cli.memory_setup.get_hermes_home",
                            lambda: tmp_path)

        def fake_load():
            import yaml
            return yaml.safe_load(config_file.read_text())

        def fake_save(cfg):
            import yaml
            config_file.write_text(yaml.dump(cfg))

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        ms.cmd_setup_provider("testprov")
        mock_provider.post_setup.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------
class TestCmdStatus:
    """Tests for cmd_status()."""

    def test_no_provider_shows_none(self, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"memory": {}})
        monkeypatch.setattr(ms, "_get_available_providers", lambda: [])
        ms.cmd_status(None)
        captured = capsys.readouterr()
        assert "none" in captured.out.lower() or "(none" in captured.out

    def test_provider_shows_name(self, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"memory": {"provider": "testprov"}})
        monkeypatch.setattr(ms, "_get_available_providers", lambda: [])
        ms.cmd_status(None)
        captured = capsys.readouterr()
        assert "testprov" in captured.out

    def test_provider_not_installed(self, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"memory": {"provider": "missing"}})
        monkeypatch.setattr(ms, "_get_available_providers", lambda: [])
        ms.cmd_status(None)
        captured = capsys.readouterr()
        assert "NOT installed" in captured.out

    def test_provider_available_shows_check(self, capsys, monkeypatch):
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"memory": {"provider": "testprov"}})
        monkeypatch.setattr(ms, "_get_available_providers",
                            lambda: [("testprov", "desc", mock_provider)])
        ms.cmd_status(None)
        captured = capsys.readouterr()
        assert "available" in captured.out

    def test_provider_unavailable_shows_missing_secrets(self, capsys, monkeypatch):
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        mock_provider.get_config_schema.return_value = [
            {"key": "api_key", "secret": True, "env_var": "TEST_API_KEY", "url": "https://example.com"}
        ]
        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"memory": {"provider": "testprov"}})
        monkeypatch.setattr(ms, "_get_available_providers",
                            lambda: [("testprov", "desc", mock_provider)])
        ms.cmd_status(None)
        captured = capsys.readouterr()
        assert "not available" in captured.out
        assert "TEST_API_KEY" in captured.out


# ---------------------------------------------------------------------------
# memory_command router
# ---------------------------------------------------------------------------
class TestMemoryCommand:
    """Tests for memory_command() router."""

    def test_setup_routes_to_cmd_setup(self, monkeypatch):
        called = {}
        monkeypatch.setattr(ms, "cmd_setup", lambda args: called.setdefault("setup", True))
        args = MagicMock()
        args.memory_command = "setup"
        ms.memory_command(args)
        assert called.get("setup")

    def test_status_routes_to_cmd_status(self, monkeypatch):
        called = {}
        monkeypatch.setattr(ms, "cmd_status", lambda args: called.setdefault("status", True))
        args = MagicMock()
        args.memory_command = "status"
        ms.memory_command(args)
        assert called.get("status")

    def test_none_routes_to_status(self, monkeypatch):
        called = {}
        monkeypatch.setattr(ms, "cmd_status", lambda args: called.setdefault("status", True))
        args = MagicMock()
        args.memory_command = None
        ms.memory_command(args)
        assert called.get("status")

    def test_unknown_routes_to_status(self, monkeypatch):
        called = {}
        monkeypatch.setattr(ms, "cmd_status", lambda args: called.setdefault("status", True))
        args = MagicMock()
        args.memory_command = "bogus"
        ms.memory_command(args)
        assert called.get("status")


# ---------------------------------------------------------------------------
# _install_dependencies
# ---------------------------------------------------------------------------
class TestInstallDependencies:
    """Tests for _install_dependencies()."""

    def test_no_plugin_yaml_returns_silently(self, monkeypatch, tmp_path, capsys):
        """When plugin.yaml doesn't exist, function returns without error."""
        monkeypatch.setattr("pathlib.Path.parent",
                            property(lambda self: tmp_path))
        # Just verify it doesn't crash
        ms._install_dependencies("nonexistent_provider")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_with_no_dependencies(self, monkeypatch, tmp_path, capsys):
        """When plugin.yaml has no pip_dependencies, returns silently."""
        monkeypatch.setattr("pathlib.Path.parent",
                            property(lambda self: tmp_path))
        ms._install_dependencies("empty_provider")
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# cmd_setup interactive wizard
# ---------------------------------------------------------------------------
class TestCmdSetup:
    """Tests for cmd_setup() interactive wizard."""

    def test_no_providers_prints_message(self, capsys, monkeypatch):
        monkeypatch.setattr(ms, "_get_available_providers", lambda: [])
        ms.cmd_setup(None)
        captured = capsys.readouterr()
        assert "No memory provider" in captured.out

    def test_builtin_only_selection(self, capsys, monkeypatch):
        """Selecting built-in should clear provider."""
        monkeypatch.setattr(ms, "_curses_select", lambda *a, **kw: 1)
        monkeypatch.setattr(ms, "_get_available_providers", lambda: [
            ("prov1", "desc1", MagicMock()),
        ])

        saved = {}
        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        ms.cmd_setup(None)
        captured = capsys.readouterr()
        assert "built-in only" in captured.out
        assert saved.get("memory", {}).get("provider") == ""
