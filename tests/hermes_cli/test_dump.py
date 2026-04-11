"""
Tests for hermes_cli/dump.py
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.dump import (
    _redact,
    _count_skills,
    _count_mcp_servers,
    _cron_summary,
    _configured_platforms,
    _memory_provider,
    _get_model_and_provider,
    _config_overrides,
    _get_git_commit,
    _gateway_status,
)


# =========================================================================
# _redact
# =========================================================================

class TestRedact:
    """Tests for _redact helper."""

    def test_empty_string(self):
        assert _redact("") == ""

    def test_short_string_under_12(self):
        assert _redact("short") == "***"

    def test_exactly_12_chars(self):
        assert _redact("abcdefghijkl") == "abcd...ijkl"

    def test_long_string(self):
        key = "sk-1234567890abcdef1234567890abcdef"
        result = _redact(key)
        assert result.startswith("sk-1")
        assert result.endswith("cdef")
        assert "..." in result

    def test_preserves_first_and_last_four(self):
        result = _redact("ABCDEFGHIJKLMNOP")
        assert result == "ABCD...MNOP"


# =========================================================================
# _count_skills
# =========================================================================

class TestCountSkills:
    """Tests for _count_skills helper."""

    def test_no_skills_dir(self, tmp_path):
        assert _count_skills(tmp_path) == 0

    def test_empty_skills_dir(self, tmp_path):
        (tmp_path / "skills").mkdir()
        assert _count_skills(tmp_path) == 0

    def test_one_skill(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "my-skill").mkdir()
        (skills / "my-skill" / "SKILL.md").write_text("# My Skill")
        assert _count_skills(tmp_path) == 1

    def test_nested_skills(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "cat" / "skill-a").mkdir(parents=True)
        (skills / "cat" / "skill-a" / "SKILL.md").write_text("# A")
        (skills / "skill-b").mkdir()
        (skills / "skill-b" / "SKILL.md").write_text("# B")
        assert _count_skills(tmp_path) == 2

    def test_ignores_non_skill_md(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "myskill").mkdir()
        (skills / "myskill" / "README.md").write_text("# not a skill")
        assert _count_skills(tmp_path) == 0


# =========================================================================
# _count_mcp_servers
# =========================================================================

class TestCountMcpServers:
    """Tests for _count_mcp_servers helper."""

    def test_empty_config(self):
        assert _count_mcp_servers({}) == 0

    def test_no_servers(self):
        assert _count_mcp_servers({"mcp": {}}) == 0

    def test_one_server(self):
        config = {"mcp": {"servers": {"my-server": {"command": "npx"}}}}
        assert _count_mcp_servers(config) == 1

    def test_multiple_servers(self):
        config = {"mcp": {"servers": {
            "s1": {"command": "npx"},
            "s2": {"command": "python"},
            "s3": {"command": "node"},
        }}}
        assert _count_mcp_servers(config) == 3


# =========================================================================
# _cron_summary
# =========================================================================

class TestCronSummary:
    """Tests for _cron_summary helper."""

    def test_no_cron_dir(self, tmp_path):
        assert _cron_summary(tmp_path) == "0"

    def test_no_jobs_file(self, tmp_path):
        (tmp_path / "cron").mkdir()
        assert _cron_summary(tmp_path) == "0"

    def test_empty_jobs_list(self, tmp_path):
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        (cron_dir / "jobs.json").write_text(json.dumps({"jobs": []}))
        result = _cron_summary(tmp_path)
        # Empty list with file present returns "0 active / 0 total"
        assert "0" in result

    def test_active_jobs(self, tmp_path):
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        jobs = [
            {"name": "a", "enabled": True},
            {"name": "b", "enabled": True},
            {"name": "c", "enabled": False},
        ]
        (cron_dir / "jobs.json").write_text(json.dumps({"jobs": jobs}))
        result = _cron_summary(tmp_path)
        assert "2 active" in result
        assert "3 total" in result

    def test_all_enabled_default(self, tmp_path):
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        jobs = [{"name": "a"}, {"name": "b"}]
        (cron_dir / "jobs.json").write_text(json.dumps({"jobs": jobs}))
        result = _cron_summary(tmp_path)
        assert "2 active" in result
        assert "2 total" in result

    def test_corrupt_json(self, tmp_path):
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        (cron_dir / "jobs.json").write_text("not json at all")
        assert _cron_summary(tmp_path) == "(error reading)"


# =========================================================================
# _configured_platforms
# =========================================================================

class TestConfiguredPlatforms:
    """Tests for _configured_platforms helper."""

    _ALL_PLATFORM_VARS = [
        "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN",
        "WHATSAPP_ENABLED", "SIGNAL_HTTP_URL", "EMAIL_ADDRESS",
        "TWILIO_ACCOUNT_SID", "MATRIX_HOMESERVER_URL", "MATTERMOST_URL",
        "HASS_TOKEN", "DINGTALK_CLIENT_ID", "FEISHU_APP_ID",
        "WECOM_BOT_ID", "WEIXIN_ACCOUNT_ID",
    ]

    def test_no_platforms(self, monkeypatch):
        for var in self._ALL_PLATFORM_VARS:
            monkeypatch.delenv(var, raising=False)
        assert _configured_platforms() == []

    def test_telegram_configured(self, monkeypatch):
        for var in self._ALL_PLATFORM_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        platforms = _configured_platforms()
        assert "telegram" in platforms

    def test_multiple_platforms(self, monkeypatch):
        for var in self._ALL_PLATFORM_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "d")
        monkeypatch.setenv("EMAIL_ADDRESS", "x@y.com")
        platforms = _configured_platforms()
        assert "telegram" in platforms
        assert "discord" in platforms
        assert "email" in platforms
        assert len(platforms) == 3


# =========================================================================
# _memory_provider
# =========================================================================

class TestMemoryProvider:
    """Tests for _memory_provider helper."""

    def test_empty_config(self):
        assert _memory_provider({}) == "built-in"

    def test_no_provider_key(self):
        assert _memory_provider({"memory": {}}) == "built-in"

    def test_empty_provider(self):
        assert _memory_provider({"memory": {"provider": ""}}) == "built-in"

    def test_custom_provider(self):
        assert _memory_provider({"memory": {"provider": "redis"}}) == "redis"


# =========================================================================
# _get_model_and_provider
# =========================================================================

class TestGetModelAndProvider:
    """Tests for _get_model_and_provider helper."""

    def test_empty_config(self):
        model, provider = _get_model_and_provider({})
        assert model == "(not set)"
        assert provider == "(auto)"

    def test_string_model(self):
        model, provider = _get_model_and_provider({"model": "gpt-4"})
        assert model == "gpt-4"
        assert provider == "(auto)"

    def test_empty_string_model(self):
        model, provider = _get_model_and_provider({"model": ""})
        assert model == "(not set)"
        assert provider == "(auto)"

    def test_dict_model_with_default(self):
        cfg = {"model": {"default": "claude-3", "provider": "anthropic"}}
        model, provider = _get_model_and_provider(cfg)
        assert model == "claude-3"
        assert provider == "anthropic"

    def test_dict_model_with_model_key(self):
        cfg = {"model": {"model": "gpt-4o"}}
        model, provider = _get_model_and_provider(cfg)
        assert model == "gpt-4o"
        assert provider == "(auto)"

    def test_dict_model_with_name_key(self):
        cfg = {"model": {"name": "glm-4"}}
        model, provider = _get_model_and_provider(cfg)
        assert model == "glm-4"
        assert provider == "(auto)"

    def test_dict_model_empty(self):
        cfg = {"model": {}}
        model, provider = _get_model_and_provider(cfg)
        assert model == "(not set)"
        assert provider == "(auto)"

    def test_non_string_non_dict_model(self):
        cfg = {"model": 42}
        model, provider = _get_model_and_provider(cfg)
        assert model == "(not set)"
        assert provider == "(auto)"

    def test_dict_model_fallback_chain(self):
        cfg = {"model": {"default": "first", "model": "second", "name": "third"}}
        model, provider = _get_model_and_provider(cfg)
        assert model == "first"


# =========================================================================
# _config_overrides
# =========================================================================

class TestConfigOverrides:
    """Tests for _config_overrides helper."""

    def test_no_overrides_empty_config(self):
        overrides = _config_overrides({})
        assert isinstance(overrides, dict)

    def test_fallback_providers_present(self):
        result = _config_overrides({"fallback_providers": ["backup1"]})
        assert "fallback_providers" in result

    def test_no_fallback_providers_empty(self):
        result = _config_overrides({"fallback_providers": []})
        assert "fallback_providers" not in result

    def test_terminal_backend_override(self):
        from hermes_cli import config as config_mod
        orig_terminal = config_mod.DEFAULT_CONFIG.get("terminal", {})
        try:
            config_mod.DEFAULT_CONFIG["terminal"] = {"backend": "local"}
            result = _config_overrides({"terminal": {"backend": "docker"}})
            assert "terminal.backend" in result
            assert result["terminal.backend"] == "docker"
        finally:
            config_mod.DEFAULT_CONFIG["terminal"] = orig_terminal

    def test_toolsets_same_as_default(self):
        from hermes_cli import config as config_mod
        default_ts = config_mod.DEFAULT_CONFIG.get("toolsets", ["hermes-cli"])
        result = _config_overrides({"toolsets": list(default_ts)})
        assert "toolsets" not in result

    def test_toolsets_different_from_default(self):
        from hermes_cli import config as config_mod
        default_ts = config_mod.DEFAULT_CONFIG.get("toolsets", ["hermes-cli"])
        modified = list(default_ts) + ["browser"]
        result = _config_overrides({"toolsets": modified})
        assert "toolsets" in result

    def test_compression_enabled_override(self):
        from hermes_cli import config as config_mod
        orig = config_mod.DEFAULT_CONFIG.get("compression", {})
        try:
            config_mod.DEFAULT_CONFIG["compression"] = {"enabled": False, "threshold": 0.7}
            result = _config_overrides({"compression": {"enabled": True}})
            assert "compression.enabled" in result
        finally:
            config_mod.DEFAULT_CONFIG["compression"] = orig


# =========================================================================
# _get_git_commit
# =========================================================================

class TestGetGitCommit:
    """Tests for _get_git_commit helper."""

    def test_success(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "f.txt").write_text("hi")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        result = _get_git_commit(tmp_path)
        assert result != "(unknown)"
        assert len(result) == 8

    def test_not_git_repo(self, tmp_path):
        result = _get_git_commit(tmp_path)
        assert result == "(unknown)"

    def test_timeout_handled(self, tmp_path):
        with patch("hermes_cli.dump.subprocess.run", side_effect=Exception("boom")):
            result = _get_git_commit(tmp_path)
        assert result == "(unknown)"


# =========================================================================
# _gateway_status
# =========================================================================

class TestGatewayStatus:
    """Tests for _gateway_status helper.

    We patch sys.platform at the function level to avoid xdist worker issues
    with module-level imports.
    """

    def test_non_linux_non_mac(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        assert _gateway_status() == "N/A"

    def test_linux_systemctl_running(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        mock_result = MagicMock()
        mock_result.stdout = "active\n"
        with patch("hermes_cli.dump.subprocess.run", return_value=mock_result) as mock_run:
            with patch("hermes_cli.dump.get_service_name", return_value="hermes-gateway", create=True):
                # Need to handle the import inside _gateway_status
                pass
        # The function does: from hermes_cli.gateway import get_service_name
        # We need to patch the gateway module itself
        import hermes_cli.dump as dump_mod
        monkeypatch.setattr("sys.platform", "linux")
        mock_result2 = MagicMock()
        mock_result2.stdout = "active\n"
        with patch.object(dump_mod, "subprocess") as mock_sp:
            mock_sp.run.return_value = mock_result2
            with patch.dict("sys.modules", {"hermes_cli.gateway": MagicMock(get_service_name=MagicMock(return_value="hermes-gateway"))}):
                result = _gateway_status()
        assert "running" in result

    def test_linux_stopped(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        mock_result = MagicMock()
        mock_result.stdout = "inactive\n"
        import hermes_cli.dump as dump_mod
        with patch.object(dump_mod, "subprocess") as mock_sp:
            mock_sp.run.return_value = mock_result
            with patch.dict("sys.modules", {"hermes_cli.gateway": MagicMock(get_service_name=MagicMock(return_value="hermes-gateway"))}):
                result = _gateway_status()
        assert result == "stopped"

    def test_linux_exception(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        import hermes_cli.dump as dump_mod
        with patch.object(dump_mod, "subprocess") as mock_sp:
            mock_sp.run.side_effect = Exception("no systemctl")
            with patch.dict("sys.modules", {"hermes_cli.gateway": MagicMock(get_service_name=MagicMock(return_value="hermes-gateway"))}):
                result = _gateway_status()
        assert result == "unknown"

    def test_macos_loaded(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        mock_result = MagicMock()
        mock_result.returncode = 0
        import hermes_cli.dump as dump_mod
        with patch.object(dump_mod, "subprocess") as mock_sp:
            mock_sp.run.return_value = mock_result
            with patch.dict("sys.modules", {"hermes_cli.gateway": MagicMock(get_launchd_label=MagicMock(return_value="com.hermes"))}):
                result = _gateway_status()
        assert "loaded" in result

    def test_macos_not_loaded(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        mock_result = MagicMock()
        mock_result.returncode = 1
        import hermes_cli.dump as dump_mod
        with patch.object(dump_mod, "subprocess") as mock_sp:
            mock_sp.run.return_value = mock_result
            with patch.dict("sys.modules", {"hermes_cli.gateway": MagicMock(get_launchd_label=MagicMock(return_value="com.hermes"))}):
                result = _gateway_status()
        assert "not loaded" in result
