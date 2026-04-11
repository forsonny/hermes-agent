"""Tests for hermes_cli/webhook.py — webhook subscription management."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for webhook subscription file I/O."""
    hh = tmp_path / ".hermes"
    hh.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hh))
    import hermes_cli.webhook as wh_mod
    monkeypatch.setattr(wh_mod, "_hermes_home", lambda: hh)
    return hh


@pytest.fixture
def webhook_enabled(monkeypatch):
    """Pretend webhook platform is enabled in config."""
    import hermes_cli.webhook as wh_mod
    monkeypatch.setattr(wh_mod, "_is_webhook_enabled", lambda: True)


@pytest.fixture
def webhook_disabled(monkeypatch):
    """Pretend webhook platform is disabled."""
    import hermes_cli.webhook as wh_mod
    monkeypatch.setattr(wh_mod, "_is_webhook_enabled", lambda: False)


@pytest.fixture
def make_args():
    """Factory for argparse-like namespace objects."""
    def _make(**overrides):
        defaults = dict(
            webhook_action="subscribe",
            name="test-hook",
            secret=None,
            events=None,
            description=None,
            prompt=None,
            skills=None,
            deliver=None,
            deliver_chat_id=None,
            payload=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)
    return _make


# ---------------------------------------------------------------------------
# _load_subscriptions / _save_subscriptions
# ---------------------------------------------------------------------------

class TestLoadSaveSubscriptions:

    def test_load_empty_when_file_missing(self, hermes_home):
        import hermes_cli.webhook as wh
        assert wh._load_subscriptions() == {}

    def test_load_valid_json(self, hermes_home):
        import hermes_cli.webhook as wh
        data = {"myhook": {"events": ["push"]}}
        (hermes_home / "webhook_subscriptions.json").write_text(
            json.dumps(data)
        )
        assert wh._load_subscriptions() == data

    def test_load_corrupt_json_returns_empty(self, hermes_home):
        import hermes_cli.webhook as wh
        (hermes_home / "webhook_subscriptions.json").write_text("not json{{{")
        assert wh._load_subscriptions() == {}

    def test_load_non_dict_returns_empty(self, hermes_home):
        import hermes_cli.webhook as wh
        (hermes_home / "webhook_subscriptions.json").write_text("[1,2,3]")
        assert wh._load_subscriptions() == {}

    def test_save_creates_file(self, hermes_home):
        import hermes_cli.webhook as wh
        subs = {"hook": {"events": []}}
        wh._save_subscriptions(subs)
        path = hermes_home / "webhook_subscriptions.json"
        assert path.exists()
        assert json.loads(path.read_text()) == subs

    def test_save_atomic_write(self, hermes_home):
        import hermes_cli.webhook as wh
        subs = {"hook": {"events": []}}
        wh._save_subscriptions(subs)
        # No leftover .tmp file
        assert not (hermes_home / "webhook_subscriptions.tmp").exists()

    def test_roundtrip(self, hermes_home):
        import hermes_cli.webhook as wh
        subs = {
            "hook-a": {"events": ["push", "pr"], "deliver": "chat"},
            "hook-b": {"events": [], "deliver": "log"},
        }
        wh._save_subscriptions(subs)
        loaded = wh._load_subscriptions()
        assert loaded == subs

    def test_save_creates_parent_dirs(self, tmp_path, monkeypatch):
        deep = tmp_path / "a" / "b" / "c"
        monkeypatch.setenv("HERMES_HOME", str(deep))
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_hermes_home", lambda: deep)
        wh._save_subscriptions({"x": {}})
        assert (deep / "webhook_subscriptions.json").exists()


# ---------------------------------------------------------------------------
# _get_webhook_config / _is_webhook_enabled / _get_webhook_base_url
# ---------------------------------------------------------------------------

class TestWebhookConfig:

    def test_get_webhook_config_when_missing(self, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_config", lambda: {})
        assert wh._get_webhook_config() == {}

    def test_is_webhook_enabled_false(self, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_config", lambda: {})
        assert wh._is_webhook_enabled() is False

    def test_is_webhook_enabled_true(self, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(
            wh, "_get_webhook_config",
            lambda: {"enabled": True, "extra": {}},
        )
        assert wh._is_webhook_enabled() is True

    def test_base_url_defaults(self, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(
            wh, "_get_webhook_config",
            lambda: {"enabled": True, "extra": {}},
        )
        url = wh._get_webhook_base_url()
        assert url == "http://localhost:8644"

    def test_base_url_custom_host(self, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(
            wh, "_get_webhook_config",
            lambda: {"enabled": True, "extra": {"host": "192.168.1.5", "port": 9999}},
        )
        url = wh._get_webhook_base_url()
        assert url == "http://192.168.1.5:9999"

    def test_base_url_wildcard_host_shows_localhost(self, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(
            wh, "_get_webhook_config",
            lambda: {"enabled": True, "extra": {"host": "0.0.0.0"}},
        )
        url = wh._get_webhook_base_url()
        assert url.startswith("http://localhost:")


# ---------------------------------------------------------------------------
# _require_webhook_enabled
# ---------------------------------------------------------------------------

class TestRequireWebhookEnabled:

    def test_returns_true_when_enabled(self, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
        assert wh._require_webhook_enabled() is True
        assert capsys.readouterr().out == ""

    def test_returns_false_and_prints_hint_when_disabled(self, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: False)
        assert wh._require_webhook_enabled() is False
        output = capsys.readouterr().out
        assert "Webhook platform is not enabled" in output


# ---------------------------------------------------------------------------
# webhook_command dispatcher
# ---------------------------------------------------------------------------

class TestWebhookCommand:

    def test_no_action_prints_usage(self, webhook_enabled, capsys):
        import hermes_cli.webhook as wh
        args = SimpleNamespace(webhook_action=None)
        wh.webhook_command(args)
        output = capsys.readouterr().out
        assert "Usage:" in output

    def test_disabled_exits_early(self, webhook_disabled, capsys):
        import hermes_cli.webhook as wh
        args = SimpleNamespace(webhook_action="subscribe")
        wh.webhook_command(args)
        output = capsys.readouterr().out
        assert "not enabled" in output

    def test_subscribe_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_subscribe", lambda a: called.append("sub"))
        args = SimpleNamespace(webhook_action="subscribe")
        wh.webhook_command(args)
        assert called == ["sub"]

    def test_add_alias_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_subscribe", lambda a: called.append("sub"))
        args = SimpleNamespace(webhook_action="add")
        wh.webhook_command(args)
        assert called == ["sub"]

    def test_list_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_list", lambda a: called.append("list"))
        args = SimpleNamespace(webhook_action="list")
        wh.webhook_command(args)
        assert called == ["list"]

    def test_ls_alias_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_list", lambda a: called.append("list"))
        args = SimpleNamespace(webhook_action="ls")
        wh.webhook_command(args)
        assert called == ["list"]

    def test_remove_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_remove", lambda a: called.append("rm"))
        args = SimpleNamespace(webhook_action="remove")
        wh.webhook_command(args)
        assert called == ["rm"]

    def test_rm_alias_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_remove", lambda a: called.append("rm"))
        args = SimpleNamespace(webhook_action="rm")
        wh.webhook_command(args)
        assert called == ["rm"]

    def test_test_dispatched(self, webhook_enabled, monkeypatch, hermes_home):
        import hermes_cli.webhook as wh
        called = []
        monkeypatch.setattr(wh, "_cmd_test", lambda a: called.append("test"))
        args = SimpleNamespace(webhook_action="test")
        wh.webhook_command(args)
        assert called == ["test"]


# ---------------------------------------------------------------------------
# _cmd_subscribe
# ---------------------------------------------------------------------------

class TestCmdSubscribe:

    def test_creates_subscription(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="my-hook", events="push,pr")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert "my-hook" in subs
        assert subs["my-hook"]["events"] == ["push", "pr"]

    def test_name_normalization(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name=" My Hook ")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert "my-hook" in subs

    def test_rejects_invalid_name(self, hermes_home, webhook_enabled, make_args, capsys):
        import hermes_cli.webhook as wh
        args = make_args(name="!!!bad!!!")
        wh._cmd_subscribe(args)
        output = capsys.readouterr().out
        assert "Invalid name" in output
        assert wh._load_subscriptions() == {}

    def test_generates_secret_when_not_provided(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert len(subs["hook1"]["secret"]) > 20

    def test_uses_provided_secret(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", secret="my-secret-123")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["secret"] == "my-secret-123"

    def test_default_deliver_is_log(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["deliver"] == "log"

    def test_custom_deliver(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", deliver="chat")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["deliver"] == "chat"

    def test_deliver_chat_id(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", deliver="chat", deliver_chat_id="12345")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["deliver_extra"] == {"chat_id": "12345"}

    def test_description_default(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert "hook1" in subs["hook1"]["description"]

    def test_custom_description(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", description="Deploy notifications")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["description"] == "Deploy notifications"

    def test_skills_parsed(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", skills="skill-a,skill-b")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["skills"] == ["skill-a", "skill-b"]

    def test_prompt_saved(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", prompt="Summarize the deployment")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert subs["hook1"]["prompt"] == "Summarize the deployment"

    def test_created_at_timestamp(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1")
        wh._cmd_subscribe(args)
        subs = wh._load_subscriptions()
        assert "T" in subs["hook1"]["created_at"]
        assert subs["hook1"]["created_at"].endswith("Z")

    def test_update_existing(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        # Create first
        args = make_args(name="hook1", events="push")
        wh._cmd_subscribe(args)
        # Update
        args2 = make_args(name="hook1", events="push,pr")
        wh._cmd_subscribe(args2)
        output = capsys.readouterr().out
        assert "Updated" in output

    def test_prints_url_and_secret(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        args = make_args(name="hook1", secret="sec123")
        wh._cmd_subscribe(args)
        output = capsys.readouterr().out
        assert "http://localhost:8644/webhooks/hook1" in output
        assert "sec123" in output


# ---------------------------------------------------------------------------
# _cmd_list
# ---------------------------------------------------------------------------

class TestCmdList:

    def test_no_subscriptions_message(self, hermes_home, webhook_enabled, make_args, capsys):
        import hermes_cli.webhook as wh
        args = make_args()
        wh._cmd_list(args)
        output = capsys.readouterr().out
        assert "No dynamic webhook subscriptions" in output

    def test_lists_existing_subscriptions(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        # Create a subscription
        wh._save_subscriptions({"deploy": {"events": ["push"], "deliver": "chat", "description": "Deploy hook"}})
        wh._cmd_list(make_args())
        output = capsys.readouterr().out
        assert "deploy" in output
        assert "push" in output
        assert "Deploy hook" in output
        assert "1 webhook subscription(s)" in output

    def test_lists_multiple(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"a": {"events": []}, "b": {"events": []}})
        wh._cmd_list(make_args())
        output = capsys.readouterr().out
        assert "2 webhook subscription(s)" in output

    def test_shows_all_events(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"h": {"events": ["push", "pr", "issue"]}})
        wh._cmd_list(make_args())
        output = capsys.readouterr().out
        assert "push, pr, issue" in output

    def test_no_events_shows_all(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"h": {"events": []}})
        wh._cmd_list(make_args())
        output = capsys.readouterr().out
        assert "(all)" in output


# ---------------------------------------------------------------------------
# _cmd_remove
# ---------------------------------------------------------------------------

class TestCmdRemove:

    def test_remove_existing(self, hermes_home, webhook_enabled, make_args, capsys):
        import hermes_cli.webhook as wh
        wh._save_subscriptions({"myhook": {"events": []}})
        args = make_args(webhook_action="remove", name="myhook")
        wh._cmd_remove(args)
        assert wh._load_subscriptions() == {}
        output = capsys.readouterr().out
        assert "Removed" in output

    def test_remove_nonexistent(self, hermes_home, webhook_enabled, make_args, capsys):
        import hermes_cli.webhook as wh
        args = make_args(webhook_action="remove", name="nope")
        wh._cmd_remove(args)
        output = capsys.readouterr().out
        assert "No subscription" in output

    def test_remove_preserves_others(self, hermes_home, webhook_enabled, make_args):
        import hermes_cli.webhook as wh
        wh._save_subscriptions({"a": {"events": []}, "b": {"events": []}})
        args = make_args(webhook_action="remove", name="a")
        wh._cmd_remove(args)
        subs = wh._load_subscriptions()
        assert "a" not in subs
        assert "b" in subs

    def test_name_stripped(self, hermes_home, webhook_enabled, make_args, capsys):
        import hermes_cli.webhook as wh
        wh._save_subscriptions({"myhook": {"events": []}})
        args = make_args(webhook_action="remove", name=" myhook ")
        wh._cmd_remove(args)
        assert "myhook" not in wh._load_subscriptions()


# ---------------------------------------------------------------------------
# _cmd_test
# ---------------------------------------------------------------------------

class TestCmdTest:

    def test_test_nonexistent_subscription(self, hermes_home, webhook_enabled, make_args, capsys):
        import hermes_cli.webhook as wh
        args = make_args(webhook_action="test", name="nope")
        wh._cmd_test(args)
        output = capsys.readouterr().out
        assert "No subscription" in output

    def test_test_sends_post(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"myhook": {"secret": "s3cret"}})

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"ok"

        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda s: mock_resp
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            args = make_args(webhook_action="test", name="myhook", payload=None)
            wh._cmd_test(args)
            assert mock_urlopen.called

        output = capsys.readouterr().out
        assert "Sending test POST" in output
        assert "200" in output

    def test_test_custom_payload(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"myhook": {"secret": "s3cret"}})

        captured_req = {}
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"ok"
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda s: mock_resp
        mock_cm.__exit__ = MagicMock(return_value=False)

        def fake_urlopen(req, **kw):
            captured_req["data"] = req.data
            captured_req["headers"] = dict(req.headers)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            custom = '{"custom": true}'
            args = make_args(webhook_action="test", name="myhook", payload=custom)
            wh._cmd_test(args)

        assert captured_req["data"] == custom.encode()
        headers_lower = {k.lower(): v for k, v in captured_req["headers"].items()}
        assert "x-hub-signature-256" in headers_lower

    def test_test_connection_error(self, hermes_home, webhook_enabled, make_args, monkeypatch, capsys):
        import hermes_cli.webhook as wh
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"myhook": {"secret": "s3cret"}})

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            args = make_args(webhook_action="test", name="myhook")
            wh._cmd_test(args)

        output = capsys.readouterr().out
        assert "Error" in output
        assert "Connection refused" in output

    def test_test_hmac_signature(self, hermes_home, webhook_enabled, make_args, monkeypatch):
        """Verify HMAC-SHA256 signature is computed correctly."""
        import hermes_cli.webhook as wh
        import hmac
        import hashlib
        monkeypatch.setattr(wh, "_get_webhook_base_url", lambda: "http://localhost:8644")
        wh._save_subscriptions({"myhook": {"secret": "s3cret"}})

        captured_headers = {}
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"ok"
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda s: mock_resp
        mock_cm.__exit__ = MagicMock(return_value=False)

        def fake_urlopen(req, **kw):
            hdrs = {k.lower(): v for k, v in req.headers.items()}
            captured_headers["sig"] = hdrs.get("x-hub-signature-256", "")
            captured_headers["data"] = req.data.decode()
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            args = make_args(webhook_action="test", name="myhook", payload=None)
            wh._cmd_test(args)

        # Verify the HMAC against the actual payload that was sent
        expected_sig = "sha256=" + hmac.new(
            b"s3cret", captured_headers["data"].encode(), hashlib.sha256
        ).hexdigest()
        assert captured_headers["sig"] == expected_sig


# ---------------------------------------------------------------------------
# _subscriptions_path edge cases
# ---------------------------------------------------------------------------

class TestSubscriptionsPath:

    def test_path_uses_hermes_home(self, hermes_home):
        import hermes_cli.webhook as wh
        path = wh._subscriptions_path()
        assert path == hermes_home / "webhook_subscriptions.json"

    def test_filename_constant(self):
        import hermes_cli.webhook as wh
        assert wh._SUBSCRIPTIONS_FILENAME == "webhook_subscriptions.json"
