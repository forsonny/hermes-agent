"""Tests for core auth utilities and auth store persistence layer.

Covers the fundamental building blocks of hermes_cli/auth.py that are
not tested by the provider-specific test files:
  - has_usable_secret(), _token_fingerprint()
  - _parse_iso_timestamp(), _is_expiring(), _coerce_ttl_seconds()
  - _optional_base_url(), _decode_jwt_claims()
  - AuthError, format_auth_error()
  - _load_auth_store(), _save_auth_store()
  - _load_provider_state(), _save_provider_state()
  - read_credential_pool(), write_credential_pool()
  - suppress_credential_source(), is_source_suppressed()
  - get_active_provider(), get_provider_auth_state()
  - resolve_provider() (alias resolution, auto-detection)
  - _oauth_trace_enabled(), _oauth_trace()
  - _resolve_kimi_base_url()
  - clear_provider_auth(), deactivate_provider()
  - _codex_access_token_is_expiring()
"""

from __future__ import annotations

import base64
import json
import os
import stat
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp dir so tests don't touch ~/.hermes."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.auth.get_hermes_home", lambda: home)
    yield home


@pytest.fixture
def auth_home(_isolate_hermes_home):
    """Return the isolated HERMES_HOME path."""
    return _isolate_hermes_home


@pytest.fixture
def auth_store_path(auth_home):
    """Return the path to auth.json in the isolated home."""
    return auth_home / "auth.json"


def _write_auth_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# has_usable_secret
# ---------------------------------------------------------------------------

class TestHasUsableSecret:
    """Tests for has_usable_secret()."""

    def test_valid_secret(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("sk-abc123def456") is True

    def test_empty_string(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("") is False

    def test_whitespace_only(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("   ") is False

    def test_too_short(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("abc") is False

    def test_min_length_custom(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("ab", min_length=3) is False
        assert has_usable_secret("abc", min_length=3) is True

    def test_placeholder_star(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("*") is False

    def test_placeholder_changeme(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("changeme") is False

    def test_placeholder_your_api_key(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("your_api_key") is False

    def test_placeholder_dummy(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("dummy") is False

    def test_placeholder_null(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("null") is False

    def test_placeholder_none(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("none") is False

    def test_non_string_returns_false(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret(12345) is False
        assert has_usable_secret(None) is False

    def test_placeholder_case_insensitive(self):
        from hermes_cli.auth import has_usable_secret
        assert has_usable_secret("CHANGEME") is False
        assert has_usable_secret("Placeholder") is False

    def test_stripped_before_check(self):
        from hermes_cli.auth import has_usable_secret
        # "  abc  " has length 3 after strip, below default min_length=4
        assert has_usable_secret("  abc  ") is False


# ---------------------------------------------------------------------------
# _token_fingerprint
# ---------------------------------------------------------------------------

class TestTokenFingerprint:
    """Tests for _token_fingerprint()."""

    def test_returns_hex_string(self):
        from hermes_cli.auth import _token_fingerprint
        result = _token_fingerprint("my-secret-token")
        assert isinstance(result, str)
        assert len(result) == 12
        # Should be hex chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_none_returns_none(self):
        from hermes_cli.auth import _token_fingerprint
        assert _token_fingerprint(None) is None

    def test_empty_returns_none(self):
        from hermes_cli.auth import _token_fingerprint
        assert _token_fingerprint("") is None

    def test_whitespace_only_returns_none(self):
        from hermes_cli.auth import _token_fingerprint
        assert _token_fingerprint("   ") is None

    def test_non_string_returns_none(self):
        from hermes_cli.auth import _token_fingerprint
        assert _token_fingerprint(12345) is None

    def test_deterministic(self):
        from hermes_cli.auth import _token_fingerprint
        fp1 = _token_fingerprint("same-token")
        fp2 = _token_fingerprint("same-token")
        assert fp1 == fp2

    def test_different_tokens_different_fingerprints(self):
        from hermes_cli.auth import _token_fingerprint
        fp1 = _token_fingerprint("token-alpha")
        fp2 = _token_fingerprint("token-beta")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# _parse_iso_timestamp
# ---------------------------------------------------------------------------

class TestParseIsoTimestamp:
    """Tests for _parse_iso_timestamp()."""

    def test_valid_iso_format(self):
        from hermes_cli.auth import _parse_iso_timestamp
        ts = "2025-01-15T12:30:00+00:00"
        result = _parse_iso_timestamp(ts)
        assert result is not None
        assert isinstance(result, float)

    def test_z_suffix(self):
        from hermes_cli.auth import _parse_iso_timestamp
        ts = "2025-01-15T12:30:00Z"
        result = _parse_iso_timestamp(ts)
        assert result is not None
        # Should be same as +00:00 version
        expected = _parse_iso_timestamp("2025-01-15T12:30:00+00:00")
        assert result == expected

    def test_none_returns_none(self):
        from hermes_cli.auth import _parse_iso_timestamp
        assert _parse_iso_timestamp(None) is None

    def test_empty_returns_none(self):
        from hermes_cli.auth import _parse_iso_timestamp
        assert _parse_iso_timestamp("") is None

    def test_non_string_returns_none(self):
        from hermes_cli.auth import _parse_iso_timestamp
        assert _parse_iso_timestamp(12345) is None

    def test_invalid_format_returns_none(self):
        from hermes_cli.auth import _parse_iso_timestamp
        assert _parse_iso_timestamp("not-a-date") is None

    def test_naive_datetime_gets_utc(self):
        from hermes_cli.auth import _parse_iso_timestamp
        ts = "2025-01-15T12:30:00"
        result = _parse_iso_timestamp(ts)
        assert result is not None
        # Naive datetime should be treated as UTC


# ---------------------------------------------------------------------------
# _is_expiring
# ---------------------------------------------------------------------------

class TestIsExpiring:
    """Tests for _is_expiring()."""

    def test_already_expired(self):
        from hermes_cli.auth import _is_expiring
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        assert _is_expiring(past, 0) is True

    def test_not_expired(self):
        from hermes_cli.auth import _is_expiring
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert _is_expiring(future, 0) is False

    def test_within_skew_is_expiring(self):
        from hermes_cli.auth import _is_expiring
        # Token expires in 30 seconds, skew is 60 seconds -> expiring
        near_future = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        assert _is_expiring(near_future, 60) is True

    def test_none_always_expiring(self):
        from hermes_cli.auth import _is_expiring
        assert _is_expiring(None, 0) is True

    def test_invalid_always_expiring(self):
        from hermes_cli.auth import _is_expiring
        assert _is_expiring("garbage", 0) is True


# ---------------------------------------------------------------------------
# _coerce_ttl_seconds
# ---------------------------------------------------------------------------

class TestCoerceTtlSeconds:
    """Tests for _coerce_ttl_seconds()."""

    def test_valid_int(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds(3600) == 3600

    def test_valid_string(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds("3600") == 3600

    def test_negative_clamped_to_zero(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds(-100) == 0

    def test_none_returns_zero(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds(None) == 0

    def test_string_non_numeric_returns_zero(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds("abc") == 0

    def test_float_truncated(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds(3600.9) == 3600

    def test_zero_stays_zero(self):
        from hermes_cli.auth import _coerce_ttl_seconds
        assert _coerce_ttl_seconds(0) == 0


# ---------------------------------------------------------------------------
# _optional_base_url
# ---------------------------------------------------------------------------

class TestOptionalBaseUrl:
    """Tests for _optional_base_url()."""

    def test_valid_url(self):
        from hermes_cli.auth import _optional_base_url
        assert _optional_base_url("https://api.example.com/v1") == "https://api.example.com/v1"

    def test_trailing_slash_removed(self):
        from hermes_cli.auth import _optional_base_url
        assert _optional_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"

    def test_empty_returns_none(self):
        from hermes_cli.auth import _optional_base_url
        assert _optional_base_url("") is None

    def test_whitespace_only_returns_none(self):
        from hermes_cli.auth import _optional_base_url
        assert _optional_base_url("   ") is None

    def test_non_string_returns_none(self):
        from hermes_cli.auth import _optional_base_url
        assert _optional_base_url(123) is None

    def test_stripped_whitespace(self):
        from hermes_cli.auth import _optional_base_url
        assert _optional_base_url("  https://api.example.com  ") == "https://api.example.com"


# ---------------------------------------------------------------------------
# _decode_jwt_claims
# ---------------------------------------------------------------------------

class TestDecodeJwtClaims:
    """Tests for _decode_jwt_claims()."""

    def _make_jwt(self, payload: dict) -> str:
        """Build a minimal JWT-like string (header.payload.sig)."""
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig = base64.urlsafe_b64encode(b"signature").decode().rstrip("=")
        return f"{header}.{body}.{sig}"

    def test_valid_jwt(self):
        from hermes_cli.auth import _decode_jwt_claims
        token = self._make_jwt({"sub": "user123", "exp": 1700000000})
        claims = _decode_jwt_claims(token)
        assert claims["sub"] == "user123"
        assert claims["exp"] == 1700000000

    def test_non_string_returns_empty(self):
        from hermes_cli.auth import _decode_jwt_claims
        assert _decode_jwt_claims(12345) == {}

    def test_wrong_dot_count_returns_empty(self):
        from hermes_cli.auth import _decode_jwt_claims
        assert _decode_jwt_claims("a.b") == {}
        assert _decode_jwt_claims("a.b.c.d") == {}

    def test_invalid_base64_returns_empty(self):
        from hermes_cli.auth import _decode_jwt_claims
        assert _decode_jwt_claims("header.!!!invalid!!!.sig") == {}

    def test_non_dict_payload_returns_empty(self):
        from hermes_cli.auth import _decode_jwt_claims
        token = self._make_jwt(["not", "a", "dict"])
        # The function checks isinstance(claims, dict)
        assert _decode_jwt_claims(token) == {}


# ---------------------------------------------------------------------------
# _codex_access_token_is_expiring
# ---------------------------------------------------------------------------

class TestCodexAccessTokenIsExpiring:
    """Tests for _codex_access_token_is_expiring()."""

    def _make_jwt(self, payload: dict) -> str:
        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig = base64.urlsafe_b64encode(b"sig").decode().rstrip("=")
        return f"{header}.{body}.{sig}"

    def test_not_expired(self):
        from hermes_cli.auth import _codex_access_token_is_expiring
        future_exp = int(time.time()) + 3600
        token = self._make_jwt({"exp": future_exp})
        assert _codex_access_token_is_expiring(token, 60) is False

    def test_expired(self):
        from hermes_cli.auth import _codex_access_token_is_expiring
        past_exp = int(time.time()) - 3600
        token = self._make_jwt({"exp": past_exp})
        assert _codex_access_token_is_expiring(token, 0) is True

    def test_within_skew(self):
        from hermes_cli.auth import _codex_access_token_is_expiring
        near_exp = int(time.time()) + 30
        token = self._make_jwt({"exp": near_exp})
        assert _codex_access_token_is_expiring(token, 60) is True

    def test_no_exp_claim_returns_false(self):
        from hermes_cli.auth import _codex_access_token_is_expiring
        token = self._make_jwt({"sub": "user"})
        assert _codex_access_token_is_expiring(token, 60) is False

    def test_non_numeric_exp_returns_false(self):
        from hermes_cli.auth import _codex_access_token_is_expiring
        token = self._make_jwt({"exp": "not-a-number"})
        assert _codex_access_token_is_expiring(token, 60) is False

    def test_non_jwt_string_returns_false(self):
        from hermes_cli.auth import _codex_access_token_is_expiring
        assert _codex_access_token_is_expiring("not-a-jwt", 60) is False


# ---------------------------------------------------------------------------
# AuthError
# ---------------------------------------------------------------------------

class TestAuthError:
    """Tests for AuthError class."""

    def test_basic_message(self):
        from hermes_cli.auth import AuthError
        err = AuthError("test error")
        assert str(err) == "test error"
        assert err.provider == ""
        assert err.code is None
        assert err.relogin_required is False

    def test_full_kwargs(self):
        from hermes_cli.auth import AuthError
        err = AuthError("auth failed", provider="nous", code="token_expired", relogin_required=True)
        assert err.provider == "nous"
        assert err.code == "token_expired"
        assert err.relogin_required is True

    def test_is_runtime_error(self):
        from hermes_cli.auth import AuthError
        err = AuthError("test")
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# format_auth_error
# ---------------------------------------------------------------------------

class TestFormatAuthError:
    """Tests for format_auth_error()."""

    def test_non_auth_error_passthrough(self):
        from hermes_cli.auth import format_auth_error
        assert format_auth_error(ValueError("oops")) == "oops"

    def test_relogin_required_hint(self):
        from hermes_cli.auth import format_auth_error, AuthError
        err = AuthError("Token expired", relogin_required=True)
        result = format_auth_error(err)
        assert "hermes model" in result.lower()

    def test_subscription_required(self):
        from hermes_cli.auth import format_auth_error, AuthError
        err = AuthError("Sub required", code="subscription_required")
        result = format_auth_error(err)
        assert "subscription" in result.lower()

    def test_insufficient_credits(self):
        from hermes_cli.auth import format_auth_error, AuthError
        err = AuthError("No credits", code="insufficient_credits")
        result = format_auth_error(err)
        assert "credits" in result.lower()

    def test_temporarily_unavailable(self):
        from hermes_cli.auth import format_auth_error, AuthError
        err = AuthError("Rate limited", code="temporarily_unavailable")
        result = format_auth_error(err)
        assert "retry" in result.lower()

    def test_plain_auth_error(self):
        from hermes_cli.auth import format_auth_error, AuthError
        err = AuthError("Something went wrong")
        result = format_auth_error(err)
        assert result == "Something went wrong"


# ---------------------------------------------------------------------------
# _oauth_trace_enabled / _oauth_trace
# ---------------------------------------------------------------------------

class TestOAuthTrace:
    """Tests for _oauth_trace_enabled() and _oauth_trace()."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_OAUTH_TRACE", raising=False)
        from hermes_cli.auth import _oauth_trace_enabled
        assert _oauth_trace_enabled() is False

    def test_enabled_with_1(self, monkeypatch):
        monkeypatch.setenv("HERMES_OAUTH_TRACE", "1")
        from hermes_cli.auth import _oauth_trace_enabled
        # Need to reimport or call directly
        import hermes_cli.auth as auth_mod
        assert auth_mod._oauth_trace_enabled() is True

    def test_enabled_with_true(self, monkeypatch):
        monkeypatch.setenv("HERMES_OAUTH_TRACE", "true")
        import hermes_cli.auth as auth_mod
        assert auth_mod._oauth_trace_enabled() is True

    def test_not_enabled_with_random(self, monkeypatch):
        monkeypatch.setenv("HERMES_OAUTH_TRACE", "nope")
        import hermes_cli.auth as auth_mod
        assert auth_mod._oauth_trace_enabled() is False

    def test_trace_does_nothing_when_disabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_OAUTH_TRACE", raising=False)
        import hermes_cli.auth as auth_mod
        # Should not raise
        auth_mod._oauth_trace("test_event", sequence_id="123")


# ---------------------------------------------------------------------------
# Auth Store: _load_auth_store / _save_auth_store
# ---------------------------------------------------------------------------

class TestAuthStore:
    """Tests for _load_auth_store() and _save_auth_store()."""

    def test_load_missing_file_returns_default(self, auth_store_path):
        from hermes_cli.auth import _load_auth_store
        result = _load_auth_store()
        assert "version" in result
        assert "providers" in result
        assert result["providers"] == {}

    def test_load_valid_file(self, auth_store_path):
        from hermes_cli.auth import _load_auth_store
        data = {"version": 2, "providers": {"nous": {"access_token": "abc"}}}
        _write_auth_json(auth_store_path, data)
        result = _load_auth_store()
        assert result["providers"]["nous"]["access_token"] == "abc"

    def test_load_invalid_json_returns_default(self, auth_store_path):
        from hermes_cli.auth import _load_auth_store
        auth_store_path.write_text("not valid json{{{")
        result = _load_auth_store()
        assert result["providers"] == {}

    def test_load_systems_format_migration(self, auth_store_path):
        from hermes_cli.auth import _load_auth_store
        data = {"systems": {"nous_portal": {"access_token": "migrated"}}}
        _write_auth_json(auth_store_path, data)
        result = _load_auth_store()
        assert "nous" in result["providers"]
        assert result["active_provider"] == "nous"

    def test_save_creates_file(self, auth_home):
        from hermes_cli.auth import _save_auth_store
        store = {"providers": {"nous": {"access_token": "test123"}}}
        path = _save_auth_store(store)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["providers"]["nous"]["access_token"] == "test123"

    def test_save_sets_version_and_timestamp(self, auth_home):
        from hermes_cli.auth import _save_auth_store, AUTH_STORE_VERSION
        store = {"providers": {}}
        _save_auth_store(store)
        loaded = json.loads((auth_home / "auth.json").read_text())
        assert loaded["version"] == AUTH_STORE_VERSION
        assert "updated_at" in loaded

    def test_save_restricts_permissions(self, auth_home):
        from hermes_cli.auth import _save_auth_store
        store = {"providers": {}}
        path = _save_auth_store(store)
        mode = path.stat().st_mode
        assert mode & stat.S_IRGRP == 0
        assert mode & stat.S_IWGRP == 0
        assert mode & stat.S_IROTH == 0
        assert mode & stat.S_IWOTH == 0

    def test_roundtrip(self, auth_home):
        from hermes_cli.auth import _load_auth_store, _save_auth_store
        store = {"providers": {"nous": {"access_token": "roundtrip"}}}
        _save_auth_store(store)
        loaded = _load_auth_store()
        assert loaded["providers"]["nous"]["access_token"] == "roundtrip"

    def test_load_with_credential_pool(self, auth_store_path):
        from hermes_cli.auth import _load_auth_store
        data = {"credential_pool": {"nous": [{"label": "test"}]}}
        _write_auth_json(auth_store_path, data)
        result = _load_auth_store()
        assert "credential_pool" in result


# ---------------------------------------------------------------------------
# Provider state: _load_provider_state / _save_provider_state
# ---------------------------------------------------------------------------

class TestProviderState:
    """Tests for _load_provider_state() and _save_provider_state()."""

    def test_load_existing_state(self):
        from hermes_cli.auth import _load_provider_state
        store = {"providers": {"nous": {"access_token": "abc", "refresh_token": "def"}}}
        result = _load_provider_state(store, "nous")
        assert result is not None
        assert result["access_token"] == "abc"

    def test_load_missing_provider_returns_none(self):
        from hermes_cli.auth import _load_provider_state
        store = {"providers": {"nous": {"access_token": "abc"}}}
        assert _load_provider_state(store, "codex") is None

    def test_load_no_providers_key_returns_none(self):
        from hermes_cli.auth import _load_provider_state
        assert _load_provider_state({}, "nous") is None

    def test_load_non_dict_provider_returns_none(self):
        from hermes_cli.auth import _load_provider_state
        store = {"providers": {"nous": "not-a-dict"}}
        assert _load_provider_state(store, "nous") is None

    def test_save_state(self):
        from hermes_cli.auth import _save_provider_state
        store = {"providers": {}}
        _save_provider_state(store, "nous", {"access_token": "new"})
        assert store["providers"]["nous"]["access_token"] == "new"
        assert store["active_provider"] == "nous"

    def test_save_state_fixes_non_dict_providers(self):
        from hermes_cli.auth import _save_provider_state
        store = {"providers": "broken"}
        _save_provider_state(store, "nous", {"access_token": "fixed"})
        assert store["providers"]["nous"]["access_token"] == "fixed"


# ---------------------------------------------------------------------------
# Credential pool
# ---------------------------------------------------------------------------

class TestCredentialPool:
    """Tests for read_credential_pool() and write_credential_pool()."""

    def test_read_empty_pool(self, auth_home):
        from hermes_cli.auth import read_credential_pool
        result = read_credential_pool()
        assert result == {}

    def test_write_and_read(self, auth_home):
        from hermes_cli.auth import read_credential_pool, write_credential_pool
        entries = [{"label": "primary", "api_key": "sk-test123"}]
        write_credential_pool("nous", entries)
        result = read_credential_pool("nous")
        assert len(result) == 1
        assert result[0]["label"] == "primary"

    def test_read_nonexistent_provider_returns_empty(self, auth_home):
        from hermes_cli.auth import read_credential_pool, write_credential_pool
        write_credential_pool("nous", [{"label": "test"}])
        assert read_credential_pool("codex") == []

    def test_write_creates_auth_json(self, auth_home):
        from hermes_cli.auth import write_credential_pool
        assert not (auth_home / "auth.json").exists()
        write_credential_pool("nous", [])
        assert (auth_home / "auth.json").exists()


# ---------------------------------------------------------------------------
# Source suppression
# ---------------------------------------------------------------------------

class TestSourceSuppression:
    """Tests for suppress_credential_source() and is_source_suppressed()."""

    def test_not_suppressed_by_default(self, auth_home):
        from hermes_cli.auth import is_source_suppressed
        assert is_source_suppressed("nous", "env") is False

    def test_suppress_and_check(self, auth_home):
        from hermes_cli.auth import suppress_credential_source, is_source_suppressed
        suppress_credential_source("nous", "env")
        assert is_source_suppressed("nous", "env") is True

    def test_different_source_not_affected(self, auth_home):
        from hermes_cli.auth import suppress_credential_source, is_source_suppressed
        suppress_credential_source("nous", "env")
        assert is_source_suppressed("nous", "config") is False

    def test_different_provider_not_affected(self, auth_home):
        from hermes_cli.auth import suppress_credential_source, is_source_suppressed
        suppress_credential_source("nous", "env")
        assert is_source_suppressed("codex", "env") is False

    def test_duplicate_suppress_is_idempotent(self, auth_home):
        from hermes_cli.auth import suppress_credential_source, is_source_suppressed
        suppress_credential_source("nous", "env")
        suppress_credential_source("nous", "env")
        # Should only appear once
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        suppressed = store.get("suppressed_sources", {}).get("nous", [])
        assert suppressed.count("env") == 1


# ---------------------------------------------------------------------------
# get_active_provider / get_provider_auth_state
# ---------------------------------------------------------------------------

class TestGetActiveProvider:
    """Tests for get_active_provider() and get_provider_auth_state()."""

    def test_no_active_provider(self, auth_home):
        from hermes_cli.auth import get_active_provider
        assert get_active_provider() is None

    def test_active_provider_after_save(self, auth_home):
        from hermes_cli.auth import _save_auth_store, get_active_provider
        _save_auth_store({"providers": {}, "active_provider": "nous"})
        assert get_active_provider() == "nous"

    def test_get_provider_auth_state(self, auth_home):
        from hermes_cli.auth import _save_auth_store, get_provider_auth_state
        _save_auth_store({"providers": {"nous": {"access_token": "abc"}}})
        state = get_provider_auth_state("nous")
        assert state is not None
        assert state["access_token"] == "abc"

    def test_get_provider_auth_state_missing(self, auth_home):
        from hermes_cli.auth import get_provider_auth_state
        assert get_provider_auth_state("nonexistent") is None


# ---------------------------------------------------------------------------
# _resolve_kimi_base_url
# ---------------------------------------------------------------------------

class TestResolveKimiBaseUrl:
    """Tests for _resolve_kimi_base_url()."""

    def test_env_override_wins(self):
        from hermes_cli.auth import _resolve_kimi_base_url
        result = _resolve_kimi_base_url("sk-kimi-xxx", "https://default.com", "https://custom.com")
        assert result == "https://custom.com"

    def test_kimi_prefix_routes_to_code_endpoint(self):
        from hermes_cli.auth import _resolve_kimi_base_url, KIMI_CODE_BASE_URL
        result = _resolve_kimi_base_url("sk-kimi-test123", "https://default.com", "")
        assert result == KIMI_CODE_BASE_URL

    def test_regular_key_uses_default(self):
        from hermes_cli.auth import _resolve_kimi_base_url
        result = _resolve_kimi_base_url("sk-regular-key", "https://default.com", "")
        assert result == "https://default.com"


# ---------------------------------------------------------------------------
# resolve_provider
# ---------------------------------------------------------------------------

class TestResolveProvider:
    """Tests for resolve_provider() alias resolution and auto-detection."""

    def test_explicit_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("openrouter") == "openrouter"

    def test_explicit_custom(self, monkeypatch):
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("custom") == "custom"

    def test_alias_glm_to_zai(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from hermes_cli.auth import resolve_provider
        # "glm" is an alias for "zai"
        # But "zai" is in PROVIDER_REGISTRY, so it returns "zai"
        # However, since zai needs GLM_API_KEY env, let's set it
        monkeypatch.setenv("GLM_API_KEY", "test-key-12345")
        assert resolve_provider("glm") == "zai"

    def test_alias_claude_to_anthropic(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("claude") == "anthropic"

    def test_alias_google_to_gemini(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-12345")
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("google") == "gemini"

    def test_unknown_provider_raises(self, monkeypatch):
        from hermes_cli.auth import resolve_provider, AuthError
        with pytest.raises(AuthError, match="Unknown provider"):
            resolve_provider("nonexistent_provider_xyz")

    def test_auto_with_openai_key(self, monkeypatch, auth_home):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-testkey12345")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Mock auth store to return empty
        monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {"providers": {}})
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("auto") == "openrouter"

    def test_auto_with_explicit_creds(self, monkeypatch, auth_home):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("auto", explicit_api_key="sk-test") == "openrouter"

    def test_auto_no_provider_raises(self, monkeypatch, auth_home):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Clear all provider env vars to force the error
        for _key in ["GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY", "MINIMAX_API_KEY",
                      "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY",
                      "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
                      "MINIMAX_CN_API_KEY", "HF_TOKEN", "AI_GATEWAY_API_KEY",
                      "OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY", "KILOCODE_API_KEY",
                      "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]:
            monkeypatch.delenv(_key, raising=False)
        monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {"providers": {}})
        from hermes_cli.auth import resolve_provider, AuthError
        with pytest.raises(AuthError, match="No inference provider configured"):
            resolve_provider("auto")

    def test_alias_deepseek(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-12345")
        from hermes_cli.auth import resolve_provider
        # "deepseek" is already a canonical name, not an alias
        assert resolve_provider("deepseek") == "deepseek"

    def test_alias_hf_to_huggingface(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("hf") == "huggingface"


# ---------------------------------------------------------------------------
# clear_provider_auth
# ---------------------------------------------------------------------------

class TestClearProviderAuth:
    """Tests for clear_provider_auth()."""

    def test_clear_existing_provider(self, auth_home):
        from hermes_cli.auth import _save_auth_store, clear_provider_auth, get_active_provider
        _save_auth_store({
            "providers": {"nous": {"access_token": "abc"}},
            "active_provider": "nous",
        })
        result = clear_provider_auth("nous")
        assert result is True
        assert get_active_provider() is None

    def test_clear_nonexistent_returns_false(self, auth_home):
        from hermes_cli.auth import clear_provider_auth
        assert clear_provider_auth("nonexistent") is False

    def test_clear_none_uses_active_provider(self, auth_home):
        from hermes_cli.auth import _save_auth_store, clear_provider_auth
        _save_auth_store({
            "providers": {"nous": {"access_token": "abc"}},
            "active_provider": "nous",
        })
        result = clear_provider_auth(None)
        assert result is True

    def test_clear_also_removes_credential_pool(self, auth_home):
        from hermes_cli.auth import _save_auth_store, clear_provider_auth, _load_auth_store
        _save_auth_store({
            "providers": {"nous": {"access_token": "abc"}},
            "credential_pool": {"nous": [{"label": "test"}]},
            "active_provider": "nous",
        })
        clear_provider_auth("nous")
        store = _load_auth_store()
        assert "nous" not in store.get("credential_pool", {})


# ---------------------------------------------------------------------------
# deactivate_provider
# ---------------------------------------------------------------------------

class TestDeactivateProvider:
    """Tests for deactivate_provider()."""

    def test_deactivate_clears_active(self, auth_home):
        from hermes_cli.auth import _save_auth_store, deactivate_provider, get_active_provider
        _save_auth_store({
            "providers": {"nous": {"access_token": "abc"}},
            "active_provider": "nous",
        })
        deactivate_provider()
        assert get_active_provider() is None

    def test_deactivate_preserves_credentials(self, auth_home):
        from hermes_cli.auth import _save_auth_store, deactivate_provider, get_provider_auth_state
        _save_auth_store({
            "providers": {"nous": {"access_token": "abc"}},
            "active_provider": "nous",
        })
        deactivate_provider()
        state = get_provider_auth_state("nous")
        assert state is not None
        assert state["access_token"] == "abc"


# ---------------------------------------------------------------------------
# ProviderConfig dataclass
# ---------------------------------------------------------------------------

class TestProviderConfig:
    """Tests for the ProviderConfig dataclass."""

    def test_basic_creation(self):
        from hermes_cli.auth import ProviderConfig
        config = ProviderConfig(
            id="test",
            name="Test Provider",
            auth_type="api_key",
        )
        assert config.id == "test"
        assert config.name == "Test Provider"
        assert config.auth_type == "api_key"
        assert config.api_key_env_vars == ()
        assert config.base_url_env_var == ""

    def test_with_env_vars(self):
        from hermes_cli.auth import ProviderConfig
        config = ProviderConfig(
            id="test",
            name="Test",
            auth_type="api_key",
            api_key_env_vars=("TEST_API_KEY", "TEST_KEY"),
            base_url_env_var="TEST_BASE_URL",
        )
        assert len(config.api_key_env_vars) == 2
        assert config.base_url_env_var == "TEST_BASE_URL"


# ---------------------------------------------------------------------------
# PROVIDER_REGISTRY
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    """Tests for the PROVIDER_REGISTRY constant."""

    def test_registry_is_dict(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert isinstance(PROVIDER_REGISTRY, dict)

    def test_registry_has_expected_providers(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        expected = {"nous", "openai-codex", "qwen-oauth", "copilot", "copilot-acp",
                    "gemini", "zai", "kimi-coding", "minimax", "anthropic",
                    "alibaba", "minimax-cn", "deepseek", "xai", "ai-gateway",
                    "opencode-zen", "opencode-go", "kilocode", "huggingface"}
        assert expected.issubset(set(PROVIDER_REGISTRY.keys()))

    def test_each_entry_is_provider_config(self):
        from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig
        for pid, config in PROVIDER_REGISTRY.items():
            assert isinstance(config, ProviderConfig), f"{pid} is not a ProviderConfig"
            assert config.id == pid

    def test_all_api_key_providers_have_env_vars(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pid, config in PROVIDER_REGISTRY.items():
            if config.auth_type == "api_key":
                assert config.api_key_env_vars, f"{pid} has no api_key_env_vars"


# ---------------------------------------------------------------------------
# ZAI_ENDPOINTS
# ---------------------------------------------------------------------------

class TestZaiEndpoints:
    """Tests for ZAI_ENDPOINTS constant."""

    def test_endpoints_is_list(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        assert isinstance(ZAI_ENDPOINTS, list)

    def test_endpoints_have_expected_ids(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        ids = {ep[0] for ep in ZAI_ENDPOINTS}
        assert "global" in ids
        assert "cn" in ids
        assert "coding-global" in ids
        assert "coding-cn" in ids

    def test_endpoints_have_valid_urls(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        for ep_id, base_url, model, label in ZAI_ENDPOINTS:
            assert base_url.startswith("https://"), f"{ep_id} has invalid URL"
            assert model, f"{ep_id} has no model"
            assert label, f"{ep_id} has no label"


# ---------------------------------------------------------------------------
# detect_zai_endpoint (mocked)
# ---------------------------------------------------------------------------

class TestDetectZaiEndpoint:
    """Tests for detect_zai_endpoint() with mocked HTTP."""

    def test_first_endpoint_succeeds(self, monkeypatch):
        from hermes_cli.auth import detect_zai_endpoint
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        monkeypatch.setattr("httpx.post", lambda *a, **kw: mock_resp)
        result = detect_zai_endpoint("test-key")
        assert result is not None
        assert result["id"] == "global"

    def test_all_fail_returns_none(self, monkeypatch):
        from hermes_cli.auth import detect_zai_endpoint
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        monkeypatch.setattr("httpx.post", lambda *a, **kw: mock_resp)
        result = detect_zai_endpoint("bad-key")
        assert result is None

    def test_network_error_returns_none(self, monkeypatch):
        from hermes_cli.auth import detect_zai_endpoint
        def raise_error(*a, **kw):
            raise ConnectionError("network error")
        monkeypatch.setattr("httpx.post", raise_error)
        result = detect_zai_endpoint("test-key")
        assert result is None

    def test_second_endpoint_succeeds(self, monkeypatch):
        from hermes_cli.auth import detect_zai_endpoint
        call_count = 0
        def mock_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200 if call_count == 2 else 401
            return mock_resp
        monkeypatch.setattr("httpx.post", mock_post)
        result = detect_zai_endpoint("test-key")
        assert result is not None
        assert result["id"] == "cn"


# ---------------------------------------------------------------------------
# Auth store lock
# ---------------------------------------------------------------------------

class TestAuthStoreLock:
    """Tests for _auth_store_lock()."""

    def test_lock_context_manager(self, auth_home):
        from hermes_cli.auth import _auth_store_lock
        with _auth_store_lock():
            # Should not raise
            pass

    def test_lock_reentrant(self, auth_home):
        from hermes_cli.auth import _auth_store_lock
        with _auth_store_lock():
            with _auth_store_lock():
                # Reentrant lock should work
                pass


# ---------------------------------------------------------------------------
# get_auth_status
# ---------------------------------------------------------------------------

class TestGetAuthStatus:
    """Tests for get_auth_status() dispatcher."""

    def test_unknown_provider_returns_not_logged_in(self, auth_home):
        from hermes_cli.auth import get_auth_status
        result = get_auth_status("unknown-provider")
        assert result["logged_in"] is False

    def test_no_provider_returns_not_logged_in(self, auth_home):
        from hermes_cli.auth import get_auth_status
        result = get_auth_status(None)
        assert result["logged_in"] is False


# ---------------------------------------------------------------------------
# resolve_api_key_provider_credentials
# ---------------------------------------------------------------------------

class TestResolveApiKeyProviderCredentials:
    """Tests for resolve_api_key_provider_credentials()."""

    def test_non_api_key_provider_raises(self):
        from hermes_cli.auth import resolve_api_key_provider_credentials, AuthError
        with pytest.raises(AuthError, match="not an API-key provider"):
            resolve_api_key_provider_credentials("nous")

    def test_unknown_provider_raises(self):
        from hermes_cli.auth import resolve_api_key_provider_credentials, AuthError
        with pytest.raises(AuthError, match="not an API-key provider"):
            resolve_api_key_provider_credentials("nonexistent")

    def test_zai_with_key(self, monkeypatch, auth_home):
        monkeypatch.setenv("GLM_API_KEY", "test-zai-key-12345")
        monkeypatch.delenv("GLM_BASE_URL", raising=False)
        # Mock detect_zai_endpoint and _load_auth_store to avoid network calls
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {"providers": {}})
        from hermes_cli.auth import resolve_api_key_provider_credentials
        result = resolve_api_key_provider_credentials("zai")
        assert result["provider"] == "zai"
        assert result["api_key"] == "test-zai-key-12345"

    def test_deepseek_with_env_url(self, monkeypatch, auth_home):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-12345")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.deepseek.com/v1")
        from hermes_cli.auth import resolve_api_key_provider_credentials
        result = resolve_api_key_provider_credentials("deepseek")
        assert result["base_url"] == "https://custom.deepseek.com/v1"

    def test_no_key_returns_empty(self, monkeypatch, auth_home):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from hermes_cli.auth import resolve_api_key_provider_credentials
        result = resolve_api_key_provider_credentials("deepseek")
        assert result["api_key"] == ""


# ---------------------------------------------------------------------------
# get_api_key_provider_status
# ---------------------------------------------------------------------------

class TestGetApiKeyProviderStatus:
    """Tests for get_api_key_provider_status()."""

    def test_non_api_key_provider(self):
        from hermes_cli.auth import get_api_key_provider_status
        result = get_api_key_provider_status("nous")
        assert result["configured"] is False

    def test_unknown_provider(self):
        from hermes_cli.auth import get_api_key_provider_status
        result = get_api_key_provider_status("nonexistent")
        assert result["configured"] is False

    def test_configured_provider(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-key-12345")
        from hermes_cli.auth import get_api_key_provider_status
        result = get_api_key_provider_status("deepseek")
        assert result["configured"] is True
        assert result["logged_in"] is True
        assert result["key_source"] == "DEEPSEEK_API_KEY"

    def test_unconfigured_provider(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from hermes_cli.auth import get_api_key_provider_status
        result = get_api_key_provider_status("deepseek")
        assert result["configured"] is False


# ---------------------------------------------------------------------------
# get_external_process_provider_status
# ---------------------------------------------------------------------------

class TestGetExternalProcessProviderStatus:
    """Tests for get_external_process_provider_status()."""

    def test_non_external_provider(self):
        from hermes_cli.auth import get_external_process_provider_status
        result = get_external_process_provider_status("nous")
        assert result["configured"] is False

    def test_copilot_acp_status(self, monkeypatch):
        monkeypatch.delenv("HERMES_COPILOT_ACP_COMMAND", raising=False)
        monkeypatch.delenv("COPILOT_CLI_PATH", raising=False)
        monkeypatch.delenv("COPILOT_ACP_BASE_URL", raising=False)
        from hermes_cli.auth import get_external_process_provider_status
        result = get_external_process_provider_status("copilot-acp")
        assert result["provider"] == "copilot-acp"
        assert result["command"] == "copilot"
        assert result["args"] == ["--acp", "--stdio"]
