"""Tests for hermes_cli/runtime_provider.py — pure utility functions."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.runtime_provider as rp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool_entry(**kwargs):
    """Create a mock PooledCredential-like object."""
    defaults = dict(
        runtime_base_url="",
        base_url="",
        runtime_api_key="",
        access_token="",
        source="pool",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _normalize_custom_provider_name
# ---------------------------------------------------------------------------

class TestNormalizeCustomProviderName:
    def test_lowercase(self):
        assert rp._normalize_custom_provider_name("MyProvider") == "myprovider"

    def test_strip_whitespace(self):
        assert rp._normalize_custom_provider_name("  my-provider  ") == "my-provider"

    def test_replace_spaces(self):
        assert rp._normalize_custom_provider_name("My Provider") == "my-provider"

    def test_already_normalized(self):
        assert rp._normalize_custom_provider_name("my-provider") == "my-provider"

    def test_empty_string(self):
        assert rp._normalize_custom_provider_name("") == ""

    def test_mixed_case_spaces(self):
        assert rp._normalize_custom_provider_name("  My Cool Provider  ") == "my-cool-provider"


# ---------------------------------------------------------------------------
# _detect_api_mode_for_url
# ---------------------------------------------------------------------------

class TestDetectApiModeForUrl:
    def test_openai_com(self):
        assert rp._detect_api_mode_for_url("https://api.openai.com/v1") == "codex_responses"

    def test_openai_com_trailing_slash(self):
        assert rp._detect_api_mode_for_url("https://api.openai.com/v1/") == "codex_responses"

    def test_openai_com_case_insensitive(self):
        assert rp._detect_api_mode_for_url("HTTPS://API.OPENAI.COM/V1") == "codex_responses"

    def test_openrouter_not_openai(self):
        # openrouter.ai URLs should NOT trigger codex_responses
        assert rp._detect_api_mode_for_url("https://openrouter.ai/api/v1") is None

    def test_custom_url(self):
        assert rp._detect_api_mode_for_url("https://my-llm.example.com/v1") is None

    def test_empty_string(self):
        assert rp._detect_api_mode_for_url("") is None

    def test_none_like(self):
        assert rp._detect_api_mode_for_url(None) is None


# ---------------------------------------------------------------------------
# _parse_api_mode
# ---------------------------------------------------------------------------

class TestParseApiMode:
    def test_chat_completions(self):
        assert rp._parse_api_mode("chat_completions") == "chat_completions"

    def test_codex_responses(self):
        assert rp._parse_api_mode("codex_responses") == "codex_responses"

    def test_anthropic_messages(self):
        assert rp._parse_api_mode("anthropic_messages") == "anthropic_messages"

    def test_case_insensitive(self):
        assert rp._parse_api_mode("Chat_Completions") == "chat_completions"

    def test_strip_whitespace(self):
        assert rp._parse_api_mode("  codex_responses  ") == "codex_responses"

    def test_invalid_mode(self):
        assert rp._parse_api_mode("invalid_mode") is None

    def test_non_string(self):
        assert rp._parse_api_mode(42) is None

    def test_none(self):
        assert rp._parse_api_mode(None) is None

    def test_empty_string(self):
        assert rp._parse_api_mode("") is None


# ---------------------------------------------------------------------------
# _provider_supports_explicit_api_mode
# ---------------------------------------------------------------------------

class TestProviderSupportsExplicitApiMode:
    def test_no_configured_provider(self):
        assert rp._provider_supports_explicit_api_mode("openrouter", None) is True

    def test_empty_configured_provider(self):
        assert rp._provider_supports_explicit_api_mode("openrouter", "") is True

    def test_matching_provider(self):
        assert rp._provider_supports_explicit_api_mode("anthropic", "anthropic") is True

    def test_matching_provider_case_insensitive(self):
        assert rp._provider_supports_explicit_api_mode("Anthropic", "ANTHROPIC") is True

    def test_mismatched_provider(self):
        assert rp._provider_supports_explicit_api_mode("anthropic", "openrouter") is False

    def test_custom_provider_with_custom_configured(self):
        assert rp._provider_supports_explicit_api_mode("custom", "custom") is True

    def test_custom_provider_with_custom_prefix(self):
        assert rp._provider_supports_explicit_api_mode("custom", "custom:local") is True

    def test_custom_provider_with_non_custom_configured(self):
        assert rp._provider_supports_explicit_api_mode("custom", "openrouter") is False

    def test_none_provider(self):
        assert rp._provider_supports_explicit_api_mode(None, "openrouter") is False

    def test_none_configured(self):
        assert rp._provider_supports_explicit_api_mode("openrouter", None) is True


# ---------------------------------------------------------------------------
# format_runtime_provider_error
# ---------------------------------------------------------------------------

class TestFormatRuntimeProviderError:
    def test_auth_error(self):
        err = rp.AuthError("test auth error")
        result = rp.format_runtime_provider_error(err)
        assert isinstance(result, str)
        assert "test auth error" in result

    def test_generic_error(self):
        err = ValueError("something went wrong")
        result = rp.format_runtime_provider_error(err)
        assert result == "something went wrong"

    def test_runtime_error(self):
        err = RuntimeError("connection timeout")
        result = rp.format_runtime_provider_error(err)
        assert result == "connection timeout"


# ---------------------------------------------------------------------------
# resolve_requested_provider
# ---------------------------------------------------------------------------

class TestResolveRequestedProvider:
    def test_explicit_requested(self):
        assert rp.resolve_requested_provider("anthropic") == "anthropic"

    def test_explicit_requested_stripped(self):
        assert rp.resolve_requested_provider("  OpenRouter  ") == "openrouter"

    def test_explicit_requested_empty_falls_through(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "nous"})
        assert rp.resolve_requested_provider("") == "nous"

    def test_none_requested_uses_config(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "anthropic"})
        assert rp.resolve_requested_provider(None) == "anthropic"

    def test_none_requested_config_empty_uses_env(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        assert rp.resolve_requested_provider(None) == "openrouter"

    def test_none_requested_falls_to_auto(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        assert rp.resolve_requested_provider(None) == "auto"


# ---------------------------------------------------------------------------
# _get_named_custom_provider
# ---------------------------------------------------------------------------

class TestGetNamedCustomProvider:
    def test_empty_string(self, monkeypatch):
        assert rp._get_named_custom_provider("") is None

    def test_plain_custom(self, monkeypatch):
        # "custom" without a name should return None
        assert rp._get_named_custom_provider("custom") is None

    def test_auto_returns_none(self, monkeypatch):
        assert rp._get_named_custom_provider("auto") is None

    def test_known_builtin_provider_returns_none(self, monkeypatch):
        # If it resolves as a built-in provider, don't treat as custom
        assert rp._get_named_custom_provider("anthropic") is None

    def test_custom_with_name_found(self, monkeypatch):
        config = {
            "custom_providers": [
                {"name": "my-llm", "base_url": "https://llm.example.com/v1"}
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        result = rp._get_named_custom_provider("custom:my-llm")
        assert result is not None
        assert result["name"] == "my-llm"
        assert result["base_url"] == "https://llm.example.com/v1"

    def test_custom_with_name_by_short_name(self, monkeypatch):
        config = {
            "custom_providers": [
                {"name": "My LLM", "base_url": "https://llm.example.com/v1"}
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        result = rp._get_named_custom_provider("my-llm")
        assert result is not None
        assert result["name"] == "My LLM"

    def test_custom_with_api_key(self, monkeypatch):
        config = {
            "custom_providers": [
                {"name": "my-llm", "base_url": "https://llm.example.com", "api_key": "sk-123"}
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        result = rp._get_named_custom_provider("custom:my-llm")
        assert result["api_key"] == "sk-123"

    def test_custom_with_api_mode(self, monkeypatch):
        config = {
            "custom_providers": [
                {"name": "my-llm", "base_url": "https://llm.example.com", "api_mode": "anthropic_messages"}
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        result = rp._get_named_custom_provider("custom:my-llm")
        assert result["api_mode"] == "anthropic_messages"

    def test_custom_with_model(self, monkeypatch):
        config = {
            "custom_providers": [
                {"name": "my-llm", "base_url": "https://llm.example.com", "model": "llama3"}
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        result = rp._get_named_custom_provider("custom:my-llm")
        assert result["model"] == "llama3"

    def test_custom_not_found(self, monkeypatch):
        config = {
            "custom_providers": [
                {"name": "my-llm", "base_url": "https://llm.example.com"}
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        assert rp._get_named_custom_provider("custom:other-llm") is None

    def test_no_custom_providers_key(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {})
        assert rp._get_named_custom_provider("custom:my-llm") is None

    def test_custom_providers_is_dict_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(rp, "load_config", lambda: {"custom_providers": {"my-llm": {"base_url": "x"}}})
        import logging
        with caplog.at_level(logging.WARNING):
            result = rp._get_named_custom_provider("custom:my-llm")
        assert result is None

    def test_invalid_entry_skipped(self, monkeypatch):
        config = {
            "custom_providers": [
                "not-a-dict",
                {"base_url": "missing-name"},
                {"name": 123, "base_url": "bad-name-type"},
                {"name": "good", "base_url": "https://good.example.com"},
            ]
        }
        monkeypatch.setattr(rp, "load_config", lambda: config)
        result = rp._get_named_custom_provider("custom:good")
        assert result is not None
        assert result["name"] == "good"


# ---------------------------------------------------------------------------
# _resolve_runtime_from_pool_entry
# ---------------------------------------------------------------------------

class TestResolveRuntimeFromPoolEntry:
    def test_openrouter_provider(self):
        entry = _make_pool_entry(access_token="sk-test")
        result = rp._resolve_runtime_from_pool_entry(
            provider="openrouter",
            entry=entry,
            requested_provider="openrouter",
        )
        assert result["provider"] == "openrouter"
        assert result["api_key"] == "sk-test"
        assert result["api_mode"] == "chat_completions"
        assert result["requested_provider"] == "openrouter"

    def test_openai_codex_provider(self):
        entry = _make_pool_entry(runtime_api_key="codex-key")
        result = rp._resolve_runtime_from_pool_entry(
            provider="openai-codex",
            entry=entry,
            requested_provider="codex",
        )
        assert result["provider"] == "openai-codex"
        assert result["api_mode"] == "codex_responses"
        assert result["api_key"] == "codex-key"

    def test_anthropic_provider(self):
        entry = _make_pool_entry(runtime_api_key="ant-key", base_url="https://api.anthropic.com")
        result = rp._resolve_runtime_from_pool_entry(
            provider="anthropic",
            entry=entry,
            requested_provider="anthropic",
        )
        assert result["provider"] == "anthropic"
        assert result["api_mode"] == "anthropic_messages"

    def test_anthropic_config_base_url(self, monkeypatch):
        entry = _make_pool_entry(runtime_api_key="ant-key")
        model_cfg = {"provider": "anthropic", "base_url": "https://custom.anthropic.com"}
        result = rp._resolve_runtime_from_pool_entry(
            provider="anthropic",
            entry=entry,
            requested_provider="anthropic",
            model_cfg=model_cfg,
        )
        assert result["base_url"] == "https://custom.anthropic.com"

    def test_nous_provider(self):
        entry = _make_pool_entry(runtime_api_key="nous-key")
        result = rp._resolve_runtime_from_pool_entry(
            provider="nous",
            entry=entry,
            requested_provider="nous",
        )
        assert result["provider"] == "nous"
        assert result["api_mode"] == "chat_completions"

    def test_qwen_provider(self):
        entry = _make_pool_entry(runtime_api_key="qwen-key")
        result = rp._resolve_runtime_from_pool_entry(
            provider="qwen-oauth",
            entry=entry,
            requested_provider="qwen",
        )
        assert result["provider"] == "qwen-oauth"
        assert result["api_mode"] == "chat_completions"

    def test_pool_entry_source(self):
        entry = _make_pool_entry(access_token="key", source="test-source")
        result = rp._resolve_runtime_from_pool_entry(
            provider="openrouter",
            entry=entry,
            requested_provider="openrouter",
        )
        assert result["source"] == "test-source"

    def test_credential_pool_passed_through(self):
        mock_pool = MagicMock()
        entry = _make_pool_entry(access_token="key")
        result = rp._resolve_runtime_from_pool_entry(
            provider="openrouter",
            entry=entry,
            requested_provider="openrouter",
            pool=mock_pool,
        )
        assert result["credential_pool"] is mock_pool

    def test_base_url_from_runtime_base_url(self):
        entry = _make_pool_entry(runtime_base_url="https://custom.example.com/v1")
        result = rp._resolve_runtime_from_pool_entry(
            provider="openrouter",
            entry=entry,
            requested_provider="openrouter",
        )
        assert result["base_url"] == "https://custom.example.com/v1"

    def test_base_url_trailing_slash_stripped(self):
        entry = _make_pool_entry(base_url="https://example.com/v1/")
        result = rp._resolve_runtime_from_pool_entry(
            provider="openrouter",
            entry=entry,
            requested_provider="openrouter",
        )
        assert not result["base_url"].endswith("/")

    def test_unknown_provider_defaults_to_chat_completions(self):
        entry = _make_pool_entry(access_token="key")
        result = rp._resolve_runtime_from_pool_entry(
            provider="unknown-provider",
            entry=entry,
            requested_provider="unknown-provider",
        )
        assert result["api_mode"] == "chat_completions"


# ---------------------------------------------------------------------------
# _resolve_named_custom_runtime
# ---------------------------------------------------------------------------

class TestResolveNamedCustomRuntime:
    def test_no_custom_provider_found(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda x: None)
        result = rp._resolve_named_custom_runtime(
            requested_provider="custom:nonexistent",
        )
        assert result is None

    def test_custom_runtime_with_explicit_key(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda x: {
            "name": "my-llm",
            "base_url": "https://llm.example.com/v1",
        })
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s))
        result = rp._resolve_named_custom_runtime(
            requested_provider="custom:my-llm",
            explicit_api_key="sk-test-key",
        )
        assert result is not None
        assert result["provider"] == "custom"
        assert result["api_key"] == "sk-test-key"
        assert result["base_url"] == "https://llm.example.com/v1"

    def test_custom_runtime_no_base_url(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda x: {
            "name": "broken",
            "base_url": "",
        })
        result = rp._resolve_named_custom_runtime(
            requested_provider="custom:broken",
        )
        assert result is None

    def test_custom_runtime_pool_credentials(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda x: {
            "name": "my-llm",
            "base_url": "https://llm.example.com/v1",
            "model": "llama3",
        })
        pool_result = {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://llm.example.com/v1",
            "api_key": "pooled-key",
            "source": "pool:custom_example",
        }
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: pool_result)
        result = rp._resolve_named_custom_runtime(
            requested_provider="custom:my-llm",
        )
        assert result is not None
        assert result["api_key"] == "pooled-key"
        assert result["model"] == "llama3"  # model propagated from custom_provider config

    def test_custom_runtime_fallback_to_openai_key(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda x: {
            "name": "my-llm",
            "base_url": "https://llm.example.com/v1",
        })
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s and s != "bad"))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
        result = rp._resolve_named_custom_runtime(
            requested_provider="custom:my-llm",
        )
        assert result is not None
        assert result["api_key"] == "sk-openai-fallback"

    def test_custom_runtime_no_key_at_all(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_named_custom_provider", lambda x: {
            "name": "my-llm",
            "base_url": "https://llm.example.com/v1",
        })
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: False)
        result = rp._resolve_named_custom_runtime(
            requested_provider="custom:my-llm",
        )
        assert result is not None
        assert result["api_key"] == "no-key-required"


# ---------------------------------------------------------------------------
# _auto_detect_local_model
# ---------------------------------------------------------------------------

class TestAutoDetectLocalModel:
    def test_empty_base_url(self):
        assert rp._auto_detect_local_model("") == ""

    def test_none_base_url(self):
        assert rp._auto_detect_local_model(None) == ""

    def test_single_model_detected(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"data": [{"id": "llama3-8b"}]}
        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp
        monkeypatch.setattr("requests.get", mock_requests.get)
        result = rp._auto_detect_local_model("http://localhost:8000")
        assert result == "llama3-8b"

    def test_multiple_models_returns_empty(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp
        monkeypatch.setattr("requests.get", mock_requests.get)
        result = rp._auto_detect_local_model("http://localhost:8000")
        assert result == ""

    def test_connection_error_returns_empty(self, monkeypatch):
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("connection refused")
        monkeypatch.setattr("requests.get", mock_requests.get)
        result = rp._auto_detect_local_model("http://localhost:8000")
        assert result == ""


# ---------------------------------------------------------------------------
# _resolve_openrouter_runtime
# ---------------------------------------------------------------------------

class TestResolveOpenrouterRuntime:
    def test_basic_openrouter(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        result = rp._resolve_openrouter_runtime(
            requested_provider="openrouter",
        )
        assert result["provider"] == "openrouter"
        assert result["api_key"] == "sk-or-test"

    def test_custom_provider_label(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-custom")
        result = rp._resolve_openrouter_runtime(
            requested_provider="custom",
        )
        assert result["provider"] == "custom"

    def test_explicit_api_key(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        result = rp._resolve_openrouter_runtime(
            requested_provider="openrouter",
            explicit_api_key="explicit-key",
        )
        assert result["api_key"] == "explicit-key"
        assert result["source"] == "explicit"

    def test_explicit_base_url(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = rp._resolve_openrouter_runtime(
            requested_provider="openrouter",
            explicit_base_url="https://custom.llm.com/v1",
        )
        assert result["base_url"] == "https://custom.llm.com/v1"

    def test_no_key_required_for_custom_non_openrouter(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: False)
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        result = rp._resolve_openrouter_runtime(
            requested_provider="custom",
            explicit_base_url="https://local.llm.com/v1",
        )
        assert result["api_key"] == "no-key-required"

    def test_config_base_url_used_when_auto(self, monkeypatch):
        monkeypatch.setattr(rp, "_get_model_config", lambda: {
            "provider": "auto",
            "base_url": "https://config-url.example.com/v1",
        })
        monkeypatch.setattr(rp, "has_usable_secret", lambda s: bool(s))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = rp._resolve_openrouter_runtime(
            requested_provider="auto",
        )
        assert result["base_url"] == "https://config-url.example.com/v1"


# ---------------------------------------------------------------------------
# _get_model_config
# ---------------------------------------------------------------------------

class TestGetModelConfig:
    def test_dict_config(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {"model": {"default": "gpt-4"}})
        result = rp._get_model_config()
        assert result["default"] == "gpt-4"

    def test_string_config(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {"model": "claude-3-opus"})
        result = rp._get_model_config()
        assert result["default"] == "claude-3-opus"

    def test_model_alias_for_default(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {"model": {"model": "gpt-4"}})
        result = rp._get_model_config()
        assert result["default"] == "gpt-4"

    def test_default_prefers_explicit_default(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {"model": {"default": "gpt-4", "model": "claude-3"}})
        result = rp._get_model_config()
        assert result["default"] == "gpt-4"

    def test_empty_config(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {})
        result = rp._get_model_config()
        assert result == {}

    def test_none_config(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {"model": None})
        result = rp._get_model_config()
        assert result == {}
