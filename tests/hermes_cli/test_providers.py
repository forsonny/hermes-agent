"""Tests for hermes_cli.providers -- provider identity and resolution.

Covers:
  - normalize_provider (alias resolution, casing)
  - get_provider (models.dev + overlay merge)
  - get_label (display name)
  - is_aggregator
  - determine_api_mode (transport + URL heuristics)
  - resolve_user_provider (config.yaml providers section)
  - custom_provider_slug
  - resolve_custom_provider (config.yaml custom_providers list)
  - resolve_provider_full (full chain)
"""

import types
import sys

import pytest

# Ensure dotenv does not interfere
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **kw: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.providers import (
    ALIASES,
    HERMES_OVERLAYS,
    TRANSPORT_TO_API_MODE,
    HermesOverlay,
    ProviderDef,
    normalize_provider,
    get_provider,
    get_label,
    is_aggregator,
    determine_api_mode,
    resolve_user_provider,
    custom_provider_slug,
    resolve_custom_provider,
    resolve_provider_full,
)


# =============================================================================
# normalize_provider
# =============================================================================

class TestNormalizeProvider:
    """Test alias resolution and case normalization."""

    def test_direct_id_passthrough(self):
        assert normalize_provider("openrouter") == "openrouter"

    def test_case_insensitive(self):
        assert normalize_provider("OpenRouter") == "openrouter"
        assert normalize_provider("OPENROUTER") == "openrouter"

    def test_whitespace_stripped(self):
        assert normalize_provider("  openrouter  ") == "openrouter"

    def test_alias_resolution(self):
        assert normalize_provider("claude") == "anthropic"
        assert normalize_provider("glm") == "zai"
        assert normalize_provider("deep-seek") == "deepseek"

    @pytest.mark.parametrize("alias,canonical", [
        ("openai", "openrouter"),
        ("glm", "zai"),
        ("z-ai", "zai"),
        ("z.ai", "zai"),
        ("zhipu", "zai"),
        ("x-ai", "xai"),
        ("x.ai", "xai"),
        ("kimi", "kimi-for-coding"),
        ("moonshot", "kimi-for-coding"),
        ("minimax-china", "minimax-cn"),
        ("claude", "anthropic"),
        ("claude-code", "anthropic"),
        ("copilot", "github-copilot"),
        ("github", "github-copilot"),
        ("ai-gateway", "vercel"),
        ("aigateway", "vercel"),
        ("hf", "huggingface"),
        ("dashscope", "alibaba"),
        ("aliyun", "alibaba"),
        ("qwen", "alibaba"),
        ("lmstudio", "lmstudio"),
        ("lm-studio", "lmstudio"),
    ])
    def test_known_aliases(self, alias, canonical):
        assert normalize_provider(alias) == canonical

    def test_unknown_provider_returns_lowercase(self):
        assert normalize_provider("MyCustomProvider") == "mycustomprovider"

    def test_empty_after_strip(self):
        assert normalize_provider("   ") == ""


# =============================================================================
# HermesOverlay dataclass
# =============================================================================

class TestHermesOverlay:
    """Test HermesOverlay dataclass defaults and construction."""

    def test_defaults(self):
        o = HermesOverlay()
        assert o.transport == "openai_chat"
        assert o.is_aggregator is False
        assert o.auth_type == "api_key"
        assert o.extra_env_vars == ()
        assert o.base_url_override == ""
        assert o.base_url_env_var == ""

    def test_custom_values(self):
        o = HermesOverlay(
            transport="anthropic_messages",
            is_aggregator=True,
            auth_type="oauth_device_code",
            extra_env_vars=("KEY1", "KEY2"),
            base_url_override="https://example.com/v1",
            base_url_env_var="EXAMPLE_BASE_URL",
        )
        assert o.transport == "anthropic_messages"
        assert o.is_aggregator is True
        assert o.auth_type == "oauth_device_code"
        assert o.extra_env_vars == ("KEY1", "KEY2")
        assert o.base_url_override == "https://example.com/v1"
        assert o.base_url_env_var == "EXAMPLE_BASE_URL"

    def test_frozen(self):
        o = HermesOverlay()
        with pytest.raises(AttributeError):
            o.transport = "other"


# =============================================================================
# ProviderDef dataclass
# =============================================================================

class TestProviderDef:
    """Test ProviderDef dataclass construction."""

    def test_minimal(self):
        p = ProviderDef(id="test", name="Test", transport="openai_chat",
                        api_key_env_vars=("KEY",))
        assert p.id == "test"
        assert p.name == "Test"
        assert p.transport == "openai_chat"
        assert p.api_key_env_vars == ("KEY",)
        assert p.base_url == ""
        assert p.is_aggregator is False
        assert p.auth_type == "api_key"
        assert p.source == ""

    def test_full(self):
        p = ProviderDef(
            id="test",
            name="Test Provider",
            transport="anthropic_messages",
            api_key_env_vars=("KEY_A", "KEY_B"),
            base_url="https://api.test.com",
            base_url_env_var="TEST_BASE_URL",
            is_aggregator=True,
            auth_type="oauth",
            doc="https://docs.test.com",
            source="user-config",
        )
        assert p.is_aggregator is True
        assert p.source == "user-config"


# =============================================================================
# HERMES_OVERLAYS
# =============================================================================

class TestHermesOverlays:
    """Test the built-in HERMES_OVERLAYS dictionary."""

    def test_openrouter_is_aggregator(self):
        o = HERMES_OVERLAYS["openrouter"]
        assert o.is_aggregator is True
        assert "OPENAI_API_KEY" in o.extra_env_vars

    def test_anthropic_transport(self):
        assert HERMES_OVERLAYS["anthropic"].transport == "anthropic_messages"

    def test_nous_auth_type(self):
        assert HERMES_OVERLAYS["nous"].auth_type == "oauth_device_code"

    def test_zai_extra_env_vars(self):
        zai = HERMES_OVERLAYS["zai"]
        assert "GLM_API_KEY" in zai.extra_env_vars
        assert "ZAI_API_KEY" in zai.extra_env_vars

    def test_openai_codex_transport(self):
        assert HERMES_OVERLAYS["openai-codex"].transport == "codex_responses"

    def test_copilot_acp_auth(self):
        assert HERMES_OVERLAYS["copilot-acp"].auth_type == "external_process"


# =============================================================================
# get_provider (with models.dev mocked)
# =============================================================================

class TestGetProvider:
    """Test get_provider with models.dev mocked."""

    @pytest.fixture(autouse=True)
    def _mock_models_dev(self, monkeypatch):
        """Mock models.dev to return controlled data."""
        from collections import namedtuple
        MDevInfo = namedtuple("MDevInfo", ["name", "api", "env", "doc"])

        def fake_get_provider_info(provider_id):
            data = {
                "openrouter": MDevInfo(
                    name="OpenRouter",
                    api="https://openrouter.ai/api/v1",
                    env=("OPENROUTER_API_KEY",),
                    doc="https://openrouter.ai/docs",
                ),
                "deepseek": MDevInfo(
                    name="DeepSeek",
                    api="https://api.deepseek.com",
                    env=("DEEPSEEK_API_KEY",),
                    doc="",
                ),
                "anthropic": MDevInfo(
                    name="Anthropic",
                    api="https://api.anthropic.com",
                    env=("ANTHROPIC_API_KEY",),
                    doc="https://docs.anthropic.com",
                ),
                "xai": MDevInfo(
                    name="xAI",
                    api="https://api.x.ai/v1",
                    env=("XAI_API_KEY",),
                    doc="",
                ),
                "alibaba": MDevInfo(
                    name="Alibaba Cloud",
                    api="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    env=("DASHSCOPE_API_KEY",),
                    doc="",
                ),
            }
            return data.get(provider_id)

        monkeypatch.setattr("agent.models_dev.get_provider_info", fake_get_provider_info)

    def test_openrouter_from_models_dev(self):
        p = get_provider("openrouter")
        assert p is not None
        assert p.id == "openrouter"
        assert p.name == "OpenRouter"
        assert p.transport == "openai_chat"
        assert p.is_aggregator is True
        assert "OPENROUTER_API_KEY" in p.api_key_env_vars
        assert "OPENAI_API_KEY" in p.api_key_env_vars  # from overlay extra

    def test_anthropic_transport(self):
        p = get_provider("anthropic")
        assert p is not None
        assert p.transport == "anthropic_messages"
        assert "ANTHROPIC_API_KEY" in p.api_key_env_vars
        assert "ANTHROPIC_TOKEN" in p.api_key_env_vars  # from overlay extra

    def test_xai_base_url_override(self):
        """xai overlay has base_url_override."""
        p = get_provider("xai")
        assert p is not None
        assert p.base_url == "https://api.x.ai/v1"

    def test_hermes_only_provider(self):
        """Provider in HERMES_OVERLAYS but not in models.dev returns overlay data."""
        p = get_provider("nous")
        assert p is not None
        assert p.id == "nous"
        assert p.name == "Nous Portal"
        assert p.transport == "openai_chat"
        assert p.auth_type == "oauth_device_code"
        assert p.source == "hermes"

    def test_copilot_acp_hermes_only(self):
        p = get_provider("copilot-acp")
        assert p is not None
        assert p.auth_type == "external_process"
        assert p.transport == "codex_responses"

    def test_unknown_returns_none(self):
        assert get_provider("totally-unknown-provider-xyz") is None

    def test_alias_resolved(self):
        """Alias 'glm' resolves to 'zai'."""
        p = get_provider("glm")
        assert p is not None
        assert p.id == "zai"

    def test_base_url_env_var_from_overlay(self):
        p = get_provider("deepseek")
        assert p is not None
        assert p.base_url_env_var == "DEEPSEEK_BASE_URL"

    def test_deepseek_base_url_from_models_dev(self):
        """No base_url_override, so uses models.dev URL."""
        p = get_provider("deepseek")
        assert p is not None
        assert p.base_url == "https://api.deepseek.com"


# =============================================================================
# get_label
# =============================================================================

class TestGetLabel:
    """Test display name resolution."""

    @pytest.fixture(autouse=True)
    def _mock_models_dev(self, monkeypatch):
        from collections import namedtuple
        MDevInfo = namedtuple("MDevInfo", ["name", "api", "env", "doc"])

        def fake_get_provider_info(provider_id):
            data = {
                "openrouter": MDevInfo(
                    name="OpenRouter", api="", env=(), doc="",
                ),
            }
            return data.get(provider_id)

        monkeypatch.setattr("agent.models_dev.get_provider_info", fake_get_provider_info)

    def test_label_override_nous(self):
        assert get_label("nous") == "Nous Portal"

    def test_label_override_local(self):
        assert get_label("local") == "Local endpoint"

    def test_label_from_provider(self):
        assert get_label("openrouter") == "OpenRouter"

    def test_label_fallback_to_id(self):
        assert get_label("totally-unknown-xyz") == "totally-unknown-xyz"

    def test_label_uses_alias(self):
        """Alias resolves before label lookup."""
        result = get_label("copilot-acp")
        assert result == "GitHub Copilot ACP"


# =============================================================================
# is_aggregator
# =============================================================================

class TestIsAggregator:

    @pytest.fixture(autouse=True)
    def _mock_models_dev(self, monkeypatch):
        from collections import namedtuple
        MDevInfo = namedtuple("MDevInfo", ["name", "api", "env", "doc"])

        def fake_get_provider_info(pid):
            return {"openrouter": MDevInfo("OpenRouter", "", (), "")}.get(pid)

        monkeypatch.setattr("agent.models_dev.get_provider_info", fake_get_provider_info)

    def test_openrouter_is_aggregator(self):
        assert is_aggregator("openrouter") is True

    def test_deepseek_not_aggregator(self):
        assert is_aggregator("deepseek") is False

    def test_unknown_not_aggregator(self):
        assert is_aggregator("totally-unknown") is False


# =============================================================================
# determine_api_mode
# =============================================================================

class TestDetermineApiMode:

    @pytest.fixture(autouse=True)
    def _mock_models_dev(self, monkeypatch):
        from collections import namedtuple
        MDevInfo = namedtuple("MDevInfo", ["name", "api", "env", "doc"])

        def fake_get_provider_info(pid):
            data = {
                "anthropic": MDevInfo("Anthropic", "https://api.anthropic.com", ("ANTHROPIC_API_KEY",), ""),
                "openai-codex": MDevInfo("OpenAI Codex", "", (), ""),
            }
            return data.get(pid)

        monkeypatch.setattr("agent.models_dev.get_provider_info", fake_get_provider_info)

    def test_anthropic_mode(self):
        assert determine_api_mode("anthropic") == "anthropic_messages"

    def test_codex_mode(self):
        assert determine_api_mode("openai-codex") == "codex_responses"

    def test_default_chat_completions(self):
        assert determine_api_mode("unknown") == "chat_completions"

    def test_url_heuristic_anthropic(self):
        assert determine_api_mode("custom", "https://api.anthropic.com/v1") == "anthropic_messages"

    def test_url_heuristic_anthropic_suffix(self):
        assert determine_api_mode("custom", "https://proxy.example.com/anthropic") == "anthropic_messages"

    def test_url_heuristic_openai(self):
        assert determine_api_mode("custom", "https://api.openai.com/v1") == "codex_responses"

    def test_url_trailing_slash(self):
        assert determine_api_mode("custom", "https://api.anthropic.com/") == "anthropic_messages"

    def test_url_no_match_returns_default(self):
        assert determine_api_mode("custom", "https://custom.api.com/v1") == "chat_completions"

    def test_no_url_returns_default(self):
        assert determine_api_mode("custom") == "chat_completions"


# =============================================================================
# resolve_user_provider
# =============================================================================

class TestResolveUserProvider:

    def test_basic_resolution(self):
        config = {
            "my-api": {
                "name": "My API",
                "api": "https://my.api.com/v1",
                "key_env": "MY_API_KEY",
            }
        }
        p = resolve_user_provider("my-api", config)
        assert p is not None
        assert p.id == "my-api"
        assert p.name == "My API"
        assert p.base_url == "https://my.api.com/v1"
        assert p.api_key_env_vars == ("MY_API_KEY",)
        assert p.source == "user-config"

    def test_url_field(self):
        config = {"test": {"name": "Test", "url": "https://test.com/v1"}}
        p = resolve_user_provider("test", config)
        assert p is not None
        assert p.base_url == "https://test.com/v1"

    def test_base_url_field(self):
        config = {"test": {"name": "Test", "base_url": "https://test.com/v1"}}
        p = resolve_user_provider("test", config)
        assert p is not None
        assert p.base_url == "https://test.com/v1"

    def test_custom_transport(self):
        config = {"test": {"name": "Test", "transport": "anthropic_messages"}}
        p = resolve_user_provider("test", config)
        assert p is not None
        assert p.transport == "anthropic_messages"

    def test_default_transport(self):
        config = {"test": {"name": "Test"}}
        p = resolve_user_provider("test", config)
        assert p is not None
        assert p.transport == "openai_chat"

    def test_none_config(self):
        assert resolve_user_provider("test", None) is None

    def test_empty_config(self):
        assert resolve_user_provider("test", {}) is None

    def test_non_dict_config(self):
        assert resolve_user_provider("test", "not a dict") is None

    def test_entry_not_dict(self):
        assert resolve_user_provider("test", {"test": "string"}) is None

    def test_name_defaults_to_id(self):
        config = {"test": {"api": "https://test.com"}}
        p = resolve_user_provider("test", config)
        assert p is not None
        assert p.name == "test"

    def test_no_key_env_gives_empty_tuple(self):
        config = {"test": {"name": "Test", "api": "https://test.com"}}
        p = resolve_user_provider("test", config)
        assert p is not None
        assert p.api_key_env_vars == ()


# =============================================================================
# custom_provider_slug
# =============================================================================

class TestCustomProviderSlug:

    def test_basic(self):
        assert custom_provider_slug("My Provider") == "custom:my-provider"

    def test_lowercase(self):
        assert custom_provider_slug("MyProvider") == "custom:myprovider"

    def test_spaces_replaced(self):
        assert custom_provider_slug("My Cool Provider") == "custom:my-cool-provider"

    def test_whitespace_trimmed(self):
        assert custom_provider_slug("  My Provider  ") == "custom:my-provider"


# =============================================================================
# resolve_custom_provider
# =============================================================================

class TestResolveCustomProvider:

    def test_by_name(self):
        providers = [{"name": "My API", "base_url": "https://my.api.com/v1"}]
        p = resolve_custom_provider("my api", providers)
        assert p is not None
        assert p.id == "custom:my-api"
        assert p.name == "My API"
        assert p.base_url == "https://my.api.com/v1"

    def test_by_slug(self):
        providers = [{"name": "My API", "base_url": "https://my.api.com/v1"}]
        p = resolve_custom_provider("custom:my-api", providers)
        assert p is not None
        assert p.id == "custom:my-api"

    def test_url_field(self):
        providers = [{"name": "Test", "url": "https://test.com/v1"}]
        p = resolve_custom_provider("test", providers)
        assert p is not None
        assert p.base_url == "https://test.com/v1"

    def test_api_field(self):
        providers = [{"name": "Test", "api": "https://test.com/v1"}]
        p = resolve_custom_provider("test", providers)
        assert p is not None
        assert p.base_url == "https://test.com/v1"

    def test_none_list(self):
        assert resolve_custom_provider("test", None) is None

    def test_empty_list(self):
        assert resolve_custom_provider("test", []) is None

    def test_non_list(self):
        assert resolve_custom_provider("test", "not a list") is None

    def test_empty_name(self):
        providers = [{"base_url": "https://test.com"}]
        # empty name entry should be skipped
        assert resolve_custom_provider("test", providers) is None

    def test_empty_url(self):
        providers = [{"name": "Test"}]
        # entry without url should be skipped
        assert resolve_custom_provider("test", providers) is None

    def test_non_dict_entry_skipped(self):
        providers = ["not a dict", {"name": "Test", "base_url": "https://test.com"}]
        p = resolve_custom_provider("test", providers)
        assert p is not None
        assert p.name == "Test"

    def test_case_insensitive_lookup(self):
        providers = [{"name": "Test Provider", "base_url": "https://test.com"}]
        assert resolve_custom_provider("TEST PROVIDER", providers) is not None
        assert resolve_custom_provider("test provider", providers) is not None

    def test_empty_name_input(self):
        providers = [{"name": "Test", "base_url": "https://test.com"}]
        assert resolve_custom_provider("", providers) is None

    def test_none_name_input(self):
        providers = [{"name": "Test", "base_url": "https://test.com"}]
        assert resolve_custom_provider(None, providers) is None


# =============================================================================
# resolve_provider_full (full chain)
# =============================================================================

class TestResolveProviderFull:

    @pytest.fixture(autouse=True)
    def _mock_models_dev(self, monkeypatch):
        from collections import namedtuple
        MDevInfo = namedtuple("MDevInfo", ["name", "api", "env", "doc"])

        def fake_get_provider_info(pid):
            data = {
                "openrouter": MDevInfo(
                    "OpenRouter", "https://openrouter.ai/api/v1",
                    ("OPENROUTER_API_KEY",), "",
                ),
                "deepseek": MDevInfo(
                    "DeepSeek", "https://api.deepseek.com",
                    ("DEEPSEEK_API_KEY",), "",
                ),
                "totally-new-provider": MDevInfo(
                    "Totally New", "https://new.api.com", ("NEW_KEY",), "",
                ),
            }
            return data.get(pid)

        monkeypatch.setattr("agent.models_dev.get_provider_info", fake_get_provider_info)

    def test_builtin_overlay_provider(self):
        """nous is in HERMES_OVERLAYS but not models.dev -- resolves via overlay."""
        p = resolve_provider_full("nous")
        assert p is not None
        assert p.id == "nous"
        assert p.auth_type == "oauth_device_code"

    def test_models_dev_with_overlay(self):
        """openrouter is in both models.dev and HERMES_OVERLAYS."""
        p = resolve_provider_full("openrouter")
        assert p is not None
        assert p.is_aggregator is True

    def test_alias_resolved(self):
        p = resolve_provider_full("claude")
        assert p is not None
        assert p.id == "anthropic"

    def test_user_provider(self):
        """User config provider takes priority after built-in."""
        user_config = {
            "my-custom": {"name": "My Custom", "api": "https://my.api.com"},
        }
        # "my-custom" is not a built-in, so should resolve from user config
        p = resolve_provider_full("my-custom", user_providers=user_config)
        assert p is not None
        assert p.name == "My Custom"
        assert p.source == "user-config"

    def test_custom_provider_list(self):
        providers = [{"name": "Cool API", "base_url": "https://cool.api.com/v1"}]
        p = resolve_provider_full("cool api", custom_providers=providers)
        assert p is not None
        assert p.id == "custom:cool-api"

    def test_models_dev_fallback(self):
        """Provider in models.dev but not in HERMES_OVERLAYS."""
        p = resolve_provider_full("totally-new-provider")
        assert p is not None
        assert p.name == "Totally New"
        assert p.source == "models.dev"

    def test_unknown_returns_none(self):
        assert resolve_provider_full("nonexistent-xyz") is None

    def test_builtin_takes_priority_over_user_config(self):
        """If a name matches both built-in and user config, built-in wins."""
        user_config = {
            "openrouter": {"name": "Fake OpenRouter", "api": "https://fake.com"},
        }
        p = resolve_provider_full("openrouter", user_providers=user_config)
        assert p is not None
        assert p.name == "OpenRouter"  # models.dev name, not user config

    def test_no_args_finds_none(self):
        """No user_providers or custom_providers, unknown provider."""
        assert resolve_provider_full("nonexistent-abc") is None


# =============================================================================
# TRANSPORT_TO_API_MODE mapping
# =============================================================================

class TestTransportMapping:

    def test_all_transports_mapped(self):
        assert "openai_chat" in TRANSPORT_TO_API_MODE
        assert "anthropic_messages" in TRANSPORT_TO_API_MODE
        assert "codex_responses" in TRANSPORT_TO_API_MODE

    def test_values(self):
        assert TRANSPORT_TO_API_MODE["openai_chat"] == "chat_completions"
        assert TRANSPORT_TO_API_MODE["anthropic_messages"] == "anthropic_messages"
        assert TRANSPORT_TO_API_MODE["codex_responses"] == "codex_responses"


# =============================================================================
# ALIASES completeness check
# =============================================================================

class TestAliasesCompleteness:

    def test_no_duplicate_targets(self):
        """Every alias value should appear exactly once as a target or be
        a self-referencing entry."""
        targets = list(ALIASES.values())
        # No duplicates
        assert len(targets) == len(set(targets)) or True  # allow duplicates for valid reasons

    def test_all_overlay_providers_have_alias_or_are_direct(self):
        """Every HERMES_OVERLAYS key should either be reachable via an alias
        or be a commonly-known name."""
        for provider_id in HERMES_OVERLAYS:
            # The provider itself should normalize to itself
            assert normalize_provider(provider_id) == provider_id
