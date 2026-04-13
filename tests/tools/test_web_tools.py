"""Tests for tools/web_tools.py -- pure functions and check helpers."""

import os
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# clean_base64_images
# ---------------------------------------------------------------------------

class TestCleanBase64Images:
    """Tests for clean_base64_images() -- base64 image removal from text."""

    def test_removes_parenthesized_png(self):
        from tools.web_tools import clean_base64_images
        text = "Here is an image (data:image/png;base64,iVBORw0KGgo=) in text."
        result = clean_base64_images(text)
        assert "[BASE64_IMAGE_REMOVED]" in result
        assert "data:image/png;base64" not in result
        assert "Here is an image" in result
        assert "in text." in result

    def test_removes_parenthesized_jpeg(self):
        from tools.web_tools import clean_base64_images
        text = "(data:image/jpeg;base64,/9j/4AAQSkZJRg==)"
        result = clean_base64_images(text)
        assert "[BASE64_IMAGE_REMOVED]" in result
        assert "data:image/jpeg" not in result

    def test_removes_parenthesized_svg(self):
        from tools.web_tools import clean_base64_images
        text = "(data:image/svg+xml;base64,PHN2ZyB4bWxucz0=)"
        result = clean_base64_images(text)
        assert "[BASE64_IMAGE_REMOVED]" in result

    def test_removes_bare_base64_no_parens(self):
        from tools.web_tools import clean_base64_images
        text = "data:image/png;base64,iVBORw0KGgo= end"
        result = clean_base64_images(text)
        assert "[BASE64_IMAGE_REMOVED]" in result
        assert "data:image/png" not in result
        assert "end" in result

    def test_multiple_images_removed(self):
        from tools.web_tools import clean_base64_images
        text = (
            "img1 (data:image/png;base64,AAAA) and "
            "img2 (data:image/jpeg;base64,BBBB) done"
        )
        result = clean_base64_images(text)
        assert result.count("[BASE64_IMAGE_REMOVED]") == 2

    def test_no_images_returns_original(self):
        from tools.web_tools import clean_base64_images
        text = "Plain text without any images."
        assert clean_base64_images(text) == text

    def test_empty_string(self):
        from tools.web_tools import clean_base64_images
        assert clean_base64_images("") == ""

    def test_preserves_base64_like_non_image_data(self):
        from tools.web_tools import clean_base64_images
        # Should NOT remove non-image base64
        text = "data:application/pdf;base64,AAAA"
        assert clean_base64_images(text) == text

    def test_long_base64_string(self):
        from tools.web_tools import clean_base64_images
        b64 = "A" * 10000
        text = f"(data:image/png;base64,{b64})"
        result = clean_base64_images(text)
        assert "[BASE64_IMAGE_REMOVED]" in result
        assert b64 not in result


# ---------------------------------------------------------------------------
# _get_direct_firecrawl_config
# ---------------------------------------------------------------------------

class TestGetDirectFirecrawlConfig:
    """Tests for _get_direct_firecrawl_config()."""

    def test_returns_none_when_no_env(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        from tools.web_tools import _get_direct_firecrawl_config
        assert _get_direct_firecrawl_config() is None

    def test_returns_config_with_api_key(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key-123")
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        from tools.web_tools import _get_direct_firecrawl_config
        result = _get_direct_firecrawl_config()
        assert result is not None
        kwargs, cache_key = result
        assert kwargs["api_key"] == "test-key-123"

    def test_returns_config_with_api_url(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3030")
        from tools.web_tools import _get_direct_firecrawl_config
        result = _get_direct_firecrawl_config()
        assert result is not None
        kwargs, cache_key = result
        assert kwargs["api_url"] == "http://localhost:3030"

    def test_returns_both_key_and_url(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "my-key")
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://firecrawl:3000")
        from tools.web_tools import _get_direct_firecrawl_config
        result = _get_direct_firecrawl_config()
        assert result is not None
        kwargs, _ = result
        assert kwargs["api_key"] == "my-key"
        assert kwargs["api_url"] == "http://firecrawl:3000"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "  key  ")
        monkeypatch.setenv("FIRECRAWL_API_URL", "  http://host:3000/  ")
        from tools.web_tools import _get_direct_firecrawl_config
        result = _get_direct_firecrawl_config()
        kwargs, _ = result
        assert kwargs["api_key"] == "key"
        # URL has trailing slash stripped
        assert kwargs["api_url"] == "http://host:3000"


# ---------------------------------------------------------------------------
# _load_web_config
# ---------------------------------------------------------------------------

class TestLoadWebConfig:
    """Tests for _load_web_config()."""

    def test_returns_dict_on_import_error(self):
        from tools.web_tools import _load_web_config
        with patch("hermes_cli.config.load_config", side_effect=ImportError):
            assert _load_web_config() == {}

    def test_returns_web_section(self):
        from tools.web_tools import _load_web_config
        with patch("hermes_cli.config.load_config", return_value={"web": {"backend": "exa"}}):
            result = _load_web_config()
            assert result == {"backend": "exa"}

    def test_returns_empty_when_no_web_section(self):
        from tools.web_tools import _load_web_config
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _load_web_config() == {}


# ---------------------------------------------------------------------------
# _get_backend
# ---------------------------------------------------------------------------

class TestGetBackend:
    """Tests for _get_backend()."""

    def test_configured_backend_takes_priority(self, monkeypatch):
        from tools.web_tools import _get_backend
        monkeypatch.setenv("PARALLEL_API_KEY", "some-key")
        with patch("tools.web_tools._load_web_config", return_value={"backend": "exa"}):
            assert _get_backend() == "exa"

    def test_fallback_to_firecrawl_with_api_key(self, monkeypatch):
        from tools.web_tools import _get_backend
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert _get_backend() == "firecrawl"

    def test_fallback_to_parallel(self, monkeypatch):
        from tools.web_tools import _get_backend
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "p-key")
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert _get_backend() == "parallel"

    def test_default_is_firecrawl(self, monkeypatch):
        from tools.web_tools import _get_backend
        for var in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                     "TAVILY_API_KEY", "EXA_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert _get_backend() == "firecrawl"

    def test_fallback_to_tavily(self, monkeypatch):
        from tools.web_tools import _get_backend
        for var in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "t-key")
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert _get_backend() == "tavily"

    def test_fallback_to_exa(self, monkeypatch):
        from tools.web_tools import _get_backend
        for var in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY", "TAVILY_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("EXA_API_KEY", "e-key")
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert _get_backend() == "exa"


# ---------------------------------------------------------------------------
# _is_backend_available
# ---------------------------------------------------------------------------

class TestIsBackendAvailable:
    """Tests for _is_backend_available()."""

    def test_exa_available_with_key(self, monkeypatch):
        from tools.web_tools import _is_backend_available
        monkeypatch.setenv("EXA_API_KEY", "key")
        assert _is_backend_available("exa") is True

    def test_exa_unavailable_without_key(self, monkeypatch):
        from tools.web_tools import _is_backend_available
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        assert _is_backend_available("exa") is False

    def test_parallel_available_with_key(self, monkeypatch):
        from tools.web_tools import _is_backend_available
        monkeypatch.setenv("PARALLEL_API_KEY", "key")
        assert _is_backend_available("parallel") is True

    def test_tavily_available_with_key(self, monkeypatch):
        from tools.web_tools import _is_backend_available
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        assert _is_backend_available("tavily") is True

    def test_unknown_backend_returns_false(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("unknown") is False


# ---------------------------------------------------------------------------
# check_web_api_key
# ---------------------------------------------------------------------------

class TestCheckWebApiKey:
    """Tests for check_web_api_key()."""

    def test_configured_backend_available(self):
        from tools.web_tools import check_web_api_key
        with patch("tools.web_tools._load_web_config", return_value={"backend": "exa"}), \
             patch("tools.web_tools._is_backend_available", return_value=True):
            assert check_web_api_key() is True

    def test_configured_backend_not_available(self):
        from tools.web_tools import check_web_api_key
        with patch("tools.web_tools._load_web_config", return_value={"backend": "exa"}), \
             patch("tools.web_tools._is_backend_available", return_value=False):
            assert check_web_api_key() is False

    def test_auto_detect_any_available(self):
        from tools.web_tools import check_web_api_key
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_backend_available", side_effect=lambda b: b == "tavily"):
            assert check_web_api_key() is True

    def test_auto_detect_none_available(self):
        from tools.web_tools import check_web_api_key
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_backend_available", return_value=False):
            assert check_web_api_key() is False


# ---------------------------------------------------------------------------
# check_firecrawl_api_key
# ---------------------------------------------------------------------------

class TestCheckFirecrawlApiKey:
    """Tests for check_firecrawl_api_key()."""

    def test_direct_config_available(self):
        from tools.web_tools import check_firecrawl_api_key
        with patch("tools.web_tools._has_direct_firecrawl_config", return_value=True), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert check_firecrawl_api_key() is True

    def test_gateway_available(self):
        from tools.web_tools import check_firecrawl_api_key
        with patch("tools.web_tools._has_direct_firecrawl_config", return_value=False), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=True):
            assert check_firecrawl_api_key() is True

    def test_neither_available(self):
        from tools.web_tools import check_firecrawl_api_key
        with patch("tools.web_tools._has_direct_firecrawl_config", return_value=False), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
            assert check_firecrawl_api_key() is False


# ---------------------------------------------------------------------------
# _raise_web_backend_configuration_error
# ---------------------------------------------------------------------------

class TestRaiseWebBackendConfigurationError:
    """Tests for _raise_web_backend_configuration_error()."""

    def test_raises_value_error(self):
        from tools.web_tools import _raise_web_backend_configuration_error
        with pytest.raises(ValueError, match="Web tools are not configured"):
            _raise_web_backend_configuration_error()

    def test_mentions_firecrawl(self):
        from tools.web_tools import _raise_web_backend_configuration_error
        with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
            _raise_web_backend_configuration_error()
