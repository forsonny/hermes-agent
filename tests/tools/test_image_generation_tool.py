"""
Tests for tools/image_generation_tool.py -- parameter validation,
URL normalization, managed client, requirements checks, error handling,
and the registry wiring.

Run with:  python -m pytest tests/tools/test_image_generation_tool.py -v
"""

import json
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest

import tools.image_generation_tool as mod


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify module-level constants are reasonable."""

    def test_default_model(self):
        assert mod.DEFAULT_MODEL == "fal-ai/flux-2-pro"

    def test_default_aspect_ratio(self):
        assert mod.DEFAULT_ASPECT_RATIO == "landscape"

    def test_default_num_inference_steps(self):
        assert mod.DEFAULT_NUM_INFERENCE_STEPS == 50

    def test_default_guidance_scale(self):
        assert mod.DEFAULT_GUIDANCE_SCALE == 4.5

    def test_default_num_images(self):
        assert mod.DEFAULT_NUM_IMAGES == 1

    def test_default_output_format(self):
        assert mod.DEFAULT_OUTPUT_FORMAT == "png"

    def test_valid_image_sizes(self):
        assert "landscape_16_9" in mod.VALID_IMAGE_SIZES
        assert "square_hd" in mod.VALID_IMAGE_SIZES
        assert "portrait_16_9" in mod.VALID_IMAGE_SIZES

    def test_valid_output_formats(self):
        assert mod.VALID_OUTPUT_FORMATS == ["jpeg", "png"]

    def test_valid_acceleration_modes(self):
        assert "none" in mod.VALID_ACCELERATION_MODES
        assert "regular" in mod.VALID_ACCELERATION_MODES
        assert "high" in mod.VALID_ACCELERATION_MODES

    def test_aspect_ratio_map_keys(self):
        assert set(mod.ASPECT_RATIO_MAP.keys()) == {"landscape", "square", "portrait"}

    def test_aspect_ratio_map_values(self):
        assert mod.ASPECT_RATIO_MAP["landscape"] == "landscape_16_9"
        assert mod.ASPECT_RATIO_MAP["square"] == "square_hd"
        assert mod.ASPECT_RATIO_MAP["portrait"] == "portrait_16_9"

    def test_upscaler_factor(self):
        assert mod.UPSCALER_FACTOR == 2

    def test_safety_checker_disabled(self):
        assert mod.ENABLE_SAFETY_CHECKER is False

    def test_safety_tolerance(self):
        assert mod.SAFETY_TOLERANCE == "5"


# ---------------------------------------------------------------------------
# Test: _normalize_fal_queue_url_format
# ---------------------------------------------------------------------------

class TestNormalizeFalQueueUrlFormat:
    """Test URL normalization for managed FAL queue."""

    def test_adds_trailing_slash(self):
        result = mod._normalize_fal_queue_url_format("https://example.com")
        assert result == "https://example.com/"

    def test_strips_whitespace(self):
        result = mod._normalize_fal_queue_url_format("  https://example.com  ")
        assert result == "https://example.com/"

    def test_no_double_slash(self):
        result = mod._normalize_fal_queue_url_format("https://example.com/")
        assert result == "https://example.com/"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="required"):
            mod._normalize_fal_queue_url_format("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="required"):
            mod._normalize_fal_queue_url_format(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="required"):
            mod._normalize_fal_queue_url_format("   ")


# ---------------------------------------------------------------------------
# Test: _validate_parameters
# ---------------------------------------------------------------------------

class TestValidateParameters:
    """Parameter validation for image generation."""

    def test_valid_defaults(self):
        result = mod._validate_parameters("landscape_16_9", 50, 4.5, 1, "png")
        assert result["image_size"] == "landscape_16_9"
        assert result["num_inference_steps"] == 50
        assert result["guidance_scale"] == 4.5
        assert result["num_images"] == 1
        assert result["output_format"] == "png"

    def test_valid_all_formats(self):
        for fmt in ("jpeg", "png"):
            result = mod._validate_parameters("square_hd", 50, 4.5, 1, fmt)
            assert result["output_format"] == fmt

    def test_valid_all_sizes(self):
        for size in mod.VALID_IMAGE_SIZES:
            result = mod._validate_parameters(size, 50, 4.5, 1, "png")
            assert result["image_size"] == size

    def test_valid_all_acceleration(self):
        for acc in mod.VALID_ACCELERATION_MODES:
            result = mod._validate_parameters("square_hd", 50, 4.5, 1, "png", acc)
            assert result["acceleration"] == acc

    def test_custom_size_dict(self):
        result = mod._validate_parameters({"width": 512, "height": 768}, 50, 4.5, 1, "png")
        assert result["image_size"] == {"width": 512, "height": 768}

    def test_guidance_scale_int(self):
        result = mod._validate_parameters("square_hd", 50, 5, 1, "png")
        assert result["guidance_scale"] == 5.0

    def test_invalid_image_size_string(self):
        with pytest.raises(ValueError, match="Invalid image_size"):
            mod._validate_parameters("tiny", 50, 4.5, 1, "png")

    def test_invalid_image_size_type(self):
        with pytest.raises(ValueError, match="preset string or a dict"):
            mod._validate_parameters(123, 50, 4.5, 1, "png")

    def test_custom_size_missing_width(self):
        with pytest.raises(ValueError, match="width"):
            mod._validate_parameters({"height": 768}, 50, 4.5, 1, "png")

    def test_custom_size_missing_height(self):
        with pytest.raises(ValueError, match="height"):
            mod._validate_parameters({"width": 512}, 50, 4.5, 1, "png")

    def test_custom_size_non_int_width(self):
        with pytest.raises(ValueError, match="integers"):
            mod._validate_parameters({"width": 512.5, "height": 768}, 50, 4.5, 1, "png")

    def test_custom_size_too_small(self):
        with pytest.raises(ValueError, match="at least 64"):
            mod._validate_parameters({"width": 32, "height": 32}, 50, 4.5, 1, "png")

    def test_custom_size_too_large(self):
        with pytest.raises(ValueError, match="exceed 2048"):
            mod._validate_parameters({"width": 4096, "height": 768}, 50, 4.5, 1, "png")

    def test_steps_zero(self):
        with pytest.raises(ValueError, match="between 1 and 100"):
            mod._validate_parameters("square_hd", 0, 4.5, 1, "png")

    def test_steps_over_100(self):
        with pytest.raises(ValueError, match="between 1 and 100"):
            mod._validate_parameters("square_hd", 101, 4.5, 1, "png")

    def test_steps_non_int(self):
        with pytest.raises(ValueError, match="between 1 and 100"):
            mod._validate_parameters("square_hd", 50.5, 4.5, 1, "png")

    def test_guidance_too_low(self):
        with pytest.raises(ValueError, match="between 0.1 and 20.0"):
            mod._validate_parameters("square_hd", 50, 0.05, 1, "png")

    def test_guidance_too_high(self):
        with pytest.raises(ValueError, match="between 0.1 and 20.0"):
            mod._validate_parameters("square_hd", 50, 21.0, 1, "png")

    def test_num_images_zero(self):
        with pytest.raises(ValueError, match="between 1 and 4"):
            mod._validate_parameters("square_hd", 50, 4.5, 0, "png")

    def test_num_images_over_4(self):
        with pytest.raises(ValueError, match="between 1 and 4"):
            mod._validate_parameters("square_hd", 50, 4.5, 5, "png")

    def test_invalid_output_format(self):
        with pytest.raises(ValueError, match="output_format"):
            mod._validate_parameters("square_hd", 50, 4.5, 1, "gif")

    def test_invalid_acceleration(self):
        with pytest.raises(ValueError, match="acceleration"):
            mod._validate_parameters("square_hd", 50, 4.5, 1, "png", "turbo")


# ---------------------------------------------------------------------------
# Test: _resolve_managed_fal_gateway
# ---------------------------------------------------------------------------

class TestResolveManagedFalGateway:
    """Managed gateway resolution logic."""

    def test_returns_none_with_fal_key(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        result = mod._resolve_managed_fal_gateway()
        assert result is None

    def test_calls_resolve_when_no_key(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        mock_gw = MagicMock()
        with patch.object(mod, "resolve_managed_tool_gateway", return_value=mock_gw):
            result = mod._resolve_managed_fal_gateway()
            assert result is mock_gw

    def test_returns_none_when_no_key_and_no_gateway(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with patch.object(mod, "resolve_managed_tool_gateway", return_value=None):
            result = mod._resolve_managed_fal_gateway()
            assert result is None


# ---------------------------------------------------------------------------
# Test: check_fal_api_key
# ---------------------------------------------------------------------------

class TestCheckFalApiKey:
    """API key availability check."""

    def test_true_with_fal_key(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        assert mod.check_fal_api_key() is True

    def test_true_with_managed_gateway(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with patch.object(mod, "_resolve_managed_fal_gateway", return_value=MagicMock()):
            assert mod.check_fal_api_key() is True

    def test_false_with_nothing(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with patch.object(mod, "_resolve_managed_fal_gateway", return_value=None):
            assert mod.check_fal_api_key() is False


# ---------------------------------------------------------------------------
# Test: check_image_generation_requirements
# ---------------------------------------------------------------------------

class TestCheckImageGenerationRequirements:
    """Full requirements check (API key + fal_client import)."""

    def test_true_when_available(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        assert mod.check_image_generation_requirements() is True

    def test_false_when_no_key(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with patch.object(mod, "_resolve_managed_fal_gateway", return_value=None):
            assert mod.check_image_generation_requirements() is False


# ---------------------------------------------------------------------------
# Test: _ManagedFalSyncClient
# ---------------------------------------------------------------------------

class TestManagedFalSyncClient:
    """Managed FAL client construction and submit method."""

    def test_init_success(self):
        import fal_client
        # Save originals
        orig_SyncClient = fal_client.SyncClient
        orig_client = fal_client.client

        try:
            mock_handle_class = MagicMock()
            mock_client_module = MagicMock()
            mock_client_module.SyncRequestHandle = mock_handle_class
            mock_client_module._maybe_retry_request = MagicMock()
            mock_client_module._raise_for_status = MagicMock()
            mock_client_module.add_hint_header = MagicMock()
            mock_client_module.add_priority_header = MagicMock()
            mock_client_module.add_timeout_header = MagicMock()

            mock_sync_client = MagicMock()
            mock_http_client = MagicMock()
            mock_sync_client._client = mock_http_client
            fal_client.SyncClient = MagicMock(return_value=mock_sync_client)
            fal_client.client = mock_client_module

            client = mod._ManagedFalSyncClient(
                key="test-key",
                queue_run_origin="https://queue.example.com",
            )
            assert client._queue_url_format == "https://queue.example.com/"
            assert client._http_client is mock_http_client
        finally:
            fal_client.SyncClient = orig_SyncClient
            fal_client.client = orig_client

    def test_init_missing_sync_client(self):
        import fal_client
        orig_SyncClient = fal_client.SyncClient
        orig_client = fal_client.client
        try:
            fal_client.SyncClient = None
            fal_client.client = MagicMock()
            with pytest.raises(RuntimeError, match="SyncClient"):
                mod._ManagedFalSyncClient(
                    key="test-key",
                    queue_run_origin="https://queue.example.com",
                )
        finally:
            fal_client.SyncClient = orig_SyncClient
            fal_client.client = orig_client

    def test_init_missing_client_module(self):
        import fal_client
        orig_client = fal_client.client
        try:
            fal_client.client = None
            with pytest.raises(RuntimeError, match="client"):
                mod._ManagedFalSyncClient(
                    key="test-key",
                    queue_run_origin="https://queue.example.com",
                )
        finally:
            fal_client.client = orig_client

    def test_submit_builds_url(self):
        import fal_client
        orig_SyncClient = fal_client.SyncClient
        orig_client = fal_client.client
        try:
            mock_handle_class = MagicMock()
            mock_client_module = MagicMock()
            mock_client_module.SyncRequestHandle = mock_handle_class
            mock_client_module._maybe_retry_request = MagicMock()
            mock_client_module._raise_for_status = MagicMock()
            mock_client_module.add_hint_header = MagicMock()
            mock_client_module.add_priority_header = MagicMock()
            mock_client_module.add_timeout_header = MagicMock()

            mock_sync_client = MagicMock()
            mock_http_client = MagicMock()
            mock_sync_client._client = mock_http_client
            fal_client.SyncClient = MagicMock(return_value=mock_sync_client)
            fal_client.client = mock_client_module

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "request_id": "req-1",
                "response_url": "https://r.example.com",
                "status_url": "https://s.example.com",
                "cancel_url": "https://c.example.com",
            }
            mock_client_module._maybe_retry_request.return_value = mock_response

            client = mod._ManagedFalSyncClient(
                key="test-key",
                queue_run_origin="https://queue.example.com",
            )
            handle = client.submit("fal-ai/flux-2-pro", {"prompt": "test"})
            assert handle is not None
            call_args = mock_client_module._maybe_retry_request.call_args
            assert "fal-ai/flux-2-pro" in call_args[0][2]
        finally:
            fal_client.SyncClient = orig_SyncClient
            fal_client.client = orig_client

    def test_submit_with_webhook_url(self):
        import fal_client
        orig_SyncClient = fal_client.SyncClient
        orig_client = fal_client.client
        try:
            mock_handle_class = MagicMock()
            mock_client_module = MagicMock()
            mock_client_module.SyncRequestHandle = mock_handle_class
            mock_client_module._maybe_retry_request = MagicMock()
            mock_client_module._raise_for_status = MagicMock()
            mock_client_module.add_hint_header = MagicMock()
            mock_client_module.add_priority_header = MagicMock()
            mock_client_module.add_timeout_header = MagicMock()

            mock_sync_client = MagicMock()
            mock_http_client = MagicMock()
            mock_sync_client._client = mock_http_client
            fal_client.SyncClient = MagicMock(return_value=mock_sync_client)
            fal_client.client = mock_client_module

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "request_id": "req-1",
                "response_url": "https://r.example.com",
                "status_url": "https://s.example.com",
                "cancel_url": "https://c.example.com",
            }
            mock_client_module._maybe_retry_request.return_value = mock_response

            client = mod._ManagedFalSyncClient(
                key="test-key",
                queue_run_origin="https://queue.example.com",
            )
            client.submit("model", {"prompt": "test"}, webhook_url="https://hook.example.com")
            call_args = mock_client_module._maybe_retry_request.call_args
            url = call_args[0][2]
            assert "fal_webhook=" in url
        finally:
            fal_client.SyncClient = orig_SyncClient
            fal_client.client = orig_client


# ---------------------------------------------------------------------------
# Test: image_generate_tool
# ---------------------------------------------------------------------------

class TestImageGenerateTool:
    """Main tool function tests (error paths and basic flow)."""

    def test_empty_prompt_returns_error(self):
        result = mod.image_generate_tool(prompt="")
        data = json.loads(result)
        assert data["success"] is False
        assert data["image"] is None
        assert "error" in data

    def test_none_prompt_returns_error(self):
        result = mod.image_generate_tool(prompt=None)
        data = json.loads(result)
        assert data["success"] is False

    def test_whitespace_prompt_returns_error(self):
        result = mod.image_generate_tool(prompt="   ")
        data = json.loads(result)
        assert data["success"] is False

    def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with patch.object(mod, "_resolve_managed_fal_gateway", return_value=None),              patch.object(mod, "managed_nous_tools_enabled", return_value=False):
            result = mod.image_generate_tool(prompt="a sunset")
            data = json.loads(result)
            assert data["success"] is False
            assert "FAL_KEY" in data.get("error", "")

    def test_invalid_aspect_ratio_uses_default(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/test.png", "width": 1024, "height": 576}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value=None):
                result = mod.image_generate_tool(
                    prompt="a sunset",
                    aspect_ratio="invalid_ratio",
                )
                data = json.loads(result)
                assert data["success"] is True
                call_args = mock_submit.call_args
                assert call_args[0][0] == "fal-ai/flux-2-pro"

    def test_successful_generation_no_upscale(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/test.png", "width": 1024, "height": 576}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value=None):
                result = mod.image_generate_tool(prompt="a mountain")
                data = json.loads(result)
                assert data["success"] is True
                assert "https://img.example.com/test.png" in data["image"]

    def test_successful_generation_with_upscale(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/orig.png", "width": 512, "height": 288}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value={
                "url": "https://img.example.com/upscaled.png",
                "width": 1024,
                "height": 576,
                "upscaled": True,
                "upscale_factor": 2,
            }):
                result = mod.image_generate_tool(prompt="a lake")
                data = json.loads(result)
                assert data["success"] is True
                assert "upscaled.png" in data["image"]

    def test_api_returns_no_images(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {"images": []}
            mock_submit.return_value = mock_handler
            result = mod.image_generate_tool(prompt="a test")
            data = json.loads(result)
            assert data["success"] is False

    def test_api_returns_none(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = None
            mock_submit.return_value = mock_handler
            result = mod.image_generate_tool(prompt="a test")
            data = json.loads(result)
            assert data["success"] is False

    def test_api_raises_exception(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_submit.side_effect = RuntimeError("API error")
            result = mod.image_generate_tool(prompt="a test")
            data = json.loads(result)
            assert data["success"] is False
            assert "API error" in data.get("error", "")

    def test_response_includes_error_type(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_submit.side_effect = RuntimeError("timeout")
            result = mod.image_generate_tool(prompt="a test")
            data = json.loads(result)
            assert data.get("error_type") == "RuntimeError"

    def test_seed_included_in_arguments(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/test.png", "width": 1024, "height": 576}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value=None):
                mod.image_generate_tool(prompt="a test", seed=42)
                call_args = mock_submit.call_args
                assert call_args[1]["arguments"]["seed"] == 42

    def test_aspect_ratio_mapping(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/test.png", "width": 1024, "height": 1024}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value=None):
                mod.image_generate_tool(prompt="a test", aspect_ratio="square")
                call_args = mock_submit.call_args
                assert call_args[1]["arguments"]["image_size"] == "square_hd"


# ---------------------------------------------------------------------------
# Test: _upscale_image
# ---------------------------------------------------------------------------

class TestUpscaleImage:
    """Upscale function tests."""

    def test_upscale_success(self):
        mock_handler = MagicMock()
        mock_handler.get.return_value = {
            "image": {
                "url": "https://img.example.com/upscaled.png",
                "width": 2048,
                "height": 1152,
            }
        }
        with patch.object(mod, "_submit_fal_request", return_value=mock_handler):
            result = mod._upscale_image("https://img.example.com/orig.png", "a test")
            assert result is not None
            assert result["upscaled"] is True
            assert result["url"] == "https://img.example.com/upscaled.png"
            assert result["upscale_factor"] == 2

    def test_upscale_no_image_key(self):
        mock_handler = MagicMock()
        mock_handler.get.return_value = {"something_else": True}
        with patch.object(mod, "_submit_fal_request", return_value=mock_handler):
            result = mod._upscale_image("https://img.example.com/orig.png", "a test")
            assert result is None

    def test_upscale_exception(self):
        with patch.object(mod, "_submit_fal_request", side_effect=RuntimeError("fail")):
            result = mod._upscale_image("https://img.example.com/orig.png", "a test")
            assert result is None


# ---------------------------------------------------------------------------
# Test: _handle_image_generate (registry handler)
# ---------------------------------------------------------------------------

class TestHandleImageGenerate:
    """Registry handler wrapper function."""

    def test_empty_prompt_returns_error(self):
        result = mod._handle_image_generate({"prompt": ""})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_missing_prompt_returns_error(self):
        result = mod._handle_image_generate({})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_passes_args_to_tool(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/test.png", "width": 1024, "height": 576}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value=None):
                result = mod._handle_image_generate({"prompt": "a test"})
                data = json.loads(result)
                assert data["success"] is True

    def test_default_aspect_ratio_is_landscape(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-key")
        with patch.object(mod, "_submit_fal_request") as mock_submit:
            mock_handler = MagicMock()
            mock_handler.get.return_value = {
                "images": [{"url": "https://img.example.com/test.png", "width": 1024, "height": 576}]
            }
            mock_submit.return_value = mock_handler
            with patch.object(mod, "_upscale_image", return_value=None):
                mod._handle_image_generate({"prompt": "a test"})
                call_args = mock_submit.call_args
                assert call_args[1]["arguments"]["image_size"] == "landscape_16_9"


# ---------------------------------------------------------------------------
# Test: _get_managed_fal_client
# ---------------------------------------------------------------------------

class TestGetManagedFalClient:
    """Client caching and reuse."""

    @pytest.fixture(autouse=True)
    def _reset_globals(self):
        """Reset module-level globals before each test."""
        orig_client = mod._managed_fal_client
        orig_config = mod._managed_fal_client_config
        yield
        mod._managed_fal_client = orig_client
        mod._managed_fal_client_config = orig_config

    def test_creates_new_client(self):
        import fal_client
        orig_SyncClient = fal_client.SyncClient
        orig_client = fal_client.client
        try:
            mock_gw = MagicMock()
            mock_gw.gateway_origin = "https://queue.example.com"
            mock_gw.nous_user_token = "test-token"

            mock_handle_class = MagicMock()
            mock_client_module = MagicMock()
            mock_client_module.SyncRequestHandle = mock_handle_class
            mock_client_module._maybe_retry_request = MagicMock()
            mock_client_module._raise_for_status = MagicMock()
            mock_client_module.add_hint_header = MagicMock()
            mock_client_module.add_priority_header = MagicMock()
            mock_client_module.add_timeout_header = MagicMock()
            fal_client.client = mock_client_module

            mock_sync = MagicMock()
            mock_sync._client = MagicMock()
            fal_client.SyncClient = MagicMock(return_value=mock_sync)

            mod._managed_fal_client = None
            mod._managed_fal_client_config = None
            client = mod._get_managed_fal_client(mock_gw)
            assert client is not None
            assert mod._managed_fal_client is client
        finally:
            fal_client.SyncClient = orig_SyncClient
            fal_client.client = orig_client

    def test_reuses_cached_client(self):
        mock_gw = MagicMock()
        mock_gw.gateway_origin = "https://queue.example.com"
        mock_gw.nous_user_token = "test-token"

        mock_cached = MagicMock()
        mod._managed_fal_client = mock_cached
        mod._managed_fal_client_config = ("https://queue.example.com", "test-token")

        client = mod._get_managed_fal_client(mock_gw)
        assert client is mock_cached


# ---------------------------------------------------------------------------
# Test: _submit_fal_request
# ---------------------------------------------------------------------------

class TestSubmitFalRequest:
    """Request submission with direct vs managed credentials."""

    def test_uses_direct_fal_when_key_available(self, monkeypatch):
        import fal_client
        monkeypatch.setenv("FAL_KEY", "test-key")
        mock_result = MagicMock()
        with patch.object(fal_client, "submit", return_value=mock_result) as mock_submit:
            result = mod._submit_fal_request("model", {"prompt": "test"})
            assert result is mock_result
            mock_submit.assert_called_once()

    def test_uses_managed_when_no_key(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        mock_gw = MagicMock()
        mock_gw.gateway_origin = "https://queue.example.com"
        mock_gw.nous_user_token = "test-token"

        with patch.object(mod, "_resolve_managed_fal_gateway", return_value=mock_gw),              patch.object(mod, "_get_managed_fal_client") as mock_get:
            mock_client = MagicMock()
            mock_get.return_value = mock_client
            result = mod._submit_fal_request("model", {"prompt": "test"})
            mock_client.submit.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Verify the tool is registered correctly."""

    def test_schema_name(self):
        schema = mod.IMAGE_GENERATE_SCHEMA
        assert schema["name"] == "image_generate"
        assert "prompt" in schema["parameters"]["properties"]

    def test_schema_required_fields(self):
        schema = mod.IMAGE_GENERATE_SCHEMA
        assert "prompt" in schema["parameters"]["required"]

    def test_schema_aspect_ratio_enum(self):
        schema = mod.IMAGE_GENERATE_SCHEMA
        ar = schema["parameters"]["properties"]["aspect_ratio"]
        assert set(ar["enum"]) == {"landscape", "square", "portrait"}


# ---------------------------------------------------------------------------
# Test: get_debug_session_info
# ---------------------------------------------------------------------------

class TestGetDebugSessionInfo:
    """Debug session info getter."""

    def test_returns_dict(self):
        result = mod.get_debug_session_info()
        assert isinstance(result, dict)
