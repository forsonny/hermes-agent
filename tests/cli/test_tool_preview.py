"""Tests for build_tool_preview entries — browser and HA tools.

These tests verify that the ``build_tool_preview`` function in
``agent/display.py`` produces correct short previews for the browser
automation and Home Assistant tool families.
"""

import pytest


class TestBuildToolPreview:
    """Verify tool preview entries for browser and ha_* tools."""

    def test_browser_press_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_press", {"key": "Enter"})
        assert result == "Enter"

    def test_browser_scroll_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_scroll", {"direction": "down"})
        assert result == "down"

    def test_browser_console_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_console", {"expression": "document.title"})
        assert result == "document.title"

    def test_browser_vision_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_vision", {"question": "what do you see?"})
        assert result == "what do you see?"

    def test_browser_snapshot_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_snapshot", {"full": True})
        assert result == "True"

    def test_browser_get_images_no_preview(self):
        """browser_get_images has no primary arg — should return None."""
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_get_images", {})
        assert result is None

    def test_browser_back_no_preview(self):
        """browser_back has no primary arg — should return None."""
        from agent.display import build_tool_preview
        result = build_tool_preview("browser_back", {})
        assert result is None

    def test_ha_call_service_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("ha_call_service", {"service": "light.turn_on"})
        assert result == "light.turn_on"

    def test_ha_get_state_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("ha_get_state", {"entity_id": "light.living_room"})
        assert result == "light.living_room"

    def test_ha_list_entities_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("ha_list_entities", {"domain": "light"})
        assert result == "light"

    def test_ha_list_services_preview(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("ha_list_services", {"domain": "switch"})
        assert result == "switch"
