"""Tests for format_tool_timing_footer — aggregate tool timing summary."""

import pytest
from unittest.mock import patch

from agent.display import format_tool_timing_footer


class TestFormatToolTimingFooter:
    """Tests for format_tool_timing_footer."""

    def test_returns_none_for_empty_list(self):
        result = format_tool_timing_footer([])
        assert result is None

    def test_returns_none_for_single_tool(self):
        result = format_tool_timing_footer([("terminal", 1.0)])
        assert result is None

    def test_returns_summary_for_two_tools(self):
        timings = [("terminal", 2.5), ("read_file", 1.0)]
        result = format_tool_timing_footer(timings)
        assert result is not None
        assert "2 tools" in result
        assert "3.5s" in result
        assert "slowest" in result
        assert "terminal" in result
        assert "2.5s" in result

    def test_returns_summary_for_many_tools(self):
        timings = [
            ("web_search", 3.2),
            ("terminal", 1.5),
            ("read_file", 0.5),
            ("write_file", 2.0),
            ("browser_navigate", 5.1),
        ]
        result = format_tool_timing_footer(timings)
        assert result is not None
        assert "5 tools" in result
        assert "12.3s" in result
        assert "browser_navigate" in result
        assert "5.1s" in result

    def test_skin_prefix_applied(self):
        timings = [("terminal", 2.0), ("read_file", 1.0)]
        with patch("agent.display.get_skin_tool_prefix", return_value="│"):
            result = format_tool_timing_footer(timings)
        assert result is not None
        assert result.startswith("│")
        assert "┊" not in result

    def test_default_prefix_unchanged(self):
        timings = [("web_search", 1.0), ("terminal", 2.0)]
        result = format_tool_timing_footer(timings)
        assert result is not None
        assert result.startswith("┊")

    def test_timing_footer_format_string(self):
        timings = [("terminal", 1.234), ("write_file", 0.567)]
        result = format_tool_timing_footer(timings)
        assert result is not None
        # Duration should be formatted to 1 decimal place
        assert "1.8s" in result
        assert "1.2s" in result
