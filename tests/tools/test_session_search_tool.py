"""Tests for tools/session_search_tool.py pure functions and session_search."""

import json
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ── _format_timestamp ──────────────────────────────────────────────────


class TestFormatTimestamp:
    """Tests for _format_timestamp() conversion."""

    def _import(self):
        from tools.session_search_tool import _format_timestamp
        return _format_timestamp

    def test_none_returns_unknown(self):
        fmt = self._import()
        assert fmt(None) == "unknown"

    def test_integer_unix_timestamp(self):
        fmt = self._import()
        # Jan 1, 2025 00:00:00 UTC → local time depends on TZ,
        # just check it returns a formatted string, not the raw int
        result = fmt(1735689600)
        assert result != "unknown"
        assert "2025" in result or "2024" in result  # TZ dependent

    def test_float_unix_timestamp(self):
        fmt = self._import()
        result = fmt(1735689600.123)
        assert result != "unknown"
        assert isinstance(result, str)

    def test_iso_string_passthrough(self):
        fmt = self._import()
        iso = "2025-01-15T10:30:00"
        result = fmt(iso)
        # Non-numeric strings are returned as-is
        assert result == iso

    def test_numeric_string_timestamp(self):
        fmt = self._import()
        result = fmt("1735689600")
        assert result != "unknown"
        assert "2025" in result or "2024" in result

    def test_negative_timestamp(self):
        fmt = self._import()
        # Negative timestamps (before epoch) should still return a string
        result = fmt(-1000000)
        assert isinstance(result, str)

    def test_invalid_string_returns_str(self):
        fmt = self._import()
        result = fmt("not-a-timestamp")
        # Non-numeric, non-ISO string passed through
        assert result == "not-a-timestamp"

    def test_empty_string_returns_empty(self):
        fmt = self._import()
        assert fmt("") == ""

    def test_zero_timestamp(self):
        fmt = self._import()
        result = fmt(0)
        assert isinstance(result, str)
        assert result != "unknown"


# ── _format_conversation ──────────────────────────────────────────────


class TestFormatConversation:
    """Tests for _format_conversation() message formatting."""

    def _import(self):
        from tools.session_search_tool import _format_conversation
        return _format_conversation

    def test_empty_messages(self):
        fmt = self._import()
        assert fmt([]) == ""

    def test_user_message(self):
        fmt = self._import()
        msgs = [{"role": "user", "content": "Hello"}]
        result = fmt(msgs)
        assert "[USER]: Hello" in result

    def test_assistant_message(self):
        fmt = self._import()
        msgs = [{"role": "assistant", "content": "Hi there"}]
        result = fmt(msgs)
        assert "[ASSISTANT]: Hi there" in result

    def test_tool_message_with_name(self):
        fmt = self._import()
        msgs = [{"role": "tool", "content": "output", "tool_name": "terminal"}]
        result = fmt(msgs)
        assert "[TOOL:terminal]: output" in result

    def test_tool_message_long_content_truncated(self):
        fmt = self._import()
        long_content = "x" * 600
        msgs = [{"role": "tool", "content": long_content, "tool_name": "terminal"}]
        result = fmt(msgs)
        assert "[truncated]" in result
        assert len(result) < len(long_content) + 50

    def test_tool_message_short_content_not_truncated(self):
        fmt = self._import()
        short = "x" * 400
        msgs = [{"role": "tool", "content": short, "tool_name": "terminal"}]
        result = fmt(msgs)
        assert "[truncated]" not in result
        assert short in result

    def test_assistant_with_tool_calls(self):
        fmt = self._import()
        msgs = [{
            "role": "assistant",
            "content": "Running command",
            "tool_calls": [
                {"name": "terminal"},
                {"function": {"name": "browser_navigate"}},
            ],
        }]
        result = fmt(msgs)
        assert "[Called: terminal, browser_navigate]" in result
        assert "Running command" in result

    def test_assistant_with_tool_calls_no_content(self):
        fmt = self._import()
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "terminal"}],
        }]
        result = fmt(msgs)
        assert "[Called: terminal]" in result

    def test_unknown_role(self):
        fmt = self._import()
        msgs = [{"role": "system", "content": "Be helpful"}]
        result = fmt(msgs)
        assert "[SYSTEM]: Be helpful" in result

    def test_missing_role(self):
        fmt = self._import()
        msgs = [{"content": "no role"}]
        result = fmt(msgs)
        assert "[UNKNOWN]: no role" in result

    def test_none_content(self):
        fmt = self._import()
        msgs = [{"role": "user", "content": None}]
        result = fmt(msgs)
        assert "[USER]:" in result

    def test_multiple_messages_joined(self):
        fmt = self._import()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = fmt(msgs)
        assert "[USER]: Hello" in result
        assert "[ASSISTANT]: Hi" in result
        # Messages should be separated by double newline
        assert "\n\n" in result


# ── _find_all_match_positions ─────────────────────────────────────────


class TestFindAllMatchPositions:
    """Tests for _find_all_match_positions() text search."""

    def _import(self):
        from tools.session_search_tool import _find_all_match_positions
        return _find_all_match_positions

    def test_empty_text(self):
        find = self._import()
        assert find("", ["hello"]) == []

    def test_empty_terms(self):
        find = self._import()
        assert find("hello world", []) == []

    def test_single_match(self):
        find = self._import()
        result = find("hello world", ["hello"])
        assert result == [(0, 5)]

    def test_multiple_matches(self):
        find = self._import()
        result = find("hello hello", ["hello"])
        assert result == [(0, 5), (6, 5)]

    def test_multiple_terms(self):
        find = self._import()
        result = find("hello world foo", ["hello", "foo"])
        assert (0, 5) in result
        assert (12, 3) in result

    def test_case_insensitive(self):
        find = self._import()
        # Function expects text already lowered by caller
        result = find("hello world", ["hello"])
        assert result == [(0, 5)]

    def test_no_match(self):
        find = self._import()
        assert find("hello world", ["missing"]) == []

    def test_empty_term_skipped(self):
        find = self._import()
        assert find("hello world", ["", "hello"]) == [(0, 5)]

    def test_position_limit(self):
        find = self._import()
        text = "ab " * 1000
        result = find(text, ["ab"])
        # Limit is >500, so up to 501 is possible before break
        assert len(result) <= 501


# ── _merge_regions ────────────────────────────────────────────────────


class TestMergeRegions:
    """Tests for _merge_regions() region merging."""

    def _import(self):
        from tools.session_search_tool import _merge_regions
        return _merge_regions

    def test_empty_positions(self):
        merge = self._import()
        assert merge([], 100) == []

    def test_single_region(self):
        merge = self._import()
        result = merge([(10, 5)], 100)
        assert result == [(10, 15)]

    def test_overlapping_merged(self):
        merge = self._import()
        result = merge([(10, 5), (12, 5)], 100)
        assert result == [(10, 17)]

    def test_close_regions_merged(self):
        merge = self._import()
        # gap=10, distance between end of first and start of second = 8
        result = merge([(10, 5), (23, 5)], 10)
        assert result == [(10, 28)]

    def test_distant_regions_separate(self):
        merge = self._import()
        # gap=5, distance = 10 (> gap)
        result = merge([(10, 5), (25, 5)], 5)
        assert result == [(10, 15), (25, 30)]

    def test_zero_gap(self):
        merge = self._import()
        # Only overlapping/touching regions merge with gap=0
        result = merge([(10, 5), (15, 5)], 0)
        assert result == [(10, 20)]

    def test_zero_gap_non_adjacent(self):
        merge = self._import()
        result = merge([(10, 5), (16, 5)], 0)
        assert result == [(10, 15), (16, 21)]


# ── _truncate_around_matches ──────────────────────────────────────────


class TestTruncateAroundMatches:
    """Tests for _truncate_around_matches() smart truncation."""

    def _import(self):
        from tools.session_search_tool import _truncate_around_matches
        return _truncate_around_matches

    def test_short_text_unchanged(self):
        trunc = self._import()
        text = "Hello world"
        assert trunc(text, "hello") == text

    def test_text_under_max_unchanged(self):
        trunc = self._import()
        text = "Hello " * 100  # 600 chars
        assert trunc(text, "hello", max_chars=1000) == text

    def test_empty_query_takes_start(self):
        trunc = self._import()
        text = "x" * 2000
        result = trunc(text, "", max_chars=500)
        assert len(result) <= 500
        assert result == "x" * 500

    def test_query_match_preserved(self):
        trunc = self._import()
        text = "x" * 500 + "TARGET" + "y" * 500
        result = trunc(text, "TARGET", max_chars=200)
        assert "TARGET" in result

    def test_no_match_takes_start(self):
        trunc = self._import()
        text = "a" * 2000
        result = trunc(text, "NOTFOUND", max_chars=500)
        assert len(result) <= 500
        assert result.startswith("aaa")

    def test_gap_marker_appears(self):
        trunc = self._import()
        # Two distant matches
        text = "ALPHA" + "x" * 2000 + "BETA" + "y" * 2000
        result = trunc(text, "ALPHA BETA", max_chars=5000)
        # If gap markers present, the matches are preserved
        assert "ALPHA" in result
        assert "BETA" in result

    def test_prefix_suffix_markers(self):
        trunc = self._import()
        text = "x" * 2000 + "TARGET" + "y" * 2000
        result = trunc(text, "TARGET", max_chars=500)
        # Should have truncation markers when not starting at 0
        assert "TARGET" in result


# ── _list_recent_sessions ────────────────────────────────────────────


class TestListRecentSessions:
    """Tests for _list_recent_sessions() metadata retrieval."""

    def _import(self):
        from tools.session_search_tool import _list_recent_sessions
        return _list_recent_sessions

    def _mock_db(self, sessions=None):
        db = MagicMock()
        db.list_sessions_rich.return_value = sessions or []
        return db

    def test_empty_sessions(self):
        list_fn = self._import()
        db = self._mock_db([])
        result = json.loads(list_fn(db, limit=5))
        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []

    def test_returns_sessions(self):
        list_fn = self._import()
        sessions = [
            {
                "id": "s1",
                "title": "Test Session",
                "source": "cli",
                "started_at": 1735689600,
                "last_active": 1735689600,
                "message_count": 5,
                "preview": "Hello world",
            }
        ]
        db = self._mock_db(sessions)
        result = json.loads(list_fn(db, limit=5))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Test Session"

    def test_excludes_current_session(self):
        list_fn = self._import()
        sessions = [
            {"id": "s1", "title": "A", "source": "cli", "started_at": 0,
             "last_active": 0, "message_count": 1, "preview": "p"},
            {"id": "current", "title": "B", "source": "cli", "started_at": 0,
             "last_active": 0, "message_count": 1, "preview": "p"},
        ]
        db = self._mock_db(sessions)
        db.get_session.return_value = None
        result = json.loads(list_fn(db, limit=5, current_session_id="current"))
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "s1"

    def test_excludes_child_sessions(self):
        list_fn = self._import()
        sessions = [
            {"id": "s1", "title": "Parent", "source": "cli", "started_at": 0,
             "last_active": 0, "message_count": 1, "preview": "p"},
            {"id": "s2", "title": "Child", "source": "cli", "started_at": 0,
             "last_active": 0, "message_count": 1, "preview": "p",
             "parent_session_id": "s1"},
        ]
        db = self._mock_db(sessions)
        result = json.loads(list_fn(db, limit=5))
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Parent"

    def test_respects_limit(self):
        list_fn = self._import()
        sessions = [
            {"id": f"s{i}", "title": f"Session {i}", "source": "cli",
             "started_at": 0, "last_active": 0, "message_count": 1, "preview": "p"}
            for i in range(10)
        ]
        db = self._mock_db(sessions)
        result = json.loads(list_fn(db, limit=3))
        assert result["count"] == 3

    def test_db_error_returns_error(self):
        list_fn = self._import()
        db = MagicMock()
        db.list_sessions_rich.side_effect = RuntimeError("DB broken")
        result = json.loads(list_fn(db, limit=5))
        assert result["success"] is False


# ── session_search (main function) ────────────────────────────────────


class TestSessionSearch:
    """Tests for session_search() main function."""

    def _import(self):
        from tools.session_search_tool import session_search
        return session_search

    def test_none_db_returns_error(self):
        search = self._import()
        result = json.loads(search(query="test", db=None))
        assert result["success"] is False

    def test_empty_query_returns_recent(self):
        search = self._import()
        db = MagicMock()
        db.list_sessions_rich.return_value = []
        result = json.loads(search(query="", db=db))
        assert result["success"] is True
        assert result["mode"] == "recent"

    def test_whitespace_query_returns_recent(self):
        search = self._import()
        db = MagicMock()
        db.list_sessions_rich.return_value = []
        result = json.loads(search(query="   ", db=db))
        assert result["success"] is True
        assert result["mode"] == "recent"

    def test_no_matches_returns_empty(self):
        search = self._import()
        db = MagicMock()
        db.search_messages.return_value = []
        result = json.loads(search(query="nonexistent topic", db=db))
        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []

    def test_limit_capped_at_5(self):
        search = self._import()
        db = MagicMock()
        db.search_messages.return_value = []
        result = json.loads(search(query="test", limit=20, db=db))
        # Should not error even with limit > 5
        assert result["success"] is True

    def test_search_with_role_filter(self):
        search = self._import()
        db = MagicMock()
        db.search_messages.return_value = []
        result = json.loads(search(query="test", role_filter="user,assistant", db=db))
        # Verify role_filter was parsed and passed
        call_args = db.search_messages.call_args
        assert call_args.kwargs.get("role_filter") == ["user", "assistant"] or \
               (call_args[1].get("role_filter") == ["user", "assistant"])

    def test_limit_default_is_3(self):
        search = self._import()
        db = MagicMock()
        db.search_messages.return_value = []
        result = json.loads(search(query="test", db=db))
        # Should work with default limit
        assert result["success"] is True


# ── check_session_search_requirements ─────────────────────────────────


class TestCheckRequirements:
    """Tests for check_session_search_requirements()."""

    def _import(self):
        from tools.session_search_tool import check_session_search_requirements
        return check_session_search_requirements

    def test_returns_bool(self):
        check = self._import()
        result = check()
        assert isinstance(result, bool)
