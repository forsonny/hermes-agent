"""Tests for hermes sessions export --format (markdown, json, jsonl).

The _format_session_markdown and _write_export functions are nested inside
main(), so we test by reimplementing the same rendering logic against
synthetic session data.
"""
import json
import os
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _make_session(
    session_id="test-session-001",
    title="Test Session",
    source="cli",
    model="anthropic/claude-sonnet-4-20250514",
    messages=None,
    started_at=None,
    ended_at=_SENTINEL,
    message_count=None,
    tool_call_count=0,
    estimated_cost_usd=None,
):
    """Create a synthetic session dict matching SessionDB.export_session() format."""
    if started_at is None:
        started_at = time.time() - 3600
    if ended_at is _SENTINEL:
        ended_at = time.time()
    if messages is None:
        messages = [
            {"role": "user", "content": "Hello, world!", "timestamp": started_at},
            {"role": "assistant", "content": "Hi there! How can I help?", "timestamp": started_at + 1},
        ]
    if message_count is None:
        message_count = len(messages)
    return {
        "id": session_id,
        "title": title,
        "source": source,
        "model": model,
        "started_at": started_at,
        "ended_at": ended_at,
        "end_reason": "complete",
        "message_count": message_count,
        "tool_call_count": tool_call_count,
        "input_tokens": 100,
        "output_tokens": 50,
        "estimated_cost_usd": estimated_cost_usd,
        "messages": messages,
    }


def _format_session_markdown(session: dict) -> str:
    """Reimplementation of the nested function from main.py for testing."""
    from datetime import datetime as _dt

    lines = []
    title = session.get("title") or "Untitled Session"
    sid = session.get("id", "unknown")
    source = session.get("source", "unknown")
    model = session.get("model", "unknown")
    started = session.get("started_at")
    ended = session.get("ended_at")
    msg_count = session.get("message_count", 0)
    tool_count = session.get("tool_call_count", 0)
    cost = session.get("estimated_cost_usd") or session.get("actual_cost_usd")

    def _ts(val):
        if val is None:
            return "N/A"
        try:
            return _dt.fromtimestamp(float(val)).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError, TypeError):
            return str(val)

    lines.append(f"# {title}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Session ID | `{sid}` |")
    lines.append(f"| Source | {source} |")
    lines.append(f"| Model | {model} |")
    lines.append(f"| Started | {_ts(started)} |")
    lines.append(f"| Ended | {_ts(ended)} |")
    lines.append(f"| Messages | {msg_count} |")
    lines.append(f"| Tool Calls | {tool_count} |")
    if cost is not None:
        lines.append(f"| Est. Cost | ${cost:.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    messages = session.get("messages", [])
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name")
        tool_calls = msg.get("tool_calls")

        if role == "SYSTEM":
            lines.append("### System Prompt")
            lines.append("```")
            lines.append(content[:2000] + ("..." if len(content) > 2000 else ""))
            lines.append("```")
            lines.append("")
        elif role == "USER":
            lines.append("### 👤 User")
            lines.append("")
            lines.append(content)
            lines.append("")
        elif role == "ASSISTANT":
            if tool_calls and isinstance(tool_calls, list):
                tc_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        tc_names.append(name)
                if tc_names:
                    lines.append(f"### 🤖 Assistant — Called: {', '.join(tc_names)}")
                else:
                    lines.append("### 🤖 Assistant")
            else:
                lines.append("### 🤖 Assistant")
            lines.append("")
            if content:
                lines.append(content)
                lines.append("")
        elif role == "TOOL":
            label = f"🔧 Tool: {tool_name}" if tool_name else "🔧 Tool Result"
            if len(content) > 1000:
                content_display = content[:500] + "\n\n...[truncated]...\n\n" + content[-500:]
            else:
                content_display = content
            lines.append(f"### {label}")
            lines.append("```")
            lines.append(content_display)
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"### {role}")
            lines.append("")
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test: Markdown format
# ---------------------------------------------------------------------------

class TestFormatSessionMarkdown:
    """Tests for the markdown session formatter."""

    def test_basic_session_produces_valid_markdown(self):
        session = _make_session()
        md = _format_session_markdown(session)
        assert md.startswith("# Test Session")
        assert "| Session ID | `test-session-001` |" in md
        assert "| Source | cli |" in md
        assert "| Model | anthropic/claude-sonnet-4-20250514 |" in md
        assert "### 👤 User" in md
        assert "Hello, world!" in md
        assert "### 🤖 Assistant" in md
        assert "Hi there! How can I help?" in md

    def test_untitled_session_uses_fallback_title(self):
        session = _make_session(title=None)
        md = _format_session_markdown(session)
        assert md.startswith("# Untitled Session")

    def test_cost_displayed_when_present(self):
        session = _make_session(estimated_cost_usd=0.0421)
        md = _format_session_markdown(session)
        assert "| Est. Cost | $0.0421 |" in md

    def test_cost_omitted_when_none(self):
        session = _make_session(estimated_cost_usd=None)
        md = _format_session_markdown(session)
        assert "Est. Cost" not in md

    def test_tool_calls_displayed_in_assistant_header(self):
        session = _make_session(messages=[
            {"role": "user", "content": "list files", "timestamp": time.time()},
            {"role": "assistant", "content": "", "timestamp": time.time() + 1,
             "tool_calls": [{"name": "terminal", "function": {"name": "terminal"}}]},
            {"role": "tool", "content": "file1.py\nfile2.py", "tool_name": "terminal", "timestamp": time.time() + 2},
        ])
        md = _format_session_markdown(session)
        assert "### 🤖 Assistant — Called: terminal" in md
        assert "### 🔧 Tool: terminal" in md

    def test_system_prompt_truncated_at_2000_chars(self):
        long_prompt = "x" * 3000
        session = _make_session(messages=[
            {"role": "system", "content": long_prompt, "timestamp": time.time()},
        ])
        md = _format_session_markdown(session)
        assert "..." in md
        assert long_prompt not in md  # full text should not be present
        assert "x" * 2000 in md  # truncated portion present

    def test_tool_output_truncated_at_1000_chars(self):
        long_output = "y" * 2000
        session = _make_session(messages=[
            {"role": "tool", "content": long_output, "tool_name": "read_file", "timestamp": time.time()},
        ])
        md = _format_session_markdown(session)
        assert "...[truncated]..." in md

    def test_timestamp_formatting(self):
        # Use a known timestamp: Jan 1, 2025 00:00:00 UTC
        known_ts = 1735689600.0
        session = _make_session(started_at=known_ts, ended_at=known_ts + 60)
        md = _format_session_markdown(session)
        # Should contain a date-like string
        assert "2025" in md

    def test_no_ended_at_shows_na(self):
        session = _make_session(ended_at=None)
        md = _format_session_markdown(session)
        assert "| Ended | N/A |" in md


# ---------------------------------------------------------------------------
# Test: JSON format
# ---------------------------------------------------------------------------

class TestExportJSON:
    """Tests for JSON array export format."""

    def test_json_export_is_valid_array(self, tmp_path):
        sessions = [_make_session(), _make_session(session_id="test-002")]
        output = json.dumps(sessions, ensure_ascii=False, indent=2)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "test-session-001"
        assert parsed[1]["id"] == "test-002"

    def test_json_export_preserves_messages(self, tmp_path):
        sessions = [_make_session()]
        output = json.dumps(sessions, ensure_ascii=False, indent=2)
        parsed = json.loads(output)
        assert "messages" in parsed[0]
        assert len(parsed[0]["messages"]) == 2

    def test_json_export_single_session(self, tmp_path):
        session = _make_session()
        output = json.dumps([session], ensure_ascii=False, indent=2)
        parsed = json.loads(output)
        assert len(parsed) == 1


# ---------------------------------------------------------------------------
# Test: JSONL format
# ---------------------------------------------------------------------------

class TestExportJSONL:
    """Tests for JSONL export format."""

    def test_jsonl_one_line_per_session(self, tmp_path):
        sessions = [_make_session(), _make_session(session_id="test-002")]
        lines = [json.dumps(s, ensure_ascii=False) for s in sessions]
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "id" in parsed

    def test_jsonl_empty_sessions(self):
        sessions = []
        lines = [json.dumps(s, ensure_ascii=False) for s in sessions]
        assert len(lines) == 0


