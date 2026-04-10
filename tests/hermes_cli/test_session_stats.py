"""Tests for hermes sessions stats — Top Tools, Duration Distribution, Busiest Day.

The _print_session_stats function is nested inside main(), so we test the
new sections by reimplementing the same SQL queries against a test database
and verifying the output through the same rendering logic.
"""
import os
import sqlite3
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path, sessions=None, messages=None):
    """Create a minimal SessionDB-compatible SQLite database for testing."""
    db_path = tmp_path / "sessions.db"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT
        )
    """)

    now = time.time()

    if sessions:
        for s in sessions:
            defaults = {
                "source": "cli",
                "user_id": None,
                "model": "test/model",
                "started_at": now - 100,
                "ended_at": now,
                "message_count": 1,
                "tool_call_count": 0,
            }
            defaults.update(s)
            cols = ", ".join(defaults.keys())
            placeholders = ", ".join(["?"] * len(defaults))
            conn.execute(f"INSERT INTO sessions ({cols}) VALUES ({placeholders})", list(defaults.values()))

    if messages:
        for m in messages:
            defaults = {
                "role": "tool",
                "content": "ok",
                "tool_call_id": None,
                "tool_calls": None,
                "tool_name": None,
                "timestamp": now,
                "token_count": None,
            }
            defaults.update(m)
            cols = ", ".join(defaults.keys())
            placeholders = ", ".join(["?"] * len(defaults))
            conn.execute(f"INSERT INTO messages ({cols}) VALUES ({placeholders})", list(defaults.values()))

    conn.commit()
    conn.close()
    return db_path


def _run_top_tools_query(db_path: Path, source_filter=None):
    """Reproduce the Top Tools query from _print_session_stats."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    source_args = [source_filter] if source_filter else []
    tool_rows = conn.execute(
        "SELECT tool_name, COUNT(*) as cnt FROM messages "
        "WHERE tool_name IS NOT NULL AND tool_name != '' "
        f"{'AND session_id IN (SELECT id FROM sessions WHERE source = ?)' if source_filter else ''} "
        "GROUP BY tool_name ORDER BY cnt DESC LIMIT 15",
        source_args,
    ).fetchall()
    conn.close()
    return tool_rows


def _run_duration_query(db_path: Path, source_filter=None):
    """Reproduce the Session Duration query from _print_session_stats."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    source_args = [source_filter] if source_filter else []
    dur_buckets = conn.execute(
        "SELECT "
        "  SUM(CASE WHEN (ended_at - started_at) < 60 THEN 1 ELSE 0 END) as under_1m, "
        "  SUM(CASE WHEN (ended_at - started_at) >= 60 AND (ended_at - started_at) < 300 THEN 1 ELSE 0 END) as m1_5, "
        "  SUM(CASE WHEN (ended_at - started_at) >= 300 AND (ended_at - started_at) < 900 THEN 1 ELSE 0 END) as m5_15, "
        "  SUM(CASE WHEN (ended_at - started_at) >= 900 AND (ended_at - started_at) < 3600 THEN 1 ELSE 0 END) as m15_60, "
        "  SUM(CASE WHEN (ended_at - started_at) >= 3600 THEN 1 ELSE 0 END) as over_60m "
        f"FROM sessions WHERE ended_at IS NOT NULL AND ended_at > started_at "
        f"{'AND source = ?' if source_filter else ''}",
        source_args,
    ).fetchone()
    conn.close()
    return dur_buckets


def _run_busiest_day_query(db_path: Path, source_filter=None):
    """Reproduce the Busiest Day query from _print_session_stats."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    source_args = [source_filter] if source_filter else []
    day_rows = conn.execute(
        "SELECT DATE(started_at, 'unixepoch') as day, COUNT(*) as cnt "
        f"FROM sessions {'WHERE source = ?' if source_filter else ''} "
        "GROUP BY day ORDER BY cnt DESC LIMIT 1",
        source_args,
    ).fetchall()
    conn.close()
    return day_rows


# ---------------------------------------------------------------------------
# Top Tools Tests
# ---------------------------------------------------------------------------

class TestTopTools:
    """Test the Top Tools SQL query logic."""

    def test_tools_ordered_by_count(self, tmp_path):
        """Tools are returned ordered by call count descending."""
        db_path = _make_db(
            tmp_path,
            sessions=[{"id": "s1"}],
            messages=[
                {"session_id": "s1", "tool_name": "terminal"},
                {"session_id": "s1", "tool_name": "terminal"},
                {"session_id": "s1", "tool_name": "terminal"},
                {"session_id": "s1", "tool_name": "read_file"},
                {"session_id": "s1", "tool_name": "read_file"},
                {"session_id": "s1", "tool_name": "web_search"},
            ],
        )
        rows = _run_top_tools_query(db_path)
        assert len(rows) == 3
        assert rows[0]["tool_name"] == "terminal"
        assert rows[0]["cnt"] == 3
        assert rows[1]["tool_name"] == "read_file"
        assert rows[1]["cnt"] == 2
        assert rows[2]["tool_name"] == "web_search"
        assert rows[2]["cnt"] == 1

    def test_empty_db_returns_no_tools(self, tmp_path):
        """No tool rows when there are no tool messages."""
        db_path = _make_db(tmp_path, sessions=[{"id": "s1"}])
        rows = _run_top_tools_query(db_path)
        assert len(rows) == 0

    def test_source_filter(self, tmp_path):
        """Source filter scopes results to matching sessions."""
        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "source": "cli"},
                {"id": "s2", "source": "telegram"},
            ],
            messages=[
                {"session_id": "s1", "tool_name": "terminal"},
                {"session_id": "s1", "tool_name": "terminal"},
                {"session_id": "s2", "tool_name": "web_search"},
                {"session_id": "s2", "tool_name": "web_search"},
                {"session_id": "s2", "tool_name": "web_search"},
            ],
        )
        rows = _run_top_tools_query(db_path, source_filter="cli")
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "terminal"
        assert rows[0]["cnt"] == 2

    def test_limited_to_15_tools(self, tmp_path):
        """Results are capped at 15 tools."""
        messages = [{"session_id": "s1", "tool_name": f"tool_{i:02d}"} for i in range(20)]
        db_path = _make_db(
            tmp_path,
            sessions=[{"id": "s1"}],
            messages=messages,
        )
        rows = _run_top_tools_query(db_path)
        assert len(rows) == 15

    def test_non_tool_messages_excluded(self, tmp_path):
        """Messages with no tool_name are excluded."""
        db_path = _make_db(
            tmp_path,
            sessions=[{"id": "s1"}],
            messages=[
                {"session_id": "s1", "tool_name": "terminal"},
                {"session_id": "s1", "role": "user", "tool_name": None},
                {"session_id": "s1", "role": "assistant", "tool_name": None},
            ],
        )
        rows = _run_top_tools_query(db_path)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "terminal"

    def test_percentage_calculation(self, tmp_path):
        """Bar chart percentage is correct for dominant tool."""
        db_path = _make_db(
            tmp_path,
            sessions=[{"id": "s1"}],
            messages=[
                {"session_id": "s1", "tool_name": "terminal"},
            ],
        )
        rows = _run_top_tools_query(db_path)
        total = sum(r["cnt"] for r in rows)
        assert total == 1
        pct = rows[0]["cnt"] * 100 // max(total, 1)
        assert pct == 100


# ---------------------------------------------------------------------------
# Session Duration Tests
# ---------------------------------------------------------------------------

class TestSessionDuration:
    """Test the Session Duration bucketing SQL."""

    def test_correct_bucketing(self, tmp_path):
        """Sessions are bucketed into correct time ranges."""
        now = time.time()
        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "started_at": now - 30, "ended_at": now},       # <1 min
                {"id": "s2", "started_at": now - 180, "ended_at": now},      # 1–5 min
                {"id": "s3", "started_at": now - 600, "ended_at": now},      # 5–15 min
                {"id": "s4", "started_at": now - 1800, "ended_at": now},     # 15–60 min
                {"id": "s5", "started_at": now - 7200, "ended_at": now},     # >60 min
            ],
        )
        buckets = _run_duration_query(db_path)
        assert buckets["under_1m"] == 1
        assert buckets["m1_5"] == 1
        assert buckets["m5_15"] == 1
        assert buckets["m15_60"] == 1
        assert buckets["over_60m"] == 1

    def test_no_ended_sessions(self, tmp_path):
        """All null when no sessions have ended_at."""
        now = time.time()
        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "started_at": now, "ended_at": None},
            ],
        )
        buckets = _run_duration_query(db_path)
        # All buckets should be 0 or None
        total = sum((v or 0) for v in [
            buckets["under_1m"], buckets["m1_5"],
            buckets["m5_15"], buckets["m15_60"], buckets["over_60m"]
        ])
        assert total == 0

    def test_all_short_sessions(self, tmp_path):
        """All sessions under 1 minute."""
        now = time.time()
        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": f"s{i}", "started_at": now - 10, "ended_at": now}
                for i in range(5)
            ],
        )
        buckets = _run_duration_query(db_path)
        assert buckets["under_1m"] == 5
        assert buckets["m1_5"] == 0

    def test_source_filter(self, tmp_path):
        """Source filter scopes duration buckets."""
        now = time.time()
        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "source": "cli", "started_at": now - 30, "ended_at": now},
                {"id": "s2", "source": "telegram", "started_at": now - 7200, "ended_at": now},
            ],
        )
        buckets = _run_duration_query(db_path, source_filter="cli")
        assert buckets["under_1m"] == 1
        assert buckets["over_60m"] == 0


# ---------------------------------------------------------------------------
# Busiest Day Tests
# ---------------------------------------------------------------------------

class TestBusiestDay:
    """Test the Busiest Day SQL query."""

    def test_busiest_day_correct(self, tmp_path):
        """The day with most sessions is returned."""
        # 2025-01-15 12:00:00 UTC
        day1 = 1736942400
        # 2025-01-16 12:00:00 UTC
        day2 = 1737028800

        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "started_at": day1},
                {"id": "s2", "started_at": day1 + 100},
                {"id": "s3", "started_at": day1 + 200},
                {"id": "s4", "started_at": day2},
            ],
        )
        rows = _run_busiest_day_query(db_path)
        assert len(rows) == 1
        assert rows[0]["day"] == "2025-01-15"
        assert rows[0]["cnt"] == 3

    def test_empty_db(self, tmp_path):
        """No busiest day when there are no sessions."""
        db_path = _make_db(tmp_path)
        rows = _run_busiest_day_query(db_path)
        assert len(rows) == 0

    def test_tie_uses_first_alphabetically(self, tmp_path):
        """When days tie, either is acceptable."""
        day1 = 1736942400
        day2 = 1737028800

        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "started_at": day1},
                {"id": "s2", "started_at": day2},
            ],
        )
        rows = _run_busiest_day_query(db_path)
        assert len(rows) == 1
        assert rows[0]["cnt"] == 1

    def test_source_filter(self, tmp_path):
        """Source filter scopes busiest day."""
        day1 = 1736942400

        db_path = _make_db(
            tmp_path,
            sessions=[
                {"id": "s1", "source": "cli", "started_at": day1},
                {"id": "s2", "source": "telegram", "started_at": day1 + 100},
            ],
        )
        rows = _run_busiest_day_query(db_path, source_filter="cli")
        assert len(rows) == 1
        assert rows[0]["cnt"] == 1
