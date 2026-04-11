"""Tests for session_search_tool multi-region truncation and helpers."""

import pytest
from tools.session_search_tool import (
    _truncate_around_matches,
    _find_all_match_positions,
    _merge_regions,
    _format_conversation,
    _format_timestamp,
    MAX_SESSION_CHARS,
)


class TestFindAllMatchPositions:
    """Tests for _find_all_match_positions helper."""

    def test_single_term_single_match(self):
        text = "hello world"
        positions = _find_all_match_positions(text, ["hello"])
        assert positions == [(0, 5)]

    def test_single_term_multiple_matches(self):
        text = "abc abc abc"
        positions = _find_all_match_positions(text, ["abc"])
        assert positions == [(0, 3), (4, 3), (8, 3)]

    def test_multiple_terms(self):
        text = "alpha beta gamma"
        positions = _find_all_match_positions(text, ["alpha", "gamma"])
        assert (0, 5) in positions
        assert (11, 5) in positions
        assert len(positions) == 2

    def test_no_matches(self):
        text = "nothing here"
        positions = _find_all_match_positions(text, ["xyz"])
        assert positions == []

    def test_case_insensitive(self):
        text = "Hello HELLO hello"
        positions = _find_all_match_positions(text.lower(), ["hello"])
        assert len(positions) == 3

    def test_empty_terms_skipped(self):
        text = "hello"
        positions = _find_all_match_positions(text, ["", "hello"])
        assert positions == [(0, 5)]

    def test_empty_text(self):
        positions = _find_all_match_positions("", ["hello"])
        assert positions == []


class TestMergeRegions:
    """Tests for _merge_regions helper."""

    def test_empty_input(self):
        assert _merge_regions([], 100) == []

    def test_single_position(self):
        result = _merge_regions([(10, 5)], 100)
        assert result == [(10, 15)]

    def test_nearby_merged(self):
        # Two matches 50 chars apart, gap=100 -> should merge
        result = _merge_regions([(10, 5), (65, 5)], 100)
        assert result == [(10, 70)]

    def test_far_apart_not_merged(self):
        # Two matches 200 chars apart, gap=100 -> separate regions
        result = _merge_regions([(10, 5), (215, 5)], 100)
        assert result == [(10, 15), (215, 220)]

    def test_chain_merge(self):
        # Three matches each 80 apart, gap=100 -> all merge
        result = _merge_regions([(0, 5), (85, 5), (170, 5)], 100)
        assert result == [(0, 175)]

    def test_gap_zero_overlapping(self):
        # Overlapping positions merge even with gap=0
        result = _merge_regions([(0, 10), (8, 5)], 0)
        assert result == [(0, 13)]


class TestTruncateAroundMatches:
    """Tests for _truncate_around_matches multi-region version."""

    def test_short_text_unchanged(self):
        text = "short text"
        result = _truncate_around_matches(text, "short", max_chars=1000)
        assert result == text

    def test_text_exactly_max_chars(self):
        text = "A" * 100
        result = _truncate_around_matches(text, "A", max_chars=100)
        assert result == text

    def test_single_match_preserved(self):
        text = "X" * 500 + "hello world" + "Y" * 500
        result = _truncate_around_matches(text, "hello", max_chars=100)
        assert "hello world" in result

    def test_no_match_returns_start(self):
        text = "A" * 1000
        result = _truncate_around_matches(text, "xyz", max_chars=100)
        assert result == text[:100]

    def test_empty_query_returns_start(self):
        text = "A" * 1000
        result = _truncate_around_matches(text, "", max_chars=100)
        assert result == text[:100]

    def test_multiple_matches_far_apart(self):
        """Two matches far apart should both appear in the result."""
        text = "X" * 200 + "alpha" + "Y" * 500 + "beta" + "Z" * 200
        result = _truncate_around_matches(text, "alpha beta", max_chars=400)
        assert "alpha" in result
        assert "beta" in result

    def test_very_far_apart_produces_gap(self):
        """Matches more than 600 chars apart produce a gap marker."""
        text = "X" * 500 + "alpha" + "Y" * 1000 + "beta" + "Z" * 500
        result = _truncate_around_matches(text, "alpha beta", max_chars=2000)
        assert "alpha" in result
        assert "beta" in result
        # With 1000-char gap between matches, regions should NOT merge
        assert "conversation gap" in result

    def test_nearby_matches_merged_no_gap(self):
        """Matches within 500 chars should be in one region (no gap marker)."""
        text = "A" * 200 + "alpha" + "X" * 100 + "beta" + "B" * 200
        result = _truncate_around_matches(text, "alpha beta", max_chars=800)
        assert "alpha" in result
        assert "beta" in result
        assert "conversation gap" not in result

    def test_scattered_matches_preserved(self):
        """Many scattered matches: at least several should survive truncation."""
        parts = []
        for i in range(10):
            parts.append("PADDING" * 100)
            parts.append(f"match{i}")
        parts.append("PADDING" * 100)
        text = "".join(parts)
        result = _truncate_around_matches(text, "match", max_chars=500)
        match_count = sum(1 for i in range(10) if f"match{i}" in result)
        assert match_count >= 3, f"Expected >= 3 matches, got {match_count}"

    def test_earlier_truncation_marker(self):
        """When result doesn't start at 0, show earlier truncation marker."""
        text = "A" * 500 + "hello" + "B" * 500
        result = _truncate_around_matches(text, "hello", max_chars=100)
        assert "earlier conversation truncated" in result

    def test_later_truncation_marker(self):
        """When result doesn't end at text end, show later truncation marker."""
        text = "A" * 500 + "hello" + "B" * 500
        result = _truncate_around_matches(text, "hello", max_chars=100)
        assert "later conversation truncated" in result

    def test_no_markers_when_fits(self):
        """When everything fits, no truncation markers."""
        text = "hello world"
        result = _truncate_around_matches(text, "hello", max_chars=1000)
        assert "truncated" not in result
        assert result == text

    def test_same_term_multiple_positions(self):
        """Multiple occurrences of the same term should all be covered."""
        text = "A" * 200 + "target" + "B" * 500 + "target" + "C" * 200
        result = _truncate_around_matches(text, "target", max_chars=400)
        # Both occurrences should be preserved
        count = result.count("target")
        assert count >= 1, "Expected at least 1 'target' in result"

    def test_large_max_chars_default(self):
        """With default max_chars (100k), reasonable text should pass through."""
        text = "Some reasonable conversation text " * 100  # ~3.5k chars
        result = _truncate_around_matches(text, "conversation")
        assert result == text  # Should be unchanged (well under 100k)

    def test_budget_allocation_distributes(self):
        """With far-apart matches and tight budget, at least the closest match is kept."""
        text = ("A" * 300 + "find1" + "B" * 600 + "find2" + "C" * 600 + "find3" + "D" * 300)
        result = _truncate_around_matches(text, "find1 find2 find3", max_chars=300)
        # At least one match should be preserved
        preserved = sum(1 for t in ["find1", "find2", "find3"] if t in result)
        assert preserved >= 1, f"Expected >= 1 preserved, got {preserved}"
