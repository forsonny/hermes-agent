"""Tests for agent/trajectory.py -- trajectory saving utilities."""

import json
import os
from pathlib import Path

import pytest

from agent.trajectory import (
    convert_scratchpad_to_think,
    has_incomplete_scratchpad,
    save_trajectory,
)


class TestConvertScratchpadToThink:
    """Tests for convert_scratchpad_to_think."""

    def test_converts_scratchpad_tags(self):
        """Should replace REASONING_SCRATCHPAD tags with think tags."""
        src = "<REASONING_SCRATCHPAD>my reasoning here</REASONING_SCRATCHPAD>"
        result = convert_scratchpad_to_think(src)
        # The function replaces with <think...> and </think...> tags
        assert "REASONING_SCRATCHPAD" not in result
        assert "my reasoning here" in result
        assert result.startswith("<t")
        # Verify close tag is present

    def test_handles_empty_string(self):
        """Empty string should return empty string."""
        assert convert_scratchpad_to_think("") == ""

    def test_handles_none(self):
        """None should return None."""
        assert convert_scratchpad_to_think(None) is None

    def test_no_scratchpad_tags_unchanged(self):
        """Content without scratchpad tags should be returned unchanged."""
        src = "Just regular content here"
        assert convert_scratchpad_to_think(src) == src

    def test_multiple_scratchpad_blocks(self):
        """Multiple scratchpad blocks should all be converted."""
        src = (
            "<REASONING_SCRATCHPAD>first</REASONING_SCRATCHPAD>"
            " some text "
            "<REASONING_SCRATCHPAD>second</REASONING_SCRATCHPAD>"
        )
        result = convert_scratchpad_to_think(src)
        assert "<REASONING_SCRATCHPAD>" not in result
        assert "</REASONING_SCRATCHPAD>" not in result
        assert "first" in result
        assert "second" in result

    def test_nested_content_preserved(self):
        """Content inside scratchpad tags should be preserved."""
        src = "<REASONING_SCRATCHPAD>Line 1\nLine 2\n<other>tags</other></REASONING_SCRATCHPAD>"
        result = convert_scratchpad_to_think(src)
        assert "Line 1" in result
        assert "<other>tags</other>" in result


class TestHasIncompleteScratchpad:
    """Tests for has_incomplete_scratchpad."""

    def test_open_without_close(self):
        """Opening tag without closing tag should return True."""
        assert has_incomplete_scratchpad("<REASONING_SCRATCHPAD>thinking...") is True

    def test_complete_scratchpad(self):
        """Both tags present should return False."""
        assert has_incomplete_scratchpad(
            "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>"
        ) is False

    def test_no_tags(self):
        """No tags should return False."""
        assert has_incomplete_scratchpad("just text") is False

    def test_empty_string(self):
        """Empty string should return False."""
        assert has_incomplete_scratchpad("") is False

    def test_none_content(self):
        """None should return False."""
        assert has_incomplete_scratchpad(None) is False

    def test_close_without_open(self):
        """Closing tag without opening should return False."""
        assert has_incomplete_scratchpad("</REASONING_SCRATCHPAD>") is False

    def test_multiple_incomplete(self):
        """Multiple opens with one close should still be incomplete."""
        content = "<REASONING_SCRATCHPAD>first<REASONING_SCRATCHPAD>second</REASONING_SCRATCHPAD>"
        assert has_incomplete_scratchpad(content) is False  # close tag present, so not incomplete


class TestSaveTrajectory:
    """Tests for save_trajectory."""

    def test_saves_to_success_file(self, tmp_path):
        """Completed trajectory should be saved to trajectory_samples.jsonl."""
        outfile = str(tmp_path / "trajectory_samples.jsonl")
        trajectory = [{"role": "user", "content": "hello"}]
        save_trajectory(trajectory, model="test-model", completed=True, filename=outfile)

        assert Path(outfile).exists()
        with open(outfile) as f:
            entry = json.loads(f.readline())
        assert entry["conversations"] == trajectory
        assert entry["model"] == "test-model"
        assert entry["completed"] is True
        assert "timestamp" in entry

    def test_saves_to_failure_file(self, tmp_path):
        """Failed trajectory should be saved to failed_trajectories.jsonl."""
        outfile = str(tmp_path / "failed_trajectories.jsonl")
        trajectory = ["data"]
        save_trajectory(trajectory, model="test", completed=False, filename=outfile)

        assert Path(outfile).exists()
        with open(outfile) as f:
            entry = json.loads(f.readline())
        assert entry["completed"] is False

    def test_appends_to_existing_file(self, tmp_path):
        """Multiple saves should append, not overwrite."""
        outfile = str(tmp_path / "trajectory_samples.jsonl")
        save_trajectory(["a"], model="m1", completed=True, filename=outfile)
        save_trajectory(["b"], model="m2", completed=True, filename=outfile)

        with open(outfile) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["conversations"] == ["a"]
        assert json.loads(lines[1])["conversations"] == ["b"]

    def test_default_filename_success(self, tmp_path):
        """Without filename, completed=True uses trajectory_samples.jsonl."""
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_trajectory(["test"], model="m", completed=True)
            assert (tmp_path / "trajectory_samples.jsonl").exists()
        finally:
            os.chdir(old_cwd)

    def test_default_filename_failure(self, tmp_path):
        """Without filename, completed=False uses failed_trajectories.jsonl."""
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_trajectory(["test"], model="m", completed=False)
            assert (tmp_path / "failed_trajectories.jsonl").exists()
        finally:
            os.chdir(old_cwd)

    def test_unicode_content(self, tmp_path):
        """Unicode content should be preserved (ensure_ascii=False)."""
        outfile = str(tmp_path / "trajectory_samples.jsonl")
        trajectory = [{"role": "user", "content": "Hello!"}]
        save_trajectory(trajectory, model="m", completed=True, filename=outfile)

        with open(outfile, encoding="utf-8") as f:
            raw = f.read()
        assert "Hello!" in raw

    def test_empty_trajectory(self, tmp_path):
        """Empty trajectory list should still save."""
        outfile = str(tmp_path / "trajectory_samples.jsonl")
        save_trajectory([], model="m", completed=True, filename=outfile)

        with open(outfile) as f:
            entry = json.loads(f.readline())
        assert entry["conversations"] == []

    def test_write_failure_handled_gracefully(self):
        """Write to invalid path should log warning, not raise."""
        save_trajectory(
            ["data"], model="m", completed=True,
            filename="/dev/null/tricked/path/trajectory.jsonl"
        )
