"""Tests for agent/display.py -- build_tool_preview(), get_cute_tool_message(), and inline diff previews."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from agent.display import (
    _detect_tool_failure,
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    render_edit_diff_with_delta,
)


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None

    def test_known_tool_with_primary_arg(self):
        """Known tool with its primary arg should return a preview string."""
        result = build_tool_preview("terminal", {"command": "ls -la"})
        assert result is not None
        assert "ls -la" in result

    def test_web_search_preview(self):
        result = build_tool_preview("web_search", {"query": "hello world"})
        assert result is not None
        assert "hello world" in result

    def test_read_file_preview(self):
        result = build_tool_preview("read_file", {"path": "/tmp/test.py", "offset": 1})
        assert result is not None
        assert "/tmp/test.py" in result

    def test_unknown_tool_with_fallback_key(self):
        """Unknown tool but with a recognized fallback key should still preview."""
        result = build_tool_preview("custom_tool", {"query": "test query"})
        assert result is not None
        assert "test query" in result

    def test_unknown_tool_no_matching_key(self):
        """Unknown tool with no recognized keys should return None."""
        result = build_tool_preview("custom_tool", {"foo": "bar"})
        assert result is None

    def test_long_value_truncated(self):
        """Preview should truncate long values."""
        long_cmd = "a" * 100
        result = build_tool_preview("terminal", {"command": long_cmd}, max_len=40)
        assert result is not None
        assert len(result) <= 43  # max_len + "..."

    def test_process_tool_with_none_args(self):
        """Process tool special case should also handle None args."""
        assert build_tool_preview("process", None) is None

    def test_process_tool_normal(self):
        result = build_tool_preview("process", {"action": "poll", "session_id": "abc123"})
        assert result is not None
        assert "poll" in result

    def test_todo_tool_read(self):
        result = build_tool_preview("todo", {"merge": False})
        assert result is not None
        assert "reading" in result

    def test_todo_tool_with_todos(self):
        result = build_tool_preview("todo", {"todos": [{"id": "1", "content": "test", "status": "pending"}]})
        assert result is not None
        assert "1 task" in result

    def test_memory_tool_add(self):
        result = build_tool_preview("memory", {"action": "add", "target": "user", "content": "test note"})
        assert result is not None
        assert "user" in result

    def test_session_search_preview(self):
        result = build_tool_preview("session_search", {"query": "find something"})
        assert result is not None
        assert "find something" in result

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestEditDiffPreview:
    def test_extract_edit_diff_for_patch(self):
        diff = extract_edit_diff("patch", '{"success": true, "diff": "--- a/x\\n+++ b/x\\n"}')
        assert diff is not None
        assert "+++ b/x" in diff

    def test_render_inline_unified_diff_colors_added_and_removed_lines(self):
        rendered = _render_inline_unified_diff(
            "--- a/cli.py\n"
            "+++ b/cli.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old line\n"
            "+new line\n"
            " context\n"
        )

        assert "a/cli.py" in rendered[0]
        assert "b/cli.py" in rendered[0]
        assert any("old line" in line for line in rendered)
        assert any("new line" in line for line in rendered)
        assert any("48;2;" in line for line in rendered)

    def test_extract_edit_diff_ignores_non_edit_tools(self):
        assert extract_edit_diff("web_search", '{"diff": "--- a\\n+++ b\\n"}') is None

    def test_extract_edit_diff_uses_local_snapshot_for_write_file(self, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("old\n", encoding="utf-8")

        snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

        target.write_text("new\n", encoding="utf-8")

        diff = extract_edit_diff(
            "write_file",
            '{"bytes_written": 4}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert diff is not None
        assert "--- a/" in diff
        assert "+++ b/" in diff
        assert "-old" in diff
        assert "+new" in diff

    def test_render_edit_diff_with_delta_invokes_printer(self):
        printer = MagicMock()

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n@@ -1 +1 @@\\n-old\\n+new\\n"}',
            print_fn=printer,
        )

        assert rendered is True
        assert printer.call_count >= 2
        calls = [call.args[0] for call in printer.call_args_list]
        assert any("a/x" in line and "b/x" in line for line in calls)
        assert any("old" in line for line in calls)
        assert any("new" in line for line in calls)

    def test_render_edit_diff_with_delta_skips_without_diff(self):
        rendered = render_edit_diff_with_delta(
            "patch",
            '{"success": true}',
        )

        assert rendered is False

    def test_render_edit_diff_with_delta_handles_renderer_errors(self, monkeypatch):
        printer = MagicMock()

        monkeypatch.setattr("agent.display._summarize_rendered_diff_sections", MagicMock(side_effect=RuntimeError("boom")))

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n"}',
            print_fn=printer,
        )

        assert rendered is False
        assert printer.call_count == 0

    def test_summarize_rendered_diff_sections_truncates_large_diff(self):
        diff = "--- a/x.py\n+++ b/x.py\n" + "".join(f"+line{i}\n" for i in range(120))

        rendered = _summarize_rendered_diff_sections(diff, max_lines=20)

        assert len(rendered) == 21
        assert "omitted" in rendered[-1]

    def test_summarize_rendered_diff_sections_limits_file_count(self):
        diff = "".join(
            f"--- a/file{i}.py\n+++ b/file{i}.py\n+line{i}\n"
            for i in range(8)
        )

        rendered = _summarize_rendered_diff_sections(diff, max_files=3, max_lines=50)

        assert any("a/file0.py" in line for line in rendered)
        assert any("a/file1.py" in line for line in rendered)
        assert any("a/file2.py" in line for line in rendered)
        assert not any("a/file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]


class TestGetCuteToolMessageNewEntries:
    """Tests for newly added get_cute_tool_message display entries."""

    def test_browser_console(self):
        msg = get_cute_tool_message("browser_console", {"expression": "document.title"}, 1.0)
        assert "document.title" in msg
        assert "console" in msg
        assert "1.0s" in msg

    def test_clarify(self):
        msg = get_cute_tool_message("clarify", {"question": "Which file?"}, 0.5)
        assert "Which file?" in msg
        assert "clarify" in msg
        assert "0.5s" in msg

    def test_skill_manage_create(self):
        msg = get_cute_tool_message("skill_manage", {"action": "create", "name": "my-skill"}, 1.2)
        assert "create" in msg
        assert "my-skill" in msg
        assert "skill" in msg

    def test_skill_manage_delete(self):
        msg = get_cute_tool_message("skill_manage", {"action": "delete", "name": "old-skill"}, 0.3)
        assert "delete" in msg
        assert "old-skill" in msg

    def test_ha_list_entities_with_domain(self):
        msg = get_cute_tool_message("ha_list_entities", {"domain": "light"}, 2.0)
        assert "light" in msg
        assert "entities" in msg
        assert "2.0s" in msg

    def test_ha_list_entities_default_domain(self):
        msg = get_cute_tool_message("ha_list_entities", {}, 1.0)
        assert "all" in msg
        assert "entities" in msg

    def test_ha_get_state(self):
        msg = get_cute_tool_message("ha_get_state", {"entity_id": "light.living_room"}, 1.5)
        assert "light.living_room" in msg
        assert "state" in msg

    def test_ha_list_services(self):
        msg = get_cute_tool_message("ha_list_services", {"domain": "switch"}, 0.8)
        assert "switch" in msg
        assert "services" in msg

    def test_ha_call_service(self):
        msg = get_cute_tool_message("ha_call_service", {"service": "light.toggle"}, 1.0)
        assert "light.toggle" in msg
        assert "call" in msg

    def test_web_crawl_uses_fallback(self):
        """web_crawl display entry was removed; should fall through to generic ⚡ handler."""
        msg = get_cute_tool_message("web_crawl", {"url": "https://example.com"}, 1.0)
        assert "⚡" in msg
        assert "web_crawl" in msg

    def test_existing_tool_not_broken(self):
        """Sanity check: existing tools like terminal still render correctly."""
        msg = get_cute_tool_message("terminal", {"command": "ls -la"}, 1.0)
        assert "ls -la" in msg
        assert "$" in msg


class TestDetectToolFailure:
    """Tests for _detect_tool_failure -- failure detection in tool results."""

    def test_none_result_is_success(self):
        is_fail, tag = _detect_tool_failure("any_tool", None)
        assert is_fail is False
        assert tag == ""

    def test_empty_string_is_success(self):
        is_fail, tag = _detect_tool_failure("any_tool", "")
        assert is_fail is False
        assert tag == ""

    def test_terminal_nonzero_exit(self):
        result = json.dumps({"exit_code": 1, "output": "error"})
        is_fail, tag = _detect_tool_failure("terminal", result)
        assert is_fail is True
        assert "exit 1" in tag

    def test_terminal_zero_exit_is_success(self):
        result = json.dumps({"exit_code": 0, "output": "ok"})
        is_fail, tag = _detect_tool_failure("terminal", result)
        assert is_fail is False

    def test_error_none_is_not_false_positive(self):
        """Tools return error: None on success -- must NOT be flagged."""
        result = json.dumps({"url": "https://example.com", "error": None})
        is_fail, tag = _detect_tool_failure("vision_analyze", result)
        assert is_fail is False
        assert tag == ""

    def test_error_empty_string_is_success(self):
        result = json.dumps({"data": "ok", "error": ""})
        is_fail, tag = _detect_tool_failure("some_tool", result)
        assert is_fail is False

    def test_error_false_is_success(self):
        result = json.dumps({"success": True, "error": False})
        is_fail, tag = _detect_tool_failure("some_tool", result)
        assert is_fail is False

    def test_error_with_message_is_failure(self):
        result = json.dumps({"error": "Something went wrong"})
        is_fail, tag = _detect_tool_failure("some_tool", result)
        assert is_fail is True
        assert "error" in tag

    def test_status_error_is_failure(self):
        result = json.dumps({"status": "error", "message": "bad"})
        is_fail, tag = _detect_tool_failure("some_tool", result)
        assert is_fail is True

    def test_status_failed_is_failure(self):
        result = json.dumps({"status": "failed", "message": "bad"})
        is_fail, tag = _detect_tool_failure("some_tool", result)
        assert is_fail is True

    def test_starts_with_error_is_failure(self):
        is_fail, tag = _detect_tool_failure("some_tool", "Error: connection refused")
        assert is_fail is True

    def test_no_error_field_is_success(self):
        result = json.dumps({"output": "hello", "status": "ok"})
        is_fail, tag = _detect_tool_failure("some_tool", result)
        assert is_fail is False

    def test_complex_success_with_null_error(self):
        result = json.dumps({
            "url": "https://example.com",
            "content": "page content here",
            "status_code": 200,
            "error": None,
        })
        is_fail, tag = _detect_tool_failure("web_extract", result)
        assert is_fail is False

    def test_error_none_with_nested_data(self):
        result = json.dumps({
            "images": [{"url": "https://img.png", "size": 1024}],
            "error": None,
            "prompt": "a sunset",
        })
        is_fail, tag = _detect_tool_failure("image_generate", result)
        assert is_fail is False

    def test_delegate_error(self):
        result = json.dumps({"error": "depth limit exceeded"})
        is_fail, tag = _detect_tool_failure("delegate_task", result)
        assert is_fail is True

    def test_delegate_partial_failure(self):
        result = json.dumps({
            "results": [
                {"summary": "ok"},
                {"error": "timeout"},
            ]
        })
        is_fail, tag = _detect_tool_failure("delegate_task", result)
        assert is_fail is True
        assert "1/2 failed" in tag

    def test_delegate_success(self):
        result = json.dumps({"results": [{"summary": "done"}]})
        is_fail, tag = _detect_tool_failure("delegate_task", result)
        assert is_fail is False

    def test_process_killed(self):
        result = json.dumps({"status": "killed"})
        is_fail, tag = _detect_tool_failure("process", result)
        assert is_fail is True
        assert "killed" in tag

    def test_process_success(self):
        result = json.dumps({"status": "done", "output": "ok"})
        is_fail, tag = _detect_tool_failure("process", result)
        assert is_fail is False

    def test_execute_code_nonzero_exit(self):
        result = json.dumps({"exit_code": 1, "error": "NameError"})
        is_fail, tag = _detect_tool_failure("execute_code", result)
        assert is_fail is True

    def test_execute_code_timeout(self):
        result = json.dumps({"exit_code": 1, "error": "Command timed out"})
        is_fail, tag = _detect_tool_failure("execute_code", result)
        assert is_fail is True
        assert "timeout" in tag

    def test_execute_code_success(self):
        result = json.dumps({"exit_code": 0, "output": "result"})
        is_fail, tag = _detect_tool_failure("execute_code", result)
        assert is_fail is False
