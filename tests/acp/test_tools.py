"""Tests for acp_adapter.tools — tool kind mapping and ACP content building."""

import pytest

from acp_adapter.tools import (
    TOOL_KIND_MAP,
    build_tool_complete,
    build_tool_start,
    build_tool_title,
    extract_locations,
    get_tool_kind,
    make_tool_call_id,
)
from acp.schema import (
    FileEditToolCallContent,
    ContentToolCallContent,
    ToolCallLocation,
    ToolCallStart,
    ToolCallProgress,
)


# ---------------------------------------------------------------------------
# TOOL_KIND_MAP coverage
# ---------------------------------------------------------------------------


COMMON_HERMES_TOOLS = [
    "read_file", "search_files", "terminal", "patch", "write_file", "process",
    "memory", "todo", "clarify", "session_search", "skill_view", "skill_manage",
    "skills_list", "send_message", "cronjob", "browser_console",
    "ha_call_service", "ha_get_state", "ha_list_entities", "ha_list_services",
    "rl_check_status", "rl_start_training", "mixture_of_agents",
]


class TestToolKindMap:
    def test_all_hermes_tools_have_kind(self):
        """Every common hermes tool should appear in TOOL_KIND_MAP."""
        for tool in COMMON_HERMES_TOOLS:
            assert tool in TOOL_KIND_MAP, f"{tool} missing from TOOL_KIND_MAP"

    def test_tool_kind_read_file(self):
        assert get_tool_kind("read_file") == "read"

    def test_tool_kind_terminal(self):
        assert get_tool_kind("terminal") == "execute"

    def test_tool_kind_patch(self):
        assert get_tool_kind("patch") == "edit"

    def test_tool_kind_write_file(self):
        assert get_tool_kind("write_file") == "edit"

    def test_tool_kind_web_search(self):
        assert get_tool_kind("web_search") == "fetch"

    def test_tool_kind_execute_code(self):
        assert get_tool_kind("execute_code") == "execute"

    def test_tool_kind_browser_navigate(self):
        assert get_tool_kind("browser_navigate") == "fetch"

    def test_unknown_tool_returns_other_kind(self):
        assert get_tool_kind("nonexistent_tool_xyz") == "other"

    # --- Newly added tool kinds ---

    def test_tool_kind_browser_console(self):
        assert get_tool_kind("browser_console") == "execute"

    def test_tool_kind_memory(self):
        assert get_tool_kind("memory") == "think"

    def test_tool_kind_todo(self):
        assert get_tool_kind("todo") == "think"

    def test_tool_kind_clarify(self):
        assert get_tool_kind("clarify") == "think"

    def test_tool_kind_session_search(self):
        assert get_tool_kind("session_search") == "search"

    def test_tool_kind_skill_view(self):
        assert get_tool_kind("skill_view") == "read"

    def test_tool_kind_skill_manage(self):
        assert get_tool_kind("skill_manage") == "edit"

    def test_tool_kind_skills_list(self):
        assert get_tool_kind("skills_list") == "read"

    def test_tool_kind_send_message(self):
        assert get_tool_kind("send_message") == "execute"

    def test_tool_kind_cronjob(self):
        assert get_tool_kind("cronjob") == "execute"

    def test_tool_kind_mixture_of_agents(self):
        assert get_tool_kind("mixture_of_agents") == "execute"

    def test_tool_kind_ha_call_service(self):
        assert get_tool_kind("ha_call_service") == "execute"

    def test_tool_kind_ha_get_state(self):
        assert get_tool_kind("ha_get_state") == "read"

    def test_tool_kind_ha_list_entities(self):
        assert get_tool_kind("ha_list_entities") == "read"

    def test_tool_kind_ha_list_services(self):
        assert get_tool_kind("ha_list_services") == "read"

    def test_tool_kind_rl_check_status(self):
        assert get_tool_kind("rl_check_status") == "read"

    def test_tool_kind_rl_start_training(self):
        assert get_tool_kind("rl_start_training") == "execute"

    def test_tool_kind_rl_stop_training(self):
        assert get_tool_kind("rl_stop_training") == "execute"

    def test_tool_kind_rl_edit_config(self):
        assert get_tool_kind("rl_edit_config") == "edit"


# ---------------------------------------------------------------------------
# make_tool_call_id
# ---------------------------------------------------------------------------


class TestMakeToolCallId:
    def test_returns_string(self):
        tc_id = make_tool_call_id()
        assert isinstance(tc_id, str)

    def test_starts_with_tc_prefix(self):
        tc_id = make_tool_call_id()
        assert tc_id.startswith("tc-")

    def test_ids_are_unique(self):
        ids = {make_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# build_tool_title
# ---------------------------------------------------------------------------


class TestBuildToolTitle:
    def test_terminal_title_includes_command(self):
        title = build_tool_title("terminal", {"command": "ls -la /tmp"})
        assert "ls -la /tmp" in title

    def test_terminal_title_truncates_long_command(self):
        long_cmd = "x" * 200
        title = build_tool_title("terminal", {"command": long_cmd})
        assert len(title) < 120
        assert "..." in title

    def test_read_file_title(self):
        title = build_tool_title("read_file", {"path": "/etc/hosts"})
        assert "/etc/hosts" in title

    def test_patch_title(self):
        title = build_tool_title("patch", {"path": "main.py", "mode": "replace"})
        assert "main.py" in title

    def test_search_title(self):
        title = build_tool_title("search_files", {"pattern": "TODO"})
        assert "TODO" in title

    def test_web_search_title(self):
        title = build_tool_title("web_search", {"query": "python asyncio"})
        assert "python asyncio" in title

    def test_unknown_tool_uses_name(self):
        title = build_tool_title("some_new_tool", {"foo": "bar"})
        assert title == "some_new_tool"

    def test_memory_title_with_action(self):
        title = build_tool_title("memory", {"action": "add", "target": "user", "content": "likes python"})
        assert "memory: add" == title

    def test_memory_title_without_action(self):
        title = build_tool_title("memory", {})
        assert title == "memory"

    def test_todo_title_with_items(self):
        title = build_tool_title("todo", {"todos": [{"id": "1", "content": "task"}]})
        assert title == "update todo"

    def test_todo_title_empty(self):
        title = build_tool_title("todo", {})
        assert title == "todo list"

    def test_clarify_title(self):
        title = build_tool_title("clarify", {"question": "What version?"})
        assert "What version?" in title

    def test_session_search_title(self):
        title = build_tool_title("session_search", {"query": "error failed"})
        assert "error failed" in title

    def test_skill_view_title(self):
        title = build_tool_title("skill_view", {"name": "fork-self-improve"})
        assert "fork-self-improve" in title

    def test_skill_manage_title(self):
        title = build_tool_title("skill_manage", {"action": "create", "name": "my-skill"})
        assert "create" in title and "my-skill" in title

    def test_skills_list_title(self):
        title = build_tool_title("skills_list", {})
        assert title == "list skills"

    def test_send_message_title(self):
        title = build_tool_title("send_message", {})
        assert title == "send message"

    def test_cronjob_title(self):
        title = build_tool_title("cronjob", {"action": "list"})
        assert "cron: list" == title

    def test_browser_console_title_with_expression(self):
        title = build_tool_title("browser_console", {"expression": "document.title"})
        assert "console: document.title" == title

    def test_browser_console_title_without_expression(self):
        title = build_tool_title("browser_console", {})
        assert title == "browser console"

    def test_ha_tool_title(self):
        title = build_tool_title("ha_get_state", {})
        assert title == "get_state"

    def test_rl_tool_title(self):
        title = build_tool_title("rl_start_training", {})
        assert title == "start training"

    def test_mixture_of_agents_title(self):
        title = build_tool_title("mixture_of_agents", {})
        assert title == "mixture of agents"


# ---------------------------------------------------------------------------
# build_tool_start
# ---------------------------------------------------------------------------


class TestBuildToolStart:
    def test_build_tool_start_for_patch(self):
        """patch should produce a FileEditToolCallContent (diff)."""
        args = {
            "path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('world')",
        }
        result = build_tool_start("tc-1", "patch", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        # The first content item should be a diff
        assert len(result.content) >= 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path == "src/main.py"
        assert diff_item.new_text == "print('world')"
        assert diff_item.old_text == "print('hello')"

    def test_build_tool_start_for_write_file(self):
        """write_file should produce a FileEditToolCallContent (diff)."""
        args = {"path": "new_file.py", "content": "print('hello')"}
        result = build_tool_start("tc-w1", "write_file", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) >= 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path == "new_file.py"

    def test_build_tool_start_for_terminal(self):
        """terminal should produce text content with the command."""
        args = {"command": "ls -la /tmp"}
        result = build_tool_start("tc-2", "terminal", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "execute"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        # The wrapped text block should contain the command
        text = content_item.content.text
        assert "ls -la /tmp" in text

    def test_build_tool_start_for_read_file(self):
        """read_file should include the path in content."""
        args = {"path": "/etc/hosts", "offset": 1, "limit": 50}
        result = build_tool_start("tc-3", "read_file", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "read"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        assert "/etc/hosts" in content_item.content.text

    def test_build_tool_start_for_search(self):
        """search_files should include pattern in content."""
        args = {"pattern": "TODO", "target": "content"}
        result = build_tool_start("tc-4", "search_files", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "search"
        assert "TODO" in result.content[0].content.text

    def test_build_tool_start_generic_fallback(self):
        """Unknown tools should get a generic text representation."""
        args = {"foo": "bar", "baz": 42}
        result = build_tool_start("tc-5", "some_tool", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "other"


# ---------------------------------------------------------------------------
# build_tool_complete
# ---------------------------------------------------------------------------


class TestBuildToolComplete:
    def test_build_tool_complete_for_terminal(self):
        """Completed terminal call should include output text."""
        result = build_tool_complete("tc-2", "terminal", "total 42\ndrwxr-xr-x 2 root root 4096 ...")
        assert isinstance(result, ToolCallProgress)
        assert result.status == "completed"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        assert "total 42" in content_item.content.text

    def test_build_tool_complete_truncates_large_output(self):
        """Very large outputs should be truncated."""
        big_output = "x" * 10000
        result = build_tool_complete("tc-6", "read_file", big_output)
        assert isinstance(result, ToolCallProgress)
        display_text = result.content[0].content.text
        assert len(display_text) < 6000
        assert "truncated" in display_text

    def test_build_tool_complete_for_patch_uses_diff_blocks(self):
        """Completed patch calls should keep structured diff content for Zed."""
        patch_result = (
            '{"success": true, "diff": "--- a/README.md\\n+++ b/README.md\\n@@ -1 +1,2 @@\\n old line\\n+new line\\n", '
            '"files_modified": ["README.md"]}'
        )
        result = build_tool_complete("tc-p1", "patch", patch_result)
        assert isinstance(result, ToolCallProgress)
        assert len(result.content) == 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path == "README.md"
        assert diff_item.old_text == "old line"
        assert diff_item.new_text == "old line\nnew line"

    def test_build_tool_complete_for_patch_falls_back_to_text_when_no_diff(self):
        result = build_tool_complete("tc-p2", "patch", '{"success": true}')
        assert isinstance(result, ToolCallProgress)
        assert isinstance(result.content[0], ContentToolCallContent)

    def test_build_tool_complete_for_write_file_uses_snapshot_diff(self, tmp_path):
        target = tmp_path / "diff-test.txt"
        snapshot = type("Snapshot", (), {"paths": [target], "before": {str(target): None}})()
        target.write_text("hello from hermes\n", encoding="utf-8")

        result = build_tool_complete(
            "tc-wf1",
            "write_file",
            '{"bytes_written": 18, "dirs_created": false}',
            function_args={"path": str(target), "content": "hello from hermes\n"},
            snapshot=snapshot,
        )
        assert isinstance(result, ToolCallProgress)
        assert len(result.content) == 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path.endswith("diff-test.txt")
        assert diff_item.old_text is None
        assert diff_item.new_text == "hello from hermes"


# ---------------------------------------------------------------------------
# extract_locations
# ---------------------------------------------------------------------------


class TestExtractLocations:
    def test_extract_locations_with_path(self):
        args = {"path": "src/app.py", "offset": 42}
        locs = extract_locations(args)
        assert len(locs) == 1
        assert isinstance(locs[0], ToolCallLocation)
        assert locs[0].path == "src/app.py"
        assert locs[0].line == 42

    def test_extract_locations_without_path(self):
        args = {"command": "echo hi"}
        locs = extract_locations(args)
        assert locs == []
