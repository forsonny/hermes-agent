"""ACP tool-call helpers for mapping hermes tools to ACP ToolKind and building content."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import acp
from acp.schema import (
    ToolCallLocation,
    ToolCallStart,
    ToolCallProgress,
    ToolKind,
)

# ---------------------------------------------------------------------------
# Map hermes tool names -> ACP ToolKind
# ---------------------------------------------------------------------------

TOOL_KIND_MAP: Dict[str, ToolKind] = {
    # File operations
    "read_file": "read",
    "write_file": "edit",
    "patch": "edit",
    "search_files": "search",
    # Terminal / execution
    "terminal": "execute",
    "process": "execute",
    "execute_code": "execute",
    # Web / fetch
    "web_search": "fetch",
    "web_extract": "fetch",
    # Browser
    "browser_navigate": "fetch",
    "browser_click": "execute",
    "browser_type": "execute",
    "browser_snapshot": "read",
    "browser_vision": "read",
    "browser_scroll": "execute",
    "browser_press": "execute",
    "browser_back": "execute",
    "browser_get_images": "read",
    "browser_console": "execute",
    # Agent internals
    "delegate_task": "execute",
    "vision_analyze": "read",
    "image_generate": "execute",
    "text_to_speech": "execute",
    # Thinking / meta
    "_thinking": "think",
    # Agent meta-tools (memory, planning, skills)
    "memory": "think",
    "todo": "think",
    "clarify": "think",
    "session_search": "search",
    "skill_view": "read",
    "skill_manage": "edit",
    "skills_list": "read",
    "send_message": "execute",
    "cronjob": "execute",
    # Multi-agent
    "mixture_of_agents": "execute",
    # Smart home
    "ha_call_service": "execute",
    "ha_get_state": "read",
    "ha_list_entities": "read",
    "ha_list_services": "read",
    # RL training
    "rl_check_status": "read",
    "rl_edit_config": "edit",
    "rl_get_current_config": "read",
    "rl_get_results": "read",
    "rl_list_environments": "read",
    "rl_list_runs": "read",
    "rl_select_environment": "execute",
    "rl_start_training": "execute",
    "rl_stop_training": "execute",
    "rl_test_inference": "execute",
}


def get_tool_kind(tool_name: str) -> ToolKind:
    """Return the ACP ToolKind for a hermes tool, defaulting to 'other'."""
    return TOOL_KIND_MAP.get(tool_name, "other")


def make_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"tc-{uuid.uuid4().hex[:12]}"


def build_tool_title(tool_name: str, args: Dict[str, Any]) -> str:
    """Build a human-readable title for a tool call."""
    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"terminal: {cmd}"
    if tool_name == "read_file":
        return f"read: {args.get('path', '?')}"
    if tool_name == "write_file":
        return f"write: {args.get('path', '?')}"
    if tool_name == "patch":
        mode = args.get("mode", "replace")
        path = args.get("path", "?")
        return f"patch ({mode}): {path}"
    if tool_name == "search_files":
        return f"search: {args.get('pattern', '?')}"
    if tool_name == "web_search":
        return f"web search: {args.get('query', '?')}"
    if tool_name == "web_extract":
        urls = args.get("urls", [])
        if urls:
            return f"extract: {urls[0]}" + (f" (+{len(urls)-1})" if len(urls) > 1 else "")
        return "web extract"
    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if goal and len(goal) > 60:
            goal = goal[:57] + "..."
        return f"delegate: {goal}" if goal else "delegate task"
    if tool_name == "execute_code":
        return "execute code"
    if tool_name == "vision_analyze":
        return f"analyze image: {args.get('question', '?')[:50]}"
    if tool_name == "memory":
        action = args.get("action", "")
        return f"memory: {action}" if action else "memory"
    if tool_name == "todo":
        return "todo list" if not args.get("todos") else "update todo"
    if tool_name == "clarify":
        return f"clarify: {args.get('question', '')[:60]}"
    if tool_name == "session_search":
        query = args.get("query", "")
        return f"session search: {query[:50]}" if query else "session search"
    if tool_name == "skill_view":
        return f"skill: {args.get('name', '?')}"
    if tool_name == "skill_manage":
        return f"skill {args.get('action', '?')}: {args.get('name', '')}"
    if tool_name == "skills_list":
        return "list skills"
    if tool_name == "send_message":
        return "send message"
    if tool_name == "cronjob":
        return f"cron: {args.get('action', '?')}"
    if tool_name == "browser_console":
        expr = args.get("expression", "")
        if expr:
            return f"console: {expr[:50]}"
        return "browser console"
    if tool_name.startswith("ha_"):
        parts = tool_name.split("_", 1)
        return parts[1] if len(parts) > 1 else tool_name
    if tool_name.startswith("rl_"):
        return tool_name[3:].replace("_", " ")
    if tool_name == "mixture_of_agents":
        return "mixture of agents"
    return tool_name


# ---------------------------------------------------------------------------
# Build ACP content objects for tool-call events
# ---------------------------------------------------------------------------


def build_tool_start(
    tool_call_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> ToolCallStart:
    """Create a ToolCallStart event for the given hermes tool invocation."""
    kind = get_tool_kind(tool_name)
    title = build_tool_title(tool_name, arguments)
    locations = extract_locations(arguments)

    if tool_name == "patch":
        mode = arguments.get("mode", "replace")
        if mode == "replace":
            path = arguments.get("path", "")
            old = arguments.get("old_string", "")
            new = arguments.get("new_string", "")
            content = [acp.tool_diff_content(path=path, new_text=new, old_text=old)]
        else:
            # Patch mode — show the patch content as text
            patch_text = arguments.get("patch", "")
            content = [acp.tool_content(acp.text_block(patch_text))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "write_file":
        path = arguments.get("path", "")
        file_content = arguments.get("content", "")
        content = [acp.tool_diff_content(path=path, new_text=file_content)]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "terminal":
        command = arguments.get("command", "")
        content = [acp.tool_content(acp.text_block(f"$ {command}"))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "read_file":
        path = arguments.get("path", "")
        content = [acp.tool_content(acp.text_block(f"Reading {path}"))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "search_files":
        pattern = arguments.get("pattern", "")
        target = arguments.get("target", "content")
        content = [acp.tool_content(acp.text_block(f"Searching for '{pattern}' ({target})"))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    # Generic fallback
    import json
    try:
        args_text = json.dumps(arguments, indent=2, default=str)
    except (TypeError, ValueError):
        args_text = str(arguments)
    content = [acp.tool_content(acp.text_block(args_text))]
    return acp.start_tool_call(
        tool_call_id, title, kind=kind, content=content, locations=locations,
        raw_input=arguments,
    )


def build_tool_complete(
    tool_call_id: str,
    tool_name: str,
    result: Optional[str] = None,
) -> ToolCallProgress:
    """Create a ToolCallUpdate (progress) event for a completed tool call."""
    kind = get_tool_kind(tool_name)

    # Truncate very large results for the UI
    display_result = result or ""
    if len(display_result) > 5000:
        display_result = display_result[:4900] + f"\n... ({len(result)} chars total, truncated)"

    content = [acp.tool_content(acp.text_block(display_result))]
    return acp.update_tool_call(
        tool_call_id,
        kind=kind,
        status="completed",
        content=content,
        raw_output=result,
    )


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------


def extract_locations(
    arguments: Dict[str, Any],
) -> List[ToolCallLocation]:
    """Extract file-system locations from tool arguments."""
    locations: List[ToolCallLocation] = []
    path = arguments.get("path")
    if path:
        line = arguments.get("offset") or arguments.get("line")
        locations.append(ToolCallLocation(path=path, line=line))
    return locations
