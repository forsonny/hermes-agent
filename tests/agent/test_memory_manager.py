"""Tests for agent.memory_manager -- MemoryManager and context helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import (
    MemoryManager,
    build_memory_context_block,
    sanitize_context,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight MemoryProvider stubs
# ---------------------------------------------------------------------------


def _make_provider(name, *, tool_schemas=None, system_prompt="", prefetch="",
                   sync_ok=True, handle_result=None):
    """Create a minimal MemoryProvider-like mock."""
    prov = MagicMock()
    prov.name = name
    prov.get_tool_schemas.return_value = tool_schemas or []
    prov.system_prompt_block.return_value = system_prompt
    prov.prefetch.return_value = prefetch
    prov.sync_turn = MagicMock()
    prov.queue_prefetch = MagicMock()
    prov.shutdown = MagicMock()
    prov.initialize = MagicMock()
    prov.on_turn_start = MagicMock()
    prov.on_session_end = MagicMock()
    prov.on_pre_compress.return_value = ""
    prov.on_memory_write = MagicMock()
    prov.on_delegation = MagicMock()
    if handle_result is not None:
        prov.handle_tool_call.return_value = handle_result
    else:
        prov.handle_tool_call.return_value = json.dumps({"ok": True})
    if not sync_ok:
        prov.sync_turn.side_effect = RuntimeError("sync failed")
    return prov


def _builtin(**kw):
    """Shortcut for a provider named 'builtin'."""
    kw.setdefault("name", "builtin")
    return _make_provider(kw.pop("name"), **kw)


# ---------------------------------------------------------------------------
# sanitize_context
# ---------------------------------------------------------------------------


class TestSanitizeContext:
    def test_strips_open_tag(self):
        text = 'before <memory-context> after'
        assert sanitize_context(text) == "before  after"

    def test_strips_close_tag(self):
        text = 'before </memory-context> after'
        assert sanitize_context(text) == "before  after"

    def test_strips_both_tags(self):
        text = 'before <memory-context> inner </memory-context> after'
        assert sanitize_context(text) == "before  inner  after"

    def test_strips_case_insensitive(self):
        text = '<Memory-Context>hello</MEMORY-CONTEXT>'
        assert sanitize_context(text) == "hello"

    def test_no_tags_unchanged(self):
        text = "just regular text"
        assert sanitize_context(text) == text

    def test_empty_string(self):
        assert sanitize_context("") == ""


# ---------------------------------------------------------------------------
# build_memory_context_block
# ---------------------------------------------------------------------------


class TestBuildMemoryContextBlock:
    def test_empty_input_returns_empty(self):
        assert build_memory_context_block("") == ""
        assert build_memory_context_block("   ") == ""

    def test_wraps_content(self):
        result = build_memory_context_block("recall data here")
        assert result.startswith("<memory-context>")
        assert result.endswith("</memory-context>")
        assert "recall data here" in result

    def test_includes_system_note(self):
        result = build_memory_context_block("data")
        assert "[System note:" in result
        assert "NOT new user input" in result

    def test_strips_existing_fence_tags(self):
        result = build_memory_context_block("<memory-context>evil</memory-context>")
        # Should not have nested tags
        assert result.count("<memory-context>") == 1
        assert result.count("</memory-context>") == 1


# ---------------------------------------------------------------------------
# MemoryManager -- registration
# ---------------------------------------------------------------------------


class TestMemoryManagerRegistration:
    def test_empty_on_creation(self):
        mgr = MemoryManager()
        assert mgr.providers == []
        assert mgr.get_all_tool_schemas() == []
        assert mgr.get_all_tool_names() == set()

    def test_add_builtin_provider(self):
        mgr = MemoryManager()
        p = _builtin(tool_schemas=[
            {"name": "memory", "description": "mem", "parameters": {}},
        ])
        mgr.add_provider(p)
        assert len(mgr.providers) == 1
        assert mgr.has_tool("memory")

    def test_add_one_external_provider(self):
        mgr = MemoryManager()
        mgr.add_provider(_builtin())
        external = _make_provider("honcho", tool_schemas=[
            {"name": "honcho_recall", "description": "recall", "parameters": {}},
        ])
        mgr.add_provider(external)
        assert len(mgr.providers) == 2
        assert mgr.has_tool("honcho_recall")

    def test_reject_second_external_provider(self):
        mgr = MemoryManager()
        mgr.add_provider(_builtin())
        mgr.add_provider(_make_provider("ext1"))
        # Second external should be rejected
        mgr.add_provider(_make_provider("ext2"))
        assert len(mgr.providers) == 2  # builtin + ext1 only

    def test_allow_multiple_builtins(self):
        """Multiple providers named 'builtin' are all accepted."""
        mgr = MemoryManager()
        mgr.add_provider(_builtin())
        mgr.add_provider(_builtin())
        assert len(mgr.providers) == 2

    def test_tool_name_conflict_uses_first(self):
        mgr = MemoryManager()
        p1 = _make_provider("p1", tool_schemas=[
            {"name": "memory", "description": "first", "parameters": {}},
        ])
        p2 = _make_provider("p2", tool_schemas=[
            {"name": "memory", "description": "second", "parameters": {}},
        ])
        mgr.add_provider(p1)
        mgr.add_provider(p2)
        # Tool routes to first provider
        assert mgr._tool_to_provider["memory"].name == "p1"

    def test_get_provider_by_name(self):
        mgr = MemoryManager()
        p = _make_provider("test_prov")
        mgr.add_provider(p)
        assert mgr.get_provider("test_prov") is p
        assert mgr.get_provider("nonexistent") is None


# ---------------------------------------------------------------------------
# MemoryManager -- system prompt
# ---------------------------------------------------------------------------


class TestMemoryManagerSystemPrompt:
    def test_empty_when_no_providers(self):
        mgr = MemoryManager()
        assert mgr.build_system_prompt() == ""

    def test_collects_blocks_from_builtin_and_external(self):
        mgr = MemoryManager()
        mgr.add_provider(_builtin(system_prompt="Block A"))
        mgr.add_provider(_make_provider("ext", system_prompt="Block B"))
        prompt = mgr.build_system_prompt()
        assert "Block A" in prompt
        assert "Block B" in prompt

    def test_skips_empty_blocks(self):
        mgr = MemoryManager()
        mgr.add_provider(_builtin(system_prompt="Keep"))
        mgr.add_provider(_make_provider("ext", system_prompt=""))
        prompt = mgr.build_system_prompt()
        assert prompt == "Keep"

    def test_provider_failure_doesnt_block(self):
        mgr = MemoryManager()
        bad = _builtin()
        bad.system_prompt_block.side_effect = RuntimeError("boom")
        mgr.add_provider(bad)
        mgr.add_provider(_make_provider("ext", system_prompt="OK"))
        prompt = mgr.build_system_prompt()
        assert prompt == "OK"


# ---------------------------------------------------------------------------
# MemoryManager -- prefetch
# ---------------------------------------------------------------------------


class TestMemoryManagerPrefetch:
    def test_merges_prefetch_results(self):
        mgr = MemoryManager()
        mgr.add_provider(_builtin(prefetch="ctx1"))
        mgr.add_provider(_make_provider("ext", prefetch="ctx2"))
        result = mgr.prefetch_all("hello")
        assert "ctx1" in result
        assert "ctx2" in result

    def test_skips_empty_prefetch(self):
        mgr = MemoryManager()
        mgr.add_provider(_builtin(prefetch=""))
        mgr.add_provider(_make_provider("ext", prefetch="data"))
        result = mgr.prefetch_all("hello")
        assert result == "data"

    def test_provider_failure_continues(self):
        mgr = MemoryManager()
        bad = _builtin()
        bad.prefetch.side_effect = RuntimeError("prefetch fail")
        mgr.add_provider(bad)
        mgr.add_provider(_make_provider("ext", prefetch="ok_data"))
        result = mgr.prefetch_all("hello")
        assert result == "ok_data"

    def test_queue_prefetch_all(self):
        mgr = MemoryManager()
        p1 = _builtin()
        p2 = _make_provider("ext")
        mgr.add_provider(p1)
        mgr.add_provider(p2)
        mgr.queue_prefetch_all("test query")
        p1.queue_prefetch.assert_called_once()
        p2.queue_prefetch.assert_called_once()


# ---------------------------------------------------------------------------
# MemoryManager -- sync
# ---------------------------------------------------------------------------


class TestMemoryManagerSync:
    def test_sync_calls_all_providers(self):
        mgr = MemoryManager()
        p1 = _builtin()
        p2 = _make_provider("ext")
        mgr.add_provider(p1)
        mgr.add_provider(p2)
        mgr.sync_all("user msg", "assistant msg")
        p1.sync_turn.assert_called_once()
        p2.sync_turn.assert_called_once()

    def test_sync_failure_continues(self):
        mgr = MemoryManager()
        bad = _builtin(sync_ok=False)
        good = _make_provider("ext")
        mgr.add_provider(bad)
        mgr.add_provider(good)
        mgr.sync_all("u", "a")
        # Good provider still synced even though bad failed
        good.sync_turn.assert_called_once()


# ---------------------------------------------------------------------------
# MemoryManager -- tools
# ---------------------------------------------------------------------------


class TestMemoryManagerTools:
    def test_get_all_tool_schemas_deduplicates(self):
        mgr = MemoryManager()
        schema = {"name": "mem", "description": "d", "parameters": {}}
        mgr.add_provider(_make_provider("p1", tool_schemas=[schema]))
        mgr.add_provider(_make_provider("p2", tool_schemas=[schema]))
        # Only one schema despite two providers with same tool name
        schemas = mgr.get_all_tool_schemas()
        assert len(schemas) == 1

    def test_has_tool(self):
        mgr = MemoryManager()
        assert not mgr.has_tool("memory")
        mgr.add_provider(_make_provider("p", tool_schemas=[
            {"name": "memory", "description": "m", "parameters": {}},
        ]))
        assert mgr.has_tool("memory")

    def test_handle_tool_call_routes(self):
        mgr = MemoryManager()
        p = _make_provider("p", tool_schemas=[
            {"name": "recall", "description": "r", "parameters": {}},
        ])
        mgr.add_provider(p)
        result = mgr.handle_tool_call("recall", {"query": "test"})
        p.handle_tool_call.assert_called_once()
        assert json.loads(result) == {"ok": True}

    def test_handle_unknown_tool_returns_error(self):
        mgr = MemoryManager()
        result = mgr.handle_tool_call("nonexistent", {})
        assert "No memory provider" in result

    def test_handle_tool_call_failure_returns_error(self):
        mgr = MemoryManager()
        p = _make_provider("p", tool_schemas=[
            {"name": "fail_tool", "description": "f", "parameters": {}},
        ])
        p.handle_tool_call.side_effect = RuntimeError("tool crashed")
        mgr.add_provider(p)
        result = mgr.handle_tool_call("fail_tool", {})
        assert "failed" in result

    def test_get_all_tool_names(self):
        mgr = MemoryManager()
        mgr.add_provider(_make_provider("p1", tool_schemas=[
            {"name": "a", "description": "", "parameters": {}},
            {"name": "b", "description": "", "parameters": {}},
        ]))
        assert mgr.get_all_tool_names() == {"a", "b"}


# ---------------------------------------------------------------------------
# MemoryManager -- lifecycle hooks
# ---------------------------------------------------------------------------


class TestMemoryManagerLifecycle:
    def test_on_turn_start(self):
        mgr = MemoryManager()
        p = _builtin()
        mgr.add_provider(p)
        mgr.on_turn_start(3, "hello", remaining_tokens=1000)
        p.on_turn_start.assert_called_once_with(3, "hello", remaining_tokens=1000)

    def test_on_session_end(self):
        mgr = MemoryManager()
        p = _builtin()
        mgr.add_provider(p)
        msgs = [{"role": "user", "content": "hi"}]
        mgr.on_session_end(msgs)
        p.on_session_end.assert_called_once_with(msgs)

    def test_on_pre_compress_merges(self):
        mgr = MemoryManager()
        p1 = _builtin()
        p1.on_pre_compress.return_value = "insight1"
        p2 = _make_provider("ext")
        p2.on_pre_compress.return_value = "insight2"
        mgr.add_provider(p1)
        mgr.add_provider(p2)
        result = mgr.on_pre_compress([{"role": "user", "content": "x"}])
        assert "insight1" in result
        assert "insight2" in result

    def test_on_memory_write_skips_builtin(self):
        mgr = MemoryManager()
        bi = _builtin()
        ext = _make_provider("ext")
        mgr.add_provider(bi)
        mgr.add_provider(ext)
        mgr.on_memory_write("add", "memory", "new fact")
        bi.on_memory_write.assert_not_called()
        ext.on_memory_write.assert_called_once_with("add", "memory", "new fact")

    def test_on_delegation(self):
        mgr = MemoryManager()
        p = _builtin()
        mgr.add_provider(p)
        mgr.on_delegation("task desc", "result", child_session_id="abc")
        p.on_delegation.assert_called_once()

    def test_shutdown_reversed(self):
        mgr = MemoryManager()
        p1 = _builtin()
        p2 = _make_provider("ext")
        mgr.add_provider(p1)
        mgr.add_provider(p2)
        mgr.shutdown_all()
        p1.shutdown.assert_called_once()
        p2.shutdown.assert_called_once()

    def test_initialize_all_injects_hermes_home(self):
        mgr = MemoryManager()
        p = _builtin()
        mgr.add_provider(p)
        # get_hermes_home is imported locally inside the method
        with patch("hermes_constants.get_hermes_home", return_value="/test/hermes"):
            mgr.initialize_all("session-1")
        call_kwargs = p.initialize.call_args
        assert call_kwargs[1]["hermes_home"] == "/test/hermes"

    def test_initialize_all_respects_explicit_hermes_home(self):
        mgr = MemoryManager()
        p = _builtin()
        mgr.add_provider(p)
        mgr.initialize_all("session-1", hermes_home="/custom/path")
        call_kwargs = p.initialize.call_args
        assert call_kwargs[1]["hermes_home"] == "/custom/path"


# ---------------------------------------------------------------------------
# MemoryManager -- providers property
# ---------------------------------------------------------------------------


class TestMemoryManagerProperties:
    def test_providers_returns_copy(self):
        mgr = MemoryManager()
        p = _builtin()
        mgr.add_provider(p)
        provs = mgr.providers
        provs.append(_make_provider("extra"))
        # Original should be unaffected
        assert len(mgr.providers) == 1
