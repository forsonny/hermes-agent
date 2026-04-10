"""Tests for the Copilot ACP client shim."""

from __future__ import annotations

import io
import json

import pytest

import agent.copilot_acp_client as copilot


class DummyProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


def _last_json(payload_stream: io.StringIO) -> dict:
    data = payload_stream.getvalue().strip().splitlines()
    assert data, "expected a JSON-RPC response to be written"
    return json.loads(data[-1])


@pytest.mark.parametrize(
    "env_vars,expected",
    [
        ({}, "copilot"),
        ({"COPILOT_CLI_PATH": "/usr/local/bin/copilot"}, "/usr/local/bin/copilot"),
        (
            {
                "COPILOT_CLI_PATH": "/usr/local/bin/copilot",
                "HERMES_COPILOT_ACP_COMMAND": "/opt/hermes-copilot",
            },
            "/opt/hermes-copilot",
        ),
    ],
)
def test_resolve_command_prefers_hermes_override(monkeypatch, env_vars, expected):
    for key in ("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    assert copilot._resolve_command() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ["--acp", "--stdio"]),
        ("", ["--acp", "--stdio"]),
        ("--acp --stdio --flag value", ["--acp", "--stdio", "--flag", "value"]),
        (
            "--acp --stdio --flag 'value with spaces'",
            ["--acp", "--stdio", "--flag", "value with spaces"],
        ),
    ],
)
def test_resolve_args(monkeypatch, raw, expected):
    monkeypatch.delenv("HERMES_COPILOT_ACP_ARGS", raising=False)
    if raw is not None:
        monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", raw)

    assert copilot._resolve_args() == expected


@pytest.mark.parametrize(
    "content,expected",
    [
        (None, ""),
        ("  hello world  ", "hello world"),
        ({"text": "  direct text  "}, "direct text"),
        ({"content": "  nested text  "}, "nested text"),
        (["first", {"text": "second"}, {"text": "   "}, 3], "first\nsecond"),
    ],
)
def test_render_message_content(content, expected):
    assert copilot._render_message_content(content) == expected


def test_format_messages_as_prompt_includes_transcript_and_tools():
    prompt = copilot._format_messages_as_prompt(
        [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": [{"text": "Hello"}, {"text": "Hermes"}]},
            {"role": "assistant", "content": {"content": "Sure"}},
            {"role": "tool", "content": {"text": "tool output"}},
            {"role": "critic", "content": "context note"},
        ],
        model="gpt-4.1",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "description": "Do the thing.",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "do_thing"}},
    )

    assert "Hermes requested model hint: gpt-4.1" in prompt
    assert "Available tools (OpenAI function schema)." in prompt
    assert '"name": "do_thing"' in prompt
    assert "Tool choice hint:" in prompt
    assert "System:\nSystem rules" in prompt
    assert "User:\nHello\nHermes" in prompt
    assert "Assistant:\nSure" in prompt
    assert "Tool:\ntool output" in prompt
    assert "Context:\ncontext note" in prompt
    assert prompt.endswith("Continue the conversation from the latest user request.")


def test_extract_tool_calls_from_xml_block():
    text = (
        "Before <tool_call>{"
        '"id":"call-1","type":"function","function":{"name":"write_file",'
        '"arguments":"{\\\"path\\\":\\\"/tmp/x\\\"}"}}'
        "</tool_call> after"
    )

    tool_calls, cleaned = copilot._extract_tool_calls_from_text(text)

    assert cleaned == "Before\nafter"
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call-1"
    assert tool_calls[0].function.name == "write_file"
    assert tool_calls[0].function.arguments == '{"path":"/tmp/x"}'


def test_extract_tool_calls_from_bare_json_block():
    text = (
        "Lead "
        '{"id":"call-2","type":"function","function":{"name":"search",'
        '"arguments":{"query":"hello"}}}'
        " tail"
    )

    tool_calls, cleaned = copilot._extract_tool_calls_from_text(text)

    assert cleaned == "Lead\ntail"
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == '{"query": "hello"}'


def test_ensure_path_within_cwd_rejects_escaping_paths(tmp_path):
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    inside = cwd / "nested" / "file.txt"

    assert copilot._ensure_path_within_cwd(str(inside), str(cwd)) == inside.resolve()

    with pytest.raises(PermissionError):
        copilot._ensure_path_within_cwd("relative.txt", str(cwd))

    with pytest.raises(PermissionError):
        copilot._ensure_path_within_cwd(str(tmp_path / "outside.txt"), str(cwd))


def test_handle_server_message_session_updates_append_chunks(tmp_path):
    client = copilot.CopilotACPClient(acp_cwd=str(tmp_path), acp_command="copilot", acp_args=[])
    process = DummyProcess()
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    handled = client._handle_server_message(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "hello"},
                }
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=text_parts,
        reasoning_parts=reasoning_parts,
    )

    handled = handled and client._handle_server_message(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"text": "thinking"},
                }
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=text_parts,
        reasoning_parts=reasoning_parts,
    )

    assert handled is True
    assert text_parts == ["hello"]
    assert reasoning_parts == ["thinking"]
    assert process.stdin.getvalue() == ""


def test_handle_server_message_permission_and_fs_roundtrip(tmp_path):
    client = copilot.CopilotACPClient(acp_cwd=str(tmp_path), acp_command="copilot", acp_args=[])
    process = DummyProcess()

    allowed = client._handle_server_message(
        {"method": "session/request_permission", "id": 7, "params": {}},
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )
    assert allowed is True
    assert _last_json(process.stdin)["result"]["outcome"]["outcome"] == "allow_once"

    process.stdin = io.StringIO()
    source = tmp_path / "notes.txt"
    source.write_text("line1\nline2\nline3\n")
    read_ok = client._handle_server_message(
        {
            "method": "fs/read_text_file",
            "id": 8,
            "params": {
                "path": str(source),
                "line": 2,
                "limit": 1,
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )
    assert read_ok is True
    assert _last_json(process.stdin)["result"]["content"] == "line2\n"

    process.stdin = io.StringIO()
    target = tmp_path / "output" / "result.txt"
    write_ok = client._handle_server_message(
        {
            "method": "fs/write_text_file",
            "id": 9,
            "params": {
                "path": str(target),
                "content": "saved",
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )
    assert write_ok is True
    assert target.read_text() == "saved"
    assert _last_json(process.stdin)["result"] is None


def test_handle_server_message_unknown_method_returns_jsonrpc_error(tmp_path):
    client = copilot.CopilotACPClient(acp_cwd=str(tmp_path), acp_command="copilot", acp_args=[])
    process = DummyProcess()

    handled = client._handle_server_message(
        {"method": "session/unknown", "id": 42, "params": {}},
        process=process,
        cwd=str(tmp_path),
        text_parts=None,
        reasoning_parts=None,
    )

    payload = _last_json(process.stdin)
    assert handled is True
    assert payload["error"]["code"] == -32601
    assert "not supported by Hermes yet" in payload["error"]["message"]
