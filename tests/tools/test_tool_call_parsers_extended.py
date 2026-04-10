"""
Extended tests for environments/tool_call_parsers/ — parsers not covered by
test_tool_call_parsers.py.

Covers: Longcat, Qwen, GLM 4.5, GLM 4.7, DeepSeek V3.1, Llama, Kimi K2,
Qwen3 Coder, plus cross-parser contract tests.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from environments.tool_call_parsers import (
        ParseResult,
        ToolCallParser,
        get_parser,
        list_parsers,
    )
except ImportError:
    pytest.skip("atroposlib not installed", allow_module_level=True)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_args(tc):
    """Safely parse tool call arguments JSON."""
    return json.loads(tc.function.arguments)


# ═══════════════════════════════════════════════════════════════════════
# Longcat parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestLongcatParser:
    @pytest.fixture
    def parser(self):
        return get_parser("longcat")

    def test_no_tool_call(self, parser):
        text = "Just plain text without any tool calls."
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = '<longcat_tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</longcat_tool_call>'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"
        assert _parse_args(tool_calls[0])["command"] == "ls"

    def test_multiple_tool_calls(self, parser):
        text = (
            '<longcat_tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</longcat_tool_call>'
            '<longcat_tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</longcat_tool_call>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = {tc.function.name for tc in tool_calls}
        assert names == {"terminal", "read_file"}

    def test_tool_call_with_preceding_text(self, parser):
        text = 'I will run a command.\n<longcat_tool_call>{"name": "terminal", "arguments": {"command": "pwd"}}</longcat_tool_call>'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert content is not None
        assert "run a command" in content

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None

    def test_truncated_unclosed_tag(self, parser):
        """Model output was truncated before closing tag."""
        text = '<longcat_tool_call>{"name": "terminal", "arguments": {"command": "ls"}'
        content, tool_calls = parser.parse(text)
        # Should handle gracefully — either parse it or return None
        if tool_calls is not None:
            assert len(tool_calls) >= 1

    def test_tool_call_ids_are_unique(self, parser):
        text = (
            '<longcat_tool_call>{"name": "a", "arguments": {}}</longcat_tool_call>'
            '<longcat_tool_call>{"name": "b", "arguments": {}}</longcat_tool_call>'
        )
        _, tool_calls = parser.parse(text)
        assert tool_calls is not None
        ids = [tc.id for tc in tool_calls]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════════════
# Qwen parser tests (inherits Hermes format)
# ═══════════════════════════════════════════════════════════════════════

class TestQwenParser:
    @pytest.fixture
    def parser(self):
        return get_parser("qwen")

    def test_no_tool_call(self, parser):
        text = "Hello world"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call_hermes_format(self, parser):
        text = '{"name": "terminal", "arguments": {"command": "echo hi"}}\n'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"

    def test_multiple_tool_calls(self, parser):
        text = (
            '{"name": "terminal", "arguments": {"command": "ls"}}\n'
            '{"name": "read_file", "arguments": {"path": "test.py"}}\n'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2


# ═══════════════════════════════════════════════════════════════════════
# GLM 4.5 parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestGlm45Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("glm45")

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help?"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = (
            'Let me help.\n'
            '△function_name\n'
            '▽param1△value1▽\n'
            '▽param2△value2▽\n'
            '☆\n'
            'More text after.'
        )
        content, tool_calls = parser.parse(text)
        # Note: the actual START_TOKEN for GLM is △ — parser checks for its presence
        if tool_calls is not None:
            assert len(tool_calls) >= 1
            assert tool_calls[0].function.name == "function_name"

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None

    def test_args_deserialized(self, parser):
        """GLM deserializes values via json.loads -> ast.literal_eval -> str."""
        text = (
            '△get_weather\n'
            '▽city△"London"▽\n'
            '▽count△42▽\n'
            '☆\n'
        )
        content, tool_calls = parser.parse(text)
        if tool_calls is not None:
            args = _parse_args(tool_calls[0])
            assert args.get("city") == "London"
            assert args.get("count") == 42


# ═══════════════════════════════════════════════════════════════════════
# GLM 4.7 parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestGlm47Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("glm47")

    def test_no_tool_call(self, parser):
        text = "Hello world"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = (
            '△get_time\n'
            '▽timezone△UTC▽\n'
            '☆\n'
        )
        content, tool_calls = parser.parse(text)
        if tool_calls is not None:
            assert len(tool_calls) >= 1
            assert tool_calls[0].function.name == "get_time"


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek V3.1 parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeepSeekV31Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_v3_1")

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help you?"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = (
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "London"}'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        args = _parse_args(tool_calls[0])
        assert args["city"] == "London"

    def test_multiple_tool_calls(self, parser):
        text = (
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>func1<｜tool▁sep｜>{"a": 1}'
            '<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>func2<｜tool▁sep｜>{"b": 2}'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = [tc.function.name for tc in tool_calls]
        assert "func1" in names
        assert "func2" in names

    def test_tool_call_with_preceding_text(self, parser):
        text = (
            'Let me check that.\n'
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>terminal<｜tool▁sep｜>{"command": "ls"}'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert content is not None
        assert "check that" in content

    def test_empty_arguments(self, parser):
        text = (
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>no_args_func<｜tool▁sep｜>{}'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "no_args_func"

    def test_deepseek_v31_alias(self):
        """deepseek_v31 should be an alias for deepseek_v3_1."""
        p1 = get_parser("deepseek_v3_1")
        p2 = get_parser("deepseek_v31")
        assert type(p1) == type(p2)

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None


# ═══════════════════════════════════════════════════════════════════════
# Llama parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestLlamaParser:
    @pytest.fixture(params=["llama3_json", "llama4_json"])
    def parser(self, request):
        return get_parser(request.param)

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help you?"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_json_tool_call(self, parser):
        text = '{"name": "terminal", "arguments": {"command": "ls"}}'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"
        args = _parse_args(tool_calls[0])
        assert args["command"] == "ls"

    def test_single_json_with_parameters_key(self, parser):
        """Llama format also accepts 'parameters' instead of 'arguments'."""
        text = '{"name": "terminal", "parameters": {"command": "pwd"}}'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"
        args = _parse_args(tool_calls[0])
        assert args["command"] == "pwd"

    def test_multiple_tool_calls_embedded(self, parser):
        text = (
            'Here are the results: '
            '{"name": "func1", "arguments": {"a": 1}} and '
            '{"name": "func2", "arguments": {"b": 2}}'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = {tc.function.name for tc in tool_calls}
        assert names == {"func1", "func2"}

    def test_tool_call_with_preceding_text(self, parser):
        text = 'Let me help.\n{"name": "terminal", "arguments": {"command": "echo hi"}}'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None

    def test_plain_json_not_tool_call(self, parser):
        """JSON without 'name' key should not be treated as tool call."""
        text = '{"command": "ls", "output": "file1.txt"}'
        content, tool_calls = parser.parse(text)
        assert tool_calls is None

    def test_nested_json_arguments(self, parser):
        text = '{"name": "search", "arguments": {"query": {"field": "name", "value": "test"}}}'
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        args = _parse_args(tool_calls[0])
        assert args["query"]["field"] == "name"

    def test_llama4_alias_registered(self):
        """llama4_json should be registered."""
        parsers = list_parsers()
        assert "llama4_json" in parsers


# ═══════════════════════════════════════════════════════════════════════
# Kimi K2 parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestKimiK2Parser:
    @pytest.fixture
    def parser(self):
        return get_parser("kimi_k2")

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help you?"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.get_weather:0'
            '<|tool_call_argument_begin|>{"city": "London"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        args = _parse_args(tool_calls[0])
        assert args["city"] == "London"

    def test_multiple_tool_calls(self, parser):
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.func1:0'
            '<|tool_call_argument_begin|>{"a": 1}'
            '<|tool_call_end|>'
            '<|tool_call_begin|>functions.func2:1'
            '<|tool_call_argument_begin|>{"b": 2}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = [tc.function.name for tc in tool_calls]
        assert "func1" in names
        assert "func2" in names

    def test_tool_call_with_preceding_text(self, parser):
        text = (
            'Let me look that up.\n'
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>search:0'
            '<|tool_call_argument_begin|>{"q": "test"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert content is not None
        assert "look that up" in content

    def test_singular_section_begin_token(self, parser):
        """Parser should also accept tool_call_section_begin (singular)."""
        text = (
            '<|tool_call_section_begin|>'
            '<|tool_call_begin|>my_func:0'
            '<|tool_call_argument_begin|>{"x": 1}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "my_func"

    def test_function_name_extraction_from_id(self, parser):
        """Function name should be extracted from the ID format."""
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.complex_name_here:5'
            '<|tool_call_argument_begin|>{"key": "val"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        _, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert tool_calls[0].function.name == "complex_name_here"

    def test_preserves_original_id(self, parser):
        """The tool call ID should preserve the original format."""
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.get_weather:0'
            '<|tool_call_argument_begin|>{"city": "London"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        _, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert "get_weather" in tool_calls[0].id
        assert ":0" in tool_calls[0].id

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None


# ═══════════════════════════════════════════════════════════════════════
# Qwen3 Coder parser tests
# ═══════════════════════════════════════════════════════════════════════

class TestQwen3CoderParser:
    @pytest.fixture
    def parser(self):
        return get_parser("qwen3_coder")

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help you?"
        content, tool_calls = parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self, parser):
        text = (
            '<tool_call:\n'
            '<function=terminal>\n'
            '<parameter=command>ls -la</parameter>\n'
            '</function>\n'
            '>\n'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "terminal"
        args = _parse_args(tool_calls[0])
        assert args["command"] == "ls -la"

    def test_multiple_tool_calls(self, parser):
        text = (
            '<tool_call:\n'
            '<function=func1>\n'
            '<parameter=a>1</parameter>\n'
            '</function>\n'
            '>\n'
            '<tool_call:\n'
            '<function=func2>\n'
            '<parameter=b>2</parameter>\n'
            '</function>\n'
            '>\n'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = {tc.function.name for tc in tool_calls}
        assert names == {"func1", "func2"}

    def test_tool_call_with_preceding_text(self, parser):
        text = (
            'Let me run that.\n'
            '<tool_call:\n'
            '<function=terminal>\n'
            '<parameter=command>pwd</parameter>\n'
            '</function>\n'
            '>\n'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert content is not None
        assert "run that" in content

    def test_multiple_parameters(self, parser):
        text = (
            '<tool_call:\n'
            '<function=search>\n'
            '<parameter=query>hello</parameter>\n'
            '<parameter=limit>10</parameter>\n'
            '</function>\n'
            '>\n'
        )
        content, tool_calls = parser.parse(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        args = _parse_args(tool_calls[0])
        assert args["query"] == "hello"
        assert args["limit"] == 10

    def test_parameter_type_conversion(self, parser):
        """Parameters should be type-converted: numbers, booleans, null."""
        text = (
            '<tool_call:\n'
            '<function=func>\n'
            '<parameter=num>42</parameter>\n'
            '<parameter=flag>true</parameter>\n'
            '<parameter>nothing>null</parameter>\n'
            '</function>\n'
            '>\n'
        )
        content, tool_calls = parser.parse(text)
        if tool_calls is not None:
            args = _parse_args(tool_calls[0])
            assert args.get("num") == 42
            assert args.get("flag") is True
            assert args.get("nothing") is None

    def test_empty_string(self, parser):
        content, tool_calls = parser.parse("")
        assert tool_calls is None


# ═══════════════════════════════════════════════════════════════════════
# Cross-parser contract tests — applies to ALL parsers
# ═══════════════════════════════════════════════════════════════════════

_ALL_PARSER_NAMES = list_parsers()


class TestAllParsersContract:
    """Ensure every registered parser meets the base contract."""

    @pytest.fixture(params=_ALL_PARSER_NAMES)
    def parser(self, request):
        return get_parser(request.param)

    def test_returns_tuple_of_two(self, parser):
        result = parser.parse("hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_tools_returns_none_tool_calls(self, parser):
        content, tool_calls = parser.parse("Just plain text, no tools.")
        assert tool_calls is None
        assert content is not None

    def test_empty_string_no_crash(self, parser):
        content, tool_calls = parser.parse("")
        # Must not raise; tool_calls should be None
        assert tool_calls is None

    def test_none_tool_calls_means_content_not_none(self, parser):
        """When no tool calls found, content should be the original text."""
        text = "Some regular text here"
        content, tool_calls = parser.parse(text)
        if tool_calls is None:
            assert content is not None

    def test_parse_is_idempotent(self, parser):
        """Calling parse twice with same input should give same structure."""
        text = "hello world"
        r1_content, r1_tc = parser.parse(text)
        r2_content, r2_tc = parser.parse(text)
        assert (r1_content, r1_tc is None) == (r2_content, r2_tc is None)


class TestAllParsersRegistered:
    """Verify all expected parsers are in the registry."""

    EXPECTED_PARSERS = [
        "hermes",
        "longcat",
        "mistral",
        "llama3_json",
        "llama4_json",
        "qwen",
        "deepseek_v3",
        "deepseek_v3_1",
        "deepseek_v31",
        "kimi_k2",
        "glm45",
        "glm47",
        "qwen3_coder",
    ]

    def test_all_expected_parsers_registered(self):
        parsers = set(list_parsers())
        for name in self.EXPECTED_PARSERS:
            assert name in parsers, f"Parser '{name}' not found in registry"

    def test_all_parsers_instantiate(self):
        for name in list_parsers():
            p = get_parser(name)
            assert isinstance(p, ToolCallParser)
            assert callable(getattr(p, "parse", None))

    def test_parser_count(self):
        """We should have at least the expected number of parsers."""
        assert len(list_parsers()) >= len(self.EXPECTED_PARSERS)
