"""Tests for hermes_cli.model_switch pure functions.

Covers parse_model_flags() and _check_hermes_model_warning() with
comprehensive edge cases including flag ordering, substring matching,
and empty inputs.
"""

import pytest
from hermes_cli.model_switch import parse_model_flags, _check_hermes_model_warning


# ---------------------------------------------------------------------------
# parse_model_flags
# ---------------------------------------------------------------------------

class TestParseModelFlagsBasic:
    """Basic flag parsing cases."""

    def test_bare_model_name(self):
        assert parse_model_flags("sonnet") == ("sonnet", "", False)

    def test_model_with_global(self):
        assert parse_model_flags("sonnet --global") == ("sonnet", "", True)

    def test_model_with_provider(self):
        assert parse_model_flags("sonnet --provider anthropic") == ("sonnet", "anthropic", False)

    def test_provider_only(self):
        assert parse_model_flags("--provider my-ollama") == ("", "my-ollama", False)

    def test_all_flags(self):
        assert parse_model_flags("sonnet --provider anthropic --global") == (
            "sonnet", "anthropic", True,
        )

    def test_empty_input(self):
        assert parse_model_flags("") == ("", "", False)


class TestParseModelFlagsOrdering:
    """Flag order independence tests."""

    def test_global_before_model(self):
        assert parse_model_flags("--global model") == ("model", "", True)

    def test_provider_before_model(self):
        assert parse_model_flags("--provider foo model") == ("model", "foo", False)

    def test_global_then_provider(self):
        assert parse_model_flags("--global --provider foo model") == (
            "model", "foo", True,
        )

    def test_provider_then_global(self):
        assert parse_model_flags("--provider foo --global model") == (
            "model", "foo", True,
        )

    def test_flags_at_end(self):
        assert parse_model_flags("model --global --provider foo") == (
            "model", "foo", True,
        )

    def test_flags_at_beginning(self):
        assert parse_model_flags("--global --provider foo model") == (
            "model", "foo", True,
        )

    def test_global_between_provider_and_value(self):
        # --global is a standalone flag, not consumed as provider value
        result = parse_model_flags("--provider --global model")
        # --provider takes the next token ("--global") as its value
        assert result == ("model", "--global", False)


class TestParseModelFlagsEdgeCases:
    """Edge cases and potential confusion patterns."""

    def test_globalx_not_matched(self):
        # BUG FIX: --globalx must NOT be treated as --global
        assert parse_model_flags("model --globalx") == ("model --globalx", "", False)

    def test_global_setting_not_matched(self):
        # BUG FIX: --global-setting must NOT be treated as --global
        assert parse_model_flags("model --global-setting") == (
            "model --global-setting", "", False,
        )

    def test_double_global(self):
        assert parse_model_flags("model --global --global") == ("model", "", True)

    def test_double_provider(self):
        # Last --provider wins
        assert parse_model_flags("model --provider foo --provider bar") == (
            "model", "bar", False,
        )

    def test_dangling_provider(self):
        # --provider without a following value is left as-is
        assert parse_model_flags("model --provider") == ("model --provider", "", False)

    def test_dangling_provider_trailing_space(self):
        assert parse_model_flags("model --provider ") == ("model --provider", "", False)

    def test_model_with_dashes(self):
        assert parse_model_flags("claude-opus-4-6") == ("claude-opus-4-6", "", False)

    def test_model_with_dashes_and_global(self):
        assert parse_model_flags("claude-opus-4-6 --global") == (
            "claude-opus-4-6", "", True,
        )

    def test_provider_with_dashes(self):
        assert parse_model_flags("model --provider my-custom-llama") == (
            "model", "my-custom-llama", False,
        )

    def test_extra_whitespace(self):
        assert parse_model_flags("  sonnet  --global  ") == ("sonnet", "", True)

    def test_model_name_with_spaces_not_possible(self):
        # Since we split on whitespace, multi-word model names collapse
        assert parse_model_flags("my model name") == ("my model name", "", False)


class TestParseModelFlagsOpenRouterVariants:
    """Test OpenRouter-style variant suffixes in model names."""

    def test_free_suffix(self):
        assert parse_model_flags("deepseek/deepseek-r1:free") == (
            "deepseek/deepseek-r1:free", "", False,
        )

    def test_extended_suffix_with_global(self):
        assert parse_model_flags("deepseek/deepseek-r1:extended --global") == (
            "deepseek/deepseek-r1:extended", "", True,
        )

    def test_vendor_model_format(self):
        assert parse_model_flags("anthropic/claude-sonnet-4-6 --provider openrouter") == (
            "anthropic/claude-sonnet-4-6", "openrouter", False,
        )


# ---------------------------------------------------------------------------
# _check_hermes_model_warning
# ---------------------------------------------------------------------------

class TestCheckHermesModelWarning:
    """Tests for the Hermes model name warning detector."""

    def test_hermes_lowercase(self):
        result = _check_hermes_model_warning("hermes-3-llama")
        assert result != ""
        assert "NOT agentic" in result

    def test_hermes_mixed_case(self):
        result = _check_hermes_model_warning("Hermes-4")
        assert result != ""

    def test_hermes_uppercase(self):
        result = _check_hermes_model_warning("HERMES")
        assert result != ""

    def test_non_hermes_model(self):
        assert _check_hermes_model_warning("claude-opus-4-6") == ""

    def test_partial_match_in_word(self):
        # "hermes" is a substring of "thermes" -- the check is intentionally
        # broad (case-insensitive substring), so it DOES trigger.
        assert _check_hermes_model_warning("thermes") != ""

    def test_no_hermes_substring(self):
        assert _check_hermes_model_warning("llama-3.3-70b") == ""
        assert _check_hermes_model_warning("gpt-5") == ""
        assert _check_hermes_model_warning("deepseek-r1") == ""

    def test_empty_string(self):
        assert _check_hermes_model_warning("") == ""
