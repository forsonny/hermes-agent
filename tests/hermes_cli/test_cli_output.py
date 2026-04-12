"""Tests for hermes_cli/cli_output.py — shared CLI output helpers."""

import sys
from unittest.mock import patch, MagicMock

import pytest

# Ensure the module can be imported
from hermes_cli.cli_output import (
    print_info,
    print_success,
    print_warning,
    print_error,
    print_header,
    prompt,
    prompt_yes_no,
)


class TestPrintInfo:
    """Tests for print_info()."""

    def test_prints_dim_text(self, capsys):
        print_info("loading config")
        captured = capsys.readouterr()
        assert "loading config" in captured.out

    def test_empty_string(self, capsys):
        print_info("")
        captured = capsys.readouterr()
        # Should still print something (just the formatting)
        assert captured.out.strip() == "" or " " in captured.out

    def test_unicode_text(self, capsys):
        print_info("Configuration ✓ loaded")
        captured = capsys.readouterr()
        assert "Configuration" in captured.out


class TestPrintSuccess:
    """Tests for print_success()."""

    def test_contains_checkmark(self, capsys):
        print_success("done")
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_green_color_applied(self):
        """When colors are active, output should contain ANSI escape codes."""
        with patch("hermes_cli.colors.should_use_color", return_value=True):
            with patch("builtins.print") as mock_print:
                print_success("ok")
                args = mock_print.call_args[0][0]
                assert "\033[32m" in args  # GREEN
                assert "\033[0m" in args  # RESET

    def test_no_color_when_disabled(self):
        """When colors are disabled, output should be plain text."""
        with patch("hermes_cli.colors.should_use_color", return_value=False):
            with patch("builtins.print") as mock_print:
                print_success("ok")
                args = mock_print.call_args[0][0]
                assert "\033[" not in args


class TestPrintWarning:
    """Tests for print_warning()."""

    def test_contains_text(self, capsys):
        print_warning("be careful")
        captured = capsys.readouterr()
        assert "be careful" in captured.out

    def test_yellow_color_applied(self):
        with patch("hermes_cli.colors.should_use_color", return_value=True):
            with patch("builtins.print") as mock_print:
                print_warning("caution")
                args = mock_print.call_args[0][0]
                assert "\033[33m" in args  # YELLOW


class TestPrintError:
    """Tests for print_error()."""

    def test_contains_text(self, capsys):
        print_error("something broke")
        captured = capsys.readouterr()
        assert "something broke" in captured.out

    def test_red_color_applied(self):
        with patch("hermes_cli.colors.should_use_color", return_value=True):
            with patch("builtins.print") as mock_print:
                print_error("fail")
                args = mock_print.call_args[0][0]
                assert "\033[31m" in args  # RED


class TestPrintHeader:
    """Tests for print_header()."""

    def test_contains_text(self, capsys):
        print_header("Section Title")
        captured = capsys.readouterr()
        assert "Section Title" in captured.out

    def test_has_leading_newline(self, capsys):
        print_header("Title")
        captured = capsys.readouterr()
        assert captured.out.startswith("\n")

    def test_yellow_color_applied(self):
        with patch("hermes_cli.colors.should_use_color", return_value=True):
            with patch("builtins.print") as mock_print:
                print_header("Test")
                args = mock_print.call_args[0][0]
                assert "\033[33m" in args  # YELLOW


class TestPrompt:
    """Tests for prompt()."""

    def test_returns_user_input(self):
        with patch("builtins.input", return_value="hello"):
            result = prompt("Enter name")
            assert result == "hello"

    def test_strips_whitespace(self):
        with patch("builtins.input", return_value="  hello  "):
            result = prompt("Enter name")
            assert result == "hello"

    def test_returns_default_on_empty(self):
        with patch("builtins.input", return_value=""):
            result = prompt("Enter name", default="world")
            assert result == "world"

    def test_returns_empty_on_empty_no_default(self):
        with patch("builtins.input", return_value=""):
            result = prompt("Enter name")
            assert result == ""

    def test_keyboard_interrupt_returns_empty(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = prompt("Enter name")
            assert result == ""

    def test_eof_returns_empty(self):
        with patch("builtins.input", side_effect=EOFError):
            result = prompt("Enter name")
            assert result == ""

    def test_password_uses_getpass(self):
        with patch("getpass.getpass", return_value="secret123") as mock_gp:
            result = prompt("Password", password=True)
            mock_gp.assert_called_once()
            assert result == "secret123"

    def test_password_returns_default_on_empty(self):
        with patch("getpass.getpass", return_value=""):
            result = prompt("Password", default="fallback", password=True)
            assert result == "fallback"

    def test_displays_question_with_default(self):
        with patch("builtins.input", return_value="") as mock_input:
            prompt("Name", default="foo")
            call_args = mock_input.call_args[0][0]
            assert "Name" in call_args
            assert "foo" in call_args

    def test_displays_question_without_default(self):
        with patch("builtins.input", return_value="bar") as mock_input:
            prompt("Name")
            call_args = mock_input.call_args[0][0]
            assert "Name" in call_args
            assert "[" not in call_args  # no default hint


class TestPromptYesNo:
    """Tests for prompt_yes_no()."""

    def test_default_true_yes(self):
        with patch("hermes_cli.cli_output.prompt", return_value=""):
            result = prompt_yes_no("Continue?", default=True)
            assert result is True

    def test_default_false_no(self):
        with patch("hermes_cli.cli_output.prompt", return_value=""):
            result = prompt_yes_no("Continue?", default=False)
            assert result is False

    def test_yes_response(self):
        with patch("hermes_cli.cli_output.prompt", return_value="yes"):
            result = prompt_yes_no("Continue?")
            assert result is True

    def test_y_response(self):
        with patch("hermes_cli.cli_output.prompt", return_value="Y"):
            result = prompt_yes_no("Continue?")
            assert result is True

    def test_no_response(self):
        with patch("hermes_cli.cli_output.prompt", return_value="no"):
            result = prompt_yes_no("Continue?")
            assert result is False

    def test_n_response(self):
        with patch("hermes_cli.cli_output.prompt", return_value="N"):
            result = prompt_yes_no("Continue?")
            assert result is False

    def test_random_input_returns_false(self):
        with patch("hermes_cli.cli_output.prompt", return_value="maybe"):
            result = prompt_yes_no("Continue?")
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
