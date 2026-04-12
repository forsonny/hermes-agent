"""Tests for hermes_cli/colors.py -- ANSI color utilities."""

import os
import sys

import pytest


class TestShouldUseColor:
    """Tests for should_use_color()."""

    def test_returns_false_when_no_color_set(self, monkeypatch):
        """NO_COLOR env var disables colors."""
        monkeypatch.setenv("NO_COLOR", "1")
        # Need to also handle isatty since we patch stdout
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert mod.should_use_color() is False

    def test_returns_false_when_no_color_empty(self, monkeypatch):
        """NO_COLOR set to empty string still disables (per no-color.org spec)."""
        monkeypatch.setenv("NO_COLOR", "")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert mod.should_use_color() is False

    def test_returns_false_when_term_dumb(self, monkeypatch):
        """TERM=dumb disables colors."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert mod.should_use_color() is False

    def test_returns_false_when_not_tty(self, monkeypatch):
        """Non-TTY stdout disables colors."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        assert mod.should_use_color() is False

    def test_returns_true_when_tty_and_no_restrictions(self, monkeypatch):
        """TTY with no NO_COLOR and TERM != dumb enables colors."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert mod.should_use_color() is True

    def test_returns_false_when_no_color_overrides_tty(self, monkeypatch):
        """NO_COLOR takes precedence over TTY check."""
        monkeypatch.setenv("NO_COLOR", "1")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert mod.should_use_color() is False

    def test_returns_false_when_term_dumb_overrides_tty(self, monkeypatch):
        """TERM=dumb takes precedence over TTY check."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert mod.should_use_color() is False

    def test_no_color_checked_before_term(self, monkeypatch):
        """NO_COLOR is checked first (short-circuits)."""
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", "dumb")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        # Both would return False, just verify it works
        assert mod.should_use_color() is False


class TestColors:
    """Tests for the Colors class constants."""

    def test_reset_is_ansi_reset(self):
        from hermes_cli.colors import Colors
        assert Colors.RESET == "\033[0m"

    def test_bold_is_ansi_bold(self):
        from hermes_cli.colors import Colors
        assert Colors.BOLD == "\033[1m"

    def test_dim_is_ansi_dim(self):
        from hermes_cli.colors import Colors
        assert Colors.DIM == "\033[2m"

    def test_red_code(self):
        from hermes_cli.colors import Colors
        assert Colors.RED == "\033[31m"

    def test_green_code(self):
        from hermes_cli.colors import Colors
        assert Colors.GREEN == "\033[32m"

    def test_yellow_code(self):
        from hermes_cli.colors import Colors
        assert Colors.YELLOW == "\033[33m"

    def test_blue_code(self):
        from hermes_cli.colors import Colors
        assert Colors.BLUE == "\033[34m"

    def test_magenta_code(self):
        from hermes_cli.colors import Colors
        assert Colors.MAGENTA == "\033[35m"

    def test_cyan_code(self):
        from hermes_cli.colors import Colors
        assert Colors.CYAN == "\033[36m"

    def test_all_codes_start_with_escape(self):
        """All color constants start with ESC character."""
        from hermes_cli.colors import Colors
        for attr in ("RESET", "BOLD", "DIM", "RED", "GREEN", "YELLOW",
                     "BLUE", "MAGENTA", "CYAN"):
            val = getattr(Colors, attr)
            assert val.startswith("\033["), f"{attr} doesn't start with ESC["


class TestColorFunction:
    """Tests for the color() helper function."""

    def test_returns_plain_text_when_no_color(self, monkeypatch):
        """When colors disabled, returns text unchanged."""
        monkeypatch.setenv("NO_COLOR", "1")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("hello", mod.Colors.RED)
        assert result == "hello"

    def test_wraps_text_with_color_codes(self, monkeypatch):
        """When colors enabled, wraps text with codes and RESET."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("hello", mod.Colors.RED)
        assert result == "\033[31mhello\033[0m"

    def test_combines_multiple_codes(self, monkeypatch):
        """Multiple codes are joined before the text."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("hello", mod.Colors.BOLD, mod.Colors.RED)
        assert result == "\033[1m\033[31mhello\033[0m"

    def test_empty_text(self, monkeypatch):
        """Empty string returns just the codes."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("", mod.Colors.GREEN)
        assert result == "\033[32m\033[0m"

    def test_empty_text_no_color(self, monkeypatch):
        """Empty string with colors disabled returns empty string."""
        monkeypatch.setenv("NO_COLOR", "1")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("", mod.Colors.GREEN)
        assert result == ""

    def test_no_codes_passed(self, monkeypatch):
        """No color codes returns text with just RESET appended."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("hello")
        assert result == "hello\033[0m"

    def test_no_codes_no_color(self, monkeypatch):
        """No codes with colors disabled returns plain text."""
        monkeypatch.setenv("NO_COLOR", "1")
        import hermes_cli.colors as mod
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = mod.color("hello")
        assert result == "hello"
