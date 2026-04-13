"""Tests for hermes_cli/pairing.py — DM pairing CLI commands."""

import pytest
from unittest.mock import MagicMock, patch

from hermes_cli.pairing import (
    _cmd_list,
    _cmd_approve,
    _cmd_revoke,
    _cmd_clear_pending,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_store():
    """Create a mock PairingStore with sensible defaults."""
    store = MagicMock()
    store.list_pending.return_value = []
    store.list_approved.return_value = []
    store.approve_code.return_value = None
    store.revoke.return_value = False
    store.clear_pending.return_value = 0
    return store


def _mock_args(action=None, platform=None, code=None, user_id=None):
    """Create a mock args namespace."""
    args = MagicMock()
    args.pairing_action = action
    args.platform = platform
    args.code = code
    args.user_id = user_id
    return args


# ---------------------------------------------------------------------------
# pairing_command — top-level dispatcher
# ---------------------------------------------------------------------------

class TestPairingCommand:
    """Tests for the pairing_command dispatcher."""

    def test_dispatches_list(self, capsys):
        store = _mock_store()
        args = _mock_args(action="list")
        with patch.dict("sys.modules", {"gateway.pairing": MagicMock(PairingStore=MagicMock(return_value=store))}):
            from hermes_cli.pairing import pairing_command
            pairing_command(args)
        out = capsys.readouterr().out
        assert "No pairing data found" in out

    def test_dispatches_approve(self, capsys):
        store = _mock_store()
        store.approve_code.return_value = None
        args = _mock_args(action="approve", platform="telegram", code="ABC123")
        with patch.dict("sys.modules", {"gateway.pairing": MagicMock(PairingStore=MagicMock(return_value=store))}):
            from hermes_cli.pairing import pairing_command
            pairing_command(args)
        out = capsys.readouterr().out
        assert "not found or expired" in out

    def test_dispatches_revoke(self, capsys):
        store = _mock_store()
        store.revoke.return_value = False
        args = _mock_args(action="revoke", platform="discord", user_id="123")
        with patch.dict("sys.modules", {"gateway.pairing": MagicMock(PairingStore=MagicMock(return_value=store))}):
            from hermes_cli.pairing import pairing_command
            pairing_command(args)
        out = capsys.readouterr().out
        assert "not found" in out

    def test_dispatches_clear_pending(self, capsys):
        store = _mock_store()
        store.clear_pending.return_value = 0
        args = _mock_args(action="clear-pending")
        with patch.dict("sys.modules", {"gateway.pairing": MagicMock(PairingStore=MagicMock(return_value=store))}):
            from hermes_cli.pairing import pairing_command
            pairing_command(args)
        out = capsys.readouterr().out
        assert "No pending requests" in out

    def test_unknown_action_prints_usage(self, capsys):
        store = _mock_store()
        args = _mock_args(action="bogus")
        with patch.dict("sys.modules", {"gateway.pairing": MagicMock(PairingStore=MagicMock(return_value=store))}):
            from hermes_cli.pairing import pairing_command
            pairing_command(args)
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_none_action_prints_usage(self, capsys):
        store = _mock_store()
        args = _mock_args(action=None)
        with patch.dict("sys.modules", {"gateway.pairing": MagicMock(PairingStore=MagicMock(return_value=store))}):
            from hermes_cli.pairing import pairing_command
            pairing_command(args)
        out = capsys.readouterr().out
        assert "Usage:" in out


# ---------------------------------------------------------------------------
# _cmd_list
# ---------------------------------------------------------------------------

class TestCmdList:
    """Tests for _cmd_list."""

    def test_empty_store(self, capsys):
        store = _mock_store()
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "No pairing data found" in out

    def test_pending_only(self, capsys):
        store = _mock_store()
        store.list_pending.return_value = [
            {"platform": "telegram", "code": "ABCD", "user_id": "u1", "user_name": "Alice", "age_minutes": 5},
        ]
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "Pending Pairing Requests (1)" in out
        assert "telegram" in out
        assert "ABCD" in out
        assert "Alice" in out
        assert "No approved users" in out

    def test_approved_only(self, capsys):
        store = _mock_store()
        store.list_approved.return_value = [
            {"platform": "discord", "user_id": "u2", "user_name": "Bob"},
        ]
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "No pending pairing requests" in out
        assert "Approved Users (1)" in out
        assert "discord" in out
        assert "Bob" in out

    def test_both_pending_and_approved(self, capsys):
        store = _mock_store()
        store.list_pending.return_value = [
            {"platform": "slack", "code": "XYZ1", "user_id": "u3", "user_name": "Charlie", "age_minutes": 2},
        ]
        store.list_approved.return_value = [
            {"platform": "telegram", "user_id": "u4", "user_name": "Diana"},
        ]
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "Pending Pairing Requests (1)" in out
        assert "Approved Users (1)" in out

    def test_pending_without_user_name(self, capsys):
        store = _mock_store()
        store.list_pending.return_value = [
            {"platform": "signal", "code": "ZZ99", "user_id": "u5", "age_minutes": 10},
        ]
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "signal" in out
        assert "ZZ99" in out

    def test_approved_without_user_name(self, capsys):
        store = _mock_store()
        store.list_approved.return_value = [
            {"platform": "matrix", "user_id": "u6"},
        ]
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "matrix" in out
        assert "u6" in out

    def test_multiple_pending_entries(self, capsys):
        store = _mock_store()
        store.list_pending.return_value = [
            {"platform": "telegram", "code": "AA11", "user_id": "u10", "user_name": "Eve", "age_minutes": 1},
            {"platform": "discord", "code": "BB22", "user_id": "u11", "user_name": "Frank", "age_minutes": 3},
        ]
        _cmd_list(store)
        out = capsys.readouterr().out
        assert "Pending Pairing Requests (2)" in out
        assert "Eve" in out
        assert "Frank" in out


# ---------------------------------------------------------------------------
# _cmd_approve
# ---------------------------------------------------------------------------

class TestCmdApprove:
    """Tests for _cmd_approve."""

    def test_successful_approval(self, capsys):
        store = _mock_store()
        store.approve_code.return_value = {"user_id": "u1", "user_name": "Alice"}
        _cmd_approve(store, "telegram", "abcd")
        out = capsys.readouterr().out
        assert "Approved!" in out
        assert "Alice" in out
        store.approve_code.assert_called_once_with("telegram", "ABCD")

    def test_successful_approval_without_name(self, capsys):
        store = _mock_store()
        store.approve_code.return_value = {"user_id": "u1"}
        _cmd_approve(store, "discord", "xyz1")
        out = capsys.readouterr().out
        assert "Approved!" in out
        assert "u1" in out

    def test_code_not_found(self, capsys):
        store = _mock_store()
        store.approve_code.return_value = None
        _cmd_approve(store, "slack", "nope")
        out = capsys.readouterr().out
        assert "not found or expired" in out
        assert "NOPE" in out

    def test_normalizes_platform_to_lowercase(self, capsys):
        store = _mock_store()
        store.approve_code.return_value = {"user_id": "u2"}
        _cmd_approve(store, "  TELEGRAM  ", "abc123")
        store.approve_code.assert_called_once_with("telegram", "ABC123")

    def test_normalizes_code_to_uppercase(self, capsys):
        store = _mock_store()
        store.approve_code.return_value = {"user_id": "u3"}
        _cmd_approve(store, "discord", "  xyz12  ")
        store.approve_code.assert_called_once_with("discord", "XYZ12")


# ---------------------------------------------------------------------------
# _cmd_revoke
# ---------------------------------------------------------------------------

class TestCmdRevoke:
    """Tests for _cmd_revoke."""

    def test_successful_revoke(self, capsys):
        store = _mock_store()
        store.revoke.return_value = True
        _cmd_revoke(store, "telegram", "u1")
        out = capsys.readouterr().out
        assert "Revoked access" in out
        assert "u1" in out

    def test_user_not_found(self, capsys):
        store = _mock_store()
        store.revoke.return_value = False
        _cmd_revoke(store, "discord", "nonexistent")
        out = capsys.readouterr().out
        assert "not found" in out
        assert "nonexistent" in out

    def test_normalizes_platform(self, capsys):
        store = _mock_store()
        store.revoke.return_value = True
        _cmd_revoke(store, "  TELEGRAM  ", "u1")
        store.revoke.assert_called_once_with("telegram", "u1")


# ---------------------------------------------------------------------------
# _cmd_clear_pending
# ---------------------------------------------------------------------------

class TestCmdClearPending:
    """Tests for _cmd_clear_pending."""

    def test_cleared_some(self, capsys):
        store = _mock_store()
        store.clear_pending.return_value = 3
        _cmd_clear_pending(store)
        out = capsys.readouterr().out
        assert "Cleared 3 pending" in out

    def test_none_to_clear(self, capsys):
        store = _mock_store()
        store.clear_pending.return_value = 0
        _cmd_clear_pending(store)
        out = capsys.readouterr().out
        assert "No pending requests" in out

    def test_cleared_one(self, capsys):
        store = _mock_store()
        store.clear_pending.return_value = 1
        _cmd_clear_pending(store)
        out = capsys.readouterr().out
        assert "Cleared 1 pending" in out
