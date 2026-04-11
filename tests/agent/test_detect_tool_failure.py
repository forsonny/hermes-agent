#!/usr/bin/env python3
"""
Tests for _detect_tool_failure in agent/display.py.

Covers tool-specific failure detection for terminal, memory,
delegate_task, process, execute_code, and the generic heuristic.
"""

import json
import unittest

from agent.display import _detect_tool_failure


class TestDetectToolFailureTerminal(unittest.TestCase):
    """Terminal tool failure detection."""

    def test_nonzero_exit_code(self):
        result = json.dumps({"exit_code": 1, "output": "error"})
        is_fail, suffix = _detect_tool_failure("terminal", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [exit 1]")

    def test_zero_exit_code(self):
        result = json.dumps({"exit_code": 0, "output": "ok"})
        is_fail, suffix = _detect_tool_failure("terminal", result)
        self.assertFalse(is_fail)
        self.assertEqual(suffix, "")

    def test_missing_exit_code(self):
        result = json.dumps({"output": "ok"})
        is_fail, suffix = _detect_tool_failure("terminal", result)
        self.assertFalse(is_fail)

    def test_non_json_result(self):
        result = "not json"
        is_fail, suffix = _detect_tool_failure("terminal", result)
        self.assertFalse(is_fail)

    def test_none_result(self):
        is_fail, suffix = _detect_tool_failure("terminal", None)
        self.assertFalse(is_fail)


class TestDetectToolFailureMemory(unittest.TestCase):
    """Memory tool failure detection."""

    def test_memory_full(self):
        result = json.dumps({"success": False, "error": "exceed the limit of 50"})
        is_fail, suffix = _detect_tool_failure("memory", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [full]")

    def test_memory_success(self):
        result = json.dumps({"success": True})
        is_fail, suffix = _detect_tool_failure("memory", result)
        self.assertFalse(is_fail)


class TestDetectToolFailureDelegate(unittest.TestCase):
    """delegate_task failure detection."""

    def test_partial_failure(self):
        result = json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "done"},
                {"task_index": 1, "status": "error", "error": "timeout"},
            ]
        })
        is_fail, suffix = _detect_tool_failure("delegate_task", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [1/2 failed]")

    def test_all_success(self):
        result = json.dumps({
            "results": [
                {"task_index": 0, "status": "completed", "summary": "done"},
                {"task_index": 1, "status": "completed", "summary": "also done"},
            ]
        })
        is_fail, suffix = _detect_tool_failure("delegate_task", result)
        self.assertFalse(is_fail)
        self.assertEqual(suffix, "")

    def test_top_level_error(self):
        result = json.dumps({
            "error": "Delegation depth limit reached (2)."
        })
        is_fail, suffix = _detect_tool_failure("delegate_task", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [error]")

    def test_all_failed(self):
        result = json.dumps({
            "results": [
                {"task_index": 0, "status": "failed", "error": "bad"},
                {"task_index": 1, "status": "error", "error": "also bad"},
            ]
        })
        is_fail, suffix = _detect_tool_failure("delegate_task", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [2/2 failed]")

    def test_empty_results(self):
        result = json.dumps({"results": []})
        is_fail, suffix = _detect_tool_failure("delegate_task", result)
        self.assertFalse(is_fail)

    def test_non_json(self):
        is_fail, suffix = _detect_tool_failure("delegate_task", "not json")
        self.assertFalse(is_fail)

    def test_single_failure(self):
        result = json.dumps({
            "results": [
                {"task_index": 0, "status": "error", "error": "boom"},
            ]
        })
        is_fail, suffix = _detect_tool_failure("delegate_task", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [1/1 failed]")


class TestDetectToolFailureProcess(unittest.TestCase):
    """Process tool failure detection."""

    def test_killed(self):
        result = json.dumps({"status": "killed"})
        is_fail, suffix = _detect_tool_failure("process", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [killed]")

    def test_timeout(self):
        result = json.dumps({"status": "timeout"})
        is_fail, suffix = _detect_tool_failure("process", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [timeout]")

    def test_running(self):
        result = json.dumps({"status": "running"})
        is_fail, suffix = _detect_tool_failure("process", result)
        self.assertFalse(is_fail)

    def test_exited(self):
        result = json.dumps({"status": "exited", "exit_code": 0})
        is_fail, suffix = _detect_tool_failure("process", result)
        self.assertFalse(is_fail)


class TestDetectToolFailureExecuteCode(unittest.TestCase):
    """execute_code failure detection."""

    def test_nonzero_exit(self):
        result = json.dumps({"exit_code": 1, "output": "error", "error": "NameError"})
        is_fail, suffix = _detect_tool_failure("execute_code", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [exit 1]")

    def test_timeout(self):
        result = json.dumps({"exit_code": 124, "error": "Command timed out after 300s"})
        is_fail, suffix = _detect_tool_failure("execute_code", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [timeout]")

    def test_success(self):
        result = json.dumps({"exit_code": 0, "output": "OK"})
        is_fail, suffix = _detect_tool_failure("execute_code", result)
        self.assertFalse(is_fail)

    def test_missing_exit_code(self):
        result = json.dumps({"output": "no exit code"})
        is_fail, suffix = _detect_tool_failure("execute_code", result)
        self.assertFalse(is_fail)


class TestDetectToolFailureGeneric(unittest.TestCase):
    """Generic heuristic for unknown tools."""

    def test_error_in_json(self):
        result = json.dumps({"error": "something went wrong"})
        is_fail, suffix = _detect_tool_failure("unknown_tool", result)
        self.assertTrue(is_fail)
        self.assertEqual(suffix, " [error]")

    def test_failed_in_json(self):
        result = json.dumps({"status": "failed"})
        is_fail, suffix = _detect_tool_failure("unknown_tool", result)
        self.assertTrue(is_fail)

    def test_error_prefix(self):
        result = "Error: connection refused"
        is_fail, suffix = _detect_tool_failure("unknown_tool", result)
        self.assertTrue(is_fail)

    def test_success_no_error(self):
        result = json.dumps({"status": "ok", "data": [1, 2, 3]})
        is_fail, suffix = _detect_tool_failure("unknown_tool", result)
        self.assertFalse(is_fail)

    def test_none_result(self):
        is_fail, suffix = _detect_tool_failure("any_tool", None)
        self.assertFalse(is_fail)
        self.assertEqual(suffix, "")


if __name__ == "__main__":
    unittest.main()
