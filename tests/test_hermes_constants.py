"""Tests for hermes_constants module."""

import os
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_constants
from hermes_constants import apply_ipv4_preference, get_default_hermes_root, is_container


class TestGetDefaultHermesRoot:
    """Tests for get_default_hermes_root() -- Docker/custom deployment awareness."""

    def test_no_hermes_home_returns_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME is not set, returns ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_default_hermes_root() == tmp_path / ".hermes"

    def test_hermes_home_is_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME = ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        native.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(native))
        assert get_default_hermes_root() == native

    def test_hermes_home_is_profile(self, tmp_path, monkeypatch):
        """When HERMES_HOME is a profile under ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        profile = native / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == native

    def test_hermes_home_is_docker(self, tmp_path, monkeypatch):
        """When HERMES_HOME points outside ~/.hermes (Docker), returns HERMES_HOME."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        assert get_default_hermes_root() == docker_home

    def test_hermes_home_is_custom_path(self, tmp_path, monkeypatch):
        """Any HERMES_HOME outside ~/.hermes is treated as the root."""
        custom = tmp_path / "my-hermes-data"
        custom.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert get_default_hermes_root() == custom

    def test_docker_profile_active(self, tmp_path, monkeypatch):
        """When a Docker profile is active (HERMES_HOME=<root>/profiles/<name>),
        returns the Docker root, not the profile dir."""
        docker_root = tmp_path / "opt" / "data"
        profile = docker_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == docker_root

import socket

from hermes_constants import apply_ipv4_preference


class TestApplyIpv4Preference:
    """Tests for apply_ipv4_preference() -- IPv4 network preference."""

    def test_no_op_when_force_false(self):
        """When force=False, socket.getaddrinfo is NOT patched."""
        original = socket.getaddrinfo
        apply_ipv4_preference(force=False)
        assert socket.getaddrinfo is original

    def test_no_op_when_force_false_explicit(self):
        """Explicit False does not patch."""
        original = socket.getaddrinfo
        apply_ipv4_preference(False)
        assert socket.getaddrinfo is original

    def test_patches_when_force_true(self):
        """When force=True, socket.getaddrinfo gets replaced."""
        original = socket.getaddrinfo
        try:
            apply_ipv4_preference(force=True)
            assert socket.getaddrinfo is not original
            assert getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False)
        finally:
            socket.getaddrinfo = original

    def test_idempotent_double_call(self):
        """Calling twice with force=True only patches once."""
        original = socket.getaddrinfo
        try:
            apply_ipv4_preference(force=True)
            first_patch = socket.getaddrinfo
            apply_ipv4_preference(force=True)
            assert socket.getaddrinfo is first_patch
        finally:
            socket.getaddrinfo = original

    def test_patched_version_prefers_ipv4(self):
        """The patched getaddrinfo calls with AF_INET when family=0 (AF_UNSPEC)."""
        original = socket.getaddrinfo
        calls = []

        def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append((host, port, family))
            return [(family, 1, 0, "", ("1.2.3.4", port))]

        try:
            socket.getaddrinfo = _fake_getaddrinfo
            apply_ipv4_preference(force=True)
            # Clear calls from the guard check
            calls.clear()
            # Make a call with family=0 (AF_UNSPEC)
            result = socket.getaddrinfo("example.com", 443)
            # Should have been called with AF_INET (2) instead of AF_UNSPEC (0)
            assert len(calls) >= 1
            assert calls[0][2] == socket.AF_INET  # 2
        finally:
            socket.getaddrinfo = original

    def test_patched_version_falls_back_on_gaierror(self):
        """When AF_INET fails (gaierror), falls back to original resolution."""
        original = socket.getaddrinfo
        calls = []

        def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append((host, port, family))
            if family == socket.AF_INET:
                raise socket.gaierror("No IPv4")
            return [(family, 1, 0, "", ("::1", port))]

        try:
            socket.getaddrinfo = _fake_getaddrinfo
            apply_ipv4_preference(force=True)
            calls.clear()
            result = socket.getaddrinfo("ipv6-only.example.com", 443)
            # Should have tried AF_INET first, then fallen back to AF_UNSPEC
            assert len(calls) == 2
            assert calls[0][2] == socket.AF_INET
            assert calls[1][2] == 0  # AF_UNSPEC fallback
        finally:
            socket.getaddrinfo = original

    def test_patched_version_passes_through_specific_family(self):
        """When family is explicitly set (not AF_UNSPEC), passes through unchanged."""
        original = socket.getaddrinfo
        calls = []

        def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append((host, port, family))
            return [(family, 1, 0, "", (host, port))]

        try:
            socket.getaddrinfo = _fake_getaddrinfo
            apply_ipv4_preference(force=True)
            calls.clear()
            # Call with AF_INET6 (explicit family)
            result = socket.getaddrinfo("example.com", 443, family=socket.AF_INET6)
            assert len(calls) == 1
            assert calls[0][2] == socket.AF_INET6  # Passed through unchanged
        finally:
            socket.getaddrinfo = original

    def test_patched_version_preserves_extra_args(self):
        """type, proto, flags are passed through to the original."""
        original = socket.getaddrinfo
        captured = {}

        def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            captured.update(host=host, port=port, type=type, proto=proto, flags=flags)
            return [(family, type, proto, "", (host, port))]

        try:
            socket.getaddrinfo = _fake_getaddrinfo
            apply_ipv4_preference(force=True)
            socket.getaddrinfo("test.com", 8080, type=socket.SOCK_STREAM, proto=6, flags=socket.AI_PASSIVE)
            assert captured["host"] == "test.com"
            assert captured["port"] == 8080
            assert captured["type"] == socket.SOCK_STREAM
            assert captured["proto"] == 6
            assert captured["flags"] == socket.AI_PASSIVE
        finally:
            socket.getaddrinfo = original

class TestIsContainer:
    """Tests for is_container() — Docker/Podman detection."""

    def _reset_cache(self, monkeypatch):
        """Reset the cached detection result before each test."""
        monkeypatch.setattr(hermes_constants, "_container_detected", None)

    def test_detects_dockerenv(self, monkeypatch, tmp_path):
        """/.dockerenv triggers container detection."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv")
        assert is_container() is True

    def test_detects_containerenv(self, monkeypatch, tmp_path):
        """/run/.containerenv triggers container detection (Podman)."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/run/.containerenv")
        assert is_container() is True

    def test_detects_cgroup_docker(self, monkeypatch, tmp_path):
        """/proc/1/cgroup containing 'docker' triggers detection."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/docker/abc123\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is True

    def test_negative_case(self, monkeypatch, tmp_path):
        """Returns False on a regular Linux host."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is False

    def test_caches_result(self, monkeypatch):
        """Second call uses cached value without re-probing."""
        monkeypatch.setattr(hermes_constants, "_container_detected", True)
        assert is_container() is True
        # Even if we make os.path.exists return False, cached value wins
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        assert is_container() is True
