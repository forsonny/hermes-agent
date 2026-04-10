"""Tests for agent.skill_utils — skill metadata, frontmatter parsing, platform matching,
disabled skills, external dirs, config vars, and file iteration."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.skill_utils as su
from agent.skill_utils import (
    SKILL_CONFIG_PREFIX,
    _normalize_string_set,
    _resolve_dotpath,
    discover_all_skill_config_vars,
    extract_skill_conditions,
    extract_skill_config_vars,
    extract_skill_description,
    get_all_skills_dirs,
    get_disabled_skill_names,
    get_external_skills_dirs,
    iter_skill_index_files,
    parse_frontmatter,
    resolve_skill_config_values,
    skill_matches_platform,
    yaml_load,
)


# ── yaml_load ──────────────────────────────────────────────────────────────


class TestYamlLoad:
    def test_simple_dict(self):
        result = yaml_load("key: value")
        assert result == {"key": "value"}

    def test_list(self):
        result = yaml_load("- a\n- b")
        assert result == ["a", "b"]

    def test_nested(self):
        result = yaml_load("outer:\n  inner: 42")
        assert result == {"outer": {"inner": 42}}

    def test_empty_string(self):
        result = yaml_load("")
        assert result is None

    def test_invalid_yaml_returns_string(self):
        # pyyaml may parse some things as strings
        result = yaml_load("just a string")
        assert result == "just a string"

    def test_lazy_load_caches(self):
        """Calling yaml_load twice should use cached loader."""
        r1 = yaml_load("x: 1")
        r2 = yaml_load("x: 2")
        assert r1 == {"x": 1}
        assert r2 == {"x": 2}


# ── parse_frontmatter ──────────────────────────────────────────────────────


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        fm, body = parse_frontmatter("Just markdown content")
        assert fm == {}
        assert body == "Just markdown content"

    def test_simple_frontmatter(self):
        content = "---\nname: myskill\ndescription: A skill\n---\nBody text"
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "myskill"
        assert fm["description"] == "A skill"
        assert "Body text" in body

    def test_frontmatter_with_lists(self):
        content = "---\nname: test\nplatforms:\n  - macos\n  - linux\n---\nbody"
        fm, body = parse_frontmatter(content)
        assert fm["platforms"] == ["macos", "linux"]

    def test_frontmatter_with_nested_metadata(self):
        content = "---\nname: test\nmetadata:\n  hermes:\n    requires_toolsets:\n      - browser\n---\nbody"
        fm, body = parse_frontmatter(content)
        assert fm["metadata"]["hermes"]["requires_toolsets"] == ["browser"]

    def test_unclosed_frontmatter(self):
        """Missing closing --- should return empty frontmatter."""
        content = "---\nname: test\n"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_malformed_yaml_fallback(self):
        """Malformed YAML should fall back to simple key:value parsing."""
        content = "---\nkey with spaces: value\nunparseable [\n---\nbody"
        fm, body = parse_frontmatter(content)
        # Should at least get the parseable line via fallback
        assert "key with spaces" in fm

    def test_empty_frontmatter(self):
        content = "---\n---\nbody text"
        fm, body = parse_frontmatter(content)
        assert fm == {}

    def test_frontmatter_with_multiline_body(self):
        content = "---\nname: x\n---\nLine 1\nLine 2\nLine 3"
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "x"
        assert "Line 1" in body
        assert "Line 3" in body


# ── skill_matches_platform ──────────────────────────────────────────────────


class TestSkillMatchesPlatform:
    def test_no_platforms_field(self):
        assert skill_matches_platform({}) is True

    def test_empty_platforms_list(self):
        assert skill_matches_platform({"platforms": []}) is True

    def test_current_platform_linux(self):
        with patch.object(sys, "platform", "linux"):
            assert skill_matches_platform({"platforms": ["linux"]}) is True

    def test_current_platform_darwin(self):
        with patch.object(sys, "platform", "darwin"):
            assert skill_matches_platform({"platforms": ["macos"]}) is True

    def test_current_platform_win32(self):
        with patch.object(sys, "platform", "win32"):
            assert skill_matches_platform({"platforms": ["windows"]}) is True

    def test_wrong_platform(self):
        with patch.object(sys, "platform", "linux"):
            assert skill_matches_platform({"platforms": ["macos"]}) is False

    def test_multiple_platforms(self):
        with patch.object(sys, "platform", "linux"):
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is True

    def test_string_instead_of_list(self):
        """Single string platform should be normalized to list."""
        with patch.object(sys, "platform", "linux"):
            assert skill_matches_platform({"platforms": "linux"}) is True

    def test_case_insensitive(self):
        with patch.object(sys, "platform", "linux"):
            assert skill_matches_platform({"platforms": ["Linux"]}) is True
            assert skill_matches_platform({"platforms": ["LINUX"]}) is True

    def test_whitespace_trimmed(self):
        with patch.object(sys, "platform", "linux"):
            assert skill_matches_platform({"platforms": [" linux "]}) is True


# ── _normalize_string_set ───────────────────────────────────────────────────


class TestNormalizeStringSet:
    def test_none(self):
        assert _normalize_string_set(None) == set()

    def test_list(self):
        assert _normalize_string_set(["a", "b", "c"]) == {"a", "b", "c"}

    def test_string(self):
        assert _normalize_string_set("single") == {"single"}

    def test_whitespace_items(self):
        assert _normalize_string_set([" a ", " b ", ""]) == {"a", "b"}

    def test_empty_list(self):
        assert _normalize_string_set([]) == set()

    def test_non_string_values(self):
        assert _normalize_string_set([1, 2, 3]) == {"1", "2", "3"}


# ── get_disabled_skill_names ───────────────────────────────────────────────


class TestGetDisabledSkillNames:
    def test_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert get_disabled_skill_names() == set()

    def test_global_disabled(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  disabled:\n    - skill_a\n    - skill_b\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        assert get_disabled_skill_names() == {"skill_a", "skill_b"}

    def test_platform_disabled(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text(
            "skills:\n  disabled:\n    - global_skill\n  platform_disabled:\n    telegram:\n      - tg_skill\n"
        )
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_PLATFORM", "telegram")
        assert get_disabled_skill_names() == {"tg_skill"}

    def test_explicit_platform_param(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text(
            "skills:\n  platform_disabled:\n    discord:\n      - disc_skill\n"
        )
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert get_disabled_skill_names("discord") == {"disc_skill"}

    def test_malformed_config_returns_empty(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("{{invalid yaml")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        # Should not crash, return empty
        assert get_disabled_skill_names() == set()


# ── get_external_skills_dirs ───────────────────────────────────────────────


class TestGetExternalSkillsDirs:
    def test_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert get_external_skills_dirs() == []

    def test_no_external_dirs_key(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  disabled: []\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert get_external_skills_dirs() == []

    def test_valid_dirs(self, tmp_path, monkeypatch):
        ext_dir = tmp_path / "external_skills"
        ext_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  external_dirs:\n    - " + str(ext_dir) + "\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == ext_dir.resolve()

    def test_nonexistent_dir_skipped(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  external_dirs:\n    - " + str(tmp_path) + "/nonexistent\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert get_external_skills_dirs() == []

    def test_duplicates_deduplicated(self, tmp_path, monkeypatch):
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            "skills:\n  external_dirs:\n    - " + str(ext_dir) + "\n    - " + str(ext_dir) + "\n"
        )
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert len(get_external_skills_dirs()) == 1

    def test_local_skills_dir_excluded(self, tmp_path, monkeypatch):
        local_skills = tmp_path / "skills"
        local_skills.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  external_dirs:\n    - " + str(local_skills) + "\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        # local skills dir should be skipped
        assert get_external_skills_dirs() == []

    def test_string_entry(self, tmp_path, monkeypatch):
        ext_dir = tmp_path / "single_skill"
        ext_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  external_dirs: " + str(ext_dir) + "\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        assert len(get_external_skills_dirs()) == 1


# ── get_all_skills_dirs ─────────────────────────────────────────────────────


class TestGetAllSkillsDirs:
    def test_local_always_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(su, "get_external_skills_dirs", lambda: [])
        dirs = get_all_skills_dirs()
        assert dirs[0] == tmp_path / "skills"

    def test_external_appended(self, tmp_path, monkeypatch):
        ext = tmp_path / "ext"
        ext.mkdir()
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(su, "get_external_skills_dirs", lambda: [ext])
        dirs = get_all_skills_dirs()
        assert len(dirs) == 2
        assert dirs[1] == ext


# ── extract_skill_conditions ───────────────────────────────────────────────


class TestExtractSkillConditions:
    def test_empty_frontmatter(self):
        result = extract_skill_conditions({})
        assert result == {
            "fallback_for_toolsets": [],
            "requires_toolsets": [],
            "fallback_for_tools": [],
            "requires_tools": [],
        }

    def test_no_metadata(self):
        result = extract_skill_conditions({"name": "test"})
        assert result["requires_toolsets"] == []

    def test_metadata_not_dict(self):
        result = extract_skill_conditions({"metadata": "not a dict"})
        assert result["requires_toolsets"] == []

    def test_hermes_not_dict(self):
        result = extract_skill_conditions({"metadata": {"hermes": "string"}})
        assert result["requires_toolsets"] == []

    def test_full_conditions(self):
        fm = {
            "metadata": {
                "hermes": {
                    "fallback_for_toolsets": ["browser"],
                    "requires_toolsets": ["terminal"],
                    "fallback_for_tools": ["web_search"],
                    "requires_tools": ["execute_code"],
                }
            }
        }
        result = extract_skill_conditions(fm)
        assert result["fallback_for_toolsets"] == ["browser"]
        assert result["requires_toolsets"] == ["terminal"]
        assert result["fallback_for_tools"] == ["web_search"]
        assert result["requires_tools"] == ["execute_code"]

    def test_partial_conditions(self):
        fm = {"metadata": {"hermes": {"requires_toolsets": ["browser"]}}}
        result = extract_skill_conditions(fm)
        assert result["requires_toolsets"] == ["browser"]
        assert result["fallback_for_toolsets"] == []


# ── extract_skill_config_vars ───────────────────────────────────────────────


class TestExtractSkillConfigVars:
    def test_no_metadata(self):
        assert extract_skill_config_vars({}) == []

    def test_no_hermes(self):
        assert extract_skill_config_vars({"metadata": {}}) == []

    def test_no_config(self):
        assert extract_skill_config_vars({"metadata": {"hermes": {}}}) == []

    def test_single_config_var(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": [
                        {
                            "key": "wiki.path",
                            "description": "Wiki directory",
                            "default": "~/wiki",
                        }
                    ]
                }
            }
        }
        result = extract_skill_config_vars(fm)
        assert len(result) == 1
        assert result[0]["key"] == "wiki.path"
        assert result[0]["description"] == "Wiki directory"
        assert result[0]["default"] == "~/wiki"
        assert result[0]["prompt"] == "Wiki directory"

    def test_custom_prompt(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": [
                        {
                            "key": "test.key",
                            "description": "A test key",
                            "prompt": "Enter value",
                        }
                    ]
                }
            }
        }
        result = extract_skill_config_vars(fm)
        assert result[0]["prompt"] == "Enter value"

    def test_missing_description_skipped(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": [{"key": "test.key"}]
                }
            }
        }
        assert extract_skill_config_vars(fm) == []

    def test_missing_key_skipped(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": [{"description": "no key"}]
                }
            }
        }
        assert extract_skill_config_vars(fm) == []

    def test_duplicate_keys_deduplicated(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": [
                        {"key": "dup.key", "description": "first"},
                        {"key": "dup.key", "description": "second"},
                    ]
                }
            }
        }
        result = extract_skill_config_vars(fm)
        assert len(result) == 1
        assert result[0]["description"] == "first"

    def test_dict_config_wrapped_in_list(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": {"key": "test", "description": "desc"}
                }
            }
        }
        result = extract_skill_config_vars(fm)
        assert len(result) == 1
        assert result[0]["key"] == "test"

    def test_non_dict_items_skipped(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": ["string_item", 42, {"key": "valid", "description": "ok"}]
                }
            }
        }
        result = extract_skill_config_vars(fm)
        assert len(result) == 1
        assert result[0]["key"] == "valid"

    def test_no_default(self):
        fm = {
            "metadata": {
                "hermes": {
                    "config": [{"key": "k", "description": "d"}]
                }
            }
        }
        result = extract_skill_config_vars(fm)
        assert "default" not in result[0]


# ── _resolve_dotpath ───────────────────────────────────────────────────────


class TestResolveDotpath:
    def test_simple_key(self):
        assert _resolve_dotpath({"a": 1}, "a") == 1

    def test_nested_key(self):
        assert _resolve_dotpath({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_missing_key(self):
        assert _resolve_dotpath({"a": 1}, "b") is None

    def test_missing_nested_key(self):
        assert _resolve_dotpath({"a": {"b": 1}}, "a.c") is None

    def test_non_dict_intermediate(self):
        assert _resolve_dotpath({"a": "string"}, "a.b") is None

    def test_list_intermediate(self):
        data = {"a": [1, 2, 3]}
        # Lists don't support key access
        assert _resolve_dotpath(data, "a.0") is None


# ── resolve_skill_config_values ─────────────────────────────────────────────


class TestResolveSkillConfigValues:
    def test_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        config_vars = [{"key": "test.path", "description": "Test", "default": "/default"}]
        result = resolve_skill_config_values(config_vars)
        assert result["test.path"] == "/default"

    def test_value_from_config(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  config:\n    wiki:\n      path: /custom/wiki\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        config_vars = [
            {"key": "wiki.path", "description": "Wiki path", "default": "~/wiki"}
        ]
        result = resolve_skill_config_values(config_vars)
        assert result["wiki.path"] == "/custom/wiki"

    def test_fallback_to_default(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  config:\n    other: value\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        config_vars = [
            {"key": "missing.key", "description": "Missing", "default": "fallback"}
        ]
        result = resolve_skill_config_values(config_vars)
        assert result["missing.key"] == "fallback"

    def test_empty_string_uses_default(self, tmp_path, monkeypatch):
        config = tmp_path / "config.yaml"
        config.write_text("skills:\n  config:\n    empty_key: ''\n")
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        config_vars = [
            {"key": "empty_key", "description": "Empty", "default": "default_val"}
        ]
        result = resolve_skill_config_values(config_vars)
        assert result["empty_key"] == "default_val"


# ── extract_skill_description ───────────────────────────────────────────────


class TestExtractSkillDescription:
    def test_normal_description(self):
        assert extract_skill_description({"description": "A cool skill"}) == "A cool skill"

    def test_truncated_description(self):
        long_desc = "A" * 100
        result = extract_skill_description({"description": long_desc})
        assert len(result) == 60
        assert result.endswith("...")

    def test_empty_description(self):
        assert extract_skill_description({"description": ""}) == ""

    def test_missing_description(self):
        assert extract_skill_description({}) == ""

    def test_quoted_description(self):
        assert extract_skill_description({"description": '"quoted"'}) == "quoted"

    def test_exactly_60_chars(self):
        desc = "A" * 60
        assert extract_skill_description({"description": desc}) == desc

    def test_61_chars_truncated(self):
        desc = "A" * 61
        result = extract_skill_description({"description": desc})
        assert len(result) == 60
        assert result.endswith("...")


# ── iter_skill_index_files ─────────────────────────────────────────────────


class TestIterSkillIndexFiles:
    def test_empty_dir(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        assert list(iter_skill_index_files(skills_dir, "SKILL.md")) == []

    def test_finds_files(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_a = skills_dir / "skill_a"
        skill_a.mkdir(parents=True)
        (skill_a / "SKILL.md").write_text("---\nname: a\n---\nbody")
        result = list(iter_skill_index_files(skills_dir, "SKILL.md"))
        assert len(result) == 1
        assert result[0].name == "SKILL.md"

    def test_excluded_dirs(self, tmp_path):
        skills_dir = tmp_path / "skills"
        git_dir = skills_dir / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "SKILL.md").write_text("should not match")
        github_dir = skills_dir / ".github"
        github_dir.mkdir(parents=True)
        (github_dir / "SKILL.md").write_text("should not match")
        assert list(iter_skill_index_files(skills_dir, "SKILL.md")) == []

    def test_sorted_output(self, tmp_path):
        skills_dir = tmp_path / "skills"
        for name in ("c_skill", "a_skill", "b_skill"):
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text("---\nname: " + name + "\n---\nbody")
        result = list(iter_skill_index_files(skills_dir, "SKILL.md"))
        names = [p.parent.name for p in result]
        assert names == sorted(names)

    def test_different_filename(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "my_skill"
        d.mkdir(parents=True)
        (d / "README.md").write_text("readme")
        result = list(iter_skill_index_files(skills_dir, "README.md"))
        assert len(result) == 1

    def test_nested_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "category" / "my_skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: nested\n---\nbody")
        result = list(iter_skill_index_files(skills_dir, "SKILL.md"))
        assert len(result) == 1


# ── discover_all_skill_config_vars ──────────────────────────────────────────


class TestDiscoverAllSkillConfigVars:
    def test_empty_skills_dir(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(su, "get_disabled_skill_names", lambda: set())
        monkeypatch.setattr(su, "get_all_skills_dirs", lambda: [skills_dir])
        assert discover_all_skill_config_vars() == []

    def test_discovers_config_vars(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skill_d = skills_dir / "test-skill"
        skill_d.mkdir(parents=True)
        (skill_d / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "metadata:\n"
            "  hermes:\n"
            "    config:\n"
            "      - key: my.key\n"
            "        description: A config key\n"
            "        default: /default\n"
            "---\n"
            "Body"
        )
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(su, "get_disabled_skill_names", lambda: set())
        monkeypatch.setattr(su, "get_all_skills_dirs", lambda: [skills_dir])
        result = discover_all_skill_config_vars()
        assert len(result) == 1
        assert result[0]["key"] == "my.key"
        assert result[0]["skill"] == "test-skill"

    def test_disabled_skills_excluded(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skill_d = skills_dir / "disabled-skill"
        skill_d.mkdir(parents=True)
        (skill_d / "SKILL.md").write_text(
            "---\n"
            "name: disabled-skill\n"
            "metadata:\n"
            "  hermes:\n"
            "    config:\n"
            "      - key: d.key\n"
            "        description: desc\n"
            "---\n"
            "Body"
        )
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(su, "get_disabled_skill_names", lambda: {"disabled-skill"})
        monkeypatch.setattr(su, "get_all_skills_dirs", lambda: [skills_dir])
        assert discover_all_skill_config_vars() == []

    def test_cross_skill_deduplication(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        for name in ("skill-a", "skill-b"):
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                "---\n"
                "name: " + name + "\n"
                "metadata:\n"
                "  hermes:\n"
                "    config:\n"
                "      - key: shared.key\n"
                "        description: desc\n"
                "---\n"
                "body"
            )
        monkeypatch.setattr(su, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(su, "get_disabled_skill_names", lambda: set())
        monkeypatch.setattr(su, "get_all_skills_dirs", lambda: [skills_dir])
        result = discover_all_skill_config_vars()
        # First skill wins the shared key
        assert len(result) == 1
        assert result[0]["skill"] == "skill-a"
