#!/usr/bin/env python3
"""
SEC-05: Tests for generic MCP restriction semantics.

Tests the delegation.mcp_denylist config field and the existing cron no_mcp
sentinel.  All tests are self-contained — no API keys, no running services.

Run with:  scripts/run_tests.sh tests/tools/test_mcp_denylist.py -v
"""

import os
import threading
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _apply_mcp_denylist,
    _build_child_agent,
    _get_mcp_denylist,
    _is_mcp_toolset_name,
)


# ---------------------------------------------------------------------------
# _is_mcp_toolset_name — existing helper, regression coverage
# ---------------------------------------------------------------------------

class TestIsMcpToolsetName(unittest.TestCase):
    """Verify the MCP toolset name predicate."""

    def test_mcp_prefix(self):
        assert _is_mcp_toolset_name("mcp-cua-driver") is True
        assert _is_mcp_toolset_name("mcp-MiniMax") is True
        assert _is_mcp_toolset_name("mcp-blender") is True

    def test_non_mcp(self):
        assert _is_mcp_toolset_name("web") is False
        assert _is_mcp_toolset_name("terminal") is False
        assert _is_mcp_toolset_name("file") is False
        assert _is_mcp_toolset_name("") is False


# ---------------------------------------------------------------------------
# _get_mcp_denylist — config reading (delegation subsection shape)
# ---------------------------------------------------------------------------

class TestGetMcpDenylist(unittest.TestCase):
    """Verify config reading for mcp_denylist.

    _load_config() returns the delegation subsection directly, so mocks
    must pass the subsection shape: {"mcp_denylist": [...]} — NOT the
    full config {"delegation": {"mcp_denylist": [...]}}.
    """

    def test_empty_when_unset(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            assert _get_mcp_denylist() == set()

    def test_empty_when_denylist_missing(self):
        with patch("tools.delegate_tool._load_config", return_value={"inherit_mcp_toolsets": True}):
            assert _get_mcp_denylist() == set()

    def test_empty_when_denylist_empty(self):
        with patch("tools.delegate_tool._load_config", return_value={"mcp_denylist": []}):
            assert _get_mcp_denylist() == set()

    def test_reads_entries(self):
        cfg = {"mcp_denylist": ["mcp-cua-driver", "mcp-blender"]}
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            result = _get_mcp_denylist()
            assert result == {"mcp-cua-driver", "mcp-blender"}

    def test_strips_whitespace(self):
        cfg = {"mcp_denylist": ["  mcp-cua-driver  ", "mcp-blender"]}
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            result = _get_mcp_denylist()
            assert result == {"mcp-cua-driver", "mcp-blender"}

    def test_ignores_empty_strings(self):
        cfg = {"mcp_denylist": ["mcp-cua-driver", "", "  "]}
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            result = _get_mcp_denylist()
            assert result == {"mcp-cua-driver"}

    def test_rejects_non_list(self):
        cfg = {"mcp_denylist": "mcp-cua-driver"}  # string, not list
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            assert _get_mcp_denylist() == set()

    def test_rejects_none(self):
        cfg = {"mcp_denylist": None}
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            assert _get_mcp_denylist() == set()


# ---------------------------------------------------------------------------
# _apply_mcp_denylist — removal logic
# ---------------------------------------------------------------------------

class TestApplyMcpDenylist(unittest.TestCase):
    """Verify the denylist removes MCP toolsets and preserves non-MCP."""

    def test_no_op_when_empty(self):
        toolsets = ["web", "terminal", "mcp-cua-driver"]
        assert _apply_mcp_denylist(toolsets, set()) == toolsets

    def test_removes_denied_mcp(self):
        toolsets = ["web", "terminal", "mcp-cua-driver", "mcp-blender"]
        denylist = {"mcp-cua-driver"}
        result = _apply_mcp_denylist(toolsets, denylist)
        assert result == ["web", "terminal", "mcp-blender"]

    def test_removes_multiple_denied(self):
        toolsets = ["web", "mcp-cua-driver", "mcp-blender", "mcp-MiniMax"]
        denylist = {"mcp-cua-driver", "mcp-blender"}
        result = _apply_mcp_denylist(toolsets, denylist)
        assert result == ["web", "mcp-MiniMax"]

    def test_preserves_non_mcp_even_if_name_matches(self):
        toolsets = ["web", "terminal", "cua-driver"]
        denylist = {"mcp-cua-driver"}
        result = _apply_mcp_denylist(toolsets, denylist)
        assert result == ["web", "terminal", "cua-driver"]

    def test_empty_input(self):
        assert _apply_mcp_denylist([], {"mcp-cua-driver"}) == []

    def test_does_not_mutate_input(self):
        toolsets = ["web", "mcp-cua-driver"]
        denylist = {"mcp-cua-driver"}
        result = _apply_mcp_denylist(toolsets, denylist)
        assert toolsets == ["web", "mcp-cua-driver"]
        assert result == ["web"]


# ---------------------------------------------------------------------------
# _build_child_agent integration — denylist applied after inheritance
# ---------------------------------------------------------------------------

def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields _build_child_agent expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.valid_tool_names = ["terminal", "file", "web", "mcp-cua-driver", "mcp-blender"]
    parent.enabled_toolsets = ["terminal", "file", "web", "mcp-cua-driver", "mcp-blender"]
    return parent


class TestBuildChildAgentMcpDenylist(unittest.TestCase):
    """Integration: _build_child_agent applies mcp_denylist after inheritance.

    Mocks use the delegation subsection shape (what _load_config returns),
    NOT the full config with a nested 'delegation' key.
    """

    @patch(
        "tools.delegate_tool._load_config",
        return_value={"mcp_denylist": ["mcp-cua-driver"]},
    )
    def test_denylist_removes_mcp_from_child(self, mock_cfg):
        parent = _make_mock_parent()
        parent.enabled_toolsets = ["terminal", "file", "web", "mcp-cua-driver", "mcp-blender"]

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            _build_child_agent(
                task_index=0,
                goal="Test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )
            result_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
            assert "mcp-cua-driver" not in result_toolsets
            assert "mcp-blender" in result_toolsets
            assert "terminal" in result_toolsets

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_no_denylist_preserves_all_mcp(self, mock_cfg):
        parent = _make_mock_parent()
        parent.enabled_toolsets = ["terminal", "file", "web", "mcp-cua-driver", "mcp-blender"]

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            _build_child_agent(
                task_index=0,
                goal="Test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )
            result_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
            assert "mcp-cua-driver" in result_toolsets
            assert "mcp-blender" in result_toolsets

    @patch(
        "tools.delegate_tool._load_config",
        return_value={
            "inherit_mcp_toolsets": True,
            "mcp_denylist": ["mcp-cua-driver"],
        },
    )
    def test_denylist_applies_after_inheritance(self, mock_cfg):
        """Child requests narrowed toolsets; MCP is inherited then denylisted."""
        parent = _make_mock_parent()
        parent.enabled_toolsets = ["terminal", "file", "web", "mcp-cua-driver", "mcp-blender"]

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child
            _build_child_agent(
                task_index=0,
                goal="Test",
                context=None,
                toolsets=["terminal", "file"],
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )
            result_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
            assert "mcp-cua-driver" not in result_toolsets
            assert "mcp-blender" in result_toolsets
            assert "terminal" in result_toolsets


# ---------------------------------------------------------------------------
# Behavior-level integration: config → child toolset via real loader
# ---------------------------------------------------------------------------

class TestMcpDenylistConfigIntegration(unittest.TestCase):
    """Prove the full config-to-child-toolset path without mocks.

    Writes a real config.yaml to a temp HERMES_HOME and verifies that
    _get_mcp_denylist reads mcp_denylist from the delegation section
    through the shared config loader.
    """

    def test_denylist_reads_from_temp_hermes_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, ".hermes")
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(textwrap.dedent("""\
                    delegation:
                      mcp_denylist:
                        - mcp-cua-driver
                        - mcp-blender
                """))

            old_home = os.environ.get("HERMES_HOME")
            try:
                os.environ["HERMES_HOME"] = config_dir
                # Force _load_config to re-read by clearing any cached state
                result = _get_mcp_denylist()
                assert result == {"mcp-cua-driver", "mcp-blender"}, f"Got: {result}"
            finally:
                if old_home is not None:
                    os.environ["HERMES_HOME"] = old_home
                else:
                    os.environ.pop("HERMES_HOME", None)

    def test_empty_denylist_from_temp_hermes_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, ".hermes")
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(textwrap.dedent("""\
                    delegation:
                      inherit_mcp_toolsets: true
                """))

            old_home = os.environ.get("HERMES_HOME")
            try:
                os.environ["HERMES_HOME"] = config_dir
                result = _get_mcp_denylist()
                assert result == set(), f"Got: {result}"
            finally:
                if old_home is not None:
                    os.environ["HERMES_HOME"] = old_home
                else:
                    os.environ.pop("HERMES_HOME", None)


# ---------------------------------------------------------------------------
# Cron no_mcp regression — existing path, focused coverage
# ---------------------------------------------------------------------------

class TestCronNoMcpRegression(unittest.TestCase):
    """Regression: existing no_mcp sentinel in cron scheduler works correctly."""

    def test_no_mcp_sentinel_strips_all_mcp(self):
        from cron.scheduler import _merge_mcp_into_per_job_toolsets

        cfg = {"mcp_servers": {
            "cua-driver": {"enabled": True, "command": "cua-driver"},
            "blender": {"enabled": True, "command": "blender"},
        }}
        result = _merge_mcp_into_per_job_toolsets(["terminal", "web", "no_mcp"], cfg)
        assert "no_mcp" not in result
        assert "terminal" in result
        assert "web" in result
        assert "mcp-cua-driver" not in result
        assert "mcp-blender" not in result

    def test_explicit_mcp_name_is_allowlist(self):
        from cron.scheduler import _merge_mcp_into_per_job_toolsets

        cfg = {"mcp_servers": {
            "cua-driver": {"enabled": True, "command": "cua-driver"},
            "blender": {"enabled": True, "command": "blender"},
        }}
        result = _merge_mcp_into_per_job_toolsets(["terminal", "mcp-cua-driver"], cfg)
        assert "mcp-cua-driver" in result
        assert "mcp-blender" not in result

    def test_no_mcp_with_empty_list(self):
        from cron.scheduler import _merge_mcp_into_per_job_toolsets

        cfg = {"mcp_servers": {
            "cua-driver": {"enabled": True, "command": "cua-driver"},
        }}
        result = _merge_mcp_into_per_job_toolsets(["no_mcp"], cfg)
        assert result == []

    def test_disabled_toolsets_always_include_cronjob_messaging_clarify(self):
        from cron.scheduler import _resolve_cron_disabled_toolsets

        result = _resolve_cron_disabled_toolsets({})
        assert "cronjob" in result
        assert "messaging" in result
        assert "clarify" in result

    def test_user_disabled_toolsets_layered_on(self):
        from cron.scheduler import _resolve_cron_disabled_toolsets

        cfg = {"agent": {"disabled_toolsets": ["web", "delegation"]}}
        result = _resolve_cron_disabled_toolsets(cfg)
        assert "web" in result
        assert "delegation" in result
        assert "cronjob" in result


if __name__ == "__main__":
    unittest.main()
