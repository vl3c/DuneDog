"""Tests for the DuneDog CLI."""

import argparse
import os
import subprocess
import sys
from unittest import mock

import pytest

from dunedog.cli import _bounded_int, _resolve_api_key, cmd_generate, cmd_soup, _build_parser


def _run_cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run dunedog CLI via python -m and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "dunedog.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ------------------------------------------------------------------ #
# Help subcommands
# ------------------------------------------------------------------ #


class TestCLIHelp:
    """Every subcommand's --help should exit 0."""

    @pytest.mark.parametrize("subcmd", ["generate", "demo", "soup", "evolve", "setup"])
    def test_subcommand_help(self, subcmd):
        result = _run_cli(subcmd, "--help")
        assert result.returncode == 0, f"{subcmd} --help failed: {result.stderr}"

    def test_top_level_help(self):
        result = _run_cli("--help")
        assert result.returncode == 0


# ------------------------------------------------------------------ #
# _bounded_int validator
# ------------------------------------------------------------------ #


class TestBoundedInt:
    """Tests for _bounded_int argument validator."""

    def test_valid_value_within_bounds(self):
        parser_type = _bounded_int(1, 100)
        assert parser_type("50") == 50

    def test_lower_bound_accepted(self):
        parser_type = _bounded_int(1, 100)
        assert parser_type("1") == 1

    def test_upper_bound_accepted(self):
        parser_type = _bounded_int(1, 100)
        assert parser_type("100") == 100

    def test_below_lower_bound_raises(self):
        parser_type = _bounded_int(1, 100)
        with pytest.raises(argparse.ArgumentTypeError, match="must be between 1 and 100"):
            parser_type("0")

    def test_above_upper_bound_raises(self):
        parser_type = _bounded_int(1, 100)
        with pytest.raises(argparse.ArgumentTypeError, match="must be between 1 and 100"):
            parser_type("101")

    def test_non_integer_string_raises_value_error(self):
        parser_type = _bounded_int(1, 100)
        with pytest.raises(ValueError):
            parser_type("abc")

    def test_float_string_raises_value_error(self):
        parser_type = _bounded_int(1, 100)
        with pytest.raises(ValueError):
            parser_type("3.14")

    def test_negative_bounds(self):
        parser_type = _bounded_int(-10, -1)
        assert parser_type("-5") == -5

    def test_name_attribute(self):
        parser_type = _bounded_int(1, 100)
        assert parser_type.__name__ == "int[1..100]"


# ------------------------------------------------------------------ #
# _resolve_api_key
# ------------------------------------------------------------------ #


class TestResolveApiKey:
    """Tests for _resolve_api_key."""

    def test_explicit_key_returned(self):
        result = _resolve_api_key("openai", "my-explicit-key")
        assert result == "my-explicit-key"

    def test_explicit_key_takes_priority_over_env(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            result = _resolve_api_key("openai", "explicit-key")
            assert result == "explicit-key"

    def test_env_var_used_for_openai(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            result = _resolve_api_key("openai", None)
            assert result == "env-openai-key"

    def test_env_var_used_for_anthropic(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-anthropic-key"}):
            result = _resolve_api_key("anthropic", None)
            assert result == "env-anthropic-key"

    def test_env_var_used_for_openrouter(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-or-key"}):
            result = _resolve_api_key("openrouter", None)
            assert result == "env-or-key"

    def test_env_var_used_for_chatgpt(self):
        with mock.patch.dict(os.environ, {"CHATGPT_TOKEN": "env-chatgpt-key"}):
            result = _resolve_api_key("chatgpt", None)
            assert result == "env-chatgpt-key"

    def test_empty_string_when_no_key_and_no_env(self):
        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                                  "OPENROUTER_API_KEY", "CHATGPT_TOKEN")}
        with mock.patch.dict(os.environ, env_clean, clear=True):
            result = _resolve_api_key("openai", None)
            assert result == ""

    def test_empty_string_when_no_provider(self):
        result = _resolve_api_key(None, None)
        assert result == ""

    def test_empty_string_for_unknown_provider(self):
        result = _resolve_api_key("unknown_provider", None)
        assert result == ""


# ------------------------------------------------------------------ #
# Soup command
# ------------------------------------------------------------------ #


class TestCLISoup:
    """Tests for the 'soup' subcommand."""

    def test_soup_with_seed(self):
        result = _run_cli("soup", "--seed", "42", "-l", "100")
        assert result.returncode == 0, f"soup failed: {result.stderr}"
        assert len(result.stdout) > 0


class TestCmdSoupFull:
    """Additional soup tests via subprocess."""

    def test_soup_with_length_and_count(self):
        result = _run_cli("soup", "--length", "50", "--count", "2", "--seed", "42")
        assert result.returncode == 0, f"soup failed: {result.stderr}"
        # Should contain output for 2 soups
        assert len(result.stdout) > 0

    def test_soup_determinism_same_seed(self):
        result1 = _run_cli("soup", "--seed", "42", "-l", "50")
        result2 = _run_cli("soup", "--seed", "42", "-l", "50")
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result1.stdout == result2.stdout, "Same seed should produce identical output"

    def test_soup_different_seeds_differ(self):
        result1 = _run_cli("soup", "--seed", "42", "-l", "50")
        result2 = _run_cli("soup", "--seed", "99", "-l", "50")
        assert result1.returncode == 0
        assert result2.returncode == 0
        # Different seeds should produce different output
        assert result1.stdout != result2.stdout


# ------------------------------------------------------------------ #
# Generate command (--no-llm)
# ------------------------------------------------------------------ #


class TestCmdGenerateNoLlm:
    """Test cmd_generate with --no-llm flag."""

    def test_generate_no_llm_via_subprocess(self):
        result = _run_cli("generate", "--preset", "quick", "--no-llm", "--seed", "42", "-n", "3")
        assert result.returncode == 0, f"generate --no-llm failed: {result.stderr}"
        assert len(result.stdout) > 0

    def test_generate_no_llm_produces_output(self):
        result = _run_cli("generate", "--preset", "quick", "--no-llm", "--seed", "42", "-n", "3",
                          "-o", "/tmp/dunedog_test_skeletons.json")
        assert result.returncode == 0, f"generate --no-llm failed: {result.stderr}"
        # Verify the output file was created
        assert os.path.exists("/tmp/dunedog_test_skeletons.json")
        # Clean up
        os.remove("/tmp/dunedog_test_skeletons.json")

    def test_generate_determinism(self):
        result1 = _run_cli("generate", "--preset", "quick", "--no-llm", "--seed", "42", "-n", "2")
        result2 = _run_cli("generate", "--preset", "quick", "--no-llm", "--seed", "42", "-n", "2")
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result1.stdout == result2.stdout


# ------------------------------------------------------------------ #
# Demo command (end-to-end, no LLM)
# ------------------------------------------------------------------ #


class TestCLIDemo:
    """Tests for the 'demo' subcommand."""

    def test_demo_no_llm(self):
        result = _run_cli("demo", "--seed", "42", "--no-llm")
        assert result.returncode == 0, f"demo failed: {result.stderr}"
        assert len(result.stdout) > 0
