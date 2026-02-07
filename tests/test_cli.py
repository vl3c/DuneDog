"""Tests for the DuneDog CLI."""

import subprocess
import sys

import pytest


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
# Soup command
# ------------------------------------------------------------------ #


class TestCLISoup:
    """Tests for the 'soup' subcommand."""

    def test_soup_with_seed(self):
        result = _run_cli("soup", "--seed", "42", "-l", "100")
        assert result.returncode == 0, f"soup failed: {result.stderr}"
        assert len(result.stdout) > 0


# ------------------------------------------------------------------ #
# Demo command (end-to-end, no LLM)
# ------------------------------------------------------------------ #


class TestCLIDemo:
    """Tests for the 'demo' subcommand."""

    def test_demo_no_llm(self):
        result = _run_cli("demo", "--seed", "42", "--no-llm")
        assert result.returncode == 0, f"demo failed: {result.stderr}"
        assert len(result.stdout) > 0
