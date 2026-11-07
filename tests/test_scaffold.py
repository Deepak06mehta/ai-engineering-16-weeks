"""Scaffold tests. These must always be green — they check the repo, not your work."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_directories_exist():
    for d in ("src", "tests", "notes", "notebooks", "benchmarks", "projects", "infra"):
        assert (ROOT / d).is_dir(), f"missing directory: {d}"


def test_env_example_exists_and_env_is_ignored():
    assert (ROOT / ".env.example").exists()
    assert ".env" in (ROOT / ".gitignore").read_text()


def test_weights_and_data_are_gitignored():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("*.gguf", "*.safetensors", "data/raw/", "models/"):
        assert pattern in ignored, f"{pattern} must never be committed"


def test_pyproject_marks_exercises_as_deselected_by_default():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    addopts = cfg["tool"]["pytest"]["ini_options"]["addopts"]
    assert "not exercise" in addopts


def test_common_helpers_importable_without_torch():
    """timing.py must not need torch — you use it before installing anything heavy."""
    from src.common.timing import BenchResult, timer  # noqa: F401
