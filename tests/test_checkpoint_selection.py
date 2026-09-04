"""Tests for held-out executable checkpoint selection."""

from pathlib import Path

from model.select_checkpoint import discover_checkpoints, mbpp_prompt, passes_mbpp


def test_discover_checkpoints_orders_steps(tmp_path: Path) -> None:
    """Only complete PEFT checkpoint directories should be selected."""
    for name in ("checkpoint-500", "checkpoint-250", "checkpoint-invalid"):
        path = tmp_path / name
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in discover_checkpoints(tmp_path)] == [
        "checkpoint-250",
        "checkpoint-500",
    ]


def test_mbpp_solution_execution() -> None:
    """Checkpoint scoring should execute all held-out assertions."""
    problem = {
        "prompt": "Write a function that adds two numbers.",
        "test_list": ["assert add(1, 2) == 3", "assert add(-2, 2) == 0"],
        "test_setup_code": "",
    }

    assert "assert add(1, 2) == 3" in mbpp_prompt(problem)
    assert passes_mbpp("def add(a, b):\n    return a + b", problem)[0] is True
    assert passes_mbpp("def add(a, b):\n    return a - b", problem)[0] is False
