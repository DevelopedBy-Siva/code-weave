"""Tests for HumanEval response handling and regression checks."""

from __future__ import annotations

import json

import pytest

from model import benchmark


PROMPT = '''def increment(value: int) -> int:
    """Return value plus one."""
'''


def test_build_solution_preserves_indented_completion() -> None:
    """Leading spaces in a completion are part of the generated program."""
    solution = benchmark.build_solution(PROMPT, "    return value + 1")

    namespace: dict[str, object] = {}
    exec(solution, namespace)
    assert namespace["increment"](2) == 3  # type: ignore[operator]


def test_build_solution_keeps_imports_before_full_function() -> None:
    """Imports must not be discarded when a model repeats the function."""
    response = "import math\nimport statistics\n\ndef increment(value: int) -> int:\n    return math.floor(value) + 1"

    solution = benchmark.build_solution(PROMPT, response)

    assert "import math" in solution
    assert "import statistics" in solution
    namespace: dict[str, object] = {}
    exec(solution, namespace)
    assert namespace["increment"](2) == 3  # type: ignore[operator]


def test_build_solution_trims_trailing_explanation() -> None:
    """Natural-language text after otherwise valid code should be ignored."""
    response = "```python\n    return value + 1\n```\nThis increments the value."

    assert benchmark.build_solution(PROMPT, response).endswith("    return value + 1\n")


def test_compare_results_fails_on_regression(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A regressed model must not pass the benchmark acceptance gate."""
    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)
    (tmp_path / "baseline_results.json").write_text(json.dumps({"pass@1": 75.0, "num_passed": 123}))
    (tmp_path / "finetuned_results.json").write_text(json.dumps({"pass@1": 70.0, "num_passed": 115}))

    with pytest.raises(SystemExit, match="Quality gate failed"):
        benchmark.compare_results()
