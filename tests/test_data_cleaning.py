"""Tests for dataset cleaning helpers."""

from data.download_datasets import (
    clean_records,
    extract_python_output,
    looks_like_python,
    normalize_hf_record,
    strip_demo_code,
    strip_eos_tokens,
)


def test_strip_eos_tokens_removes_known_tokens() -> None:
    """EOS markers should be removed from text."""
    cleaned, removed = strip_eos_tokens("def f(): pass<EOS_TOKEN></s>")
    assert cleaned == "def f(): pass"
    assert removed == 2


def test_clean_records_deduplicates_examples() -> None:
    """Duplicate normalized examples should be removed."""
    records = [
        {"instruction": "Write a Python function", "input": "", "output": "def add(a, b):\n    return a + b"},
        {"instruction": "Write a Python function", "input": "", "output": "def add(a, b):\n    return a + b"},
    ]
    cleaned, stats, _ = clean_records(records)
    assert len(cleaned) == 1
    assert stats.duplicate_removed == 1


def test_clean_records_filters_non_python() -> None:
    """Non-Python examples should not be kept."""
    records = [
        {"instruction": "Write a Python function", "input": "", "output": "def ok():\n    return True"},
        {"instruction": "Write a short poem", "input": "", "output": "Roses are red and violets are blue"},
    ]
    cleaned, stats, _ = clean_records(records)
    assert len(cleaned) == 1
    assert stats.non_python_removed == 1
    assert looks_like_python(cleaned[0])


def test_clean_records_deduplicates_same_prompt_with_different_answers() -> None:
    """The same task must not leak across train and validation via answer variants."""
    records = [
        {"instruction": "Write a Python add function", "input": "two numbers", "output": "def add(a, b):\n    return a + b"},
        {"instruction": "Write a Python add function", "input": "two numbers", "output": "def add(x, y):\n    return sum((x, y))"},
    ]

    cleaned, stats, _ = clean_records(records)

    assert len(cleaned) == 1
    assert stats.duplicate_removed == 1


def test_clean_records_rejects_incomplete_python() -> None:
    """Syntax-invalid targets should not teach the model broken code."""
    records = [
        {"instruction": "Write a Python function please", "input": "", "output": "def broken(value):\n    if value:"},
    ]

    cleaned, stats, _ = clean_records(records)

    assert cleaned == []
    assert stats.invalid_python_removed == 1


def test_extract_python_output_removes_markdown_prose() -> None:
    """A fenced solution should become a code-only training target."""
    response = "Here is the solution:\n```python\ndef square(x):\n    return x * x\n```\nThis is O(1)."

    assert extract_python_output(response) == "def square(x):\n    return x * x"


def test_strip_demo_code_removes_generated_tests() -> None:
    """Training targets should stop after the reusable implementation."""
    code = "def square(x):\n    return x * x\n\nassert square(2) == 4\nprint(square(3))"

    cleaned, removed = strip_demo_code(code)

    assert removed is True
    assert cleaned == "def square(x):\n    return x * x"


def test_normalize_verified_opencode_record() -> None:
    """OpenCodeInstruct's input field is the task, not auxiliary input."""
    record = {
        "input": "Write a function that adds two integers.",
        "output": "def add(a, b):\n    return a + b",
        "average_test_score": "1.0",
    }

    assert normalize_hf_record("LLMSafety/OpenCodeInstruct-50k", record) == {
        "instruction": record["input"],
        "input": "",
        "output": record["output"],
    }


def test_normalize_open_r1_uses_decontaminated_problem_field() -> None:
    """Open-R1's decontaminated split calls its prompt column ``problem``."""
    record = {
        "problem": "Solve this Python problem from standard input.",
        "gold_standard_solution": "```python\nprint(input())\n```",
    }

    assert normalize_hf_record(
        "open-r1/verifiable-coding-problems-python_decontaminated-tested", record
    ) == {
        "instruction": record["problem"],
        "input": "",
        "output": record["gold_standard_solution"],
        "_solution_style": "script",
    }


def test_cleaning_stats_report_source_retention() -> None:
    """Per-source counts expose schema errors that aggregate counts can hide."""
    records = [
        {
            "instruction": "Write a Python function that adds two integers.",
            "input": "",
            "output": "def add(a, b):\n    return a + b",
            "_source": "verified-source",
        }
    ]

    _, stats, _ = clean_records(records)

    assert stats.source_raw == {"verified-source": 1}
    assert stats.source_kept == {"verified-source": 1}


def test_clean_records_preserves_competitive_script_entrypoint() -> None:
    """A required solve() call is not a removable function demonstration."""
    output = "def solve():\n    print(input())\n\nsolve()"
    records = [
        {
            "instruction": "Read one string from standard input and print it.",
            "input": "",
            "output": output,
            "_solution_style": "script",
        }
    ]

    cleaned, _, _ = clean_records(records)

    assert cleaned[0]["output"] == output
