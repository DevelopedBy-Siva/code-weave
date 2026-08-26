"""Tests for dataset cleaning helpers."""

from data.download_datasets import clean_records, looks_like_python, strip_eos_tokens


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
    cleaned, stats = clean_records(records)
    assert len(cleaned) == 1
    assert stats.duplicate_removed == 1


def test_clean_records_filters_non_python() -> None:
    """Non-Python examples should not be kept."""
    records = [
        {"instruction": "Write a Python function", "input": "", "output": "def ok():\n    return True"},
        {"instruction": "Write a short poem", "input": "", "output": "Roses are red and violets are blue"},
    ]
    cleaned, stats = clean_records(records)
    assert len(cleaned) == 1
    assert stats.non_python_removed == 1
    assert looks_like_python(cleaned[0])
