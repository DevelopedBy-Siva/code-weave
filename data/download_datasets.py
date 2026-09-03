"""Download, clean, deduplicate, and validate Python instruction datasets."""

from __future__ import annotations

import argparse
import json
import logging
import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_PATH = RAW_DIR / "combined_data.json"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.json"
STATS_PATH = PROCESSED_DIR / "cleaning_stats.json"

EOS_TOKENS = ("<EOS_TOKEN>", "</s>", "<eos>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>")
PYTHON_HINTS = ("def ", "class ", "import ", "from ", "return ", "print(", "for ", "while ", "if ", "try:", "except ")
DATASETS: tuple[dict[str, str], ...] = (
    {"path": "Vezora/Tested-22k-Python-Alpaca", "split": "train"},
    {"path": "iamtarun/python_code_instructions_18k_alpaca", "split": "train"},
    {"path": "flytech/python-codes-25k", "split": "train"},
    {"path": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train"},
    {"path": "sahil2801/CodeAlpaca-20k", "split": "train"},
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CleaningStats:
    """Counts recorded while cleaning the merged dataset."""

    raw: int = 0
    eos_removed: int = 0
    non_python_removed: int = 0
    quality_removed: int = 0
    duplicate_removed: int = 0
    benchmark_removed: int = 0
    kept: int = 0


def strip_eos_tokens(text: str) -> tuple[str, int]:
    """Remove known foreign EOS tokens and return the cleaned text plus count."""
    cleaned = text
    removed = 0
    for token in EOS_TOKENS:
        count = cleaned.count(token)
        removed += count
        cleaned = cleaned.replace(token, "")
    return cleaned.strip(), removed


def looks_like_python(example: dict[str, str]) -> bool:
    """Return whether an example appears to contain Python code."""
    combined = f"{example.get('instruction', '')}\n{example.get('input', '')}\n{example.get('output', '')}"
    return "python" in combined.lower() or any(hint in combined for hint in PYTHON_HINTS)


def normalize_for_dedup(example: dict[str, str]) -> str:
    """Build a stable deduplication key from instruction, input, and output."""
    text = "\n".join([example.get("instruction", ""), example.get("input", ""), example.get("output", "")])
    return re.sub(r"\s+", " ", text).strip().lower()


def quality_ok(example: dict[str, str]) -> bool:
    """Filter out examples that are too small, too large, or instructionless."""
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    return 10 <= len(instruction) and 20 <= len(output) <= 8_000

def normalize_code(text: str) -> str:
    """Collapse whitespace for code-level comparison, ignoring formatting differences."""
    return re.sub(r"\s+", " ", text).strip().lower()


FUNC_DEF_RE = re.compile(r"def\s+(\w+)\s*\(")


def load_benchmark_signals(min_solution_chars: int = 40) -> dict[str, dict[str, str]]:
    """Load HumanEval solutions and problem text, keyed by entry_point, to screen training data against."""
    from datasets import load_dataset

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    signals: dict[str, dict[str, str]] = {}
    for row in humaneval:
        solution_norm = normalize_code(row["canonical_solution"])
        if len(solution_norm) < min_solution_chars:
            continue
        signals[row["entry_point"]] = {
            "task_id": row["task_id"],
            "prompt_norm": normalize_code(row["prompt"]),
            "solution_norm": solution_norm,
        }
    return signals


def is_benchmark_contaminated(
    example: dict[str, str],
    signals: dict[str, dict[str, str]],
    code_ratio_threshold: float = 0.75,
    text_window: int = 60,
) -> tuple[bool, str]:
    """Flag a training example whose code closely matches a HumanEval solution under the
    same function name, or whose instruction text closely matches a HumanEval prompt.
    Returns (is_contaminated, reason) for auditability.
    """
    output = example.get("output", "")
    output_norm = normalize_code(output)

    for name in FUNC_DEF_RE.findall(output):
        signal = signals.get(name)
        if signal is None:
            continue
        ratio = difflib.SequenceMatcher(None, output_norm, signal["solution_norm"]).quick_ratio()
        if ratio >= code_ratio_threshold:
            return True, f"code match: {signal['task_id']} (entry_point={name}, ratio={ratio:.2f})"

    instruction_input_norm = normalize_code(f"{example.get('instruction', '')} {example.get('input', '')}")
    for signal in signals.values():
        prefix = signal["prompt_norm"][:text_window]
        if prefix and prefix in instruction_input_norm:
            return True, f"prompt-text match: {signal['task_id']}"

    return False, ""

def clean_records(
    records: Iterable[dict[str, Any]],
    benchmark_signals: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], CleaningStats, list[dict[str, str]]]:
    """Clean records, remove non-Python data, and deduplicate examples."""
    stats = CleaningStats()
    seen: set[str] = set()
    cleaned_records: list[dict[str, str]] = []
    flagged: list[dict[str, str]] = []

    for record in records:
        stats.raw += 1
        instruction, removed_instruction = strip_eos_tokens(str(record.get("instruction", "") or ""))
        input_text, removed_input = strip_eos_tokens(str(record.get("input", "") or ""))
        output, removed_output = strip_eos_tokens(str(record.get("output", "") or ""))
        stats.eos_removed += removed_instruction + removed_input + removed_output

        example = {"instruction": instruction, "input": input_text, "output": output}
        if not looks_like_python(example):
            stats.non_python_removed += 1
            continue
        if not quality_ok(example):
            stats.quality_removed += 1
            continue
        if benchmark_signals:
            contaminated, reason = is_benchmark_contaminated(example, benchmark_signals)
            if contaminated:
                stats.benchmark_removed += 1
                flagged.append({**example, "reason": reason})
                continue
        dedup_key = normalize_for_dedup(example)
        if dedup_key in seen:
            stats.duplicate_removed += 1
            continue
        seen.add(dedup_key)
        cleaned_records.append(example)

    stats.kept = len(cleaned_records)
    return cleaned_records, stats, flagged


def normalize_hf_record(dataset_name: str, record: dict[str, Any]) -> dict[str, str] | None:
    """Map a Hugging Face dataset row into instruction/input/output fields."""
    if dataset_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
        if str(record.get("lang", "")).lower() != "python":
            return None
        return {"instruction": str(record.get("problem", "") or ""), "input": "", "output": str(record.get("solution", "") or "")}

    instruction = record.get("instruction") or record.get("prompt") or record.get("question") or ""
    input_text = record.get("input") or record.get("context") or ""
    output = record.get("output") or record.get("response") or record.get("completion") or record.get("code") or ""
    return {"instruction": str(instruction), "input": str(input_text), "output": str(output)}


def download_records(load_dataset_fn: Callable[..., Any] | None = None) -> list[dict[str, str]]:
    """Download and normalize the configured Hugging Face datasets."""
    if load_dataset_fn is None:
        from datasets import load_dataset as load_dataset_fn

    all_records: list[dict[str, str]] = []
    for spec in DATASETS:
        dataset_name = spec["path"]
        logger.info("Downloading %s [%s]", dataset_name, spec["split"])
        dataset = load_dataset_fn(dataset_name, split=spec["split"])
        before = len(dataset)
        kept = 0
        for row in dataset:
            normalized = normalize_hf_record(dataset_name, dict(row))
            if normalized is None:
                continue
            all_records.append(normalized)
            kept += 1
        logger.info("  kept %s / %s rows", kept, before)
    return all_records


def write_json(path: Path, records: Any) -> None:
    """Write JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_download() -> list[dict[str, str]]:
    """Download datasets and save the merged raw JSON file."""
    records = download_records()
    write_json(RAW_DATA_PATH, records)
    logger.info("Saved %s raw records to %s", len(records), RAW_DATA_PATH)
    return records


def run_clean(records: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, str]], CleaningStats, list[dict[str, str]]]:
    """Clean raw records and save processed data plus stats."""
    if records is None:
        records = run_download() if not RAW_DATA_PATH.exists() else read_json(RAW_DATA_PATH)
    logger.info("Loading HumanEval to screen for contamination")
    benchmark_signals = load_benchmark_signals()
    cleaned, stats, flagged = clean_records(records, benchmark_signals=benchmark_signals)
    write_json(CLEANED_DATA_PATH, cleaned)
    write_json(STATS_PATH, asdict(stats))
    if flagged:
        write_json(PROCESSED_DIR / "benchmark_contamination_flagged.json", flagged)
    logger.info("Cleaning stats: %s", asdict(stats))
    logger.info("Saved %s cleaned records to %s", len(cleaned), CLEANED_DATA_PATH)
    return cleaned, stats, flagged


def run_validate() -> None:
    """Validate that processed data exists and matches the expected schema."""
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {CLEANED_DATA_PATH}")
    records = read_json(CLEANED_DATA_PATH)
    if not isinstance(records, list) or not records:
        raise ValueError("Processed dataset must be a non-empty JSON list.")
    required = {"instruction", "input", "output"}
    bad_rows = [idx for idx, row in enumerate(records[:1_000]) if set(row) != required]
    if bad_rows:
        raise ValueError(f"Rows with invalid schema in first 1000 records: {bad_rows[:10]}")
    logger.info("Validated %s cleaned samples at %s", len(records), CLEANED_DATA_PATH)


def parse_args() -> argparse.Namespace:
    """Parse command line flags for the data pipeline."""
    parser = argparse.ArgumentParser(description="Download, clean, and validate Python datasets.")
    parser.add_argument("--download", action="store_true", help="Download and save raw HF datasets.")
    parser.add_argument("--clean", action="store_true", help="Clean raw data into data/processed/cleaned_data.json.")
    parser.add_argument("--validate", action="store_true", help="Validate processed data schema.")
    return parser.parse_args()


def main() -> None:
    """Run selected data pipeline stages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if not any((args.download, args.clean, args.validate)):
        args.download = True
        args.clean = True
        args.validate = True
    records = run_download() if args.download else None
    if args.clean:
        run_clean(records)
    if args.validate:
        run_validate()


if __name__ == "__main__":
    main()
