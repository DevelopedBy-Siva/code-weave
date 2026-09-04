"""Download, clean, deduplicate, and validate Python instruction datasets."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import difflib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_PATH = RAW_DIR / "combined_data.json"
RAW_METADATA_PATH = RAW_DIR / "dataset_manifest.json"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.json"
STATS_PATH = PROCESSED_DIR / "cleaning_stats.json"
DATA_CLEANING_VERSION = 4
RAW_NORMALIZATION_VERSION = 2

EOS_TOKENS = ("<EOS_TOKEN>", "</s>", "<eos>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>")
PYTHON_HINTS = ("def ", "class ", "import ", "from ", "return ", "print(", "for ", "while ", "if ", "try:", "except ")
# Quality matters more than volume when adapting an already strong code model.
# The Open-R1 set is decontaminated and its reference answers passed executable
# tests. OpenCodeInstruct-50k supplies HumanEval-like function/class tasks whose
# solutions passed all ten generated tests and strict quality judging. The
# previous unverified 100k mixture caused the observed catastrophic forgetting
# and is intentionally excluded.
DATASETS: tuple[dict[str, str], ...] = (
    {
        "path": "open-r1/verifiable-coding-problems-python_decontaminated-tested",
        "split": "train",
    },
    {"path": "LLMSafety/OpenCodeInstruct-50k", "split": "train"},
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CleaningStats:
    """Counts recorded while cleaning the merged dataset."""

    cleaning_version: int = DATA_CLEANING_VERSION
    raw: int = 0
    eos_removed: int = 0
    non_python_removed: int = 0
    quality_removed: int = 0
    invalid_python_removed: int = 0
    demo_sections_removed: int = 0
    duplicate_removed: int = 0
    benchmark_removed: int = 0
    kept: int = 0
    source_raw: dict[str, int] = field(default_factory=dict)
    source_kept: dict[str, int] = field(default_factory=dict)


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
    """Build a stable key for a task, regardless of small answer differences.

    Deduplicating on the answer as well as the prompt allowed the same task with
    several near-identical answers to land in both train and validation.  That
    made validation loss look better without measuring generalization.
    """
    text = "\n".join([example.get("instruction", ""), example.get("input", "")])
    return re.sub(r"\s+", " ", text).strip().lower()


def quality_ok(example: dict[str, str]) -> bool:
    """Filter out examples that are too small, too large, or instructionless."""
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    return 10 <= len(instruction) and 20 <= len(output) <= 8_000


CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_output(text: str) -> str:
    """Return code from a model-style answer, removing Markdown commentary.

    Instruction datasets frequently wrap otherwise good solutions in prose and
    fences.  Training that presentation style makes a completion benchmark less
    reliable, so cleaned targets contain code only.  When several blocks exist,
    the largest parseable Python block is preferred.
    """
    blocks = [block.strip() for block in CODE_FENCE_RE.findall(text)]
    parseable = [block for block in blocks if is_valid_python(block)]
    if parseable:
        return max(parseable, key=len)
    return text.strip()


def is_valid_python(code: str) -> bool:
    """Return whether *code* is a complete, substantive Python program."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return False

    substantive_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.For,
        ast.While,
        ast.If,
        ast.Try,
        ast.With,
        ast.Expr,
    )
    nodes = [node for node in tree.body if isinstance(node, substantive_nodes)]
    # A lone string is usually prose or a docstring rather than a solution.
    return bool(nodes) and not (
        len(nodes) == 1
        and isinstance(nodes[0], ast.Expr)
        and isinstance(nodes[0].value, ast.Constant)
        and isinstance(nodes[0].value.value, str)
    )


def strip_demo_code(code: str) -> tuple[str, bool]:
    """Remove obvious top-level tests/examples after function definitions.

    Many otherwise useful Alpaca answers append assertions, ``print`` calls, or
    a ``__main__`` demo. Teaching those suffixes wastes the generation budget
    and makes benchmark completions less reliable.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False

    defined_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not defined_names:
        return code, False

    def call_name(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
        return None

    def is_main_guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
        )

    kept: list[ast.stmt] = []
    removed = False
    for node in tree.body:
        remove = isinstance(node, ast.Assert) or is_main_guard(node)
        if isinstance(node, ast.Expr):
            remove = remove or call_name(node.value) in {*defined_names, "print"}
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            remove = remove or call_name(value) in defined_names
        if remove:
            removed = True
        else:
            kept.append(node)

    if not removed:
        return code, False
    tree.body = kept
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip(), True


def normalize_code(text: str) -> str:
    """Collapse whitespace for code-level comparison, ignoring formatting differences."""
    return re.sub(r"\s+", " ", text).strip().lower()


FUNC_DEF_RE = re.compile(r"def\s+(\w+)\s*\(")


def load_benchmark_signals(min_solution_chars: int = 40) -> dict[str, dict[str, str]]:
    """Load held-out benchmark text used to screen the training corpus."""
    from datasets import load_dataset

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    signals: dict[str, dict[str, str]] = {}
    for row in humaneval:
        solution_norm = normalize_code(row["canonical_solution"])
        if len(solution_norm) < min_solution_chars:
            continue
        signals[row["entry_point"]] = {
            "task_id": row["task_id"],
            "entry_point": row["entry_point"],
            "prompt_norm": normalize_code(row["prompt"]),
            "solution_norm": solution_norm,
        }

    # MBPP validation is used only for checkpoint selection. Remove exact prompt
    # matches from SFT data so that checkpoint selection remains held out.
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
    for row in mbpp:
        function_names = FUNC_DEF_RE.findall(row["code"])
        signals[f"mbpp:{row['task_id']}"] = {
            "task_id": f"MBPP/{row['task_id']}",
            "entry_point": function_names[0] if function_names else "",
            "prompt_norm": normalize_code(row["prompt"]),
            "solution_norm": normalize_code(row["code"]),
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
        matching_signals = [
            signal for signal in signals.values() if signal.get("entry_point") == name
        ]
        for signal in matching_signals:
            ratio = difflib.SequenceMatcher(
                None, output_norm, signal["solution_norm"]
            ).quick_ratio()
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
        source = str(record.get("_source", "unknown") or "unknown")
        stats.source_raw[source] = stats.source_raw.get(source, 0) + 1
        instruction, removed_instruction = strip_eos_tokens(str(record.get("instruction", "") or ""))
        input_text, removed_input = strip_eos_tokens(str(record.get("input", "") or ""))
        output, removed_output = strip_eos_tokens(str(record.get("output", "") or ""))
        stats.eos_removed += removed_instruction + removed_input + removed_output

        code_output = extract_python_output(output)
        if record.get("_solution_style") == "script":
            demo_removed = False
        else:
            code_output, demo_removed = strip_demo_code(code_output)
        stats.demo_sections_removed += int(demo_removed)
        example = {
            "instruction": instruction,
            "input": input_text,
            "output": code_output,
        }
        if not looks_like_python(example):
            stats.non_python_removed += 1
            continue
        if not quality_ok(example):
            stats.quality_removed += 1
            continue
        if not is_valid_python(example["output"]):
            stats.invalid_python_removed += 1
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
        stats.source_kept[source] = stats.source_kept.get(source, 0) + 1

    stats.kept = len(cleaned_records)
    return cleaned_records, stats, flagged


def normalize_hf_record(dataset_name: str, record: dict[str, Any]) -> dict[str, str] | None:
    """Map a Hugging Face dataset row into instruction/input/output fields."""
    if dataset_name == "open-r1/verifiable-coding-problems-python_decontaminated-tested":
        return {
            # The decontaminated dataset renamed this field from
            # ``problem_statement`` to ``problem``. Supporting both names also
            # makes the adapter resilient to upstream schema variants.
            "instruction": str(
                record.get("problem") or record.get("problem_statement") or ""
            ),
            "input": "",
            "output": str(record.get("gold_standard_solution", "") or ""),
            "_solution_style": "script",
        }

    if dataset_name == "LLMSafety/OpenCodeInstruct-50k":
        if float(record.get("average_test_score", 0.0) or 0.0) < 1.0:
            return None
        return {
            "instruction": str(record.get("input", "") or ""),
            "input": "",
            "output": str(record.get("output", "") or ""),
        }

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
            normalized["_source"] = dataset_name
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
    write_json(
        RAW_METADATA_PATH,
        {
            "datasets": list(DATASETS),
            "normalization_version": RAW_NORMALIZATION_VERSION,
            "record_count": len(records),
        },
    )
    logger.info("Saved %s raw records to %s", len(records), RAW_DATA_PATH)
    return records


def run_clean(records: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, str]], CleaningStats, list[dict[str, str]]]:
    """Clean raw records and save processed data plus stats."""
    if records is None:
        if not RAW_DATA_PATH.exists():
            records = run_download()
        else:
            manifest = read_json(RAW_METADATA_PATH) if RAW_METADATA_PATH.exists() else {}
            if (
                manifest.get("datasets") != list(DATASETS)
                or manifest.get("normalization_version") != RAW_NORMALIZATION_VERSION
            ):
                raise RuntimeError(
                    "Raw data came from an obsolete dataset mixture or schema adapter; rerun with "
                    "`python data/download_datasets.py --download --clean --validate`."
                )
            records = read_json(RAW_DATA_PATH)
    logger.info("Loading held-out HumanEval/MBPP signals to screen for contamination")
    benchmark_signals = load_benchmark_signals()
    cleaned, stats, flagged = clean_records(records, benchmark_signals=benchmark_signals)
    starved_sources = {
        spec["path"]: (
            stats.source_kept.get(spec["path"], 0),
            stats.source_raw.get(spec["path"], 0),
        )
        for spec in DATASETS
        if stats.source_raw.get(spec["path"], 0) > 0
        and stats.source_kept.get(spec["path"], 0)
        < 0.10 * stats.source_raw[spec["path"]]
    }
    if starved_sources:
        details = ", ".join(
            f"{source} ({kept}/{raw} kept)"
            for source, (kept, raw) in sorted(starved_sources.items())
        )
        raise RuntimeError(
            "Cleaning retained under 10% of configured source(s): "
            f"{details}. Check the upstream schema before training."
        )
    write_json(CLEANED_DATA_PATH, cleaned)
    write_json(STATS_PATH, asdict(stats))
    write_json(PROCESSED_DIR / "benchmark_contamination_flagged.json", flagged)
    logger.info("Cleaning stats: %s", asdict(stats))
    logger.info("Saved %s cleaned records to %s", len(cleaned), CLEANED_DATA_PATH)
    return cleaned, stats, flagged


def run_validate() -> None:
    """Validate that processed data exists and matches the expected schema."""
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {CLEANED_DATA_PATH}")
    records = read_json(CLEANED_DATA_PATH)
    stats = read_json(STATS_PATH) if STATS_PATH.exists() else {}
    if stats.get("cleaning_version") != DATA_CLEANING_VERSION:
        raise ValueError(
            "Processed data was created by an obsolete cleaner; rerun "
            "`python data/download_datasets.py --clean --validate`."
        )
    if not isinstance(records, list) or not records:
        raise ValueError("Processed dataset must be a non-empty JSON list.")
    required = {"instruction", "input", "output"}
    seen_tasks: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"Row {index} has an invalid schema")
        if not is_valid_python(row["output"]):
            raise ValueError(f"Row {index} has an invalid Python target")
        task_key = normalize_for_dedup(row)
        if task_key in seen_tasks:
            raise ValueError(f"Row {index} duplicates an earlier instruction/input pair")
        seen_tasks.add(task_key)
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
