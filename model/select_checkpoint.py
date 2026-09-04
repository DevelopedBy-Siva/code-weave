"""Select and merge a LoRA checkpoint using held-out executable MBPP tasks."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

try:  # Support both script and package execution.
    from .benchmark import extract_code, load_benchmark_model, run_program_safely
    from .config import default_config
    from .model_registry import format_prompt
except ImportError:
    from benchmark import extract_code, load_benchmark_model, run_program_safely
    from config import default_config
    from model_registry import format_prompt

logger = logging.getLogger(__name__)
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def discover_checkpoints(output_dir: Path) -> list[Path]:
    """Return Trainer checkpoints ordered by training step."""
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = CHECKPOINT_RE.search(path.name)
        if path.is_dir() and match and (path / "adapter_config.json").exists():
            checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]


def mbpp_prompt(problem: dict[str, Any]) -> str:
    """Build an executable-code prompt with public example assertions."""
    tests = "\n".join(str(test) for test in problem.get("test_list", [])[:3])
    return f"{problem['prompt'].strip()}\n\nThe code must satisfy these examples:\n{tests}"


def generate_code(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    """Generate one deterministic code-only answer."""
    import torch

    input_text = format_prompt(prompt, tokenizer, task="generate")
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return extract_code(response)


def passes_mbpp(code: str, problem: dict[str, Any]) -> tuple[bool, str | None]:
    """Execute an MBPP candidate against all validation assertions."""
    test_setup = str(problem.get("test_setup_code", "") or "")
    tests = "\n".join(str(test) for test in problem.get("test_list", []))
    result = run_program_safely(f"{test_setup}\n{code}\n{tests}")
    return bool(result["passed"]), result["error"] if isinstance(result["error"], str) else None


def evaluate_checkpoint(
    model_path: str,
    tokenizer_path: str,
    problems: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Return deterministic MBPP validation accuracy for one model."""
    model, tokenizer = load_benchmark_model(model_path, tokenizer_path=tokenizer_path)
    passed = 0
    failures: list[dict[str, Any]] = []
    for problem in problems:
        code = generate_code(model, tokenizer, mbpp_prompt(dict(problem)), max_new_tokens)
        ok, error = passes_mbpp(code, dict(problem))
        if ok:
            passed += 1
        else:
            failures.append({"task_id": problem["task_id"], "error": error})
    total = len(problems)
    score = 100.0 * passed / total
    logger.info("%s: %.2f%% (%s/%s)", model_path, score, passed, total)

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {
        "model_path": model_path,
        "score": round(score, 2),
        "passed": passed,
        "total": total,
        "failures": failures,
    }


def promote_checkpoint(checkpoint: Path, tokenizer_path: str, output_dir: Path) -> None:
    """Merge the selected PEFT adapter and save a deployable model."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix model shards into non-empty directory: {output_dir}"
        )
    model, tokenizer = load_benchmark_model(str(checkpoint), tokenizer_path=tokenizer_path)
    if not hasattr(model, "merge_and_unload"):
        raise TypeError(f"{checkpoint} is not a mergeable PEFT checkpoint")
    merged_model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Promoted %s to %s", checkpoint, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse checkpoint-selection options."""
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output containing checkpoint-* dirs.")
    parser.add_argument("--tokenizer", default=cfg.model.model_name)
    parser.add_argument("--baseline", default=cfg.model.model_name)
    parser.add_argument("--max-problems", type=int, default=90)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.inference.max_new_tokens)
    parser.add_argument("--min-improvement", type=float, default=2.0, help="Required MBPP gain in percentage points.")
    parser.add_argument("--promote-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Evaluate checkpoints, enforce improvement, and merge the winner."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    from datasets import load_dataset

    checkpoints = discover_checkpoints(args.output_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No PEFT checkpoints found under {args.output_dir}")

    problems = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="validation"
    )
    if args.max_problems:
        problems = problems.select(range(min(args.max_problems, len(problems))))

    results = [
        evaluate_checkpoint(args.baseline, args.tokenizer, problems, args.max_new_tokens)
    ]
    for checkpoint in checkpoints:
        results.append(
            evaluate_checkpoint(str(checkpoint), args.tokenizer, problems, args.max_new_tokens)
        )

    baseline_score = results[0]["score"]
    candidates = results[1:]
    winner = max(candidates, key=lambda result: result["score"])
    report = {
        "suite": "google-research-datasets/mbpp sanitized validation",
        "baseline_score": baseline_score,
        "required_improvement": args.min_improvement,
        "winner": winner,
        "results": results,
    }
    report_path = args.output_dir / "checkpoint_selection.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    improvement = winner["score"] - baseline_score
    if improvement < args.min_improvement:
        raise SystemExit(
            f"No checkpoint passed the quality gate: best improvement was {improvement:+.2f} "
            f"points; required {args.min_improvement:+.2f}. See {report_path}."
        )

    promote_dir = args.promote_dir or args.output_dir / "merged_model"
    promote_checkpoint(Path(winner["model_path"]), args.tokenizer, promote_dir)
    logger.info("Selected checkpoint improved MBPP validation by %+.2f points", improvement)


if __name__ == "__main__":
    main()
