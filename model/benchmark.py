"""Run HumanEval benchmarks for baseline and fine-tuned models."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from model.config import default_config
from model.model_registry import format_prompt, load_model as registry_load_model, load_tokenizer

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "benchmark_results"
BENCHMARK_LOG = BASE_DIR / "benchmark.log"
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure benchmark logging."""
    RESULTS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(BENCHMARK_LOG)],
    )


def load_benchmark_model(model_path: str, tokenizer_path: str | None = None) -> tuple[Any, Any]:
    """Load benchmark model and tokenizer with a base-tokenizer fallback."""
    logger.info("Loading model: %s", model_path)
    base_model = default_config().model.model_name
    try:
        tokenizer = load_tokenizer(tokenizer_path or model_path)
    except Exception:
        if tokenizer_path:
            raise
        logger.warning("Tokenizer load failed for %s; falling back to %s", model_path, base_model)
        tokenizer = load_tokenizer(base_model)
    model = registry_load_model(model_path)
    return model, tokenizer


def extract_code(response: str) -> str:
    """Clean the generated response into executable code."""
    if "```python" in response:
        response = response.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in response:
        response = response.split("```", 1)[1].split("```", 1)[0]

    function_match = re.search(r"def\s+\w+\s*\(", response)
    if function_match:
        response = response[function_match.start() :]

    clean_lines: list[str] = []
    for line in response.splitlines():
        if clean_lines and re.match(r"^[A-Za-z]", line) and not line.startswith(" "):
            break
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def generate_solution(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
) -> list[str]:
    """Generate one or more HumanEval solutions."""
    input_text = format_prompt(prompt, task="generate")
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    solutions: list[str] = []

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            solutions.append(f"{prompt}\n{extract_code(response)}")
    return solutions


def run_tests_safely(solution: str, test_code: str, entry_point: str) -> dict[str, bool | str | None]:
    """Execute HumanEval tests for one generated solution."""
    result: dict[str, bool | str | None] = {"passed": False, "error": None}
    exec_globals: dict[str, Any] = {}
    try:
        exec(solution, exec_globals)
        if entry_point not in exec_globals:
            result["error"] = f"Function '{entry_point}' not found in output"
            return result
        exec(test_code, exec_globals)
        exec(f"check({entry_point})", exec_globals)
        result["passed"] = True
    except AssertionError as exc:
        result["error"] = f"AssertionError: {exc}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k for n samples with c correct generations."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_benchmark(
    model_path: str,
    label: str,
    tokenizer_path: str | None = None,
    num_samples: int = 1,
    temperature: float | None = None,
    max_problems: int | None = None,
    max_new_tokens: int | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    """Run HumanEval and save a JSON benchmark report."""
    cfg = default_config()
    temperature = cfg.inference.temperature if temperature is None else temperature
    top_p = cfg.inference.top_p if top_p is None else top_p
    max_new_tokens = cfg.inference.max_new_tokens if max_new_tokens is None else max_new_tokens

    logger.info("Running HumanEval benchmark | label=%s", label)
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
    logger.info("Problems to evaluate: %s", len(dataset))

    model, tokenizer = load_benchmark_model(model_path, tokenizer_path=tokenizer_path)
    results: list[dict[str, Any]] = []
    start_time = time.time()

    for index, problem in enumerate(tqdm(dataset, desc="Evaluating")):
        solutions = generate_solution(
            model,
            tokenizer,
            problem["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
        )
        problem_results = [run_tests_safely(solution, problem["test"], problem["entry_point"]) for solution in solutions]
        num_correct = sum(1 for result in problem_results if result["passed"])
        results.append(
            {
                "task_id": problem["task_id"],
                "num_correct": num_correct,
                "num_samples": num_samples,
                "pass@1": pass_at_k(num_samples, num_correct, 1),
                "errors": [result["error"] for result in problem_results if result["error"]],
            }
        )
        if (index + 1) % 20 == 0:
            pass_so_far = sum(result["pass@1"] for result in results) / len(results)
            logger.info("Progress %s/%s | pass@1 so far: %.3f", index + 1, len(dataset), pass_so_far)

    total = len(results)
    elapsed = time.time() - start_time
    pass1 = sum(result["pass@1"] for result in results) / total
    pass10 = sum(pass_at_k(result["num_samples"], result["num_correct"], 10) for result in results) / total if num_samples >= 10 else None
    failed = [result for result in results if result["pass@1"] == 0]
    summary = {
        "label": label,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "num_problems": total,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "pass@1": round(pass1 * 100, 2),
        "pass@10": round(pass10 * 100, 2) if pass10 is not None else None,
        "num_passed": sum(1 for result in results if result["pass@1"] > 0),
        "num_failed": len(failed),
        "elapsed_sec": round(elapsed, 1),
        "failed_tasks": [result["task_id"] for result in failed],
        "per_problem": results,
    }

    out_path = RESULTS_DIR / f"{label}_results.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("pass@1=%s%% | passed=%s/%s | saved=%s", summary["pass@1"], summary["num_passed"], total, out_path)
    return summary


def compare_results() -> None:
    """Compare saved baseline and fine-tuned benchmark reports."""
    baseline_path = RESULTS_DIR / "baseline_results.json"
    finetuned_path = RESULTS_DIR / "finetuned_results.json"
    if not baseline_path.exists() or not finetuned_path.exists():
        logger.error("Run both baseline and finetuned benchmarks first.")
        sys.exit(1)

    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with finetuned_path.open("r", encoding="utf-8") as handle:
        finetuned = json.load(handle)

    delta_pass1 = finetuned["pass@1"] - baseline["pass@1"]
    print("\n" + "=" * 60)
    print("HUMANEVAL BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 60)
    print(f"{'pass@1 (%)':<20} {baseline['pass@1']:>12.2f} {finetuned['pass@1']:>12.2f} {delta_pass1:>+10.2f}")
    print(f"{'Problems passed':<20} {baseline['num_passed']:>12} {finetuned['num_passed']:>12} {finetuned['num_passed'] - baseline['num_passed']:>+10}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    cfg = default_config()
    parser = argparse.ArgumentParser(description="HumanEval benchmarking for PyCodeGen models.")
    parser.add_argument("--model", type=str, default=None, help="Model path or Hugging Face name.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path or Hugging Face name.")
    parser.add_argument("--label", type=str, default="baseline", help="Result label.")
    parser.add_argument("--samples", type=int, default=1, help="Samples per problem.")
    parser.add_argument("--temperature", type=float, default=cfg.inference.temperature)
    parser.add_argument("--top-p", type=float, default=cfg.inference.top_p)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.inference.max_new_tokens)
    parser.add_argument("--max", type=int, default=None, help="Maximum HumanEval problems.")
    parser.add_argument("--compare", action="store_true", help="Compare baseline and fine-tuned reports.")
    return parser.parse_args()


def main() -> None:
    """Run benchmark CLI."""
    setup_logging()
    args = parse_args()
    if args.compare:
        compare_results()
        return
    if not args.model:
        print("Error: --model is required unless using --compare")
        sys.exit(1)
    run_benchmark(
        model_path=args.model,
        label=args.label,
        tokenizer_path=args.tokenizer,
        num_samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_problems=args.max,
    )


if __name__ == "__main__":
    main()
