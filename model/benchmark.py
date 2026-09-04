"""Run HumanEval benchmarks for baseline and fine-tuned models."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import multiprocessing
import re
import sys
import time
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any

try:  # Support both script and package execution.
    from .config import default_config
    from .model_registry import format_prompt, load_model as registry_load_model, load_tokenizer
except ImportError:
    from config import default_config
    from model_registry import format_prompt, load_model as registry_load_model, load_tokenizer

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


CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
CODE_START_RE = re.compile(
    r"^(?:\s+\S|@|#|def\s|async\s+def\s|class\s|from\s|import\s|return\s|raise\s|if\s|for\s|while\s|try:|with\s)"
)


def extract_code(response: str) -> str:
    """Remove response prose while preserving valid imports and indentation.

    Leading indentation is significant for HumanEval because many models emit a
    function *completion* instead of repeating the full definition.  The old
    ``strip()`` converted ``"    return x"`` to invalid top-level code.  It also
    discarded imports before the first function and stopped at a second import.
    """
    fenced = CODE_FENCE_RE.findall(response)
    if fenced:
        return max(fenced, key=len).strip("\r\n")

    lines = response.strip("\r\n").splitlines()
    for index, line in enumerate(lines):
        if CODE_START_RE.match(line):
            lines = lines[index:]
            break
    return "\n".join(lines).rstrip()


def build_solution(prompt: str, response: str) -> str:
    """Combine a HumanEval prompt with the longest syntactically valid response.

    The response may be either an indented continuation or a complete function.
    Appending a complete definition is valid Python and intentionally replaces
    the stub from the prompt.  Trailing natural-language lines are removed only
    when needed to make the combined program parse.
    """
    code = extract_code(response)
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        solution = f"{prompt.rstrip()}\n{chr(10).join(lines[:end]).rstrip()}\n"
        try:
            ast.parse(solution)
        except SyntaxError:
            continue
        return solution
    return f"{prompt.rstrip()}\n{code.rstrip()}\n"


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
    import torch

    input_text = format_prompt(prompt, tokenizer, task="generate")
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    solutions: list[str] = []

    with torch.no_grad():
        for _ in range(num_samples):
            do_sample = temperature is not None and temperature > 0

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p

            outputs = model.generate(**generation_kwargs)

            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            solutions.append(build_solution(prompt, response))
    return solutions


def _execution_worker(program: str, connection: Any) -> None:
    """Execute generated code in an isolated child process."""
    result: dict[str, bool | str | None] = {"passed": False, "error": None}
    exec_globals: dict[str, Any] = {}
    try:
        exec(program, exec_globals)
        result["passed"] = True
    except AssertionError as exc:
        result["error"] = f"AssertionError: {exc}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    connection.send(result)
    connection.close()


def run_program_safely(program: str, timeout_seconds: float = 10.0) -> dict[str, bool | str | None]:
    """Run generated Python with a timeout so one loop cannot stall an evaluation."""
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_execution_worker, args=(program, child_connection))
    process.start()
    child_connection.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join()
        parent_connection.close()
        return {"passed": False, "error": f"Timeout after {timeout_seconds:g}s"}
    if parent_connection.poll():
        result = parent_connection.recv()
    else:
        result = {"passed": False, "error": f"Worker exited with code {process.exitcode}"}
    parent_connection.close()
    return result


def run_tests_safely(solution: str, test_code: str, entry_point: str) -> dict[str, bool | str | None]:
    """Execute HumanEval tests for one generated solution."""
    program = (
        f"{solution}\n{test_code}\n"
        f"assert {entry_point!r} in globals(), \"Function {entry_point!r} not found in output\"\n"
        f"check({entry_point})\n"
    )
    return run_program_safely(program)


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
    from datasets import load_dataset
    from tqdm import tqdm

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
                "entry_point": problem["entry_point"],
                "num_correct": num_correct,
                "num_samples": num_samples,
                "pass@1": pass_at_k(num_samples, num_correct, 1),
                "errors": [result["error"] for result in problem_results if result["error"]],
                "solutions": solutions,
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


def compare_results(min_improvement: float = 2.0, min_score: float = 80.0) -> None:
    """Compare reports and require a meaningful fine-tuning improvement."""
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
    failures: list[str] = []
    if finetuned["pass@1"] < min_score:
        failures.append(
            f"fine-tuned pass@1 {finetuned['pass@1']:.2f} is below {min_score:.2f}"
        )
    if delta_pass1 < min_improvement:
        failures.append(
            f"gain {delta_pass1:+.2f} is below required {min_improvement:+.2f} points"
        )
    if failures:
        raise SystemExit(f"Quality gate failed: {'; '.join(failures)}.")


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    cfg = default_config()
    parser = argparse.ArgumentParser(description="HumanEval benchmarking for PyCodeGen models.")
    parser.add_argument("--model", type=str, default=None, help="Model path or Hugging Face name.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path or Hugging Face name.")
    parser.add_argument("--label", type=str, default="baseline", help="Result label.")
    parser.add_argument("--samples", type=int, default=1, help="Samples per problem.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.inference.max_new_tokens)
    parser.add_argument("--max", type=int, default=None, help="Maximum HumanEval problems.")
    parser.add_argument("--compare", action="store_true", help="Compare baseline and fine-tuned reports.")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=2.0,
        help="Minimum pass@1 percentage-point gain required by --compare.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=80.0,
        help="Minimum fine-tuned pass@1 percentage required by --compare.",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmark CLI."""
    setup_logging()
    args = parse_args()
    if args.compare:
        compare_results(
            min_improvement=args.min_improvement,
            min_score=args.min_score,
        )
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
