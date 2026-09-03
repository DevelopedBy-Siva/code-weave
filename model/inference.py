"""SageMaker-ready inference wrapper for merged PyCodeGen models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import InferenceConfig, default_config
from model_registry import format_prompt, load_model, load_tokenizer

SUPPORTED_TASKS = ("generate", "explain", "review", "chat")


@dataclass(slots=True)
class PredictionRequest:
    """JSON-serializable inference request."""

    prompt: str
    task: str = "generate"
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class PyCodeGenPredictor:
    """Small prediction facade around a merged causal language model."""

    def __init__(self, model_name_or_path: str | Path, generation_config: InferenceConfig | None = None) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.generation_config = generation_config or default_config().inference
        self.tokenizer = load_tokenizer(self.model_name_or_path)
        self.model = load_model(self.model_name_or_path)

    def predict(
        self,
        prompt: str,
        task: str = "generate",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, str | int]:
        """Generate model output for a supported task."""
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Expected one of {SUPPORTED_TASKS}.")

        max_tokens = max_new_tokens or self.generation_config.max_new_tokens
        temp = self.generation_config.temperature if temperature is None else temperature
        nucleus = self.generation_config.top_p if top_p is None else top_p
        input_text = format_prompt(prompt, self.tokenizer, task=task)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=nucleus,
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {"output": text, "tokens": int(len(generated_ids))}


_PREDICTOR: PyCodeGenPredictor | None = None


def predict(prompt: str, task: str = "generate", model_path: str | Path | None = None, **kwargs: Any) -> dict[str, str | int]:
    """Module-level prediction helper for scripts and services."""
    global _PREDICTOR
    target_model = str(model_path or default_config().output_dir / "merged_model")
    if _PREDICTOR is None or _PREDICTOR.model_name_or_path != target_model:
        _PREDICTOR = PyCodeGenPredictor(target_model)
    return _PREDICTOR.predict(prompt=prompt, task=task, **kwargs)


def model_fn(model_dir: str) -> PyCodeGenPredictor:
    """SageMaker model loader."""
    return PyCodeGenPredictor(model_dir)


def input_fn(request_body: str, request_content_type: str = "application/json") -> PredictionRequest:
    """Parse a SageMaker request body."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    return PredictionRequest(**json.loads(request_body))


def predict_fn(request: PredictionRequest, predictor: PyCodeGenPredictor) -> dict[str, str | int]:
    """Run SageMaker prediction."""
    return predictor.predict(**asdict(request))


def output_fn(prediction: dict[str, str | int], response_content_type: str = "application/json") -> str:
    """Serialize a SageMaker prediction response."""
    if response_content_type != "application/json":
        raise ValueError(f"Unsupported response type: {response_content_type}")
    return json.dumps(prediction)


def parse_args() -> argparse.Namespace:
    """Parse command line inference arguments."""
    parser = argparse.ArgumentParser(description="Run PyCodeGen inference.")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", type=str, default=str(default_config().output_dir / "merged_model"))
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="generate")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Run local command line inference."""
    args = parse_args()
    result = predict(args.prompt, task=args.task, model_path=args.model, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
