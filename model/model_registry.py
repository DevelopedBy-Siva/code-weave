"""Model and tokenizer loading helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import default_config

INSTRUCTION_TEMPLATES = {
    "generate": "Write Python code for the following request.",
    "explain": "Explain the following Python code clearly and concisely.",
    "review": "Review the following Python code for correctness, clarity, and risks.",
    "chat": "You are a helpful Python coding assistant.",
}


def torch_dtype(precision: str) -> Any:
    """Map precision config strings to torch dtype values."""
    import torch

    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float16":
        return torch.float16
    if precision == "float32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def format_prompt(prompt: str, tokenizer: Any, task: str = "generate") -> str:
    """Format user input using the model's native chat template."""
    if task not in INSTRUCTION_TEMPLATES:
        raise ValueError(
            f"Unsupported task '{task}'. Expected one of {sorted(INSTRUCTION_TEMPLATES)}."
        )

    messages = [
        {"role": "system", "content": INSTRUCTION_TEMPLATES[task]},
        {"role": "user", "content": prompt.strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_tokenizer(model_name_or_path: str | Path) -> Any:
    """Load a tokenizer and ensure it has a pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name_or_path: str | Path) -> Any:
    """Load a causal language model for inference or evaluation."""
    from transformers import AutoModelForCausalLM

    cfg = default_config()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_name_or_path),
        torch_dtype=torch_dtype(cfg.model.precision),
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.eval()
    return model


def save_merged_model(model: Any, tokenizer: Any, output_path: str | Path) -> None:
    """Save a merged model and tokenizer to an output directory."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(str(output), tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))