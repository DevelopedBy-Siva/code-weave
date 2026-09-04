"""Central configuration for dataset preparation, training, and inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = MODEL_DIR / "qwen-python-finetuned"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.json"


@dataclass(slots=True)
class ModelConfig:
    """Model loading configuration."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    precision: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass(slots=True)
class LoRAConfig:
    """LoRA adapter hyperparameters."""

    # Keep the adapter deliberately small.  The base Coder model is already
    # strong; a 161M-parameter adapter at 2e-4 caused catastrophic forgetting.
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters."""

    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 250
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 25
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 20
    load_best_model_at_end: bool = True
    optim: str = "adamw_8bit"
    seed: int = 42
    val_split: float = 0.02
    max_samples: int | None = None
    completion_augmentation_ratio: float = 0.25


@dataclass(slots=True)
class InferenceConfig:
    """Default generation settings."""

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


@dataclass(slots=True)
class AppConfig:
    """Complete project configuration."""

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    inference: InferenceConfig
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    output_dir: Path = OUTPUT_DIR
    cleaned_data_path: Path = CLEANED_DATA_PATH


def default_config() -> AppConfig:
    """Return the default application config."""
    return AppConfig(model=ModelConfig(), lora=LoRAConfig(), training=TrainingConfig(), inference=InferenceConfig())


def parse_bool(value: str | bool) -> bool:
    """Parse booleans from CLI-friendly strings."""
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def apply_overrides(config: AppConfig, overrides: argparse.Namespace) -> AppConfig:
    """Apply known argparse overrides to the nested config object."""
    for group_name in ("model", "lora", "training", "inference"):
        group = getattr(config, group_name)
        for field in fields(group):
            value = getattr(overrides, f"{group_name}_{field.name}", None)
            if value is not None:
                current: Any = getattr(group, field.name)
                if isinstance(current, bool):
                    value = parse_bool(value)
                setattr(group, field.name, value)
    for path_name in ("data_dir", "model_dir", "output_dir", "cleaned_data_path"):
        value = getattr(overrides, path_name, None)
        if value is not None:
            setattr(config, path_name, Path(value))
    return config


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Register common config override arguments on an existing parser."""
    cfg = default_config()
    for group_name in ("model", "lora", "training", "inference"):
        group = getattr(cfg, group_name)
        for field in fields(group):
            value = getattr(group, field.name)
            if isinstance(value, tuple):
                continue
            arg_type = str if value is None else parse_bool if isinstance(value, bool) else type(value)
            parser.add_argument(f"--{group_name}-{field.name.replace('_', '-')}", dest=f"{group_name}_{field.name}", type=arg_type)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cleaned-data-path", type=Path)
