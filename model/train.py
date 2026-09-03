"""Fine-tune Qwen2.5-Coder-7B-Instruct on cleaned Python instruction data."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments
from unsloth import FastLanguageModel

from config import AppConfig, add_config_arguments, apply_overrides, default_config
from model_registry import save_merged_model

TRAINING_LOG = Path(__file__).resolve().parent / "training.log"
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console and file logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(TRAINING_LOG)],
    )


def load_clean_dataset(config: AppConfig) -> Dataset:
    """Load processed instruction data from data/processed/cleaned_data.json."""
    if not config.cleaned_data_path.exists():
        raise FileNotFoundError(
            f"Missing {config.cleaned_data_path}. Run `python data/download_datasets.py --download --clean --validate` first."
        )
    with config.cleaned_data_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    logger.info("Loaded %s cleaned samples from %s", len(records), config.cleaned_data_path)
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, config: AppConfig) -> Dataset:
    """Pre-tokenize examples for Hugging Face Trainer."""
    eos_token = tokenizer.eos_token

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        input_section = f"\n\n### Input:\n{example['input'].strip()}" if example.get("input", "").strip() else ""
        text = (
            f"### Instruction:\n{example['instruction'].strip()}"
            f"{input_section}\n\n"
            f"### Response:\n{example['output'].strip()}"
            f"{eos_token}"
        )
        result = tokenizer(text, truncation=True, max_length=config.model.max_seq_length, padding=False)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, remove_columns=["instruction", "input", "output"], num_proc=4)
    logger.info("Tokenized %s samples", len(tokenized))
    return tokenized


def prepare_dataset(dataset: Dataset, tokenizer: Any, config: AppConfig) -> tuple[Dataset, Dataset]:
    """Tokenize and split the dataset into train/eval sets."""
    if config.training.max_samples and len(dataset) > config.training.max_samples:
        dataset = dataset.shuffle(seed=config.training.seed).select(range(config.training.max_samples))

    tokenized = tokenize_dataset(dataset, tokenizer, config)
    split = tokenized.train_test_split(test_size=config.training.val_split, seed=config.training.seed)
    logger.info("Train: %s | Val: %s", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


def load_lora_model(config: AppConfig) -> tuple[Any, Any]:
    """Load the base model and attach LoRA adapters."""
    logger.info("Loading model: %s", config.model.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.training.seed,
    )

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info("Trainable parameters: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model, tokenizer


def train_model(model: Any, tokenizer: Any, train_dataset: Dataset, eval_dataset: Dataset, config: AppConfig) -> Trainer:
    """Train the model with Hugging Face Trainer."""
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        warmup_steps=config.training.warmup_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        optim=config.training.optim,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_strategy="steps",
        save_total_limit=config.training.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.training.seed,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0), gpu_memory)

    stats = trainer.train()
    logger.info("Training complete: loss=%.4f runtime=%.0fs", stats.metrics["train_loss"], stats.metrics["train_runtime"])
    return trainer


def save_model_artifacts(model: Any, tokenizer: Any, config: AppConfig) -> None:
    """Save LoRA adapter and merged model artifacts."""
    adapter_path = config.output_dir / "lora_adapter"
    merged_path = config.output_dir / "merged_model"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved to %s", adapter_path)
    save_merged_model(model, tokenizer, merged_path)
    logger.info("Merged model saved to %s", merged_path)


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune PyCodeGen with LoRA.")
    add_config_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end training job."""
    setup_logging()
    config = apply_overrides(default_config(), parse_args())
    logger.info("Qwen2.5-Coder 7B | Python fine-tuning | Unsloth LoRA")
    model, tokenizer = load_lora_model(config)
    dataset = load_clean_dataset(config)
    train_dataset, eval_dataset = prepare_dataset(dataset, tokenizer, config)
    train_model(model, tokenizer, train_dataset, eval_dataset, config)
    save_model_artifacts(model, tokenizer, config)
    logger.info("All done: %s", config.output_dir)


if __name__ == "__main__":
    main()
