"""Fine-tune Qwen2.5-Coder-7B-Instruct on cleaned Python instruction data."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments

try:  # Support both ``python model/train.py`` and ``python -m model.train``.
    from .config import AppConfig, add_config_arguments, apply_overrides, default_config
    from .model_registry import INSTRUCTION_TEMPLATES, torch_dtype
    from .training_data import make_completion_variant
except ImportError:
    from config import AppConfig, add_config_arguments, apply_overrides, default_config
    from model_registry import INSTRUCTION_TEMPLATES, torch_dtype
    from training_data import make_completion_variant

TRAINING_LOG = Path(__file__).resolve().parent / "training.log"
logger = logging.getLogger(__name__)
REQUIRED_CLEANING_VERSION = 3


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
    stats_path = config.cleaned_data_path.with_name("cleaning_stats.json")
    stats = {}
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
    if stats.get("cleaning_version") != REQUIRED_CLEANING_VERSION:
        raise RuntimeError(
            "The dataset predates the code-quality cleaner. Rerun "
            "`python data/download_datasets.py --download --clean --validate` before training."
        )
    logger.info("Loaded %s cleaned samples from %s", len(records), config.cleaned_data_path)
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, config: AppConfig) -> Dataset:
    """Tokenize complete chat turns and train only on assistant tokens.

    The full assistant turn is rendered by the tokenizer rather than manually
    appending an EOS token.  This keeps Qwen's ``<|im_end|>`` boundaries exact.
    Over-length rows are discarded instead of truncating the end of a solution
    and teaching the model incomplete code.
    """

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        user_content = example["instruction"].strip()
        if example.get("input", "").strip():
            user_content += f"\n\n{example['input'].strip()}"
        prompt_messages = [
            {"role": "system", "content": INSTRUCTION_TEMPLATES["generate"]},
            {"role": "user", "content": user_content},
        ]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": example["output"].strip("\r\n")},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        result = tokenizer(full_text, truncation=False, add_special_tokens=False)
        full_ids = result["input_ids"]

        # Tokenization at the prompt/answer string boundary can merge a token for
        # some tokenizers.  The actual common prefix is the safe masking boundary.
        prompt_len = 0
        for prompt_id, full_id in zip(prompt_ids, full_ids):
            if prompt_id != full_id:
                break
            prompt_len += 1

        labels = full_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        result["labels"] = labels
        result["supervised_tokens"] = len(labels) - prompt_len
        result["within_context"] = len(full_ids) <= config.model.max_seq_length
        return result

    tokenized = dataset.map(
        tokenize,
        remove_columns=["instruction", "input", "output"],
        num_proc=4,
    )

    before_filter = len(tokenized)
    tokenized = tokenized.filter(
        lambda example: example["within_context"] and example["supervised_tokens"] > 1,
        num_proc=4,
    )
    supervised_tokens = sum(tokenized["supervised_tokens"]) if len(tokenized) else 0
    tokenized = tokenized.remove_columns(["within_context", "supervised_tokens"])

    logger.info(
        "Tokenized %s samples; kept %s complete rows with %s supervised tokens",
        before_filter,
        len(tokenized),
        f"{supervised_tokens:,}",
    )
    if not len(tokenized):
        raise ValueError("No complete, supervised examples fit within max_seq_length")
    return tokenized


def augment_completion_examples(dataset: Dataset, ratio: float) -> Dataset:
    """Add deterministic completion variants without crossing split boundaries."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("completion_augmentation_ratio must be between 0 and 1")
    records = [dict(example) for example in dataset]
    variants: list[dict[str, str]] = []
    threshold = int(ratio * 10_000)
    for example in records:
        digest = hashlib.sha256(
            f"{example['instruction']}\n{example.get('input', '')}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        if bucket >= threshold:
            continue
        variant = make_completion_variant(example)
        if variant is not None:
            variants.append(variant)
    logger.info("Added %s completion-style variants to %s source tasks", len(variants), len(records))
    return Dataset.from_list([*records, *variants])


def prepare_dataset(dataset: Dataset, tokenizer: Any, config: AppConfig) -> tuple[Dataset, Dataset]:
    """Split by source task, augment within each split, and tokenize."""
    if config.training.max_samples and len(dataset) > config.training.max_samples:
        dataset = dataset.shuffle(seed=config.training.seed).select(range(config.training.max_samples))

    if len(dataset) < 2:
        raise ValueError("At least two examples are required for a train/eval split")
    raw_split = dataset.train_test_split(
        test_size=config.training.val_split, seed=config.training.seed
    )
    train_source = augment_completion_examples(
        raw_split["train"], config.training.completion_augmentation_ratio
    )
    eval_source = augment_completion_examples(
        raw_split["test"], config.training.completion_augmentation_ratio
    )
    train_dataset = tokenize_dataset(train_source, tokenizer, config)
    eval_dataset = tokenize_dataset(eval_source, tokenizer, config)
    if not len(train_dataset) or not len(eval_dataset):
        raise ValueError("At least two tokenized examples are required for a train/eval split")
    logger.info("Train: %s | Val: %s", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset


def load_lora_model(config: AppConfig) -> tuple[Any, Any]:
    """Load the base model and attach LoRA adapters."""
    # Keep this import local so data/tokenization tests do not require a CUDA
    # Unsloth installation.
    from unsloth import FastLanguageModel

    logger.info("Loading model: %s", config.model.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        dtype=torch_dtype(config.model.precision),
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
    """Save the eval-loss winner as an adapter for diagnostics.

    A merged deployment model is intentionally created only after executable
    checkpoint selection; token-level validation loss cannot promote a model.
    """
    adapter_path = config.output_dir / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved to %s", adapter_path)
    logger.info(
        "Run model/select_checkpoint.py to select by executable validation and create the merged model"
    )


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
