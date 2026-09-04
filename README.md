# PyCodeGen

Fine-tune `Qwen/Qwen2.5-Coder-7B-Instruct` on curated Python instruction data, then export a merged model for benchmark and SageMaker-ready inference.

## Architecture

```text
Hugging Face datasets
        |
        v
data/download_datasets.py
  verified sources -> normalize -> validate Python -> decontaminate -> deduplicate
        |
        v
data/processed/cleaned_data.json
        |
        v
model/train.py ---------------> model/qwen-python-finetuned/checkpoint-*/
        |                         lora_adapter/
        v
model/select_checkpoint.py ---> merged_model/
        v
model/benchmark.py
        |
        v
model/inference.py
```

## Project Structure

```text
pycode-gen/
├── data/
│   ├── raw/
│   ├── processed/
│   └── download_datasets.py
├── model/
│   ├── train.py
│   ├── benchmark.py
│   ├── inference.py
│   ├── select_checkpoint.py
│   ├── training_data.py
│   ├── config.py
│   └── model_registry.py
├── tests/
│   ├── test_data_cleaning.py
│   ├── test_benchmark.py
│   ├── test_checkpoint_selection.py
│   ├── test_training_data.py
│   └── test_inference.py
├── .github/workflows/
│   └── train.yml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
└── README.md
```

## Setup

```bash
conda create -n pycode-gen python=3.11 -y
conda activate pycode-gen
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Data Pipeline

```bash
python data/download_datasets.py --download --clean --validate
```

The pipeline uses two higher-confidence sources: Open-R1's decontaminated solutions that passed executable tests and a 50k OpenCodeInstruct subset whose answers passed all generated tests and strict quality judging. The previous unverified instruction mixture is intentionally excluded. It extracts code from Markdown answers, removes generated demo/test suffixes, rejects syntax-invalid targets, removes HumanEval and held-out MBPP-validation contamination, and deduplicates by prompt. Cleaning stats and a cleaner version are written to `data/processed/cleaning_stats.json`.

Regenerate both raw and processed data before training. Dataset manifests prevent an old raw mixture from being relabeled as newly cleaned data:

```bash
python data/download_datasets.py --download --clean --validate
```

## Training

```bash
python model/train.py
```

All model, LoRA, training, path, and inference defaults live in `model/config.py`. The defaults use a conservative rank-16 adapter and `5e-5` learning rate to preserve the already-strong base model. A deterministic 25% of eligible verified functions also receive body-completion variants to match HumanEval without using HumanEval answers. The raw-task split happens before augmentation, targets use complete native Qwen chat turns, prompt tokens are masked from loss, and examples that would truncate a solution are dropped. Hyperparameters can be overridden from the command line:

```bash
python model/train.py --output-dir model/qwen-python-finetuned-v3
```

Training saves checkpoints every 250 optimizer steps. It does not create a deployable merged model because token-level validation loss is not a sufficient correctness metric.

## Checkpoint Selection

Select checkpoints on the 90 held-out, executable MBPP validation tasks. The command compares every retained checkpoint with the untouched base model, requires at least a two-percentage-point improvement, and only then creates `merged_model/`:

```bash
python model/select_checkpoint.py \
  --output-dir model/qwen-python-finetuned-v3 \
  --min-improvement 2
```

Selection details are saved to `checkpoint_selection.json`. If no checkpoint clears the gate, no merged model is created.

## Benchmark

```bash
python model/benchmark.py --model Qwen/Qwen2.5-Coder-7B-Instruct --label baseline
python model/benchmark.py --model model/qwen-python-finetuned-v3/merged_model --label finetuned
python model/benchmark.py --compare
```

The comparison command is a final acceptance gate: by default it requires at least 80% pass@1 and a gain of two percentage points over baseline. Use the same tokenizer, prompt code, decoding parameters, and full 164-problem set for both runs. Do not deploy the merged model until this gate passes.

## Inference

```bash
python model/inference.py "write a binary search function" --task generate
python model/inference.py "explain this code: print(sum(range(10)))" --task explain
```

`model/inference.py` returns SageMaker-ready JSON:

```json
{
  "output": "generated text",
  "tokens": 128
}
```

Supported tasks are `generate`, `explain`, `review`, and `chat`.

## Roadmap

| Phase | Goal |
| --- | --- |
| Phase 1 | Project setup, dataset pipeline, config, inference wrapper, tests |
| Phase 2 | Full LoRA training, merged-model export, HumanEval benchmark tracking |
| Phase 3 | AWS/SageMaker packaging, endpoint deployment, artifact storage |
| Phase 4 | Monitoring, evaluation automation, prompt/task expansion |

## Development

```bash
pytest -q
black data model tests
mypy data model tests
```
