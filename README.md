# PyCodeGen

Fine-tune `Qwen/Qwen2.5-Coder-7B-Instruct` on curated Python instruction data, then export a merged model for benchmark and SageMaker-ready inference.

## Architecture

```text
Hugging Face datasets
        |
        v
data/download_datasets.py
  download -> normalize -> remove EOS -> filter Python -> deduplicate
        |
        v
data/processed/cleaned_data.json
        |
        v
model/train.py ---------------> model/qwen-python-finetuned/
        |                         lora_adapter/
        |                         merged_model/
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
│   ├── config.py
│   └── model_registry.py
├── tests/
│   ├── test_data_cleaning.py
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

The pipeline downloads five Hugging Face datasets, normalizes rows into `instruction`, `input`, and `output`, removes foreign EOS tokens, filters non-Python and low-quality examples, deduplicates records, and writes `data/processed/cleaned_data.json`. Cleaning stats are written to `data/processed/cleaning_stats.json`.

## Training

```bash
python model/train.py
```

All model, LoRA, training, path, and inference defaults live in `model/config.py`. Hyperparameters can be overridden from the command line:

```bash
python model/train.py --training-learning-rate 0.0001 --training-max-samples 5000
```

## Benchmark

```bash
python model/benchmark.py --model Qwen/Qwen2.5-Coder-7B-Instruct --label baseline
python model/benchmark.py --model model/qwen-python-finetuned/merged_model --label finetuned
python model/benchmark.py --compare
```

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
