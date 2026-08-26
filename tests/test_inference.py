"""Tests for inference request and output format."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

from model import inference


class FakeInputs(dict):
    """Minimal tokenizer output with a .to method."""

    def to(self, _: str) -> "FakeInputs":
        return self


class FakeIds:
    """Tiny object exposing the shape protocol used by inference."""

    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class FakeGenerated:
    """Tiny generated output object with input slicing support."""

    def __getitem__(self, _: slice) -> list[int]:
        return [10, 11, 12]


class FakeTokenizer:
    """Tokenizer double used to avoid loading a real model."""

    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str) -> FakeInputs:
        assert return_tensors == "pt"
        assert "### Instruction:" in text
        return FakeInputs({"input_ids": FakeIds(length=3)})

    def decode(self, ids: Any, skip_special_tokens: bool) -> str:
        assert ids == [10, 11, 12]
        assert skip_special_tokens is True
        return "def hello():\n    return 'world'"


class FakeModel:
    """Model double returning deterministic token ids."""

    device = "cpu"

    def generate(self, **_: Any) -> list[FakeGenerated]:
        return [FakeGenerated()]


def test_predictor_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """predict returns SageMaker-ready output and token count."""
    monkeypatch.setattr(inference, "load_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(inference, "load_model", lambda _: FakeModel())
    predictor = inference.PyCodeGenPredictor("fake-model")

    result = predictor.predict("write a hello function")

    assert result == {"output": "def hello():\n    return 'world'", "tokens": 3}


def test_sagemaker_request_round_trip() -> None:
    """SageMaker helpers should parse and serialize JSON payloads."""
    request = inference.input_fn('{"prompt": "x", "task": "review"}')
    assert asdict(request) == {
        "prompt": "x",
        "task": "review",
        "max_new_tokens": None,
        "temperature": None,
        "top_p": None,
    }
    assert inference.output_fn({"output": "ok", "tokens": 1}) == '{"output": "ok", "tokens": 1}'
