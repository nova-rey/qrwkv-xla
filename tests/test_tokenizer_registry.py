from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from qrwkv_xla.generation.tokenizer import (
    TokenizerConfig,
    TokenizerMetadata,
    available_tokenizer_backends,
    create_tokenizer,
    normalize_tokenizer_config,
)


def test_tokenizer_registry_keeps_smoke_default() -> None:
    tokenizer = create_tokenizer()

    assert tokenizer.metadata.backend == "smoke"
    assert tokenizer.metadata.vocab_size == 512
    assert tokenizer.encode("A") == [66]
    assert "smoke" in available_tokenizer_backends()


def test_tokenizer_config_normalizes_string_and_mapping_forms() -> None:
    assert normalize_tokenizer_config("qwen").backend == "hf"

    config = normalize_tokenizer_config(
        {
            "backend": "qwen",
            "tokenizer_id": "Qwen/Qwen2.5-0.5B",
            "vocab_size": 151936,
            "eos_token_id": 151643,
            "pad_token_id": 151643,
            "revision": "main",
            "trust_remote_code": True,
        }
    )

    assert config == TokenizerConfig(
        backend="hf",
        tokenizer_id="Qwen/Qwen2.5-0.5B",
        vocab_size=151936,
        eos_token_id=151643,
        pad_token_id=151643,
        revision="main",
        trust_remote_code=True,
    )

    config = normalize_tokenizer_config(
        {
            "backend": "hf",
            "tokenizer_id": "Qwen/Qwen2.5-0.5B",
            "local_files_only": True,
            "use_fast": False,
        }
    )

    assert config.local_files_only is True
    assert config.use_fast is False


def test_hf_tokenizer_backend_raises_install_hint_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(RuntimeError, match=r"\.\[teacher-hf\]"):
        create_tokenizer({"backend": "hf", "tokenizer_id": "local/tokenizer"})


def test_hf_tokenizer_backend_can_be_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tokenizer = _FakeHFTokenizer()
    calls: list[dict[str, Any]] = []

    def _from_pretrained(*_args: object, **kwargs: object) -> _FakeHFTokenizer:
        calls.append(dict(kwargs))
        return fake_tokenizer

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=_from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    tokenizer = create_tokenizer(
        {
            "backend": "hf",
            "tokenizer_id": "local/tokenizer",
            "vocab_size": 1024,
            "eos_token_id": 9,
            "pad_token_id": 8,
            "revision": "main",
            "local_files_only": True,
            "use_fast": False,
        }
    )

    assert calls == [
        {
            "revision": "main",
            "trust_remote_code": False,
            "local_files_only": True,
            "use_fast": False,
        }
    ]
    assert tokenizer.metadata == TokenizerMetadata(
        backend="hf",
        tokenizer_id="local/tokenizer",
        vocab_size=1024,
        eos_token_id=9,
        pad_token_id=8,
        revision="main",
        unk_token_id=7,
    )
    assert tokenizer.encode("abc", max_length=2) == [1, 2]
    assert tokenizer.decode([1, 2, 3]) == "decoded:1,2,3"


@pytest.mark.skipif(
    os.environ.get("QRWKV_XLA_RUN_HF_TOKENIZER_INTEGRATION") != "1",
    reason=(
        "set QRWKV_XLA_RUN_HF_TOKENIZER_INTEGRATION=1 "
        "to run optional HF tokenizer integration"
    ),
)
def test_hf_tokenizer_optional_integration() -> None:
    pytest.importorskip("transformers")

    tokenizer = create_tokenizer(
        {
            "backend": "hf",
            "tokenizer_id": os.environ.get(
                "QRWKV_XLA_HF_TOKENIZER_ID", "Qwen/Qwen2.5-0.5B"
            ),
            "revision": "main",
            "local_files_only": os.environ.get("QRWKV_XLA_HF_LOCAL_FILES_ONLY") == "1",
        }
    )

    assert tokenizer.metadata.vocab_size > 0
    assert tokenizer.encode("hello")


@dataclass
class _FakeHFTokenizer:
    vocab_size: int = 321
    eos_token_id: int = 0
    pad_token_id: int | None = None
    unk_token_id: int | None = 7
    eos_token: str = "<eos>"
    pad_token: str | None = None

    def encode(self, text: str, **kwargs: object) -> list[int]:
        token_ids = [index + 1 for index, _char in enumerate(text)]
        max_length = kwargs.get("max_length")
        if max_length is not None:
            token_ids = token_ids[: int(max_length)]
        return token_ids

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)
