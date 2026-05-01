from __future__ import annotations

import pytest

from qrwkv_xla.generation import SmokeTokenizer


def test_smoke_tokenizer_round_trips_ascii() -> None:
    tokenizer = SmokeTokenizer(vocab_size=512)

    token_ids = tokenizer.encode("Hello")

    assert token_ids == [73, 102, 109, 109, 112]
    assert tokenizer.decode(token_ids) == "Hello"


def test_smoke_tokenizer_handles_empty_string() -> None:
    tokenizer = SmokeTokenizer(vocab_size=512)

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_smoke_tokenizer_unknown_token_decode() -> None:
    tokenizer = SmokeTokenizer(vocab_size=512)

    assert tokenizer.decode([73, 300, 74]) == "H<tok_300>I"


def test_smoke_tokenizer_requires_byte_vocab() -> None:
    with pytest.raises(ValueError, match="vocab_size >= 257"):
        SmokeTokenizer(vocab_size=256)


def test_smoke_tokenizer_truncates() -> None:
    tokenizer = SmokeTokenizer(vocab_size=512)

    assert tokenizer.encode("abcd", max_length=2) == [98, 99]
