from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from qrwkv_xla.generation import greedy_generate
from qrwkv_xla.students.base import StudentOutput


@dataclass(frozen=True)
class FakeLogitStudent:
    next_token_id: int
    vocab_size: int = 8

    def apply(self, params, input_ids, attention_mask=None):
        del params, attention_mask
        batch_size, sequence_length = input_ids.shape
        logits = jnp.zeros((batch_size, sequence_length, self.vocab_size))
        logits = logits.at[:, -1, self.next_token_id].set(1.0)
        return StudentOutput(
            hidden_states=jnp.zeros((batch_size, 1, sequence_length, 2)), logits=logits
        )


@dataclass(frozen=True)
class FakeHiddenOnlyStudent:
    def apply(self, params, input_ids, attention_mask=None):
        del params, attention_mask
        batch_size, sequence_length = input_ids.shape
        return StudentOutput(
            hidden_states=jnp.zeros((batch_size, 1, sequence_length, 2)),
            logits=None,
        )


def test_greedy_generation_returns_expected_lengths() -> None:
    result = greedy_generate(
        student=FakeLogitStudent(next_token_id=3),
        params={},
        prompt_token_ids=[1, 2],
        max_new_tokens=4,
        eos_token_id=None,
    )

    assert result.prompt_token_ids == (1, 2)
    assert result.generated_token_ids == (3, 3, 3, 3)
    assert result.full_token_ids == (1, 2, 3, 3, 3, 3)


def test_greedy_generation_stops_at_eos() -> None:
    result = greedy_generate(
        student=FakeLogitStudent(next_token_id=0),
        params={},
        prompt_token_ids=[1],
        max_new_tokens=4,
        eos_token_id=0,
    )

    assert result.generated_token_ids == (0,)
    assert result.full_token_ids == (1, 0)


def test_greedy_generation_raises_without_logits() -> None:
    with pytest.raises(ValueError, match="logits-capable"):
        greedy_generate(
            student=FakeHiddenOnlyStudent(),
            params={},
            prompt_token_ids=[1],
            max_new_tokens=1,
        )


def test_greedy_generation_validates_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_generate(
            student=FakeLogitStudent(next_token_id=1),
            params={},
            prompt_token_ids=[1],
            max_new_tokens=0,
        )
