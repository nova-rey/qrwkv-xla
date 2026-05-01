from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]
    decoded_text: str | None = None


def greedy_generate(
    *,
    student,
    params: dict,
    prompt_token_ids: list[int] | tuple[int, ...],
    max_new_tokens: int = 16,
    eos_token_id: int | None = 0,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    token_ids = list(prompt)
    if not token_ids:
        if eos_token_id is None:
            raise ValueError(
                "prompt_token_ids must be non-empty when eos_token_id is None"
            )
        token_ids.append(int(eos_token_id))

    generated: list[int] = []
    for _ in range(max_new_tokens):
        input_ids = jnp.asarray([token_ids], dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        output = student.apply(params, input_ids, attention_mask)
        if output.logits is None:
            raise ValueError("generation requires a logits-capable student checkpoint")
        logits = jnp.asarray(output.logits)
        if logits.ndim != 3:
            raise ValueError(
                f"student logits must have shape [B,S,V], got {logits.shape}"
            )
        next_token_id = int(jnp.argmax(logits[0, -1, :]))
        generated.append(next_token_id)
        token_ids.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return GenerationResult(
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        full_token_ids=tuple(token_ids),
    )
