from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TinyTextExample:
    example_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")


def batch_tiny_text_examples(
    examples: Sequence[TinyTextExample],
    *,
    batch_size: int,
) -> tuple[tuple[TinyTextExample, ...], ...]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return tuple(
        tuple(examples[index : index + batch_size])
        for index in range(0, len(examples), batch_size)
    )
