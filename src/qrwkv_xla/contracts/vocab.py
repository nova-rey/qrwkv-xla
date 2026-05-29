from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from qrwkv_xla.targets.schema import TargetStoreMetadata


@dataclass(frozen=True)
class VocabContract:
    tokenizer_id: str
    vocab_size: int
    tokenizer_hash: str | None = None
    model_id: str | None = None
    model_family: str | None = None
    special_tokens: Mapping[str, int] = field(default_factory=dict)
    chat_template_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.tokenizer_id.strip():
            raise ValueError("tokenizer_id must be non-empty")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        for name, token_id in self.special_tokens.items():
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"special token {name!r} id {token_id} is outside vocab_size "
                    f"{self.vocab_size}"
                )


def vocab_contract_from_metadata(metadata: TargetStoreMetadata) -> VocabContract:
    return VocabContract(
        tokenizer_id=metadata.tokenizer_id,
        tokenizer_hash=metadata.tokenizer_hash,
        vocab_size=metadata.vocab_size,
        model_id=metadata.model_id,
        model_family=metadata.model_family,
    )
