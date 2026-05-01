from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeTokenizer:
    vocab_size: int = 512
    eos_token_id: int = 0
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size < 257:
            raise ValueError("SmokeTokenizer byte mode requires vocab_size >= 257")
        if self.eos_token_id != 0 or self.pad_token_id != 0:
            raise ValueError("SmokeTokenizer reserves token 0 for EOS/PAD")

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        token_ids = [byte + 1 for byte in text.encode("utf-8")]
        if max_length is not None:
            if max_length < 0:
                raise ValueError("max_length must be >= 0")
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        chunks: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id == self.eos_token_id:
                break
            if 1 <= token_id <= 256:
                byte_buffer.append(token_id - 1)
                continue
            flush_bytes()
            chunks.append(f"<tok_{token_id}>")
        flush_bytes()
        return "".join(chunks)
