from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

INSTALL_HINT = 'python -m pip install -e ".[teacher-hf]"'


class TokenizerLoadError(RuntimeError):
    """Raised when an optional tokenizer backend cannot be loaded."""


@dataclass(frozen=True)
class TokenizerConfig:
    backend: str = "smoke"
    tokenizer_id: str | None = None
    vocab_size: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    use_fast: bool = True


@dataclass(frozen=True)
class TokenizerMetadata:
    backend: str
    tokenizer_id: str | None
    vocab_size: int
    eos_token_id: int | None
    pad_token_id: int | None
    revision: str | None = None
    unk_token_id: int | None = None


class Tokenizer(Protocol):
    metadata: TokenizerMetadata

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]: ...

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str: ...


@dataclass(frozen=True)
class SmokeTokenizer:
    vocab_size: int = 512
    eos_token_id: int = 0
    pad_token_id: int = 0

    @property
    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata(
            backend="smoke",
            tokenizer_id="smoke",
            vocab_size=self.vocab_size,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

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


class HFTokenizer:
    def __init__(self, config: TokenizerConfig) -> None:
        if not config.tokenizer_id:
            raise ValueError("HF tokenizer requires tokenizer_id")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerLoadError(
                "HF tokenizer backend requires optional transformers dependency. "
                f"Install it with: {INSTALL_HINT}"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            use_fast=config.use_fast,
        )
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(self._tokenizer, "eos_token", None)
            if (
                eos_token is not None
                and getattr(self._tokenizer, "pad_token", None) is None
            ):
                self._tokenizer.pad_token = eos_token
        vocab_size = config.vocab_size or _hf_vocab_size(self._tokenizer)
        eos_token_id = config.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        pad_token_id = config.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        self._metadata = TokenizerMetadata(
            backend=config.backend,
            tokenizer_id=config.tokenizer_id,
            vocab_size=int(vocab_size),
            eos_token_id=None if eos_token_id is None else int(eos_token_id),
            pad_token_id=None if pad_token_id is None else int(pad_token_id),
            revision=config.revision,
            unk_token_id=_optional_int(getattr(self._tokenizer, "unk_token_id", None)),
        )

    @property
    def metadata(self) -> TokenizerMetadata:
        return self._metadata

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        kwargs: dict[str, object] = {"add_special_tokens": False}
        if max_length is not None:
            if max_length < 0:
                raise ValueError("max_length must be >= 0")
            kwargs.update({"max_length": max_length, "truncation": True})
        token_ids = self._tokenizer.encode(text, **kwargs)
        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=True))


TokenizerFactory = Callable[[TokenizerConfig], Tokenizer]

_TOKENIZER_REGISTRY: dict[str, TokenizerFactory] = {}
_ALIASES = {"qwen": "hf"}


def normalize_tokenizer_config(
    value: str | Mapping[str, Any] | TokenizerConfig,
) -> TokenizerConfig:
    if isinstance(value, TokenizerConfig):
        return value
    if isinstance(value, str):
        return TokenizerConfig(backend=_canonical_backend(value))
    if not isinstance(value, Mapping):
        raise ValueError("tokenizer config must be a string or mapping")
    raw_backend = value.get("backend", value.get("type", value.get("name", "smoke")))
    backend = _canonical_backend(str(raw_backend))
    tokenizer_id = value.get("tokenizer_id", value.get("id", value.get("model_id")))
    return TokenizerConfig(
        backend=backend,
        tokenizer_id=None if tokenizer_id is None else str(tokenizer_id),
        vocab_size=_optional_int(value.get("vocab_size")),
        eos_token_id=_optional_int(value.get("eos_token_id")),
        pad_token_id=_optional_int(value.get("pad_token_id")),
        revision=_optional_str(value.get("revision")),
        trust_remote_code=bool(value.get("trust_remote_code", False)),
        local_files_only=bool(value.get("local_files_only", False)),
        use_fast=bool(value.get("use_fast", True)),
    )


def register_tokenizer_backend(name: str, factory: TokenizerFactory) -> None:
    canonical = _canonical_backend(name)
    _TOKENIZER_REGISTRY[canonical] = factory


def available_tokenizer_backends() -> tuple[str, ...]:
    return tuple(sorted([*_TOKENIZER_REGISTRY, "qwen"]))


def create_tokenizer(
    value: str | Mapping[str, Any] | TokenizerConfig = "smoke",
) -> Tokenizer:
    config = normalize_tokenizer_config(value)
    try:
        factory = _TOKENIZER_REGISTRY[config.backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tokenizer backend {config.backend!r}; "
            f"available backends: {', '.join(available_tokenizer_backends())}"
        ) from exc
    return factory(config)


def _create_smoke_tokenizer(config: TokenizerConfig) -> SmokeTokenizer:
    return SmokeTokenizer(
        vocab_size=config.vocab_size or 512,
        eos_token_id=0 if config.eos_token_id is None else config.eos_token_id,
        pad_token_id=0 if config.pad_token_id is None else config.pad_token_id,
    )


def _create_hf_tokenizer(config: TokenizerConfig) -> HFTokenizer:
    return HFTokenizer(config)


def _canonical_backend(value: str) -> str:
    backend = value.strip().lower()
    return _ALIASES.get(backend, backend)


def _hf_vocab_size(tokenizer: Any) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    try:
        return int(len(tokenizer))
    except TypeError as exc:
        raise ValueError("HF tokenizer does not expose vocab_size or __len__") from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


register_tokenizer_backend("smoke", _create_smoke_tokenizer)
register_tokenizer_backend("hf", _create_hf_tokenizer)
