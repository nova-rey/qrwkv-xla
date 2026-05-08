from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "int32": 4,
}

ModelBackend = Literal[
    "rwkv7_qwen_reference",
    "rwkv7_radlads_reference",
    "rwkv7_reference",
]
ModelStatus = Literal[
    "debug",
    "local_candidate",
    "tpu_smoke_candidate",
    "grant_candidate",
    "stretch_candidate",
    "placeholder_estimate",
]
TargetLayers = Literal["all", "final", "selected"]
Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    backend: ModelBackend
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_size: int | None = None
    mlp_hidden_size: int | None = None
    sequence_length_default: int = 128
    dtype_default: str = "fp32"
    status: ModelStatus = "debug"
    tie_embeddings: bool = False
    emit_logits: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def effective_head_size(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def resolved_head_size(self) -> int:
        return self.effective_head_size if self.head_size is None else self.head_size

    @property
    def effective_mlp_hidden_size(self) -> int:
        return (
            self.hidden_size * 4
            if self.mlp_hidden_size is None
            else self.mlp_hidden_size
        )

    @property
    def kv_hidden_size(self) -> int:
        return self.num_kv_heads * self.resolved_head_size


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    device_kind: Literal["cpu", "gpu", "tpu"]
    memory_gb: float
    device_count: int = 1
    per_device_memory_gb: float | None = None
    dtype_preference: str = "fp32"
    supports_bf16: bool = False
    supports_pjit: bool = False
    confidence: Confidence = "medium"
    notes: list[str] = field(default_factory=list)

    @property
    def resolved_per_device_memory_gb(self) -> float:
        return (
            self.memory_gb / self.device_count
            if self.per_device_memory_gb is None
            else self.per_device_memory_gb
        )

    @property
    def usable_memory_fraction(self) -> float:
        return 0.70 if self.device_kind == "tpu" else 0.80

    @property
    def usable_memory_bytes(self) -> int:
        return int(self.memory_gb * (1024**3) * self.usable_memory_fraction)


@dataclass(frozen=True)
class LossProfile:
    hidden_mse: bool = True
    logits_kl: bool = False
    ce_loss: bool = False


@dataclass(frozen=True)
class TrainingMode:
    name: str
    losses: LossProfile
    optimizer: Literal["sgd", "adam", "adamw"]
    dtype: str
    activation_checkpointing: bool = False
    gradient_accumulation: bool = False
    target_layers: TargetLayers = "all"
    notes: list[str] = field(default_factory=list)

    @property
    def model_dtype(self) -> str:
        return self.dtype

    @property
    def activation_dtype(self) -> str:
        return self.dtype

    @property
    def target_dtype(self) -> str:
        return self.dtype

    @property
    def include_hidden_targets(self) -> bool:
        return self.losses.hidden_mse

    @property
    def include_logits_targets(self) -> bool:
        return self.losses.logits_kl

    @property
    def include_optimizer_state(self) -> bool:
        return self.optimizer in {"sgd", "adam", "adamw"}

    @property
    def uses_fp32_master_weights(self) -> bool:
        return self.optimizer in {"adam", "adamw"} and self.dtype in {
            "bf16",
            "bfloat16",
            "fp16",
            "float16",
        }

    @property
    def activation_multiplier(self) -> float:
        if self.activation_checkpointing:
            return 4.0 if self.dtype in {"bf16", "bfloat16", "fp16", "float16"} else 5.0
        return 7.0 if self.dtype in {"bf16", "bfloat16", "fp16", "float16"} else 8.0


def validate_model_profile(profile: ModelProfile) -> None:
    _require_name(profile.name, "model profile")
    _require_name(profile.backend, "model backend")
    for field_name in (
        "vocab_size",
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
    ):
        value = getattr(profile, field_name)
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0, got {value}")
    if profile.hidden_size % profile.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if profile.num_heads % profile.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if profile.num_kv_heads > profile.num_heads:
        raise ValueError("num_kv_heads must be <= num_heads")
    if profile.resolved_head_size != profile.effective_head_size:
        raise ValueError("head_size must equal hidden_size // num_heads")
    if profile.resolved_head_size % 2 != 0:
        raise ValueError("head_size must be even for RoPE")
    if profile.effective_mlp_hidden_size <= profile.hidden_size:
        raise ValueError("mlp_hidden_size must be > hidden_size")
    if profile.sequence_length_default <= 0:
        raise ValueError("sequence_length_default must be > 0")
    _dtype_bytes(profile.dtype_default, "dtype_default")


def validate_hardware_profile(profile: HardwareProfile) -> None:
    _require_name(profile.name, "hardware profile")
    _require_name(profile.device_kind, "device kind")
    if profile.memory_gb <= 0:
        raise ValueError("memory_gb must be > 0")
    if profile.device_count <= 0:
        raise ValueError("device_count must be > 0")
    if profile.resolved_per_device_memory_gb <= 0:
        raise ValueError("per_device_memory_gb must be > 0")
    if (
        abs(
            profile.resolved_per_device_memory_gb * profile.device_count
            - profile.memory_gb
        )
        > 1e-6
    ):
        raise ValueError("memory_gb must match per_device_memory_gb * device_count")
    _dtype_bytes(profile.dtype_preference, "dtype_preference")


def validate_training_mode(mode: TrainingMode) -> None:
    _require_name(mode.name, "training mode")
    _dtype_bytes(mode.dtype, "dtype")
    if not (mode.losses.hidden_mse or mode.losses.logits_kl or mode.losses.ce_loss):
        raise ValueError("at least one target family must be enabled")


def dtype_bytes(dtype: str) -> int:
    return _dtype_bytes(dtype, "dtype")


def _dtype_bytes(dtype: str, field_name: str) -> int:
    key = dtype.lower()
    if key not in DTYPE_BYTES:
        raise ValueError(f"{field_name} must be one of {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[key]


def _require_name(value: str, label: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{label} name must be non-empty")


MODEL_PROFILES: dict[str, ModelProfile] = {
    "tiny_debug": ModelProfile(
        name="tiny_debug",
        backend="rwkv7_qwen_reference",
        vocab_size=512,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        mlp_hidden_size=512,
        sequence_length_default=32,
        dtype_default="fp32",
        status="debug",
        notes=["Tiny local shape for CLI/tests; not a quality target."],
    ),
    "p39_tiny_hf_qwen_rope_smoke": ModelProfile(
        name="p39_tiny_hf_qwen_rope_smoke",
        backend="rwkv7_qwen_reference",
        vocab_size=50257,
        hidden_size=2,
        num_layers=2,
        num_heads=1,
        num_kv_heads=1,
        mlp_hidden_size=8,
        sequence_length_default=8,
        dtype_default="fp32",
        status="tpu_smoke_candidate",
        emit_logits=True,
        notes=[
            "P39 real tiny HF TPU smoke shape based on sshleifer/tiny-gpt2 targets.",
            "RoPE-valid for rwkv7_qwen_reference: hidden_size=2, num_heads=1.",
            "Execution target is tiny smoke only, not a quality or scale target.",
        ],
    ),
    "small_cpu": ModelProfile(
        name="small_cpu",
        backend="rwkv7_qwen_reference",
        vocab_size=8192,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        mlp_hidden_size=2048,
        sequence_length_default=128,
        dtype_default="fp32",
        status="local_candidate",
        notes=["CPU-oriented development estimate, not hardware validated."],
    ),
    "colab_tpu_smoke": ModelProfile(
        name="colab_tpu_smoke",
        backend="rwkv7_qwen_reference",
        vocab_size=16384,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=4,
        mlp_hidden_size=3072,
        sequence_length_default=256,
        dtype_default="bf16",
        status="tpu_smoke_candidate",
        notes=["TPU smoke planning estimate only."],
    ),
    "qwen_0_5b_candidate": ModelProfile(
        name="qwen_0_5b_candidate",
        backend="rwkv7_qwen_reference",
        vocab_size=151936,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=8,
        mlp_hidden_size=2816,
        sequence_length_default=1024,
        dtype_default="bf16",
        status="tpu_smoke_candidate",
        notes=["This is a planning estimate, not a validated runnable config."],
    ),
    "qwen_1_5b_candidate": ModelProfile(
        name="qwen_1_5b_candidate",
        backend="rwkv7_qwen_reference",
        vocab_size=151936,
        hidden_size=1536,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        mlp_hidden_size=8960,
        sequence_length_default=1024,
        dtype_default="bf16",
        status="grant_candidate",
        notes=["This is a planning estimate, not a validated runnable config."],
    ),
    "qwen_7b_stretch": ModelProfile(
        name="qwen_7b_stretch",
        backend="rwkv7_qwen_reference",
        vocab_size=151936,
        hidden_size=3584,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        mlp_hidden_size=18944,
        sequence_length_default=1024,
        dtype_default="bf16",
        status="stretch_candidate",
        notes=["This is a planning estimate, not a validated runnable config."],
    ),
    "qrwkv_qwen_0_5b_candidate": ModelProfile(
        name="qrwkv_qwen_0_5b_candidate",
        backend="rwkv7_qwen_reference",
        vocab_size=151936,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=8,
        mlp_hidden_size=2816,
        sequence_length_default=1024,
        dtype_default="bf16",
        status="tpu_smoke_candidate",
        notes=[
            "P45 explicit QRWKV student profile alias for qwen_0_5b_candidate.",
            "Planning-only profile; not a validated training or release config.",
        ],
    ),
    "qrwkv_qwen_1_5b_candidate": ModelProfile(
        name="qrwkv_qwen_1_5b_candidate",
        backend="rwkv7_qwen_reference",
        vocab_size=151936,
        hidden_size=1536,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        mlp_hidden_size=8960,
        sequence_length_default=1024,
        dtype_default="bf16",
        status="grant_candidate",
        notes=[
            "P45 explicit QRWKV student profile alias for qwen_1_5b_candidate.",
            "Planning-only profile; not a validated training or release config.",
        ],
    ),
    "qrwkv_qwen_7b_stretch": ModelProfile(
        name="qrwkv_qwen_7b_stretch",
        backend="rwkv7_qwen_reference",
        vocab_size=151936,
        hidden_size=3584,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        mlp_hidden_size=18944,
        sequence_length_default=1024,
        dtype_default="bf16",
        status="stretch_candidate",
        notes=[
            "P45 explicit QRWKV stretch profile alias for qwen_7b_stretch.",
            "Planning-only profile; do not infer one-device 7B training support.",
        ],
    ),
}


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "local_cpu_debug": HardwareProfile(
        name="local_cpu_debug",
        device_kind="cpu",
        memory_gb=8,
        device_count=1,
        per_device_memory_gb=8,
        dtype_preference="fp32",
        supports_bf16=False,
        supports_pjit=False,
        confidence="high",
        notes=["Small local CPU budget for metadata-only and tiny debug dry-runs."],
    ),
    "local_cpu_16gb": HardwareProfile(
        name="local_cpu_16gb",
        device_kind="cpu",
        memory_gb=16,
        device_count=1,
        per_device_memory_gb=16,
        dtype_preference="fp32",
        supports_bf16=False,
        supports_pjit=False,
        confidence="high",
        notes=["Conservative local process budget."],
    ),
    "local_cpu_32gb": HardwareProfile(
        name="local_cpu_32gb",
        device_kind="cpu",
        memory_gb=32,
        device_count=1,
        per_device_memory_gb=32,
        dtype_preference="fp32",
        supports_bf16=False,
        supports_pjit=False,
        confidence="high",
    ),
    "local_cpu_64gb": HardwareProfile(
        name="local_cpu_64gb",
        device_kind="cpu",
        memory_gb=64,
        device_count=1,
        per_device_memory_gb=64,
        dtype_preference="fp32",
        supports_bf16=False,
        supports_pjit=False,
        confidence="high",
    ),
    "colab_tpu_v2_8": HardwareProfile(
        name="colab_tpu_v2_8",
        device_kind="tpu",
        memory_gb=64,
        device_count=8,
        per_device_memory_gb=8,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="medium",
        notes=["Aggregate TPU memory estimate; no sharding implementation implied."],
    ),
    "colab_tpu_v3_8": HardwareProfile(
        name="colab_tpu_v3_8",
        device_kind="tpu",
        memory_gb=128,
        device_count=8,
        per_device_memory_gb=16,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="medium",
        notes=["Aggregate TPU memory estimate; no sharding implementation implied."],
    ),
    "colab_tpu_v2_or_v3": HardwareProfile(
        name="colab_tpu_v2_or_v3",
        device_kind="tpu",
        memory_gb=64,
        device_count=8,
        per_device_memory_gb=8,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="low",
        notes=[
            "Conservative Colab TPU v2/v3 planning floor.",
            "Aggregate memory estimate only; no pjit/model sharding is implied.",
        ],
    ),
    "kaggle_tpu_v5e_8": HardwareProfile(
        name="kaggle_tpu_v5e_8",
        device_kind="tpu",
        memory_gb=128,
        device_count=8,
        per_device_memory_gb=16,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="medium",
        notes=[
            "Kaggle TPU v5e planning profile for manual smoke runs.",
            "Aggregate memory estimate only; P39 does not add sharding or pjit.",
        ],
    ),
    "single_l4_24gb": HardwareProfile(
        name="single_l4_24gb",
        device_kind="gpu",
        memory_gb=24,
        device_count=1,
        per_device_memory_gb=24,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="medium",
    ),
    "single_a100_40gb": HardwareProfile(
        name="single_a100_40gb",
        device_kind="gpu",
        memory_gb=40,
        device_count=1,
        per_device_memory_gb=40,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="medium",
    ),
    "grant_tpu_v5e_8": HardwareProfile(
        name="grant_tpu_v5e_8",
        device_kind="tpu",
        memory_gb=128,
        device_count=8,
        per_device_memory_gb=16,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="low",
        notes=[
            "Planning placeholder for likely grant TPU shape; "
            "validate on real grant hardware.",
            "Aggregate memory estimate only; no sharded runtime support is claimed.",
        ],
    ),
    "grant_tpu_v5e_32": HardwareProfile(
        name="grant_tpu_v5e_32",
        device_kind="tpu",
        memory_gb=512,
        device_count=32,
        per_device_memory_gb=16,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="low",
        notes=[
            "Planning placeholder for a 32-device grant TPU v5e slice.",
            "Aggregate memory estimate only; no sharded runtime support is claimed.",
        ],
    ),
    "grant_tpu_v5e_64": HardwareProfile(
        name="grant_tpu_v5e_64",
        device_kind="tpu",
        memory_gb=1024,
        device_count=64,
        per_device_memory_gb=16,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="low",
        notes=[
            "Planning placeholder for a 64-device grant TPU v5e slice.",
            "Aggregate memory estimate only; no sharded runtime support is claimed.",
        ],
    ),
    "big_budget_tpu_placeholder": HardwareProfile(
        name="big_budget_tpu_placeholder",
        device_kind="tpu",
        memory_gb=256,
        device_count=8,
        per_device_memory_gb=32,
        dtype_preference="bf16",
        supports_bf16=True,
        supports_pjit=False,
        confidence="low",
        notes=["Placeholder budget only; not tied to a live provisioned system."],
    ),
}


TRAINING_MODES: dict[str, TrainingMode] = {
    "smoke_hidden_sgd": TrainingMode(
        name="smoke_hidden_sgd",
        losses=LossProfile(hidden_mse=True, logits_kl=False, ce_loss=False),
        optimizer="sgd",
        dtype="fp32",
        activation_checkpointing=False,
        gradient_accumulation=False,
        target_layers="all",
    ),
    "smoke_hidden_logits_sgd": TrainingMode(
        name="smoke_hidden_logits_sgd",
        losses=LossProfile(hidden_mse=True, logits_kl=True, ce_loss=False),
        optimizer="sgd",
        dtype="fp32",
        activation_checkpointing=False,
        gradient_accumulation=False,
        target_layers="all",
        notes=["Full-vocab logits can dominate memory even at tiny scale."],
    ),
    "local_hidden_adamw": TrainingMode(
        name="local_hidden_adamw",
        losses=LossProfile(hidden_mse=True, logits_kl=False, ce_loss=False),
        optimizer="adamw",
        dtype="fp32",
        activation_checkpointing=False,
        gradient_accumulation=False,
        target_layers="all",
    ),
    "tpu_hidden_bf16_adamw": TrainingMode(
        name="tpu_hidden_bf16_adamw",
        losses=LossProfile(hidden_mse=True, logits_kl=False, ce_loss=False),
        optimizer="adamw",
        dtype="bf16",
        activation_checkpointing=True,
        gradient_accumulation=True,
        target_layers="all",
    ),
    "tpu_hidden_logits_bf16_adamw": TrainingMode(
        name="tpu_hidden_logits_bf16_adamw",
        losses=LossProfile(hidden_mse=True, logits_kl=True, ce_loss=False),
        optimizer="adamw",
        dtype="bf16",
        activation_checkpointing=True,
        gradient_accumulation=True,
        target_layers="all",
        notes=["Planner only; full logits are expected to be expensive."],
    ),
    "scale_hidden_only_bf16_adamw": TrainingMode(
        name="scale_hidden_only_bf16_adamw",
        losses=LossProfile(hidden_mse=True, logits_kl=False, ce_loss=False),
        optimizer="adamw",
        dtype="bf16",
        activation_checkpointing=True,
        gradient_accumulation=True,
        target_layers="all",
        notes=["Scale planning mode with conservative activation reserve."],
    ),
    "scale_sampled_logits_bf16_adamw_placeholder": TrainingMode(
        name="scale_sampled_logits_bf16_adamw_placeholder",
        losses=LossProfile(hidden_mse=True, logits_kl=True, ce_loss=False),
        optimizer="adamw",
        dtype="bf16",
        activation_checkpointing=True,
        gradient_accumulation=True,
        target_layers="selected",
        notes=[
            "Placeholder planning mode representing future sampled-logits work, "
            "not implemented runtime behavior."
        ],
    ),
}
