# Decisions

## D001 — Rebuild instead of port
QRWKV-XLA will be a new JAX/XLA-first implementation inspired by RADLADS, not a direct port of the PyTorch/CUDA/Triton repo.

## D002 — Teacher extraction remains PyTorch/HF
Teacher models will initially be loaded using PyTorch/Hugging Face tooling and exported into reusable target artifacts.

## D003 — Student training is JAX/XLA-first
The recurrent student trainer will be implemented in JAX and designed for CPU debug and TPU execution.

## D004 — Primary student architecture is RWKV7-style
RWKV7-style recurrence is the primary destination architecture.

## D005 — Qwen3.latest is a policy label
The primary teacher target is the latest viable Qwen3-line open-weight model available at experiment time. Each run must resolve this to a concrete model ID in metadata.

teacher:
  family: qwen
  primary_policy: latest_qwen3_open_weight_available_at_experiment_time
  current_primary_label: Qwen3.latest
  fallback_label: Qwen3.0

student:
  primary_architecture: RWKV7-style
  fallback_architecture: none currently
  optional_reference_architecture: RWKV6-style only if needed for debugging/comparison

## D006 — No disposable toy architecture
Early configs may be tiny, but all modules should be shaped like the real system.
