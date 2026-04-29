# Decisions

## D001 — Rebuild instead of port
QRWKV-XLA will be a new JAX/XLA-first implementation inspired by RADLADS, not a
direct port of the PyTorch/CUDA/Triton repo.

## D002 — Teacher extraction remains PyTorch/HF
Teacher models will initially be loaded using PyTorch/Hugging Face tooling and
exported into reusable target artifacts.

## D003 — Student training is JAX/XLA-first
The recurrent student trainer will be implemented in JAX and designed for CPU
debug and TPU execution.

## D004 — Primary student architecture is RWKV7-style
RWKV7-style recurrence is the primary destination architecture.

## D005 — Qwen3.latest is a policy label
The primary teacher target is the latest viable Qwen3-line open-weight model
available at experiment time. Each run must resolve this to a concrete model ID
in metadata.

teacher:
  family: qwen
  primary_policy: latest_qwen3_open_weight_available_at_experiment_time
  current_primary_label: Qwen3.latest
  fallback_label: Qwen3.0

student:
  primary_architecture: RWKV7-style
  fallback_architecture: none currently
  optional_reference_architecture: RWKV6-style only if needed for
    debugging/comparison

## D006 — No disposable toy architecture
Early configs may be tiny, but all modules should be shaped like the real
system.

## D007 — Nyx/Subagent Workflow
QRWKV-XLA uses Nyx as the primary implementation agent, with Codex available as
a sub-agent for mechanical implementation, tests, and lint cleanup. The older
A/B/C prompt split is optional and only used when it helps.

## D008 — Dataclasses Before Heavy Schema Frameworks
Initial config and artifact schemas use standard-library dataclasses plus
explicit validation. Pydantic or other schema frameworks may be considered later
if the project needs stronger runtime validation or serialization features.

## D009 — NPZ Shards for Initial Target Bundles
Initial teacher target bundles use a manifest JSON plus NumPy `.npz` shards.
This keeps the first artifact store simple, CPU-only, inspectable, and testable.
Larger-scale storage formats such as Zarr, safetensors, or memory-mapped arrays
may be considered later when real teacher extraction scale demands it.

## D010 — Fake Exporter as Permanent Test Double
QRWKV-XLA will keep a deterministic fake teacher exporter as a permanent test
double. It exercises the same artifact-writing pathway as future real
exporters, allowing the pipeline to be tested without network, GPU, TPU,
PyTorch, or Hugging Face dependencies.

## D011 — Real Teacher Loading Deferred
P2 intentionally does not load Qwen, PyTorch, Hugging Face Transformers, or
tokenizers. The project will first stabilize exporter contracts and artifact
writing before adding heavyweight model dependencies.

## D012 — Strategy A Editable Install Workflow
QRWKV-XLA uses editable install as the blessed development and CI workflow:
`python -m pip install -e ".[dev]"`.

Scripts, tests, and validation commands should import `qrwkv_xla` through the
installed package metadata instead of setting `PYTHONPATH=src` or mutating
`sys.path` inside entrypoint scripts. `scripts/validate_local.py` mirrors CI but
does not install dependencies; environment setup is an explicit caller
responsibility.

## D013 — Local Validation Mirrors CI
Local validation should mirror the CI command sequence through
`scripts/validate_local.py` or the equivalent individual command list. CI and
local handoff checks should stay aligned so failures reproduce before handoff.
