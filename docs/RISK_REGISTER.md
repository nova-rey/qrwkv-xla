# Risk Register

## R1 — XLA incompatibility from dynamic shapes
Mitigation: static config discipline, tiny smoke tests, shape contracts.

## R2 — RWKV7 recurrence is hard to express efficiently in JAX
Mitigation: document math first, implement scan-based reference, optimize later.

## R3 — Teacher model target keeps moving
Mitigation: use Qwen3.latest as policy label, resolve to concrete model IDs per run.

## R4 — No guaranteed TPU access before grant
Mitigation: CPU-first development, Kaggle/Colab TPU smoke scripts, no hardware assumptions in Tier 0 tests.

## R5 — Original repo features are broad
Mitigation: preserve feature categories in roadmap, implement incrementally by subsystem.

## R6 — Accidental CUDA/Triton dependency creep
Mitigation: isolate PyTorch/HF to teacher export path; student path remains JAX/XLA.

## R7 — Real tokenizer dependency creep
Mitigation: keep `smoke` as the default tokenizer, lazy-import HF tokenizers,
gate real HF tests behind an environment variable, and require `.[teacher-hf]`
only for optional real-tokenizer runs.

## R8 — Target loss masking drift
Mitigation: target-bundle `loss_mask` is loaded into JAX training batches and
used for token-level target loss averages; focused tests place bad target values
behind masked positions to catch regressions.

## R9 — RADLADS reference overclaiming
Mitigation: keep `rwkv7_radlads_reference` separate from the existing
`rwkv7_reference` smoke backend, document it as a RADLADS-shaped JAX reference
backend, not full RADLADS parity, and require explicit future parity work before
claiming checkpoint or numerical compatibility with RADLADS torch/CUDA/Triton
outputs.

## R10 — Config-relative export paths surprise old callers
Mitigation: only config-loaded paths changed to YAML-relative semantics; CLI
path overrides such as `--out` remain cwd-relative, and tests cover
config-relative inputs, config-relative output directories, absolute paths, and
CLI override behavior.

## R11 — Full-vocab logits smokes become too heavy for default CPU CI
Mitigation: keep real tiny-GPT2 full-vocab export and logits distill execution
behind `QRWKV_RUN_HF_INTEGRATION=1`; default tests use tiny fake logits to cover
the same loss plumbing without requiring torch, transformers, network, cache, or
large target arrays.

## R12 — Kernel work starts before RADLADS block math is compatible
Mitigation: P31 classifies `rwkv7_radlads_reference` as a
RADLADS-shaped JAX reference backend, not full RADLADS parity. P32 adds
`rwkv7_qwen_reference` as a Qwen/RADLADS-compatible slow JAX reference path,
not optimized kernel parity. It closes the highest-priority slow-reference
block-compatibility gaps with a Qwen RMSNorm/MLP decoder shell, RoPE, grouped
KV heads, shift-state cache, nested parameter surface, and focused CPU tests.
Pallas, TPU kernels, `pjit`, sharding, export, and quality work remain
deferred until the slow reference can catch math regressions.

## R13 — Qwen reference cache or mask semantics drift silently
Mitigation: P33 adds deterministic tiny `rwkv7_qwen_reference` fixtures with
no-mask, interior-mask, and prefix/left-padding mask-shape cases. The manifest
records full-vs-step output/logit/state equivalence, payload hashes, and a
parameter-surface hash so local behavior changes are visible before kernel or
checkpoint work builds on them.

## R14 — Scale planner estimates are mistaken for runnable proof
Mitigation: P34 labels planner output as planning-only and not hardware
validated. The estimator reports assumptions, component memory, full-logits
dominance warnings, conservative yes/maybe/no thresholds, and provisional TPU
aggregate-memory caveats. It does not claim XLA peak-memory accuracy, pjit/model
sharding support, Qwen-scale training readiness, or full RADLADS parity.
