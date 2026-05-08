# P46 pjit / Sharding Compile Smoke

P46 adds a tiny sharding-aware compile smoke for QRWKV-XLA. It creates a named
JAX mesh, applies an explicit data-parallel single-axis sharding policy to a
tiny batch, compiles a tiny forward/loss path, optionally runs one SGD-style
update, verifies finite loss, and writes JSON plus Markdown reports.

The canonical local command is:

```bash
.venv/bin/python scripts/run_pjit_sharding_smoke.py --out artifacts/p46_pjit_sharding_smoke --overwrite
```

The report files are:

- `artifacts/p46_pjit_sharding_smoke/P46_RESULTS.md`
- `artifacts/p46_pjit_sharding_smoke/pjit_sharding_smoke_report.json`

## What P46 Proves

- JAX devices can be inspected and represented as a named mesh.
- Single-device CPU runs take an explicit fallback path instead of pretending to
  be multi-device sharding.
- A `data_parallel_single_axis` policy can annotate/compile a tiny batch with
  explicit shardings.
- The current JAX pin can compile the tiny forward/loss/update path using the
  selected compile API.
- The smoke records backend, platform, device counts, device kinds, mesh shape,
  mesh axis names, multi-device status, fallback reason, compile API, policy,
  finite loss, and update status.

## What P46 Does Not Prove

P46 does not prove large-model sharding, large-batch execution, throughput,
production sharded checkpointing, multi-host training, full distributed
optimizer state strategy, Pallas WKV7 kernels, WandB integration, official
evaluation, or that P45 0.5B / 1.5B / 7B profiles are trainable.

## Local Single-Device Fallback

Most local CPU environments expose one JAX device. In that case P46 still
creates a one-device named mesh, compiles with explicit shardings, and reports:

- `multi_device: false`
- `multi_device_execution: false`
- `fallback_reason: single_device_fallback`

This is a compile smoke only. It is not reported as real multi-device sharding.

## TPU Multi-Device Mode

On Kaggle or Colab TPU runtimes, run the same command after JAX sees multiple
TPU devices:

```bash
.venv/bin/python scripts/run_pjit_sharding_smoke.py \
  --require-multi-device \
  --mesh-axis data \
  --batch-size 8 \
  --seq-len 8 \
  --out artifacts/p46_pjit_sharding_smoke \
  --overwrite
```

`--require-multi-device` fails cleanly when fewer than two devices are visible.
On a valid TPU runtime, the report should show `multi_device: true`,
`multi_device_execution: true`, and a TPU platform/device kind.

## Mesh Creation

`qrwkv_xla.sharding.mesh.create_named_mesh` inspects `jax.devices()`, creates a
one-axis `jax.sharding.Mesh`, records topology metadata, and preserves a
single-device fallback reason. The default mesh axis is `data`; override it with
`--mesh-axis`.

## Compile API

The CLI accepts:

- `--compile-api auto`
- `--compile-api jit`
- `--compile-api pjit`

On the current JAX pin, `auto` selects `jit_with_shardings` because
`jax.experimental.pjit.pjit` is present but deprecated in favor of `jax.jit`
with explicit shardings. Explicit `--compile-api pjit` still uses pjit and the
report records `compile_api: pjit`.

## Policy

P46 supports `--policy data_parallel_single_axis`. Parameters are replicated and
the tiny batch is sharded on the first axis over the named mesh axis.
Placeholder policies such as `model_parallel_placeholder` and
`fsdp_placeholder` are intentionally unsupported and fail instead of reporting
fake success.

## How P46 Feeds Later Work

P46 is the first explicit sharding-aware compile doorway for later training
phases. It gives future work a concrete mesh/sharding/report surface to extend
without pretending that tiny local success already proves large QRWKV training.
