# QRWKV-XLA

QRWKV-XLA is a JAX/XLA-first recurrent conversion pipeline inspired by RADLADS.

It aims to distill Qwen-family transformer teachers into RWKV7-style recurrent
students using TPU-friendly training infrastructure.

## Current Status

Phase 5: distillation stage runtime.

The project can define, write, read, validate, inspect, and test fake teacher
target bundles on CPU through a reusable exporter interface. It also has a JAX
student runtime path and an XLA-friendly `rwkv7_reference` recurrent reference
implementation for CPU/JIT/gradient coverage and smoke training. The current
distillation runtime loads stage configs, composes weighted hidden-state losses,
plumbs optional logits KL, and runs a CPU-only stage smoke over target bundles.
The reference core is not a final optimized RWKV7 kernel.

## Design Principles

- Full-system architecture from day one
- Tiny configs, not disposable toy systems
- JAX/XLA-first student training
- PyTorch/Hugging Face teacher extraction later
- CPU local development
- TPU smoke tests when available
- No CUDA/Triton dependency in student training path
- Simple, inspectable artifact formats first

## Quick Usage

```bash
python -m pip install -e ".[dev]"
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/inspect_targets.py artifacts/teacher_targets/fake_export
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --student-architecture rwkv7_reference --max-steps 2
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml
python scripts/validate_local.py
```

The current exporter path uses the deterministic fake exporter. Real Qwen /
PyTorch / Hugging Face teacher loading is intentionally deferred.

`scripts/run_distill_stage.py` is now the primary entrypoint for staged distillation. It currently supports hidden-state distillation against fake teacher bundles with `tiny_student` or `rwkv7_reference` students.

Generated bundles are written under `artifacts/`, which is gitignored.

See `docs/CI.md` for the exact CI command sequence and local mirror.

## Local Development

Use the blessed editable-install workflow:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

The current tests are CPU-only. They require JAX CPU through the `dev` extra,
but do not require PyTorch, GPU, TPU, or network access.

Individual checks:

```bash
python -m compileall src scripts tests
python scripts/print_env.py
python scripts/smoke_cpu.py
python scripts/smoke_tpu.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/inspect_targets.py artifacts/teacher_targets/fake_export
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --student-architecture rwkv7_reference --max-steps 2
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Reference

This project uses `nova-rey/radlads-TPU-adapter` as a conceptual and
architectural reference, not as code to directly port.

The reference RADLADS lineage includes RAD-RWKV6/RAD-RWKV7 components,
Hugging Face conversion scripts, staged configs, Lightning trainer flows,
`lm_eval` support, and inference support. QRWKV-XLA is being rebuilt around XLA
and TPU constraints from day one instead of carrying over GPU-shaped internals.
