# QRWKV-XLA

QRWKV-XLA is a JAX/XLA-first recurrent conversion pipeline inspired by RADLADS.

It aims to distill Qwen-family transformer teachers into RWKV7-style recurrent
students using TPU-friendly training infrastructure.

## Current Status

Phase 2.5: stabilization of the teacher exporter interface, fake export
pipeline, package import path, local validation, and CI.

The project can now define, write, read, validate, inspect, and test fake
teacher target bundles on CPU through a reusable exporter interface. The
stabilized development path uses an editable install and does not require JAX,
PyTorch, TPU, GPU, or network access for core validation.

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
python scripts/validate_local.py
```

The current exporter path uses the deterministic fake exporter. Real Qwen /
PyTorch / Hugging Face teacher loading is intentionally deferred.

Generated bundles are written under `artifacts/`, which is gitignored.

See `docs/CI.md` for the exact CI command sequence and local mirror.

## Local Development

Use the blessed editable-install workflow:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

The current tests are CPU-only and do not require JAX, PyTorch, GPU, TPU, or
network access.

Individual checks:

```bash
python -m compileall src scripts tests
python scripts/print_env.py
python scripts/smoke_cpu.py
python scripts/smoke_tpu.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/inspect_targets.py artifacts/teacher_targets/fake_export
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
