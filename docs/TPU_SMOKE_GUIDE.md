# TPU Smoke Guide

## Purpose

This project is not notebook-first. Kaggle and Colab TPU sessions should act as
launchers for repo scripts.

## Local CPU smoke

```bash
python -m pip install -e ".[dev]"
python scripts/xla_inspect.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/tpu_distill_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2
```

## Kaggle / Colab launcher flow

```bash
git clone https://github.com/nova-rey/qrwkv-xla
cd qrwkv-xla
python -m pip install -e ".[dev]"
python scripts/xla_inspect.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/tpu_distill_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2 --require-tpu
```

If `--require-tpu` is omitted, the script runs on whatever JAX backend is
available.

## Notes

- CI does not require TPU.
- `tpu_distill_smoke.py` is TPU-ready, but CPU-safe by default.
- Real TPU success should only be claimed when run with `--require-tpu` on an
  actual TPU backend.
