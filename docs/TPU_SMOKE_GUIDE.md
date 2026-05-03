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

The canonical hard TPU pipeline check is:

```bash
python scripts/validate_pipeline.py --require-tpu
```

This passes `--require-tpu` to the TPU distillation smoke and should only be
used in an actual TPU environment.

## Multi-device pmap smoke

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_attention.yaml
python scripts/pmap_distill_smoke.py --config configs/distill_stage1_attention_pmap_smoke.yaml
python scripts/pmap_lm_smoke.py --config configs/lm_stage3_pmap_smoke.yaml
```

Hard multi-device validation:

```bash
python scripts/pmap_distill_smoke.py \
  --config configs/distill_stage1_attention_pmap_smoke.yaml \
  --require-multiple-devices \
  --min-device-count 2
```

## Notes

- CI does not require TPU.
- `tpu_distill_smoke.py` is TPU-ready, but CPU-safe by default.
- Real TPU success should only be claimed when run with `--require-tpu` on an
  actual TPU backend.
