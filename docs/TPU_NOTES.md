# TPU / XLA Notes

- CPU support is mandatory for local development.
- TPU support must be tested through tiny smoke scripts before scale-up.
- Avoid dynamic shapes in training-critical paths.
- Avoid Python-side token loops in final training paths where `jax.lax.scan` or vectorized forms are appropriate.
- Inspect JAX devices and default backend before assuming accelerator behavior.
- Use static-shape JIT smoke checks before attempting stage training.
- Do not assume notebook-first development.
- Kaggle/Colab notebooks may be used only as launch wrappers for repo scripts.
- No CUDA, Triton, or flash-attention dependency in the JAX student path.
- PyTorch/Hugging Face is allowed for teacher target extraction only.

See also:

- `docs/XLA_DISCIPLINE.md`
- `docs/TPU_SMOKE_GUIDE.md`

## Smoke Behavior

`scripts/smoke_tpu.py` is a graceful environment check. For an actual TPU-ready
distillation smoke path, use `scripts/tpu_distill_smoke.py` and pass
`--require-tpu` only on real TPU hardware.
