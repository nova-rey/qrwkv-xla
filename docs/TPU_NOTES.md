# TPU / XLA Notes

- CPU support is mandatory for local development.
- TPU support must be tested through tiny smoke scripts before scale-up.
- Avoid dynamic shapes in training-critical paths.
- Avoid Python-side token loops in final training paths where `jax.lax.scan` or vectorized forms are appropriate.
- Do not assume notebook-first development.
- Kaggle/Colab notebooks may be used only as launch wrappers for repo scripts.
- No CUDA, Triton, or flash-attention dependency in the JAX student path.
- PyTorch/Hugging Face is allowed for teacher target extraction only.
