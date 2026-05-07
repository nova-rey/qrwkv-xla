# Colab TPU Smoke

P36 adds a manual, opt-in Colab TPU smoke for the tiny hidden-only
`rwkv7_qwen_reference` distillation path. It proves only that the current slow
reference student can run one TPU step, checkpoint, resume for one more TPU
step, and write the expected local artifacts.

It does not prove Qwen-scale training, model quality, logits KL, pjit,
multi-host execution, sharding, Pallas/WKV7 optimized kernels, or HF student
export.

## Purpose

Use this when you want a reproducible Colab TPU regression harness for the P35
train/resume proof. Normal local CI stays CPU-only.

## Prerequisites

- A Google Colab runtime with **TPU** selected.
- A fresh runtime restart after changing accelerator type.
- Do **not** import `jax` in the notebook kernel before running the smoke.
- Run the harness once per fresh runtime.

## Runtime Setup

1. **Runtime → Change runtime type → TPU**
2. **Runtime → Disconnect and delete runtime** or **Restart runtime**
3. Run the one-cell entrypoint below

## Recommended one-cell Colab entrypoint

```python
%%bash
set -euo pipefail

if [ ! -d /content/qrwkv-xla ]; then
  git clone https://github.com/nova-rey/qrwkv-xla.git /content/qrwkv-xla
fi

cd /content/qrwkv-xla
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e . --no-deps
python -m pip install -q pytest pyyaml ruff black

python scripts/run_colab_tpu_smoke.py
```

If you are testing a branch, add `git checkout <branch>` before installing.
The repo `dev` extra installs CPU JAX for local CI, so this Colab flow keeps
TPU JAX selected by installing the repo with `--no-deps`.

## What it runs

The harness is `scripts/run_colab_tpu_smoke.py`.

It prints Python, JAX, backend, device, and git metadata; requires the JAX
default backend to be TPU; runs a tiny JAX matmul; exports deterministic fake
teacher hidden targets; runs a first one-step distill; resumes from that
checkpoint for one more step; validates checkpoint/run artifacts; validates
finite loss and hidden MSE metrics; validates optimizer-step progression `1 → 2`;
and then writes:

- `artifacts/p36_colab_tpu_smoke/P36_RESULTS.md`
- `artifacts/p36_colab_tpu_smoke/p36_results_bundle.tar.gz`
- `checkpoints/p36_tpu_qwen_reference_first`
- `checkpoints/p36_tpu_qwen_reference_resume`
- `runs/p36/p36_tpu_qwen_reference_first`
- `runs/p36/p36_tpu_qwen_reference_resume`

The checked-in config is:

```bash
configs/distill_stage0_qwen_reference_colab_tpu_smoke.yaml
```

## Expected pass criteria

A successful manual run should show all of the following:

- `backend: tpu`
- `TPU sanity: PASS` behavior via the tiny JAX matmul succeeding
- first run optimizer step `1`
- resume run optimizer step `2`
- `P36_RESULTS.md` written
- `p36_results_bundle.tar.gz` written

## Known warnings

- Transparent hugepage warnings are not a failure for this smoke test.
- `backend: cpu` is a failure for this smoke test.
- `Expected JAX backend 'tpu', got 'cpu'. In Colab, select Runtime → Change runtime type → TPU, then restart the runtime.` means the runtime is not actually on TPU.
- `The TPU is already in use by process with pid ...` usually means the notebook kernel imported JAX first or another Python process already owns the TPU.
- The harness intentionally keeps TPU work in one process to avoid TPU ownership conflicts.
- Existing P36 smoke output directories are replaced so repeated runs are reproducible at stable paths.
- Normal CI remains CPU-only. Do not add this script to required local or CI validation.

## What failures mean

- Non-TPU backend: Colab runtime is wrong or needs a restart.
- TPU already in use: restart the runtime and rerun without importing JAX in the notebook kernel first.
- Missing artifacts or bad step progression: the distill/checkpoint path regressed.
- Non-finite loss or hidden MSE: the tiny smoke run became numerically unstable.

## Scope honesty

This smoke proves tiny TPU execution only. It does not prove scale,
performance, Pallas kernels, real Qwen target training on TPU, logits KL on
TPU, or model quality.
