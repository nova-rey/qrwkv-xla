# Colab TPU Logits-KL Smoke

P37 extends the reproducible manual Colab TPU smoke path from P36 hidden-only
training to tiny hidden + logits-KL distillation. It remains manual and opt-in;
normal CI stays CPU-only.

## What P37 proves

A successful run proves only that a tiny `rwkv7_qwen_reference` student can:

- see a Colab TPU through JAX/XLA
- export deterministic fake targets that include `logits`
- run one TPU distill step with `hidden_mse` and `logits_kl` enabled
- checkpoint, resume, and run one more TPU step
- write `P37_RESULTS.md` plus a compressed results bundle

## What P37 does not prove

- real Qwen-scale training
- model quality
- real Qwen target training on TPU
- Pallas or optimized WKV7 kernels
- `pjit`, sharding, or multi-device/multi-host TPU training
- real HF teacher export on TPU
- WandB, lm-eval, or HF student export

## Runtime setup

Before running the harness in Colab:

1. `Runtime -> Change runtime type -> TPU`
2. `Runtime -> Restart runtime` or `Disconnect and delete runtime`
3. run the smoke cell once
4. avoid importing `jax` in the notebook kernel before starting the script

If another process already owns the TPU, Colab commonly reports:

- `The TPU is already in use by process with pid ...`

That usually means a previous notebook/kernel/subprocess still owns the TPU.
Restart the runtime and run the cell again.

Transparent hugepage warnings are noisy but not a failure.

If the script reports `backend: cpu`, that is a failure. The harness will exit
with:

`Expected JAX backend 'tpu', got 'cpu'. In Colab, select Runtime → Change runtime type → TPU, then restart the runtime.`

## One-cell Colab workflow

```python
%%bash
set -euo pipefail

if [ ! -d /content/qrwkv-xla ]; then
  git clone https://github.com/nova-rey/qrwkv-xla.git /content/qrwkv-xla
fi

cd /content/qrwkv-xla
python -m pip install -e . --no-deps
python -m pip install -q pytest pyyaml ruff

python scripts/run_colab_tpu_logits_smoke.py
```

## Expected pass criteria

A passing manual run should show:

- `backend: tpu`
- target manifest/shard validation confirms `logits`
- first run metrics include finite `loss`, `hidden_mse`, and `logits_kl`
- first run `optimizer_step == 1`
- resume run metrics include finite `loss`, `hidden_mse`, and `logits_kl`
- resume run `optimizer_step == 2`
- resume summary records the first checkpoint as the resume source

Stable output paths:

- `artifacts/p37_colab_tpu_logits_smoke/P37_RESULTS.md`
- `artifacts/p37_colab_tpu_logits_smoke/p37_results_bundle.tar.gz`
- `artifacts/teacher_targets/p37_colab_tpu_logits_smoke/manifest.json`
- `artifacts/teacher_targets/p37_colab_tpu_logits_smoke/shards/shard_000000.npz`
- `checkpoints/p37_tpu_qwen_reference_logits_first`
- `checkpoints/p37_tpu_qwen_reference_logits_resume`
- `runs/p37/p37_tpu_qwen_reference_logits_first`
- `runs/p37/p37_tpu_qwen_reference_logits_resume`
