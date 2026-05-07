# Real Tiny HF TPU Distill Smoke

P38 adds a manual, opt-in TPU smoke that exports real Hugging Face teacher
targets from `sshleifer/tiny-gpt2`, validates the target bundle, and runs a tiny
`rwkv7_qwen_reference` hidden + logits-KL distill train/resume proof.

Normal CI stays CPU-only. The live HF download and TPU execution happen only
when you run `scripts/run_tiny_hf_tpu_smoke.py` manually.

## What P38 proves

A successful run proves only that:

- JAX sees a TPU backend
- the existing HF exporter can write a tiny `sshleifer/tiny-gpt2` target bundle
- the target bundle includes `input_ids`, `attention_mask`, `loss_mask`,
  `hidden_states`, and `logits`
- target shapes match the manifest's batch, sequence, layer, hidden, and logits
  dimensions
- one TPU distill step produces finite `loss`, `hidden_mse`, and `logits_kl`
- checkpoint resume advances optimizer/checkpoint step progression from `1 -> 2`
- `P38_RESULTS.md` and `p38_results_bundle.tar.gz` are written

## What P38 does not prove

- Qwen-scale teacher export or training
- model quality or lm-eval results
- multi-host TPU, `pjit`, or sharded training
- Pallas or optimized WKV7 kernels
- full RADLADS numerical parity
- WandB logging or HF student export

## Colab Workflow

Before running the harness:

1. `Runtime -> Change runtime type -> TPU`
2. restart the runtime
3. avoid importing `jax` before running the script

```python
%%bash
set -euo pipefail

if [ ! -d /content/qrwkv-xla ]; then
  git clone https://github.com/nova-rey/qrwkv-xla.git /content/qrwkv-xla
fi

cd /content/qrwkv-xla
python -m pip install -e ".[teacher-hf]" --no-deps
python -m pip install -q pytest pyyaml ruff torch transformers safetensors accelerate

python scripts/run_tiny_hf_tpu_smoke.py
```

If the script reports `backend: cpu`, that is a failure. The harness exits with
the unchanged message:

`Expected JAX backend 'tpu', got 'cpu'. In Colab, select Runtime → Change runtime type → TPU, then restart the runtime.`

## Kaggle Protocol

Kaggle TPU sessions should use the same repository command. Keep the run
manual, with internet enabled for the first `sshleifer/tiny-gpt2` download.
Use a persistent Kaggle dataset or cached working directory only as an optional
speedup; the harness itself writes all required proof artifacts under stable
repo-relative paths.

```bash
set -euo pipefail

cd /kaggle/working/qrwkv-xla
python -m pip install -e ".[teacher-hf]" --no-deps
python -m pip install -q pytest pyyaml ruff torch transformers safetensors accelerate

python scripts/run_tiny_hf_tpu_smoke.py

kaggle kernels push -p <kernel_dir> --accelerator TpuV5E8 -t 3600
kaggle kernels status <kernel_id>
kaggle kernels output <kernel_id> -p <output_dir> -o
```

Expected stable outputs:

- `artifacts/p38_tiny_hf_tpu_smoke/P38_RESULTS.md`
- `artifacts/p38_tiny_hf_tpu_smoke/p38_results_bundle.tar.gz`
- `artifacts/teacher_targets/p38_tiny_hf_logits_smoke/manifest.json`
- `artifacts/teacher_targets/p38_tiny_hf_logits_smoke/shards/shard_000000.npz`
- `checkpoints/p38_tpu_qwen_reference_tiny_hf_first`
- `checkpoints/p38_tpu_qwen_reference_tiny_hf_resume`
- `runs/p38/p38_tpu_qwen_reference_tiny_hf_first`
- `runs/p38/p38_tpu_qwen_reference_tiny_hf_resume`

Known warnings and failure modes:

- transparent hugepage warnings are noisy but not a harness failure
- `backend: cpu` is a real failure
- `TPU already in use` and `/dev/vfio/0 busy` indicate runtime ownership problems, not a model bug
- P38 proves tiny real-HF target TPU train/resume only; it does not prove scale or quality
