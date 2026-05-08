# Planner-Generated Tiny HF TPU Smoke

P39 adds a manual, opt-in Colab TPU smoke that runs the scale planner first,
writes planner artifacts, exports real `sshleifer/tiny-gpt2` teacher targets,
and runs the planner-generated tiny `rwkv7_qwen_reference` distill config for
one step plus one resumed step.

Normal CI stays CPU-only. Live HF download and TPU execution happen only when
you run `scripts/run_planner_tpu_smoke.py` manually.

## What P39 Proves

- JAX sees a TPU backend
- a tiny JAX matmul runs on that backend
- the P39 planner profile `p39_tiny_hf_qwen_rope_smoke` is RoPE-valid
- the Kaggle/TPU v5e profile can be selected by the planner
- planner outputs are written under `artifacts/p39_planner_tpu_smoke/`
- the generated distill config runs a tiny hidden + logits-KL train/resume
- real tiny HF targets include `input_ids`, `attention_mask`, `loss_mask`,
  `hidden_states`, and `logits`
- finite `loss`, `hidden_mse`, and `logits_kl` are recorded
- optimizer/checkpoint step progression advances `1 -> 2`

## What P39 Does Not Prove

- Qwen-scale export, fit, or training
- long training stability or model quality
- `pjit`, sharding, or multi-host TPU
- Pallas or optimized WKV7 kernels
- lm-eval, WandB, or HF student export

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

python scripts/run_planner_tpu_smoke.py
```

If the script reports `backend: cpu`, that is a failure. The harness exits with
the unchanged message:

`Expected JAX backend 'tpu', got 'cpu'. In Colab, select Runtime → Change runtime type → TPU, then restart the runtime.`

Expected stable outputs:

- `artifacts/p39_planner_tpu_smoke/scale_plan.yaml`
- `artifacts/p39_planner_tpu_smoke/scale_plan.json`
- `artifacts/p39_planner_tpu_smoke/generated_distill.yaml`
- `artifacts/p39_planner_tpu_smoke/teacher_export.yaml`
- `artifacts/p39_planner_tpu_smoke/P39_RESULTS.md`
- `artifacts/p39_planner_tpu_smoke/p39_results_bundle.tar.gz`
- `artifacts/teacher_targets/p39_tiny_hf_logits_smoke/manifest.json`
- `checkpoints/p39_planner_tpu_smoke_first`
- `checkpoints/p39_planner_tpu_smoke_resume`
- `runs/p39/p39_planner_tpu_smoke_first`
- `runs/p39/p39_planner_tpu_smoke_resume`

The planner output is still a planning estimate. P39 treats it as executable
only for the tiny checked harness path.
