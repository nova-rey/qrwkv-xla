# Kaggle TPU Planner Smoke

P39 can be run on a Kaggle TPU v5e session with the same repository entrypoint
as Colab: `scripts/run_planner_tpu_smoke.py`. The harness generates planner
artifacts, exports real tiny HF targets, runs one TPU distill step, resumes for
one more step, and writes a result bundle.

## Kernel Setup

Use a TPU v5e accelerator, enable internet for the first Hugging Face download,
and keep the run manual. The command below assumes the repo is already under
`/kaggle/working/qrwkv-xla`.

```bash
set -euo pipefail

cd /kaggle/working/qrwkv-xla
python -m pip install -e ".[teacher-hf]" --no-deps
python -m pip install -q pytest pyyaml ruff torch transformers safetensors accelerate

python scripts/run_planner_tpu_smoke.py
```

For Kaggle CLI submission from a prepared kernel directory:

```bash
kaggle kernels push -p <kernel_dir> --accelerator TpuV5E8 -t 3600
kaggle kernels status <kernel_id>
kaggle kernels output <kernel_id> -p <output_dir> -o
```

## Outputs To Copy

- `artifacts/p39_planner_tpu_smoke/P39_RESULTS.md`
- `artifacts/p39_planner_tpu_smoke/p39_results_bundle.tar.gz`
- `artifacts/p39_planner_tpu_smoke/scale_plan.yaml`
- `artifacts/p39_planner_tpu_smoke/scale_plan.json`
- `artifacts/p39_planner_tpu_smoke/generated_distill.yaml`
- `artifacts/p39_planner_tpu_smoke/teacher_export.yaml`
- `artifacts/teacher_targets/p39_tiny_hf_logits_smoke/manifest.json`
- `artifacts/teacher_targets/p39_tiny_hf_logits_smoke/shards/shard_000000.npz`

## Caveats

- `backend: cpu` is a real failure; select a TPU runtime and restart.
- Transparent hugepage warnings are noisy but not a harness failure.
- `TPU already in use` and `/dev/vfio/0 busy` usually mean the runtime owns the
  TPU incorrectly.
- The Kaggle v5e planner profile is an aggregate memory planning profile. P39
  does not add `pjit`, sharding, or multi-host execution.
- P39 validates only the tiny planner-generated execution path, not Qwen-scale
  training or model quality.
