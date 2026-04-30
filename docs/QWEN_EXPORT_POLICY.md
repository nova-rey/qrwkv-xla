# Qwen Export Policy

Qwen export is manual-only. `Qwen3.latest` is a local policy label in
`configs/qwen_policy.yaml`; it is not a web lookup, API lookup, model search, or
automatic claim about the newest available Qwen release.

## Offline Policy Resolution

Inspect the local policy file without installing `teacher-hf`:

```bash
python scripts/resolve_qwen_policy.py Qwen3.latest --allow-unresolved
python scripts/export_teacher_targets.py --config configs/teacher_export_qwen_dryrun.yaml --dry-run --resolve-qwen-policy --allow-unresolved-policy
```

The default policy intentionally stores `resolved_model_id: null`. A real Qwen
run must supply an explicit model id in config or with `--model-id`.

## Manual Export

Install optional HF dependencies only when performing a real teacher export:

```bash
python -m pip install -e ".[dev,teacher-hf]"
python scripts/export_teacher_targets.py --config configs/teacher_export_qwen_small_manual.yaml --model-id <explicit-qwen-model-id> --resolve-qwen-policy
```

Generated bundles are written under `artifacts/`, which is gitignored and must
not be committed.

## Warnings

Qwen-family teachers may require substantial CPU/GPU memory and disk cache
space. `trust_remote_code` is explicit policy/config state; enable it only after
reviewing the selected model repository and accepting the code execution risk.

Default validation does not run real Qwen export, does not require
torch/transformers, and does not contact the network.
