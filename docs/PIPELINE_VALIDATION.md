# Pipeline Validation

## Purpose

`scripts/validate_pipeline.py` is the canonical end-to-end validation harness
for the runnable QRWKV-XLA pipeline. The default path is CPU-safe, offline, and
requires only the `.[dev]` extra.

## Default Path

```bash
python -m pip install -e ".[dev]"
python scripts/validate_pipeline.py
```

The default pipeline runs environment/runtime inspection, CPU and TPU-safe
smokes, offline Qwen policy dry-runs, prompt corpus inspection and manifest
creation, fake target export and inspection, both student smoke architectures,
the distillation stage smoke, and the TPU-ready distillation smoke without
requiring a TPU.

## Optional Hugging Face Path

```bash
python -m pip install -e ".[dev,teacher-hf]"
python scripts/validate_pipeline.py --include-hf
```

`--include-hf` adds the tiny HF teacher export, inspects
`artifacts/teacher_targets/hf_tiny`, and runs the distillation stage against
that bundle. This path may require torch/transformers and local or downloadable
model assets. Missing optional dependencies should fail clearly.

## Optional Hard TPU Check

```bash
python scripts/validate_pipeline.py --require-tpu
```

`--require-tpu` passes `--require-tpu` to `scripts/tpu_distill_smoke.py`, making
TPU availability a hard requirement. Do not use this flag in default CI or
CPU-only handoff validation.

## Artifacts

Generated bundles live under `artifacts/`, which is gitignored. They are local
validation outputs and must not be committed.

The default pipeline also runs a checkpoint save/resume smoke. Generated
checkpoint artifacts live under `checkpoints/`, which is gitignored.

## CI-Safe vs Optional

CI should run:

```bash
python scripts/validate_pipeline.py
```

The default path now includes:

```bash
python scripts/inspect_prompt_corpus.py corpora/smoke_prompts.jsonl
python scripts/create_prompt_manifest.py corpora/smoke_prompts.jsonl --out artifacts/prompt_manifests/smoke_prompts.manifest.json --description "Pipeline smoke corpus." --overwrite
python scripts/export_teacher_targets.py --config configs/teacher_export_qwen_dryrun_corpus.yaml --dry-run --resolve-qwen-policy --allow-unresolved-policy
```

CI should not pass `--include-hf` or `--require-tpu` by default. Optional HF and
hard TPU outcomes should be reported separately from the required validation
result.

## Troubleshooting

- Missing `jax`: install `python -m pip install -e ".[dev]"`.
- Missing `torch` or `transformers` with `--include-hf`: install
  `python -m pip install -e ".[dev,teacher-hf]"`.
- No TPU with `--require-tpu`: rerun on a TPU-backed JAX environment or omit the
  flag for CPU-safe validation.
- Generated bundle issues: delete local `artifacts/` and rerun the pipeline.
## Tracked run smoke

The default validation pipeline includes a one-step tracked distillation run
under `runs/pipeline_smoke`. This verifies local `run.json`, `metrics.jsonl`,
`summary.json`, and the tracked default checkpoint path without enabling any
external service.
## Logits KL Smoke

The default pipeline exports fake logits targets, inspects them, runs
`configs/distill_stage0_logits_stub.yaml`, and smokes hidden-only to logits
checkpoint continuation. This remains CPU-only and network-free.
