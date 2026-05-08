# Multi-Scale Model Dry-Runs

P45 adds a bounded, metadata-first planning surface for QRWKV student profiles
at the 0.5B, 1.5B, and 7B-stretch planning scales. The default artifact root is
`artifacts/scale/p45_multiscale_dry_run`.

Generate config and fit artifacts:

```bash
.venv/bin/python scripts/generate_multiscale_configs.py \
  --out artifacts/scale/p45_multiscale_dry_run \
  --overwrite
```

Run metadata-only shape/checkpoint dry-runs from the generated scale plan:

```bash
.venv/bin/python scripts/run_multiscale_shape_dry_run.py \
  --scale-plan artifacts/scale/p45_multiscale_dry_run/scale_plan_report.json \
  --metadata-only \
  --out artifacts/scale/p45_multiscale_dry_run \
  --overwrite
```

Both CLIs accept repeatable `--profiles` selections. The config generator also
accepts repeatable `--hardware`; the dry-run command uses `--scale-plan` when
provided and otherwise computes fit metadata from the selected hardware set.

## Artifacts

The complete P45 artifact set is:

- `P45_RESULTS.md`
- `P45_SCALE_PLAN_REPORT.md`
- `scale_plan_report.json`
- `fit_matrix.json`
- `configs/*.yaml`
- `dry_runs/*/metadata_dry_run.json`
- `checkpoint_skeletons/*/checkpoint_manifest.json`
- `checkpoint_skeletons/*/model_config.yaml`
- `checkpoint_skeletons/*/checkpoint_metadata.json`

`fit_matrix.json` records per model x hardware memory components for estimated
parameters, optimizer state and gradients, activation/sequence memory,
hidden/logits target memory, checkpoint size, overhead reserve, and total
estimate. Fit is classified as `yes`, `maybe`, or `no` for the planning memory
estimate, while metadata-only dry-run support remains explicit and separate.

## Safety

P45 does not allocate full large model arrays by default. `--materialize-init`
is safe for `tiny_debug` and debug-status profiles; large profile init is
blocked unless `--allow-large-materialization` is passed explicitly. That flag
is only an escape hatch and is not part of normal validation.

Checkpoint skeletons are metadata bundles only. The dry-run writes the manifest,
model config YAML, and checkpoint metadata JSON, then reads them back and records
the readback status in `metadata_dry_run.json` and `P45_RESULTS.md`.

## Scope Limits

P45 does not prove real training, `pjit`/sharding, distributed execution,
Pallas kernels, WandB integration, full-scale measured memory, Qwen teacher
target generation, or one-device 7B training. TPU rows are aggregate planning
estimates unless a future phase adds and profiles sharded runtime support.
