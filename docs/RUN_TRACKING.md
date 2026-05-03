# Run Tracking

QRWKV-XLA supports lightweight local run tracking for distillation stages. It is
disabled by default and writes only durable local files under `runs/`.

## Files

Each tracked run creates:

- `run.json`: run metadata, config snapshot, git metadata, and environment metadata.
- `metrics.jsonl`: one JSON object per training step.
- `summary.json`: final status, metrics, checkpoint path, and lineage.
- `checkpoints/final/`: default final checkpoint when tracking is enabled and no
  explicit `checkpoint_out` is provided.

All pretty JSON files use `indent=2`. Metrics use JSONL so they can be appended
and streamed safely.

## CLI

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --max-steps 2 \
  --track-run \
  --run-name stage0-smoke \
  --run-tag smoke \
  --run-note "local run"
```

Optional flags:

- `--run-root PATH`: defaults to `runs/` when tracking is enabled.
- `--run-name NAME`: used in the run id slug.
- `--run-tag TAG`: repeatable.
- `--run-note NOTE`: repeatable.
- `--run-overwrite`: allow reuse of an existing run directory.

Run ids use UTC timestamps: `YYYYMMDD_HHMMSS_<slug>`.

## Config

```yaml
distillation:
  tracking:
    enabled: true
    run_root: runs
    run_name: stage0-smoke
    overwrite: false
    tags: [smoke]
    notes: [local only]
```

`run_root` must be `runs/` or a path below a `runs/` directory. This keeps
generated run artifacts under the gitignored local run tree.

## Behavior

When tracking is disabled, distillation behavior and checkpoint behavior are
unchanged. When tracking is enabled and `checkpoint.checkpoint_out` is unset,
the runner saves the final checkpoint to:

```text
runs/<run_id>/checkpoints/final
```

Git and JAX environment metadata are best effort. Missing git, a non-git
directory, or incomplete JAX metadata will be recorded as unavailable metadata
instead of failing training.

Tracked distillation runs should reference teacher target manifests that now, in
turn, can reference prompt corpus metadata through `prompt_source` when corpus
exports are used.

## Logits Metrics

When logits KL is enabled, `metrics.jsonl` and `summary.json` include
`logits_kl` / `final_logits_kl` alongside hidden loss metrics. Hidden-only runs
continue to omit logits metrics.

## Optimizer Metrics

Per-step metrics include numeric `learning_rate` and `optimizer_step`. Metric
record `extra` includes `optimizer_type`, and `summary.json` records the final
optimizer type and learning rate.

## Scheduler Metrics

Tracked distillation runs record `lr_schedule` metadata in `run.json` and
`summary.json`. Per-step metrics include `learning_rate`,
`base_learning_rate`, `global_step`, and `local_step`; metric `extra` includes
`lr_schedule_type`.

## Gradient Clipping Metrics

Tracked distillation runs record gradient clipping config in `run.json` and
`summary.json`. Per-step metrics include `grad_global_norm`,
`grad_clipped_global_norm`, `grad_clip_scale`, `grad_was_clipped`, and
`max_grad_norm`.

## Distributed metadata

The Phase 21 standalone `pmap` smokes do not attempt full run-tracking parity,
but when their results are reported they should include backend, visible device
count, active device count, per-device batch size, and whether checkpoint state
was unreplicated before saving.

## Generation Outputs

P14 generation smoke writes local `generations.jsonl` and `summary.json`
artifacts under `eval_outputs/` by default. Future phases may attach generation
outputs under run directories, but P14 keeps eval artifacts separate and
gitignored.
## Evaluation Snapshots

Evaluation snapshots currently write under gitignored `eval_outputs/` by
default. They can be copied under `runs/<run_id>/evals/<eval_id>/` when a run
needs a local record of generated outputs. Full automatic run attachment is
deferred.
