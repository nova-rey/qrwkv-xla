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
