# Experiment Tracking

P47 adds a bounded experiment tracking smoke on top of the existing
`qrwkv_xla.tracking` package. Local files are the source of truth. WandB is an
optional adapter and is not required for normal development or CI.

## Local Mode

The default command uses local tracking only:

```bash
.venv/bin/python scripts/run_tracking_smoke.py \
  --out artifacts/p47_experiment_tracking_smoke \
  --tracking local \
  --overwrite
```

Local mode writes:

```text
artifacts/p47_experiment_tracking_smoke/
  P47_RESULTS.md
  tracking_smoke_report.json
  local_run/
    run_metadata.json
    config.json
    metrics.jsonl
    summary.json
    artifacts_manifest.json
    files/
```

`config.json` is the exact smoke config used for the run. `metrics.jsonl`
contains one JSON object per step and includes `step`, `train/loss`,
`train/loss_is_finite`, `train/tokens_seen`, and `train/examples_seen`.
`summary.json` records final status, final loss, finite-loss status, and token
and example counts.

## WandB

WandB modes are opt-in:

```bash
.venv/bin/python scripts/run_tracking_smoke.py --tracking wandb-offline
.venv/bin/python scripts/run_tracking_smoke.py --tracking wandb-online
```

The adapter imports `wandb` only when a WandB mode is requested. If WandB is not
installed, local mode still works. No normal test requires WandB credentials,
online login, or network access.

WandB offline/online runs still write the same local files first. Those local
files are the durable record used by P47 tests and CI.

## Metadata

`run_metadata.json` records:

- phase and UTC creation time
- repo commit and git status metadata
- `git_dirty` classification
- Python and JAX/runtime metadata when available
- backend and default backend
- device counts, local device counts, device kinds, and platforms
- hostname when available
- command/script name
- tracking mode and artifact root

`git_dirty` values:

- `clean`: `git status --short` had no entries.
- `untracked_artifacts_only`: the only status entries were untracked files.
- `tracked_dirty`: at least one tracked file was modified, deleted,
  renamed, or staged.
- `unknown`: git metadata could not be collected.

## Artifact Manifest

Artifacts registered with the local tracker are copied under
`local_run/files/`. `artifacts_manifest.json` records each artifact path
relative to `local_run`, its `kind`, byte `size`, source path, and SHA-256 hash.

## Scope

P47 proves the tracking plumbing with a tiny deterministic local run. It does
not perform real Qwen0.5B training, benchmark model quality, implement sweeps,
require production dashboards, or replace existing run/artifact conventions
outside this bounded smoke layer.
