# First Serious Compute Burn Harness

P112 provides the first serious compute burn harness and dry-run gate. It builds
the launchpad, writes reports, and documents the exact manual commands for a
human-reviewed burn attempt.

The actual serious compute burn requires an explicit human-run command with
`--confirm-serious-burn`. A P112 dry-run pass does not guarantee real burn
success.

## Purpose

The harness makes the first serious burn launch intentional, reproducible, and
guarded. It reads a burn configuration, checks P111 readiness status, runs a
cheap deterministic dry run, writes reports, and emits launch commands.

## P111 Relationship

P112 is gated by the P111 Big Burn Readiness Report. Real mode is blocked when
readiness fails. Warning readiness requires explicit warning acceptance unless
the config disables the strict readiness requirement.

## Dry-Run vs Real Mode

`dry_run` is the default mode. It runs only cheap local evidence:

- P109 read-only runtime environment preflight
- P108.1 checkpoint/resume update closure
- P110 mini eval harness on built-in tiny target artifacts
- P112 launch command generation

`real` mode is never the default. Without `--confirm-serious-burn`, real mode is
blocked. With confirmation, the harness reports that the gate is armed and
writes launch documentation; baseline code does not perform expensive training.

## Configuration Fields

The JSON config maps to `FirstSeriousBurnConfig`:

- `phase`
- `run_name`
- `mode`
- `teacher_model_id`
- `local_files_only`
- `allow_downloads`
- `architecture_id`
- `runtime`
- `target_store_path`
- `output_dir`
- `readiness_report_path`
- `max_steps`
- `batch_size`
- `sequence_length`
- `checkpoint_every_steps`
- `eval_every_steps`
- `require_readiness_pass`
- `accepted_warnings`

Defaults are conservative: `dry_run`, `local_files_only=True`,
`allow_downloads=False`, `runtime=reference`, and `max_steps=1`.

## Confirmation Flag

Real mode requires both:

- `--mode real`
- `--confirm-serious-burn`

Without the confirmation flag, the report status is `blocked` with the blocker
`real burn mode requires --confirm-serious-burn`.

## Launch Command Sequence

```bash
python scripts/run_big_burn_readiness_report.py \
  --output artifacts/p111_big_burn_readiness/readiness_report.json

python scripts/run_runtime_environment_preflight.py \
  --output artifacts/p109_runtime_environment/runtime_environment_report.json

python scripts/run_first_serious_burn.py \
  --config artifacts/p112_first_serious_burn/burn_config.json \
  --output artifacts/p112_first_serious_burn/dry_run \
  --mode dry_run

python scripts/run_first_serious_burn.py \
  --config artifacts/p112_first_serious_burn/burn_config.json \
  --output artifacts/p112_first_serious_burn/run_001 \
  --mode real \
  --confirm-serious-burn
```

Run the real command only after reviewing P111 readiness output and accepting
any warnings.

## Reports and Artifacts

The dry run writes:

- `burn_report.json`
- `readiness_report.json` when one is generated
- `preflight_report.json`
- checkpoint rehearsal output
- `mini_eval_report.json`
- `launch_commands.md`

## Stop Conditions

Stop before real mode when:

- P111 readiness status is `fail`
- P111 readiness status is `warn` and warnings are not accepted
- the real command omits `--confirm-serious-burn`
- dry-run report status is not `dry_run_pass`

## What P112 Proves

P112 proves that the repo has a guarded first-burn launch harness, conservative
dry-run evidence, structured reports, and explicit manual launch commands.

## What P112 Does Not Prove

P112 does not prove training success, model quality, production training
readiness, large-scale performance, distributed training readiness, Qwen
support, tokenizer remapping, Pallas default readiness, or that the serious
burn has completed.
