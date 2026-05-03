# Distillation Runtime

## Purpose

The Phase 5 distillation runtime turns the earlier smoke-training path into a
reusable staged distillation entrypoint for QRWKV-XLA.

The canonical package name is `qrwkv_xla.distill`. The older
`qrwkv_xla.distillation` namespace is a thin compatibility alias and should not
be used in new docs or primary scripts.

Canonical script: `scripts/run_distill_stage.py`

Current runtime shape:

```text
DistillStageConfig
 -> TargetBundleDataset
 -> create_student(...)
 -> weighted distill loss computation
 -> jitted train step
 -> DistillStageResult
```

This is intentionally CPU-only and built around fake teacher bundles for now.
It is the pathway future RADLADS-style stages should extend.

## Config Shape

`configs/distill_stage0_stub.yaml` uses a top-level `distillation:` key.

Supported sections:

- `stage`
- `targets_dir`
- `student`
- `optimizer`
- `lr_schedule`
- `gradients`
- `training`
- `losses`

The optimizer section supports `sgd`, `adam`, and `adamw`. SGD remains the
default smoke optimizer.

`student.hidden_size` and `student.num_layers` may be `null`. In that case the
runner infers them from the target bundle manifest.

## Supported Students

- `tiny_student`
- `rwkv7_reference`

## Supported Losses

### Hidden MSE

Enabled by default. This uses the existing hidden-state MSE objective against
teacher hidden states from the bundle.

### Logits KL

Implemented and plumbed, but disabled by default.

It requires:

- `student_output.logits` to be present
- teacher `logits` in the target bundle
- matching student/teacher logits shapes

This keeps the runtime ready for future behavior-distillation work without
forcing premature logits-head work in Phase 5.

### Attention / Mixer Distillation

Phase 20 enables `attention_or_mixer` as a masked MSE loss between:

- `student_output.mixer_outputs`
- `teacher attention_targets`

Both use `[batch, num_layers, sequence_length, hidden_size]`.

## Runner Flow

1. Load and validate a `DistillStageConfig`
2. Validate the target bundle
3. Read the manifest
4. Infer or verify student `hidden_size` and `num_layers`
5. Build the selected student
6. Initialize params from the configured seed
7. Run a jitted train step with weighted distillation loss
8. Cycle shards deterministically until `max_steps`
9. Return `DistillStageResult`

## CLI Usage

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml
```

`scripts/run_distill_stage.py` is the canonical script name. The older
`scripts/run_distillation_stage.py` is kept only as a compatibility alias.

Optional overrides:

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --targets artifacts/teacher_targets/fake_export \
  --student-architecture rwkv7_reference \
  --max-steps 2 \
  --optimizer adamw \
  --learning-rate 0.001 \
  --weight-decay 0.01 \
  --seed 0
```

## Limitations

Phase 5 does **not** yet include:

- real Qwen teacher loading
- PyTorch or Hugging Face target extraction
- TPU sharding
- multi-stage orchestration
- production RWKV optimization
- lm_eval integration
- full RADLADS parity

## Checkpoint/Resume

`scripts/run_distill_stage.py` accepts `--checkpoint-out`, `--resume-from`, and
`--checkpoint-overwrite`. Fresh runs initialize student params from the training
seed. Resume loads params and the starting step from the checkpoint, validates
architecture plus `vocab_size`, `hidden_size`, and `num_layers`, then runs
`max_steps` additional updates.

Optimizer state is saved and resumed when present. Older checkpoints without
optimizer state initialize fresh optimizer slots at the checkpoint step.

The final checkpoint step is the loaded start step plus the current invocation's
`max_steps`. Use a different output directory from the resume source unless
overwrite is explicitly enabled.

## Learning Rate Scheduling

Distillation supports `constant` and `warmup_cosine` learning-rate schedules via
`distillation.lr_schedule` or CLI flags. The runner evaluates the schedule once
per update using the global step, applies the scheduled value to the optimizer,
and records both base and scheduled learning rates in tracked metrics.

## Gradient Clipping

Distillation supports optional global norm clipping via `distillation.gradients`
or CLI flags. Clipping is applied after gradient computation and before the
optimizer update, so Adam/AdamW moments see clipped gradients. Per-step metrics
record pre-clip norm, post-clip norm, clip scale, whether clipping occurred, and
the configured max norm. See `docs/GRADIENT_CLIPPING.md`.
## Run tracking

The distillation runner accepts an opt-in `tracking` config. When enabled, it
creates a run directory under `runs/`, writes metadata before the training loop,
appends per-step metrics to JSONL, and writes a final summary after checkpoint
save. If no checkpoint output is configured, the final checkpoint defaults to
`runs/<run_id>/checkpoints/final`.
## Logits KL Continuation

Students can opt into logits output with `student.emit_logits=true`. When
`losses.logits_kl` is enabled with positive weight, the runner requires both a
logits-capable student and teacher targets that include logits. Per-step
metrics include `logits_kl` when that loss is active.

Hidden-only checkpoints can be resumed into logits-enabled runs when the shared
student config still matches. The runner initializes fresh current params,
loads matching checkpoint params, and keeps fresh LM head params.
