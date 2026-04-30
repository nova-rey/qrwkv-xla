# Distillation Runtime

## Purpose

The Phase 5 distillation runtime turns the earlier smoke-training path into a
reusable staged distillation entrypoint for QRWKV-XLA.

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
- `training`
- `losses`

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

Not implemented yet.

If `attention_or_mixer` is enabled, the runtime raises a clear error instead of
silently ignoring it.

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

Optional overrides:

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --targets artifacts/teacher_targets/fake_export \
  --student-architecture rwkv7_reference \
  --max-steps 2 \
  --learning-rate 0.001 \
  --seed 0
```

## Limitations

Phase 5 does **not** yet include:

- real Qwen teacher loading
- PyTorch or Hugging Face target extraction
- TPU sharding
- multi-stage orchestration
- checkpointing overhaul
- production RWKV optimization
- lm_eval integration
- full RADLADS parity
