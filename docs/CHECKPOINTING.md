# Checkpointing

QRWKV-XLA uses a simple local checkpoint format for the current distillation
stage runner:

- `checkpoint.json`: JSON manifest with schema, student architecture/config,
  training step, learning rate, loss config, target manifest summary, and array
  metadata.
- `params.npz`: NumPy archive containing parameter arrays as `arr_000000`,
  `arr_000001`, and so on.

Checkpoint directories must live under `checkpoints/`, and `checkpoints/` is
gitignored. The format is intended for local staged continuation and smoke
validation, not for long-term model release.

## CLI

Save a checkpoint after a short stage:

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --max-steps 2 \
  --checkpoint-out checkpoints/stage0_hidden \
  --checkpoint-overwrite
```

Resume and run additional steps:

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --max-steps 2 \
  --resume-from checkpoints/stage0_hidden \
  --checkpoint-out checkpoints/stage0_hidden_resume \
  --checkpoint-overwrite
```

On resume, `--max-steps` means additional steps for that invocation. A
checkpoint saved at step 2 and resumed with `--max-steps 2` ends at step 4.

## Validation

Resume checks fail clearly if the checkpoint student architecture differs from
the requested architecture, or if `vocab_size`, `hidden_size`, or `num_layers`
differs from the resolved student config. The loader also validates that both
files exist, the schema version is supported, all recorded array keys exist,
and each array has the recorded shape and dtype.

If `checkpoint_out` and `resume_from` point to the same directory, overwrite
must be enabled. Without overwrite this fails before training.

## Why Not Orbax Yet

Orbax is intentionally deferred. The current project needs a CPU-safe,
offline, dependency-light checkpoint surface for single-process staged
distillation. JSON plus NPZ is enough for the local hidden-state-only stage and
keeps the artifact easy to inspect. Orbax or a richer checkpoint manager can be
revisited after multi-device training, optimizer state, and durable release
requirements are concrete.

## Staged Continuation

The expected path is hidden-only continuation first: export targets, train on
hidden-state MSE, checkpoint, then resume for more hidden-state steps. Later,
when student logits are implemented, the same staged flow can continue into a
logits-aware phase using a new checkpoint output directory.
