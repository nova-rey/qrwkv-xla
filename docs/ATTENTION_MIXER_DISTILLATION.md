# Attention / Mixer Distillation

Phase 20 adds the Stage 1 RADLADS-style path for QRWKV-XLA.

## Target contract

Stage 1 uses teacher attention or mixer **output vectors**, not attention weights.

```text
attention_targets: [batch, num_layers, sequence_length, hidden_size]
student.mixer_outputs: [batch, num_layers, sequence_length, hidden_size]
```

This matches the recurrent student's hidden-sized mixer pathway and avoids head-shaped attention-weight targets.

## Default validation path

Default CI and local smoke stay CPU-only, network-free, and fake-target based:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_attention.yaml
python scripts/run_distill_stage.py --config configs/distill_stage1_attention_stub.yaml --max-steps 2
```

## Student behavior

- `StudentOutput` now supports `mixer_outputs`.
- `TinyStudent` emits pre-activation per-layer mixer-style outputs when enabled.
- `RWKV7ReferenceStudent` emits the pre-residual recurrent mixer contribution when enabled.

## Loss

`attention_or_mixer` now runs a masked layerwise MSE over `[B, L, S, H]` tensors.

## HF / Qwen capture

Real HF attention target export is manual-only.

- enable `attention_capture`
- use `auto_qwen` or explicit module names
- no default model downloads in CI
- hook/module layouts are architecture-specific and may require manual adjustment

## Multi-device pmap smoke

Phase 21 adds a separate skip-safe Stage 1 `pmap` smoke:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_attention.yaml
python scripts/pmap_distill_smoke.py --config configs/distill_stage1_attention_pmap_smoke.yaml
```

This keeps the normal single-device distillation runner intact while proving the
attention/mixer loss can execute with replicated params and batch sharding.

## Current limits

Not included in Phase 20:

- attention-weight distillation as the primary target
- multi-device Stage 1 sharding
- FFN freeze / unfreeze scheduling
- default real Qwen export in CI
