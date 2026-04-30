# QRWKV-XLA Architecture

QRWKV-XLA is intended to be a full-featured long-term codebase for transformer-to-recurrent distillation under JAX/XLA constraints. The early phases focus on contracts and pipeline boundaries before heavyweight model integration.

## Major subsystems

### Teacher Exporter
- **Current shape:** protocol/interface plus deterministic fake implementation
- **Later framework:** PyTorch + Hugging Face
- **Purpose:** export reusable teacher targets through a stable exporter contract.

Teacher Exporter is now represented by a protocol/interface and a fake deterministic implementation. Real PyTorch/Hugging Face exporters will later implement the same contract.

### Target Artifact Store
- **Purpose:** Store input ids, masks, hidden states, logits, attention or mixer targets, metadata, tokenizer references, teacher identity, and stage info.

### JAX/XLA Student Trainer
- **Framework:** JAX
- **Purpose:** Train RWKV7-style recurrent students using XLA-friendly code paths with CPU debug, TPU smoke, and eventual TPU scale-up support.

### Student Model Core
- **Current shape:** `tiny_student` trainer test double plus `rwkv7_reference`
  recurrent reference implementation.
- **Purpose:** Define RWKV7-style recurrent model interfaces and host
  scan-based recurrent blocks plus later stateful inference contracts.

The `rwkv7_reference` core is an XLA-friendly recurrent reference
implementation. It exists to lock down shapes, masking, JIT behavior, and
gradient flow for the student path. It is not a final optimized RWKV7 kernel.

### Distillation Engine
- **Current shape:** stage config dataclasses/YAML loading, weighted loss
  registry/composition, hidden-state distillation runtime, optional logits KL
  validation plumbing, metrics summaries, and a CPU-only stage CLI over target
  bundles.
- **Purpose:** Coordinate staged losses for hidden-state distillation, logit distillation, attention/mixer behavior distillation, and optional instruction/behavior preservation.

### XLA Discipline / TPU Smoke
- **Current shape:** runtime inspection utilities, distillation smoke wrapper,
  CPU-safe default behavior, and explicit TPU hard-fail mode.
- **Purpose:** Verify XLA/JAX runtime visibility and single-device smoke
  readiness without turning TPU launcher checks into a second training stack.

The TPU smoke path is intentionally CPU-safe by default. Real TPU validation is
opt-in through `--require-tpu`. Sharding, `pjit`, and Pallas work remain
deferred.

### Checkpointing
- **Purpose:** Save and resume JAX student state, tracking optimizer state, config, RNG state, global step, artifact IDs, and model metadata.

### Evaluation
- **Purpose:** Compare student vs teacher with loss curves, generation sanity checks, recurrent inference checks, and later `lm_eval`-style integration.

### Export
- **Purpose:** Package trained recurrent students for inference and sharing once the training pipeline exists.

## Architectural framing

The earliest runnable configs may be tiny, but the architecture is real. Small configurations are allowed; disposable toy architecture is not.
