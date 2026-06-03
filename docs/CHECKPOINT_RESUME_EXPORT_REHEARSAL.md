# Checkpoint / Resume / Export Rehearsal

P108 adds a tiny local rehearsal for checkpoint, resume, and export plumbing.

## Scope

The rehearsal creates a tiny `tiny_student` checkpoint, reloads it through the
existing simple checkpoint loader, compares checkpoint-resumed outputs against
the original parameters, exports the checkpoint through the existing
HF/safetensors interchange path, reloads the export, and compares exported
outputs against the checkpoint outputs.

Primary APIs:

- `run_checkpoint_resume_export_rehearsal()`:
  `src/qrwkv_xla/checkpointing/rehearsal.py`
- CLI: `scripts/run_checkpoint_resume_export_rehearsal.py`

## Behavior

The rehearsal writes:

```text
checkpoints/p108_tiny_student/checkpoint.json
checkpoints/p108_tiny_student/params.npz
exports/p108_tiny_student_hf_safetensors/config.json
exports/p108_tiny_student_hf_safetensors/model.safetensors
exports/p108_tiny_student_hf_safetensors/qrwkv_xla_export.json
exports/p108_tiny_student_hf_safetensors/weight_map.json
```

If `safetensors` is installed, the result is `pass` only when checkpoint
resume and export reload both preserve student hidden-state and logits outputs.
If `safetensors` is absent, checkpoint resume still runs and the report returns
`unavailable` for the export portion with the dependency message.

## Claims Not Made

P108 does not add or claim:

- production checkpointing readiness
- distributed checkpointing readiness
- production Hugging Face export readiness
- training or optimizer readiness
- Qwen-specific support
- tokenizer remapping support
- runtime, WKV math, Pallas, fixture, or tolerance changes

Reference remains the default runtime and Pallas remains opt-in.
