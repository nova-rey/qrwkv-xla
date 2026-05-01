# Logits Distillation

Hidden-state distillation trains student internal states to match teacher
targets. Logits distillation trains student output distributions with KL loss
when teacher target bundles include logits.

The supported staged path is:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_logits.yaml
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --targets artifacts/teacher_targets/fake_export --max-steps 2 --checkpoint-out checkpoints/hidden_only_for_logits --checkpoint-overwrite
python scripts/run_distill_stage.py --config configs/distill_stage0_logits_stub.yaml --targets artifacts/teacher_targets/fake_export_logits --resume-from checkpoints/hidden_only_for_logits --checkpoint-out checkpoints/hidden_plus_logits --checkpoint-overwrite --max-steps 2
```

If a hidden-only checkpoint is resumed with `student.emit_logits=true`, the
runner initializes fresh current-config parameters, merges matching checkpoint
parameters, and keeps fresh LM head parameters. Architecture, vocabulary size,
hidden size, and layer count must still match.

Logits target bundles are larger than hidden-only bundles. Default validation
uses fake logits only and remains CPU-only and offline-safe. Real HF/Qwen logits
exports are optional manual work.

Current limitations:

- No generation or sampling CLI is included.
- No tokenizer-aware evaluation harness is included.
- No advanced loss scheduling is included.
- Large Qwen logits exports are not part of default validation.
