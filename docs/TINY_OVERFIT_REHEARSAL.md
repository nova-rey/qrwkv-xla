# Tiny Overfit Rehearsal

P96 adds a tiny deterministic overfit rehearsal for the offline target path.
It loads stored synthetic teacher targets, builds an offline target batch,
computes logits from a tiny trainable head, applies local SGD updates, and
verifies that the logits MSE loss moves down.

The implementation lives in `src/qrwkv_xla/training/tiny_overfit.py`:

- `TinyOverfitResult`
- `run_tiny_overfit_rehearsal()`

The P96 update strategy is `tiny_trainable_logit_head`. This fallback head uses
deterministic row, position, token, and vocabulary biases to produce logits with
the same shape as the stored teacher logits. The update loop uses actual JAX
gradients and local SGD; it does not manually set the final loss.

P96 proves a tiny stored-target loss/update loop can move loss. It does not
prove full QRWKV student training, real training readiness, Qwen support,
production distillation, model quality, large-scale performance, or optimized
runtime behavior.

An optional script writes a small JSON report:

```bash
PYTHONPATH=src python scripts/run_tiny_overfit_rehearsal.py \
  --output artifacts/p96_tiny_overfit/tiny_overfit_report.json
```

P96 does not add Hugging Face or Qwen loading, external APIs, GPU/TPU
requirements, large datasets, distributed training, trainer replacement,
recurrence math changes, WKV equation changes, fixture edits, tolerance
changes, StudentBackend behavior changes, StudentRuntime behavior changes, or
Pallas promotion.
