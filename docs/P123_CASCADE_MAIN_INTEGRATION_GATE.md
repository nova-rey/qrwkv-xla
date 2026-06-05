# P123 Cascade/Main Integration Gate

## Purpose

P123 reconciles the P119-P122 compressed/cascaded target branch with main after
P117.1 and P118.

## Inputs

- `main` with P117.1 actual configurable train-step burn.
- `main` with P118 burn result analysis.
- `p119-p120-cascaded-targets` with P119-P122 compressed and cascaded target
  work.

## Merge Strategy

The integration preserves P117.1/P118 mainline behavior and brings in the
cascade branch capabilities. Shared status docs were reconciled into one
timeline rather than leaving branch-only language.

## Integrated Capabilities

Main now contains:

- P117.1 real-mode train-step harness knobs and zero-step safeguards.
- P118 burn result analysis and compact summary.
- `topk_with_tail_v0` TeacherTextbook builder and validator support.
- Sparse top-k target consumption with optional tail mass loss.
- `cascaded_soft_labels_v1` bucketed artifact contract.
- Optional Bucket Shape Loss for cascaded targets.

Canonical TeacherTextbook target types:

```text
dense_logits
topk_with_tail_v0
cascaded_soft_labels_v1
```

## Verification Commands

```bash
python scripts/run_first_serious_burn.py --help | grep -E "teacher-textbook|max-steps|batch-size|reuse"
python -m pytest tests/test_first_serious_burn_training.py
python -m pytest tests/test_cascaded_bucket_projection.py
python -m pytest tests/test_cascaded_soft_labels_loss.py
python -m pytest tests/test_target_dispatch.py
python -m pytest tests/test_sparse_target_loss.py
python -m pytest tests/test_cascaded_soft_labels_textbook.py
python -m pytest tests/test_topk_tail_textbook.py
python -m pytest tests/test_teacher_textbook_builder.py
python -m pytest tests/test_teacher_textbook_artifact.py
python -m ruff check .
python -m black --check .
```

## What P123 Proves

P123 proves that the mainline P117.1/P118 burn harness and analysis can coexist
with the P119-P122 compressed/cascaded target pipeline in one repository state.

## What P123 Does Not Prove

P123 does not prove model quality, true distributed training, large-scale
performance, Pallas default readiness, Qwen/Gemma support, tokenizer remapping,
Rosetta/Vocab C, or cascaded target quality.

## Next Steps

- P124 - 100-example dense TPU smoke.
- P125 - radjax-tome repo extraction.
- P126 - Cascaded Target Evaluation Smoke.
