# Bucket Shape Loss

## Purpose

P122 adds student-side bucket projection and optional bucket-shape loss for
`cascaded_soft_labels_v1` TeacherTextbooks.

## Tail Weather, Not Tail Atoms

The cascaded contract stores the exact top-k head and a lossy summary of the
tail. P122 compares the student's tail through the same lossy bucket lens. It
does not reconstruct or match exact non-top-k token probabilities.

## Student Bucket Projection

`project_student_tail_buckets` computes student probabilities, excludes teacher
`top_token_ids`, assigns remaining token probabilities to artifact
`bucket_edges`, and reports student top mass, tail mass, bucket mass, and
bucket count.

Bucket membership is hard and derived from stop-gradient probabilities. Bucket
mass remains differentiable inside the current assignment.

## Loss Components

`cascaded_soft_labels_v1` uses:

```text
total_loss =
  head_loss_weight * head_kl
  + tail_mass_loss_weight * tail_mass_loss
  + bucket_shape_loss_weight * bucket_shape_loss
```

The head term reuses sparse top-k KL. Tail mass loss compares log student tail
mass to log teacher tail mass. Bucket shape loss compares normalized teacher
and student tail bucket distributions.

## Default Weights

Defaults:

```text
head_loss_weight = 1.0
tail_mass_loss_weight = 0.0
bucket_shape_loss_weight = 0.0
bucket_shape_loss_type = kl
```

Bucket shape loss is opt-in.

## Hard vs Soft Bucket Assignment

P122 v0 uses hard probability bucket assignment. This is simple and stable for
small same-vocab smoke tests. A future phase may replace it with soft bucket
assignment if training requires smoother boundary behavior.

## Reporting

Cascaded reports include target type, loss type, top-k, bucket count, head
loss, tail loss, bucket shape loss, loss weights, teacher/student tail mass,
teacher/student bucket mass, and mean bucket KL.

Dense and `topk_with_tail_v0` reports do not invent cascaded-only bucket
metrics.

## What P122 Implements

- Student-side non-top-k tail bucket projection.
- Optional bucket-shape loss for `cascaded_soft_labels_v1`.
- Mini-eval dispatch and reporting for cascaded bucket metrics.
- Config/CLI knobs for bucket shape loss weight and type.
- Preservation of dense logits and `topk_with_tail_v0` behavior.

## What P122 Defers

- Real burn execution.
- Long training.
- Cross-vocab tail mapping.
- Soft bucket assignment.
- Cascaded evaluation comparison across dense/top-k/cascaded variants.

## Claims Not Made

P122 does not claim quality improvement, real burn success, tokenizer
remapping, Rosetta, Vocab C, cross-vocab tail mapping, Qwen/Gemma scale-up,
full HF-native integration, WKV/runtime behavior changes, or Pallas promotion.
