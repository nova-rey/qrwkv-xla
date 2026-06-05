# Cascaded Soft Labels v1

## Purpose

P121 added `cascaded_soft_labels_v1`, a TeacherTextbook target format that keeps
the high-resolution top-k head and stores the tail through lossy aggregate
buckets. P122 adds student-side bucket projection and optional
`bucket_shape_loss`.

The student is compared through the same lossy bucket lens as the teacher
artifact. The loss matches tail weather, not exact tail atoms.

## High-Resolution Head, Lossy Tail Lens

The head remains exact through `top_token_ids`, `top_log_probs`, and
`top_mass`. The tail is summarized as bucket weather through `bucket_mass`,
`bucket_count`, and `bucket_mean_logp`.

Bucket membership token IDs are not stored.

## Relationship to TopK+Tail v0

`cascaded_soft_labels_v1` includes every `topk_with_tail_v0` shard field and
adds bucketed tail summaries. It does not mutate `topk_with_tail_v0`.

P122 consumes cascaded artifacts through the sparse top-k head path plus
optional tail mass and bucket shape terms. Bucket shape loss is off by default
and becomes active only when its weight is nonzero.

## Target Type

```text
cascaded_soft_labels_v1
```

Dense `dense_logits` and compressed `topk_with_tail_v0` remain supported.

## Shard Arrays

```text
input_ids           [N,T]
attention_mask      [N,T]
top_token_ids       [N,T,K]
top_log_probs       [N,T,K]
top_mass            [N,T]
tail_mass           [N,T]
teacher_entropy     [N,T]
bucket_mass         [N,T,B]
bucket_count        [N,T,B]
bucket_mean_logp    [N,T,B]
```

Dense `logits` are not persisted for cascaded artifacts.

## Bucket Edges

Default probability bucket edges:

```text
[1.0, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]
```

This defines five descending probability buckets over non-top-k tail tokens.
Empty buckets use `bucket_mean_logp = 0.0` with `bucket_count = 0`.

## Builder CLI

```bash
python scripts/build_teacher_textbook.py \
  --teacher-mode fake \
  --output artifacts/p121_cascade_fake_smoke \
  --sequence-length 16 \
  --batch-size 2 \
  --max-examples 4 \
  --target-type cascaded_soft_labels_v1 \
  --top-k 8 \
  --vocab-size 64 \
  --top-log-probs-dtype float32 \
  --overwrite
```

Optional flags:

```text
--bucket-edges "1,1e-3,1e-6,1e-9,1e-12,0"
--bucket-edge-type probability
--bucket-mass-dtype float32
--bucket-mean-logp-dtype float32
```

## Validation Rules

Validation checks all TopK+Tail v0 rules plus bucket metadata, bucket shape,
bucket count sums, non-negative bucket mass/count, mass integrity, and the
empty-bucket `0.0` mean-log-prob sentinel.

## What P121 Implements

- `cascaded_soft_labels_v1` target type.
- Fake and HF builder emission for bucketed tails.
- Metadata, manifest, and emission config bucket fields.
- TeacherTargetStore and TeacherTextbook validation.
- Loader exposure of bucket fields.
- Top-k head-only eval compatibility with `bucket_loss_weight=0.0`.

## What P121 Defers

- Bucket-shape trainer loss.
- Student-side bucket projection.
- Cross-vocab/Rosetta tail handling.
- Qwen/Gemma scale-up.
- Full HF-native wrapper work.
- Real burn execution.

## What P122 Adds

- Student-side bucket projection over non-top-k probabilities.
- Optional normalized bucket-shape KL over teacher/student tail bucket mass.
- `bucket_shape_loss_weight` defaulting to `0.0`.
- Cascaded mini-eval reporting for teacher/student tail and bucket mass.

## Same-Vocab Scope

P121 assumes teacher and student vocabularies match. Tokenizer remapping,
Vocab C, and cross-vocab tail mapping are deferred.

## Claims Not Made

P121 does not claim model quality, P117 success, real burn completion, training
readiness, bucket-loss readiness, dense path replacement, WKV math changes,
StudentRuntime semantic changes, StudentBackend behavior changes, full
HF-native integration, or Pallas promotion.
