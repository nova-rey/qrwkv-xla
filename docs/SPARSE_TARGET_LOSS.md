# Sparse Target Loss for TopK+Tail TeacherTextbooks

## Purpose

P120 adds trainer/eval-side consumption for P119 `topk_with_tail_v0`
TeacherTextbooks. It lets student logits be supervised by the teacher's
compressed top-k head without pretending the artifact contains dense logits.

The dense P117 path remains canonical and unchanged for the official mini
textbook burn.

## Target Type Detection

The dispatch layer reads `target_type` from the loaded teacher target batch:

```text
dense_logits/full_logits/synthetic -> dense logits KL path
topk_with_tail_v0                  -> sparse top-k head KL path
cascaded_soft_labels_v1            -> top-k head-only path until P122
```

Unsupported target types fail clearly. Compressed targets must provide
`top_token_ids`, `top_log_probs`, and `attention_mask`; optional reporting fields
include `top_mass`, `tail_mass`, and `teacher_entropy`.

## Head KL Loss

For `topk_with_tail_v0`, the loss gathers student logits at teacher
`top_token_ids`:

```text
student_top_logits = gather(student_logits, top_token_ids)
```

Both teacher and student top-k heads are renormalized over K:

```text
teacher_head_log_probs = log_softmax(teacher_top_log_probs, axis=-1)
student_head_log_probs = log_softmax(student_top_logits, axis=-1)
head_kl = sum(exp(teacher_head_log_probs) *
              (teacher_head_log_probs - student_head_log_probs))
```

The loss is masked with `attention_mask != 0`.

## Tail Mass Regularization

Tail regularization is optional. When enabled, the student top-k mass is
estimated from gathered logits and full student logits, then compared to
teacher `tail_mass` in log space.

P120 does not use tail token membership because `topk_with_tail_v0` does not
store it.

## Default Weights

Defaults:

```text
sparse_head_loss_weight = 1.0
tail_loss_weight = 0.0
```

Tail regularization is off by default.

## Reporting

Sparse reports include:

```text
teacher_target_type
distillation_loss_type
mean_distillation_loss
head_loss
tail_loss
tail_loss_weight
top_k
mean_top_mass
mean_tail_mass
mean_teacher_entropy
```

Dense reports remain valid and do not invent sparse-only metrics.

For `cascaded_soft_labels_v1`, P121 reports `bucket_loss_weight=0.0` because
bucket-shape loss is intentionally deferred to P122.

## What P120 Implements

- Dense versus sparse target dispatch.
- Pure top-k-tail sparse loss over gathered student logits.
- Optional tail-mass regularization with default weight `0.0`.
- Dispatch-aware TeacherTargetStore batch loading.
- Mini eval consumption of `topk_with_tail_v0` artifacts without dense `logits`.
- Sparse report fields for target type, loss type, head/tail losses, and teacher
  mass/entropy metrics.

## What P120 Defers

- Bucketed cascaded tail losses.
- Rosetta/Vocab C/tokenizer remapping.
- Qwen/Gemma scale-up.
- Full HF-native model integration.
- lm_eval and generation integration.
- Real burn execution.

## Claims Not Made

P120 does not claim model quality, P117 success, real burn completion, dense
path replacement, production training readiness, compressed-target quality
improvement, WKV math changes, StudentRuntime semantic changes, StudentBackend
behavior changes, or Pallas promotion.
