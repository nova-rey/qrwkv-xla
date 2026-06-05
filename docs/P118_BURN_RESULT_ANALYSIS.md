# P118 Burn Result Analysis

## Summary

P118 records the post-flight analysis for the first successful actual
train-step hardware smoke. P117 proved allocation and launch plumbing but
allowed a real-mode pass with `steps_completed=0`. P117.1 replaced that with an
actual 8-step TPU-backed train-step smoke against the dense mini
TeacherTextbook.

This is execution evidence. It is not model-quality evidence and it is not
proof of coordinated distributed or sharded training.

## Input Artifacts

The analysis is based on the archived P117.1 worker-0 result described as:

```text
p117_1_real_train_steps_v6e16_worker0
p117_1_burn_report_v6e16_worker0.json
```

The large run archive is not committed. A compact machine-readable summary is
stored at `docs/results/P118_P117_1_RESULT_SUMMARY.json`.

## Hardware / Runtime

The archived report came from a v6e-16 TPU allocation. The burn report recorded
`device_backend=tpu`, and the hardware report listed 16 TPU devices visible
across 4 JAX processes. The worker-0 report is the archived analysis source.

This does not prove true sharded/distributed training correctness. The run
likely executed the same tiny smoke on each worker unless a separate phase adds
explicit distributed coordination, synchronized optimizer checks, and
rank-disciplined checkpointing.

## Burn Configuration

```text
mode: real
dry_run: false
max_steps: 8
batch_size: 1
allow_textbook_reuse: false
teacher_textbook_examples: 8
```

The run consumed exactly the 8 available examples without reuse.

## Result Table

| Field | Value |
| --- | --- |
| phase | P117.1 |
| status | pass |
| mode | real |
| dry_run | false |
| readiness_status | pass |
| device_backend | tpu |
| steps_requested | 8 |
| steps_completed | 8 |
| batch_size | 1 |
| allow_textbook_reuse | false |
| examples_available | 8 |
| examples_consumed | 8 |
| unique_examples_consumed | 8 |
| reuse_count | 0 |
| loss_initial | 0.00031686286092735827 |
| loss_final | 0.0003037375572603196 |
| loss_delta | -0.000013125303667038679 |
| checkpoint_written | true |
| checkpoint_nonzero | true |
| real_training_executed | true |
| blockers | [] |
| warnings | [] |
| teacher_target_type | dense TeacherTextbook |
| teacher_textbook_examples | 8 |
| teacher_textbook_sequence_length | recorded in source artifact |
| teacher_textbook_vocab_size | recorded in source artifact |

## Loss Trace Summary

```text
loss_initial = 0.00031686286092735827
loss_final   = 0.0003037375572603196
loss_delta   = -0.000013125303667038679
```

This is a finite, slight decrease in a tiny smoke test. It should be treated as
execution evidence, not quality evidence.

## Checkpoint Evidence

The run reported:

```text
checkpoint_written: true
checkpoint_nonzero: true
real_training_executed: true
```

That proves the P117.1 harness wrote a non-empty checkpoint after actual train
steps. It does not prove checkpoint discipline for multi-worker or production
distributed training.

## What P117.1 Proved

- P117.1 real mode executed actual training steps.
- The run completed 8/8 requested steps.
- The run consumed 8 unique examples from the mini TeacherTextbook.
- Textbook reuse was disabled and `reuse_count` was 0.
- A checkpoint was written and nonzero.
- A loss trace was written.
- Loss was finite.
- The backend reported TPU.
- JAX saw 16 TPU devices across 4 processes.
- Blockers and warnings were empty.

## What P117.1 Did Not Prove

- Model quality.
- Meaningful learning.
- Production training readiness.
- Large-scale performance.
- True distributed or sharded training correctness.
- Synchronized optimizer state across workers.
- Rank-0-only checkpoint discipline.
- Real QRWKV/RWKV student architecture training.
- Pallas default readiness.
- Qwen/Gemma support.
- Tokenizer remapping.
- Rosetta or Vocab C.
- Cascaded target effectiveness.

## Known Caveats

The run is best described as a TPU-backed train-step smoke. The 8-example
TeacherTextbook is intentionally tiny, the optimizer path is a smoke-scale
student path, and the loss movement is too small and too scoped to support any
quality claim.

The hardware launch reported multiple TPU devices and processes, but P117.1
did not add explicit distributed coordination checks. Treat the result as TPU
execution proof, not distributed training proof.

## Recommended Next Steps

1. Run P117.2 as a 100-example TPU data-shard smoke to prove broader textbook
   consumption and reuse/epoch accounting under a still-small budget.
2. Add explicit multi-worker discipline checks before claiming distributed
   training: synchronized optimizer state, rank-0 checkpointing, per-rank
   metrics, and deterministic shard assignment.
3. Merge or reconcile the P119-P122 cascade branch only as a separate reviewed
   phase, then plan P123 cascaded target evaluation smoke.

## Claims Not Made

P118 does not claim model quality, production training readiness, true
distributed training correctness, real QRWKV architecture training, Qwen/Gemma
support, tokenizer remapping, Rosetta/Vocab C, cascaded target effectiveness,
Pallas default readiness, or large-scale performance.

P118 did not run another burn, start training, change training behavior, merge
the cascade branch, or add new model/runtime functionality.
