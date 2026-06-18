# P140 Real Student Fingerprint Forward Smoke

P140 starts the next behavioral fingerprint arc: real student integration plus
teacher-pure capture. This phase only takes the first step in that arc. It
forwards a validated fingerprint corridor batch through the actual registered
QRWKV student backend and computes the existing P134/P135 diagnostics from the
resulting logits.

This is forward-only. It does not run an optimizer, train a checkpoint, call a
teacher, invoke Hugging Face, require an accelerator, or wire the main staged
runner.

## Path

```text
FingerprintTargetDataset
  -> VocabContract from artifact metadata
  -> create_student_backend(current_qrwkv)
  -> backend.forward_full(params, batch.input_ids)
  -> backend.logits(output)
  -> select target-position logits
  -> P134 distribution stats
  -> P135 corridor loss
  -> metrics/report/summary artifacts
```

The default backend is the real `CurrentQRWKVStudentBackend`, selected through
the existing student backend registry. The smoke uses artifact `input_ids` and
requires logits shaped `[batch, seq, vocab]`.

## CLI

```bash
python scripts/run_real_student_fingerprint_forward_smoke.py \
  --artifact tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny \
  --batch-size 2 \
  --seed 0 \
  --output-dir /tmp/qrwkv_fingerprint_p140_real_student_forward
```

Outputs:

- `metrics.json`
- `real_student_fingerprint_forward_report.json`
- `fingerprint_run_summary.md`

## Report Contract

The report identifies this path explicitly:

- `phase: P140`
- `report_type: real_student_fingerprint_forward_smoke_report`
- `training_path_kind: real_student_fingerprint_forward_smoke`
- `smoke_student_kind: real_student_backend`
- `smoke_student_uses_input_ids: true`
- `real_student_backend_integrated: true`
- `main_runner_integrated: false`
- `teacher_required: false`
- `hf_required: false`
- `accelerator_required: false`
- `optimizer_steps_completed: 0`
- `exemplar_forward_enabled: false`

Nested `student`, `artifact`, `forward`, `corridor`, and `corridor_metrics`
sections are written for the P139 summary/report helpers.

## Metrics

P140 emits real-forward shape/finite metrics:

- `fingerprint/real_student_forward/logits_finite`
- `fingerprint/real_student_forward/logits_shape_batch`
- `fingerprint/real_student_forward/logits_shape_seq`
- `fingerprint/real_student_forward/logits_shape_vocab`

It also preserves the canonical corridor metric surface:

- `fingerprint/corridor/loss_total`
- `fingerprint/corridor/loss_entropy`
- `fingerprint/corridor/loss_top1_margin`
- `fingerprint/corridor/loss_top8_mass`
- `fingerprint/corridor/loss_top32_mass`
- `fingerprint/corridor/loss_tail_mass`
- `fingerprint/corridor/inside_all_rate`

## Guards

The smoke fails before metric computation when:

- artifact and student vocab sizes differ
- artifact max sequence length exceeds an explicit student sequence cap
- batch token IDs are outside the student vocab
- target positions fall outside the returned logits sequence dimension
- returned logits are not rank-3 `[batch, seq, vocab]`

## Claims

P140 proves only that the real registered QRWKV student backend can be
instantiated from a fingerprint artifact vocab contract, consume fingerprint
`input_ids`, emit finite logits of compatible shape, and feed P134/P135
corridor diagnostics.

P140 does not prove training, main-runner integration, checkpoint semantics,
teacher-side fingerprint capture, quality improvement, artifact scaling,
TPU/GPU behavior, Pallas readiness, or production readiness.

## Next

P141 should wire the fingerprint objective mode into the main staged runner
without weakening the clear boundaries introduced here.
