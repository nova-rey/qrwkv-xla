# P141 Main Runner Fingerprint Mode

P141 makes corridor-only behavioral fingerprint training a first-class main
runner mode.

The new `fingerprint_corridor` mode keeps the P140 real-student boundary but
moves from forward-only diagnostics into optimizer/checkpoint/report plumbing
through `run_distill_stage`.

## Mode

```bash
python scripts/run_distill_stage.py \
  --distill-mode fingerprint_corridor \
  --fingerprint-artifact tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny \
  --student-backend current_qrwkv \
  --steps 3 \
  --batch-size 2 \
  --learning-rate 0.01 \
  --output-dir /tmp/qrwkv_p141_fingerprint_corridor
```

Outputs:

- `metrics.json`
- `fingerprint_corridor_report.json`
- `fingerprint_run_summary.md`
- `checkpoints/final/checkpoint.json`
- `checkpoints/final/params.npz`

## Training Path

```text
run_distill_stage(mode=fingerprint_corridor)
  -> validate fingerprint artifact
  -> load P133 corridor batches
  -> instantiate current_qrwkv through the student backend registry
  -> backend.forward_full(params, batch.input_ids)
  -> P134 distribution stats at target positions
  -> P135 corridor loss
  -> existing optimizer update
  -> existing checkpoint writer
  -> P141 metrics/report/summary
```

The mode cycles finite fingerprint batches until the requested optimizer steps
complete. It fails clearly if no batches are yielded.

## Teacher-Free Contract

`fingerprint_corridor` does not load `TargetBundleDataset`, does not read a
teacher target manifest, does not construct a teacher backend, and does not
require Hugging Face, internet, Qwen, torch, GPU, or TPU.

Required inputs are only:

- fingerprint artifact directory
- registered student backend id
- optimizer/training settings

## Metrics

The final result and `metrics.json` include normal runner metrics:

- `loss`
- `train/loss`
- `learning_rate`
- `global_step`
- optimizer and gradient metrics

For this mode, `train/loss` equals
`fingerprint/corridor/loss_total`.

Canonical fingerprint metrics are emitted:

- `fingerprint/corridor/loss_total`
- `fingerprint/corridor/loss_entropy`
- `fingerprint/corridor/loss_top1_margin`
- `fingerprint/corridor/loss_top8_mass`
- `fingerprint/corridor/loss_top32_mass`
- `fingerprint/corridor/loss_tail_mass`
- `fingerprint/corridor/inside_entropy_rate`
- `fingerprint/corridor/inside_top1_margin_rate`
- `fingerprint/corridor/inside_top8_mass_rate`
- `fingerprint/corridor/inside_top32_mass_rate`
- `fingerprint/corridor/inside_tail_mass_rate`
- `fingerprint/corridor/inside_all_rate`
- `fingerprint/runner/optimizer_steps_completed`
- `fingerprint/runner/batches_consumed`
- `fingerprint/runner/artifact_num_records`

## Report

`fingerprint_corridor_report.json` identifies the boundary:

- `phase: P141`
- `distill_mode: fingerprint_corridor`
- `training_path_kind: main_runner_fingerprint_corridor`
- `real_student_backend_integrated: true`
- `main_runner_integrated: true`
- `teacher_required: false`
- `exemplar_reservoir_enabled: false`

The Markdown summary states that this is main-runner corridor-only fingerprint
training, no exemplar reservoir training is active, no teacher backend is
required, and teacher-side fingerprint capture remains future work.

## Guards

P141 fails before training when:

- artifact and student vocab sizes differ
- artifact max sequence length exceeds an explicit student sequence cap
- input token IDs are outside the student vocab
- target positions exceed the emitted sequence length
- batch iteration yields zero batches

There is no token remapping and no silent resizing.

## Claims

P141 proves:

- the main staged runner can accept a fingerprint artifact
- the runner can train the real registered student backend with corridor loss
- optimizer updates, checkpoint writing, metrics, report, and summary artifacts
  work in fingerprint corridor mode
- no teacher backend is required

P141 does not prove exemplar training, mixed objectives, teacher-side
fingerprint capture, real teacher artifacts, quality improvement, artifact
convergence, storage/compute wins, TPU/GPU behavior, or production readiness.

## Next

P142 should use `fingerprint_corridor` for an input-conditioned tiny student
rehearsal. P143 remains the start of teacher-side fingerprint capture.
