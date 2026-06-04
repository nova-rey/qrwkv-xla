# P116 Revised Burn Launch Plan

## Executive Summary

P116 turns the P115 target decision into a concrete two-box launch plan:

```text
TeacherTextbook artifact -> Radjax student burn -> StudentArtifact
```

P117 uses the existing `current_qrwkv` / RWKV-family student under a matched
vocab contract. The runtime default remains reference JAX and Pallas remains
opt-in only. The output must be an HF-shaped Level 0/1 StudentArtifact, not a
full HF-native model.

P116 adds validation gates for the two boxes. It does not run the real burn,
train, download a teacher, add a remote teacher service, add tokenizer
remapping, implement KVM/FLA/Vocab C, or change model/runtime/math behavior.

## Selected Payload

P117 payload:

```text
student architecture: current_qrwkv
student family: RWKV-family
vocab: matched from TeacherTextbook
runtime: reference
pallas: disabled by default / opt-in only
artifact expectation: HF-shaped Level 0/1 StudentArtifact
```

## Selected Teacher

P117 should use a tiny real HF causal-LM teacher. Preferred exact model:

```text
sshleifer/tiny-gpt2
```

Reason:

- small enough for a conservative first burn
- exercises the generic HF causal-LM teacher path
- has normal tokenizer/vocab behavior
- avoids Qwen-specific first-burn coupling
- acts as a lab specimen, not a sacred architecture target

P117 should not start with tiny-random-only targets, `distilgpt2`, full `gpt2`,
or Qwen unless a later reviewed plan changes the teacher choice.

## Two-Box Workflow

TeacherTextbook input:

- existing `TeacherTargetStore`
- vocab contract
- teacher manifest
- emission config
- validation report

StudentArtifact output:

- student config
- vocab contract copied/matched from TeacherTextbook
- runtime metadata
- checkpoint or params
- burn, eval, export, and validation reports

TeacherTextbook generation follows teacher hardware requirements and can happen
on a separate CPU/GPU VM. TPU is not required to generate the textbook. The
student burn consumes the already validated textbook.

## Revised Burn Config

Expected P117 config shape:

```yaml
teacher_textbook:
  path: artifacts/p117_teacher_textbook
  required: true
  validate_before_burn: true

teacher:
  model_id: sshleifer/tiny-gpt2
  local_files_only: false
  allow_downloads: true
  tokenizer_source: teacher_model

student:
  architecture_id: current_qrwkv
  student_family: rwkv_family
  vocab_contract_source: teacher_textbook
  runtime: reference
  pallas_enabled: false

artifact_expectations:
  teacher_textbook_contract_version: 0
  student_artifact_contract_version: 0
  hf_shaped_level: "0/1"

burn:
  mode: real only when manually confirmed
  max_steps: conservative
  batch_size: conservative
  sequence_length: 128
  checkpoint_every_steps: conservative
  eval_every_steps: conservative
  output_dir: artifacts/p117_student_burn

stop_conditions:
  - vocab mismatch
  - teacher_textbook validation failure
  - student artifact validation failure
  - NaN/inf loss
  - checkpoint/resume failure
  - eval/report failure
  - runtime/preflight fail
  - budget exceeded
```

## P117 Launch Sequence

```bash
# 1. Build/readiness report
python scripts/run_big_burn_readiness_report.py \
  --output artifacts/p111_big_burn_readiness/readiness_report.json

# 2. Build teacher textbook
python scripts/build_teacher_textbook.py \
  --teacher-model sshleifer/tiny-gpt2 \
  --dataset artifacts/p116/input_texts.jsonl \
  --output artifacts/p117_teacher_textbook \
  --sequence-length 128 \
  --batch-size 8 \
  --max-examples 1000 \
  --logits-dtype float32

# 3. Validate teacher textbook
python scripts/validate_teacher_textbook.py \
  --path artifacts/p117_teacher_textbook \
  --write-report

# 4. Optional runtime environment preflight
python scripts/run_runtime_environment_preflight.py \
  --output artifacts/p109_runtime_environment/runtime_environment_report.json

# 5. Dry run burn harness
python scripts/run_first_serious_burn.py \
  --config artifacts/p116_revised_burn_config.json \
  --output artifacts/p117_dry_run \
  --mode dry_run

# 6. Manual real burn
python scripts/run_first_serious_burn.py \
  --config artifacts/p116_revised_burn_config.json \
  --output artifacts/p117_student_burn \
  --mode real \
  --confirm-serious-burn

# 7. Validate student artifact
python scripts/validate_student_artifact.py \
  --path artifacts/p117_student_burn/student_artifact \
  --teacher-textbook artifacts/p117_teacher_textbook \
  --write-report
```

The P112 harness does not currently accept a `--teacher-textbook` flag. P116
therefore treats TeacherTextbook location as a config requirement and keeps the
validator as the explicit pre-burn gate. P117 should not run until the config
and launch wrapper route the validated textbook path into the burn.

## Stop Conditions

Stop before or during P117 if:

- TeacherTextbook validation fails
- student artifact validation fails
- teacher/student vocab contracts differ
- selected architecture is not `current_qrwkv`
- runtime defaults to Pallas
- tokenizer remapping or cross-vocab projection is required
- loss becomes NaN or inf
- checkpoint/resume fails
- eval/report generation fails
- runtime preflight fails
- budget is exceeded
- build command would require Qwen or a remote teacher service

## Success Claim Boundary

A P117 pass may claim only that the selected matched-vocab
`current_qrwkv` / RWKV-family burn completed under the reviewed config and
produced a validated StudentArtifact.

P117 does not prove Qwen parity, KVM, FLA, Vocab C, full HF-native integration,
generation quality, benchmark quality, production Pallas readiness, or model
quality.

## Deferred Future Tracks

Deferred:

- Qwen-scale teacher burn
- KVM backend work
- FLA/hybrid backend work
- Vocab C / cross-vocab support
- tokenizer remapping
- full HF-native student wrapper
- generation and official lm_eval
- pjit/sharding and production Pallas

## Claims Not Made

P116 does not claim:

- real burn started or passed
- training added
- remote teacher service added
- tokenizer remapping added
- full HF wrapper added
- KVM implemented
- FLA dependency added
- Vocab C implemented
- WKV math changed
- StudentRuntime or StudentBackend behavior changed
- Pallas promoted
