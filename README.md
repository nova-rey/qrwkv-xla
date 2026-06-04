# QRWKV-XLA

QRWKV-XLA is a JAX/XLA-first recurrent conversion pipeline inspired by RADLADS.

It aims to distill Qwen-family transformer teachers into RWKV7-style recurrent
students using TPU-friendly training infrastructure.

## Current Status

Current phase: P116.3, TPU VM Bootstrap + P117 Preflight Scripts. The Pallas runway is
closed after a recorded real TPU v5 lite smoke pass for the tiny opt-in Pallas
WKV path, and the project is now the validated core of a Radjax-shaped modular
recurrent distillation platform.

Current emphasis is teacher backend modularity, vocab contracts, target stores,
the student backend registry, runtime separation, and the burn-readiness arc.
The project can turn tiny text examples into deterministic batches, emit
fake-HF-style sharded teacher target artifacts, validate and iterate multi-shard
target stores, consume shards through the offline target path, checkpoint and
resume a tiny student, export/reload that checkpoint through the HF/safetensors
interchange path, inspect runtime environment hygiene for JAX/TPU visibility
and transparent hugepage readiness, evaluate tiny target artifacts through a
compatibility-gated registry-selected student path, reconstruct vocab contracts
from metadata, select student backends by architecture id, keep runtime
selection separate, aggregate burn-readiness evidence into a pass/warn/fail
report, provide a guarded first serious burn dry-run/launch harness, reassess
the HF-compatible student interface, select the first serious burn target,
define the TeacherTextbook input / StudentArtifact output contracts for P117,
build a validated fake-mode TeacherTextbook artifact from tiny text input, and
build a guarded real-HF TeacherTextbook artifact for tiny causal-LM teachers.
P116.3 adds human-run TPU VM bootstrap and P117 preflight scripts for the
released mini textbook handoff.
Current arc: post-P112 research alignment. The P112 harness exists, but actual
real burn execution is deferred until the P116 launch plan is reviewed and a
validated TeacherTextbook is available. Baseline tests remain
CPU-safe and do not require Hugging Face downloads, internet, Qwen, GPU, or TPU.

P117 is the first serious burn using a validated TeacherTextbook input and a
validated HF-shaped Level 0/1 StudentArtifact output.

Runtime policy is unchanged: `reference` remains the default WKV runtime and
`pallas` remains opt-in. The Pallas TPU smoke result does not claim production
Pallas readiness, training readiness, throughput, full-model quality, or Pallas
default readiness. The reference core is not a final optimized RWKV7 kernel.

## Design Principles

- Full-system architecture from day one
- Tiny configs, not disposable toy systems
- JAX/XLA-first student training
- PyTorch/Hugging Face teacher extraction optional, never required by default
- CPU local development
- TPU smoke tests when available
- No CUDA/Triton dependency in student training path
- Simple, inspectable artifact formats first

## Quick Usage

```bash
python -m pip install -e ".[dev]"
python scripts/xla_inspect.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/inspect_targets.py artifacts/teacher_targets/fake_export
python scripts/tpu_distill_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2
python scripts/validate_local.py
```

The default exporter path uses the deterministic fake exporter. The optional HF
backend is installed with `python -m pip install -e ".[dev,teacher-hf]"` and is
documented in `docs/HF_TEACHER_EXPORT.md`.

Prompt corpora are documented in `docs/PROMPT_CORPORA.md`. The checked-in smoke
corpus lives at `corpora/smoke_prompts.jsonl`.

The tiny dataset pipeline smoke is documented in
`docs/TINY_DATASET_PIPELINE.md`.

The checkpoint/resume/export rehearsal is documented in
`docs/CHECKPOINT_RESUME_EXPORT_REHEARSAL.md`.

The runtime environment preflight is documented in
`docs/RUNTIME_ENVIRONMENT_PREFLIGHT.md`.

The mini eval harness smoke is documented in `docs/MINI_EVAL_HARNESS.md`.

The big burn readiness report is documented in
`docs/BIG_BURN_READINESS_REPORT.md`.

The first serious compute burn harness is documented in
`docs/FIRST_SERIOUS_COMPUTE_BURN.md`.

The post-P112 research intake is documented in
`docs/RADLADS2_FLA_KVM_RESEARCH_INTAKE.md`.

The HF-compatible student interface reassessment is documented in
`docs/HF_COMPATIBLE_STUDENT_INTERFACE_REASSESSMENT.md`.

`scripts/run_distill_stage.py` is the primary entrypoint for staged
distillation. It currently supports hidden-state distillation against fake
teacher bundles with `tiny_student` or `rwkv7_reference` students, and can use
SGD, Adam, or AdamW without adding an optimizer dependency. Global gradient
norm clipping is configured under `distillation.gradients` or with
`--max-grad-norm`; see `docs/GRADIENT_CLIPPING.md`.

`scripts/run_lm_stage.py` runs Stage 3 student-only next-token CE fine-tuning
from a prompt corpus using `SmokeTokenizer`; it does not require teacher target
bundles. See `docs/STAGE3_CE_TRAINING.md`.

`scripts/tpu_distill_smoke.py` runs on the available JAX backend by default and
only requires TPU when `--require-tpu` is passed.

## Student Backend Boundary

P91/P92 add behavior-preserving student abstractions for the current QRWKV path:

- protocol: `src/qrwkv_xla/students/backend.py`
- wrapper: `src/qrwkv_xla/students/current_backend.py`
- runtime protocol: `src/qrwkv_xla/students/student_runtime.py`
- tests: `tests/test_student_backend.py`
- runtime tests: `tests/test_student_runtime.py`

`CurrentQRWKVStudentBackend` delegates to the existing student implementation
for parameter initialization, state initialization, full forward, step forward,
state export/import, and logits access. `StudentRuntime` separates execution
choice from student architecture: default/`reference` selects the reference JAX
runtime and explicit `pallas` selects the opt-in Pallas runtime wrapper.
TeacherBackend and TeacherTargetStore remain future phases.

## Multi-device pmap smoke

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_attention.yaml
python scripts/pmap_distill_smoke.py --config configs/distill_stage1_attention_pmap_smoke.yaml
python scripts/pmap_lm_smoke.py --config configs/lm_stage3_pmap_smoke.yaml
```

Hard multi-device validation:

```bash
python scripts/pmap_distill_smoke.py \
  --config configs/distill_stage1_attention_pmap_smoke.yaml \
  --require-multiple-devices \
  --min-device-count 2
```

Generated bundles are written under `artifacts/`, which is gitignored.

Regression evaluation snapshots are documented in
`docs/EVALUATION_HARNESS.md`:

```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/eval_smoke --config configs/eval_regression_smoke.yaml
python scripts/compare_eval_snapshots.py --baseline eval_outputs/eval_a --candidate eval_outputs/eval_b
```

These commands are sanity/regression checks only, not quality benchmarks.

See `docs/CI.md` for the exact CI command sequence and local mirror.

## Pallas Runtime Status

The opt-in Pallas runway has passed the scoped feasibility gates through P90:

- P87 fixture-family opt-in integration passed.
- P88 added the TPU compile/execution smoke harness.
- P89 fixed the TPU tracing-boundary issue.
- P90 recorded a real TPU v5 lite smoke pass with `max_abs_error=0.0`.

Pallas remains opt-in and is not the default runtime. See
`docs/PALLAS_RUNTIME_ROADMAP.md` and
`artifacts/p90_pallas_runway_closure/P90_PALLAS_RUNWAY_CLOSURE_REPORT.md`.

## Local Development

Use the blessed editable-install workflow:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

## End-to-End Validation

The canonical whole-pipeline validation command is:

```bash
python scripts/validate_pipeline.py
```

The default path is CPU-safe, offline, and requires only `.[dev]`. Optional
checks are explicit:

```bash
python scripts/validate_pipeline.py --include-hf
python scripts/validate_pipeline.py --require-tpu
```

`--include-hf` requires the optional `teacher-hf` dependencies and validates the
tiny HF export path. `--require-tpu` makes TPU availability a hard requirement
for the TPU distillation smoke. Neither flag is used by default CI.

The current default tests are CPU-only. They require JAX CPU through the `dev`
extra, but do not require PyTorch, GPU, TPU, or network access. HF integration
coverage is opt-in through `QRWKV_RUN_HF_INTEGRATION=1`.

Individual checks:

```bash
python -m compileall src scripts tests
python scripts/validate_pipeline.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Optional Hugging Face Teacher Export

Install the optional backend:

```bash
python -m pip install -e ".[dev,teacher-hf]"
```

Run a tiny HF smoke export:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_hf_tiny.yaml --backend hf
python scripts/inspect_targets.py artifacts/teacher_targets/hf_tiny
```

This uses a tiny public model for backend validation. Qwen export is
intentionally not the default smoke path. See `docs/QWEN_EXPORT_POLICY.md` for
offline policy resolution and manual-only Qwen export prep.

Prompt corpus examples:

```bash
python scripts/inspect_prompt_corpus.py corpora/smoke_prompts.jsonl
python scripts/export_teacher_targets.py --config configs/teacher_export_qwen_dryrun_corpus.yaml --dry-run --resolve-qwen-policy --allow-unresolved-policy
```

## TPU Launcher Smoke

Kaggle and Colab TPU sessions should be treated as launch wrappers for the repo
scripts:

```bash
python -m pip install -e ".[dev]"
python scripts/xla_inspect.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/tpu_distill_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2
python scripts/tpu_distill_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2 --require-tpu
```

Without `--require-tpu`, `scripts/tpu_distill_smoke.py` exits successfully on
whatever JAX backend is available. See `docs/TPU_SMOKE_GUIDE.md`.

## Naming

The canonical distillation package is `qrwkv_xla.distill`, and the canonical
stage runner is `scripts/run_distill_stage.py`. The older
`qrwkv_xla.distillation` package and `scripts/run_distillation_stage.py` remain
thin compatibility aliases only.

## Checkpointing

Distillation stages can save and resume local JSON + NPZ checkpoints under
`checkpoints/`:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --max-steps 2 --checkpoint-out checkpoints/stage0 --checkpoint-overwrite
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --max-steps 2 --resume-from checkpoints/stage0 --checkpoint-out checkpoints/stage0_resume --checkpoint-overwrite
```

AdamW can be selected from config or CLI:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --optimizer adamw --learning-rate 0.0003 --weight-decay 0.01
```

Warmup-cosine scheduling is available for AdamW distillation smokes:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_logits.yaml
python scripts/run_distill_stage.py --config configs/distill_stage0_adamw_schedule_stub.yaml --max-steps 2
```

The same schedule can be supplied from CLI:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_logits_stub.yaml --optimizer adamw --learning-rate 0.001 --weight-decay 0.01 --lr-schedule warmup_cosine --warmup-steps 10 --total-steps 1000 --min-learning-rate 0.00001
```

Run tracking is available as an opt-in local file feature. It writes
`run.json`, `metrics.jsonl`, `summary.json`, and, when no checkpoint output is
provided, `runs/<run_id>/checkpoints/final`:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --max-steps 2 --track-run --run-name stage0-smoke
```

See `docs/RUN_TRACKING.md`.

On resume, `--max-steps` means additional steps for that invocation. Optimizer
state is saved and loaded with checkpoints. See `docs/CHECKPOINTING.md` and
`docs/OPTIMIZERS.md`.

Logits KL distillation uses target bundles that include teacher logits and a
student with `emit_logits=true`:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_logits.yaml
python scripts/run_distill_stage.py --config configs/distill_stage0_logits_stub.yaml --max-steps 2
```

Hidden-only checkpoints can continue into logits KL runs. Missing LM head
parameters are initialized during that controlled resume path:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --targets artifacts/teacher_targets/fake_export --max-steps 2 --checkpoint-out checkpoints/hidden_only_for_logits --checkpoint-overwrite
python scripts/run_distill_stage.py --config configs/distill_stage0_logits_stub.yaml --targets artifacts/teacher_targets/fake_export_logits --resume-from checkpoints/hidden_only_for_logits --checkpoint-out checkpoints/hidden_plus_logits --checkpoint-overwrite --max-steps 2
```

See `docs/LOGITS_DISTILLATION.md`.

## Generation Smoke

Logits-capable checkpoints can run a tiny greedy generation smoke with the
dependency-free `SmokeTokenizer`:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_logits.yaml
python scripts/run_distill_stage.py --config configs/distill_stage0_logits_stub.yaml --max-steps 1 --checkpoint-out checkpoints/generation_smoke --checkpoint-overwrite
python scripts/generate_from_checkpoint.py --checkpoint checkpoints/generation_smoke --prompt "Hello QRWKV" --max-new-tokens 8 --output-dir eval_outputs/generation_smoke
```

The output is a wiring sanity check, not model quality evidence. See
`docs/GENERATION_SMOKE.md`.

## Reference

This project uses `recursal/RADLADS-paper` as a conceptual and
architectural reference, not as code to directly port.

The reference RADLADS lineage includes RAD-RWKV6/RAD-RWKV7 components,
Hugging Face conversion scripts, staged configs, Lightning trainer flows,
`lm_eval` support, and inference support. QRWKV-XLA is being rebuilt around XLA
and TPU constraints from day one instead of carrying over GPU-shaped internals.
## Stage 1 attention / mixer smoke

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_attention.yaml
python scripts/run_distill_stage.py --config configs/distill_stage1_attention_stub.yaml --max-steps 2
```

Real HF/Qwen attention capture is manual-only and not part of default CI.
