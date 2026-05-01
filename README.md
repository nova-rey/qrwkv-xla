# QRWKV-XLA

QRWKV-XLA is a JAX/XLA-first recurrent conversion pipeline inspired by RADLADS.

It aims to distill Qwen-family transformer teachers into RWKV7-style recurrent
students using TPU-friendly training infrastructure.

## Current Status

Phase 18: global gradient norm clipping on top of learning-rate scheduling,
lightweight SGD/Adam/AdamW optimizer support, evaluation harness, generation
smoke, prompt corpus provenance, logits KL distillation, checkpoint/resume, XLA
discipline, optional HF exporter, and local run tracking.

The project can define, write, read, validate, inspect, and test fake teacher
target bundles on CPU through a reusable exporter interface. It also has an
optional Hugging Face / PyTorch exporter backend behind the `teacher-hf` extra;
that path is not part of default CI/local validation. Qwen policy labels are
resolved only from local YAML and never by automatic internet lookup. It also
has a JAX
student runtime path and an XLA-friendly `rwkv7_reference` recurrent reference
implementation for CPU/JIT/gradient coverage and smoke training. The current
distillation runtime loads stage configs, composes weighted hidden-state losses,
plumbs optional logits KL, and runs a CPU-only stage smoke over target bundles.
It also exposes CPU-safe JAX runtime inspection and static-shape JIT smoke
helpers used by local, CI, and TPU launcher checks.
The reference core is not a final optimized RWKV7 kernel.

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

`scripts/run_distill_stage.py` is the primary entrypoint for staged
distillation. It currently supports hidden-state distillation against fake
teacher bundles with `tiny_student` or `rwkv7_reference` students, and can use
SGD, Adam, or AdamW without adding an optimizer dependency. Global gradient
norm clipping is configured under `distillation.gradients` or with
`--max-grad-norm`; see `docs/GRADIENT_CLIPPING.md`.

`scripts/tpu_distill_smoke.py` runs on the available JAX backend by default and only requires TPU when `--require-tpu` is passed.

Generated bundles are written under `artifacts/`, which is gitignored.

Regression evaluation snapshots are documented in
`docs/EVALUATION_HARNESS.md`:

```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/eval_smoke --config configs/eval_regression_smoke.yaml
python scripts/compare_eval_snapshots.py --baseline eval_outputs/eval_a --candidate eval_outputs/eval_b
```

These commands are sanity/regression checks only, not quality benchmarks.

See `docs/CI.md` for the exact CI command sequence and local mirror.

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

This project uses `nova-rey/radlads-TPU-adapter` as a conceptual and
architectural reference, not as code to directly port.

The reference RADLADS lineage includes RAD-RWKV6/RAD-RWKV7 components,
Hugging Face conversion scripts, staged configs, Lightning trainer flows,
`lm_eval` support, and inference support. QRWKV-XLA is being rebuilt around XLA
and TPU constraints from day one instead of carrying over GPU-shaped internals.
