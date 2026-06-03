# HF Teacher Specimen Swap Smoke

P105 adds a second teacher-specimen swap smoke for the generic
`HFTeacherBackend` specimen path.

## Purpose

P105 proves the generic HF teacher specimen path is model-id/configuration
driven. Two teacher specimens can run through the same report path while
preserving distinct model id, tokenizer id, and vocab-size metadata.

## Teacher Specimen Policy

Use the term teacher specimen. Tiny GPT-2-style models are convenient lab
specimens, not special architecture targets. Qwen is a future model-id or
configuration case, not a special path.

## Two-Specimen Flow

```text
HFTeacherSpecimenConfig A
HFTeacherSpecimenConfig B
  -> run_hf_teacher_specimen_swap_smoke()
  -> run_hf_teacher_specimen_smoke() per specimen
  -> per-specimen TeacherTargetStore/report
  -> aggregate P105 report
```

Baseline tests use fake HF specimens:

```text
fake-hf-specimen-a -> fake-tokenizer-a -> vocab 8
fake-hf-specimen-b -> fake-tokenizer-b -> vocab 16
```

Both use the same generic code path.

## Report Fields

The aggregate report includes phase, status, scope, specimen count, pass /
unavailable / failure counts, model ids, per-specimen P104 reports, and claims
not made.

## Optional Live/Cache-Local Behavior

```bash
PYTHONPATH=src python scripts/run_hf_teacher_specimen_swap_smoke.py \
  --model-id hf-internal-testing/tiny-random-gpt2 \
  --model-id sshleifer/tiny-gpt2 \
  --output artifacts/p105_teacher_specimen_swap/specimen_swap_report.json \
  --target-store-root artifacts/p105_teacher_specimen_swap/stores \
  --sequence-length 8 \
  --prompt "hello" \
  --local-files-only
```

`local_files_only` defaults true. Downloads are opt-in only with
`--allow-downloads`. Missing optional dependencies or local model cache is
reported as unavailable, not as a baseline CI failure.

## What P105 Proves

- the P104 specimen path can represent multiple teacher specimens
- model id, tokenizer id, and vocab-size metadata stay specimen-specific
- the target-store/vocab-contract report shape is generic
- swapping teacher specimens does not require architecture surgery
- baseline tests require no `transformers`, internet, downloaded model, GPU,
  TPU, Qwen, student consumption, or training

## What P105 Does Not Prove

P105 does not make GPT-2, Qwen, or any teacher family special. It does not train,
consume with a student, remap tokenizers, add cross-vocab adapters, change WKV
math, change StudentBackend or StudentRuntime behavior, or change
TeacherTargetStore layout.

## Next Phases

P106 can prove target artifacts can span multiple shards with validation,
iteration, and consumption.
