# HF Teacher Specimen Smoke

P104 adds an optional tiny HF causal-LM specimen smoke for the generic
`HFTeacherBackend`.

## Purpose

P104 uses a tiny HF causal-LM specimen only to prove the generic HF backend can
emit a real `full_logits` `TeacherTargetStore` artifact with tokenizer and vocab
metadata. The specimen is not a special architecture target.

## Specimen Model Policy

The default specimen model id is:

```text
hf-internal-testing/tiny-random-gpt2
```

The model id is configurable. Tiny GPT-2-style models are lab specimens, not
student architectures and not target-store schema branches. Qwen remains a
future model-id case, not a special path in P104.

## Local/Cache-Only Behavior

The smoke defaults to `local_files_only=True`. Missing `transformers` or missing
local model cache is reported as `unavailable`, not as a baseline CI failure.
Downloads are opt-in only with `--allow-downloads`.

## Optional Live Smoke

```bash
PYTHONPATH=src python scripts/run_hf_teacher_specimen_smoke.py \
  --model-id hf-internal-testing/tiny-random-gpt2 \
  --output artifacts/p104_hf_teacher_specimen/hf_teacher_specimen_report.json \
  --target-store artifacts/p104_hf_teacher_specimen/target_store \
  --sequence-length 8 \
  --prompt "hello" \
  --local-files-only
```

The optional live pytest is disabled by default:

```bash
QRWKV_RUN_OPTIONAL_HF_SPECIMEN=1 \
  PYTHONPATH=src python -m pytest tests/test_hf_teacher_specimen_smoke_optional.py
```

## Report Fields

The JSON report includes phase, status, model id, local-files-only/download
flags, target-store path, validation flags, tokenizer id/hash, vocab size,
sequence length, example count, target type, logits shape, claims not made, and
any unavailable/failure reason.

## What P104 Proves

- the generic `HFTeacherBackend` can be exercised against a tiny cache-local HF
  causal-LM specimen when optional dependencies and local files are available
- a `full_logits` `TeacherTargetStore` artifact can be emitted and validated
- emitted metadata can reconstruct a `VocabContract`
- metadata shapes and logits shapes are checked for consistency
- baseline tests require no `transformers`, internet, downloaded model, GPU,
  TPU, Qwen, student consumption, or training

## What P104 Does Not Prove

P104 does not consume with a student or train. It does not add tokenizer
remapping, cross-vocab adapters, GPT-2-specific architecture assumptions,
Qwen-specific code, recurrence math changes, runtime changes, or model-quality
claims.

## Future Phases

P105 adds the second teacher-specimen swap smoke. See
`docs/HF_TEACHER_SPECIMEN_SWAP_SMOKE.md`.
