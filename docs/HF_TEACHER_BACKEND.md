# HF Teacher Backend

P98 adds a generic Hugging Face-style teacher backend emission smoke. It is a
teacher-side target emission path only: no student consumption, training,
tokenizer remapping, or Qwen-specific behavior is added.

## Purpose

`HFTeacherBackend` proves that a non-synthetic teacher interface can declare a
tokenizer/vocab contract, run a tiny causal-LM-style forward pass, and emit a
canonical `TeacherTargetStore` artifact with `full_logits` targets.

## Generic Backend Behavior

The backend lives in `src/qrwkv_xla/teachers/hf.py`. It accepts a configurable
`model_id`, optional `revision`, and `local_files_only=True` by default. It
does not import `transformers` at package import time. Loading is lazy and only
happens when `load()`, `vocab_contract()`, `build_metadata()`, or
`emit_targets()` needs the tokenizer/model.

P98 is generic HF emission, not Qwen-specific. Qwen-family support should later
be a `model_id`/configuration case, not a special architectural path.

## Optional Dependency And Cache Policy

Baseline tests use fake tokenizer/model objects and do not require
`transformers`, PyTorch, internet, GPU, TPU, or downloaded models.

When live HF objects are needed, unavailable dependencies or missing local cache
raise `HFTeacherUnavailable` with a clear message. `local_files_only=True`
keeps the default path cache-only; downloads must be enabled explicitly by
future tooling.

## VocabContract Extraction

`HFTeacherBackend.vocab_contract()` extracts:

- tokenizer id from `tokenizer.name_or_path` or `model_id`
- vocab size from `tokenizer.vocab_size` or `len(tokenizer)`
- model id from the backend `model_id`
- model family as `hf`
- best-effort special token ids

P98 does not compute a strong tokenizer hash. When no cheap stable hash is
available, `tokenizer_hash` remains `None`.

## TeacherTargetStore Emission

The backend encodes tiny prompts with max-length padding/truncation, emits
`input_ids` and `attention_mask`, runs a no-gradient model forward when PyTorch
is present, and stores float32 logits shaped `[N,T,V]` as `full_logits`.

The output layout remains:

```text
metadata.json
shards/shard-00000.npz
```

## What P98 Proves

P98 proves that generic HF-style teacher emission can produce valid
`TeacherTargetStore` artifacts with metadata that round-trips through the P97
`VocabContract` extraction path.

## What P98 Does Not Prove

P98 does not prove Qwen-specific support, training readiness, student
consumption of HF artifacts, tokenizer remapping, production distillation,
model quality, large-scale performance, GPU/TPU execution, or runtime changes.

## Future Phases

Likely next work is hardening teacher/student compatibility checks for direct
logit eligibility, then adding carefully gated live/cache-only teacher smokes
without making any single model family architectural.
