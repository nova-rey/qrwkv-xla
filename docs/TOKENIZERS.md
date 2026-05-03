# Tokenizers

QRWKV-XLA uses a small tokenizer registry so offline smoke tests and optional
real-tokenizer runs share the same LM data path.

## Backends

- `smoke`: default dependency-free UTF-8 byte tokenizer. It maps bytes to
  token IDs `1..256`, reserves token `0` for EOS/PAD, and defaults to
  `vocab_size=512`.
- `hf`: optional Hugging Face `AutoTokenizer` wrapper. It is lazy-imported and
  requires `python -m pip install -e ".[teacher-hf]"`. It supports
  `revision`, `trust_remote_code`, `local_files_only`, and `use_fast`.
- `qwen`: alias for `hf`.

`SmokeTokenizer` remains importable from `qrwkv_xla.generation` and
`qrwkv_xla.generation.tokenizer`.

## LM Config Forms

String form keeps the CI-safe default:

```yaml
lm:
  data:
    tokenizer: smoke
```

Mapping form records real-tokenizer metadata and avoids hardcoded EOS/PAD/vocab
assumptions:

```yaml
lm:
  data:
    tokenizer:
      backend: qwen
      tokenizer_id: Qwen/Qwen2.5-0.5B
      vocab_size: 151936
      eos_token_id: 151643
      pad_token_id: 151643
      revision: main
      local_files_only: false
      use_fast: true
```

Stage 3 validates `student.vocab_size` against tokenizer metadata when the
metadata is known. Real HF/Qwen tokenizer integration stays opt-in via
`QRWKV_XLA_RUN_HF_TOKENIZER_INTEGRATION=1`; optionally set
`QRWKV_XLA_HF_TOKENIZER_ID` and `QRWKV_XLA_HF_LOCAL_FILES_ONLY=1` for local
cache-only checks.
