# Generation Smoke

P14 adds a minimal inference-facing smoke path for logits-capable QRWKV-XLA
student checkpoints. It checks that a checkpoint can be loaded, a student can
emit logits, a short greedy autoregressive loop can produce token ids, and
local JSONL/JSON artifacts can be written.

This is not a quality benchmark. The default students and fake targets are tiny
CPU smoke fixtures, so decoded output may be nonsensical.

## Smoke Tokenizer

The default `SmokeTokenizer` is dependency-free. It maps UTF-8 bytes to token
ids `byte + 1` and reserves token `0` for EOS/PAD. It requires
`vocab_size >= 257`. Decode maps ids `1..256` back to bytes and renders unknown
ids as `<tok_N>`.

No Hugging Face tokenizer, transformers package, torch package, chat template,
or network access is required.

## Greedy Generation

`qrwkv_xla.generation.greedy_generate` runs a small greedy loop. At each step it
recomputes the full current sequence, reads logits from the last position, picks
`argmax`, appends the next token id, and stops early on EOS.

There is no sampling, beam search, cached recurrent state, or optimized long
context inference in P14.

## Checkpoint Requirement

Generation requires a checkpoint whose manifest has
`student_config.emit_logits=true`. Hidden-only checkpoints fail with a clear
error because they do not have an LM head for logits.

## CLI

Create a tiny logits checkpoint:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_logits.yaml
python scripts/run_distill_stage.py --config configs/distill_stage0_logits_stub.yaml --max-steps 1 --checkpoint-out checkpoints/generation_smoke --checkpoint-overwrite
```

Generate from one prompt:

```bash
python scripts/generate_from_checkpoint.py --checkpoint checkpoints/generation_smoke --prompt "Hello QRWKV" --max-new-tokens 8 --output-dir eval_outputs/generation_smoke
```

Run the prompt-corpus smoke eval:

```bash
python scripts/eval_generation_smoke.py --checkpoint checkpoints/generation_smoke --config configs/generation_smoke.yaml
```

Artifacts are local files only:

```text
eval_outputs/<run>/
  generations.jsonl
  summary.json
```

`eval_outputs/` is gitignored.
