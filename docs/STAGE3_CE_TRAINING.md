# Stage 3 CE Training

Stage 3 is student-only next-token cross-entropy fine-tuning. It reads prompt
corpus text, tokenizes with the dependency-free `SmokeTokenizer`, runs a
logits-capable student, and minimizes masked next-token CE. It does not read
teacher hidden states, teacher logits, or target bundles.

## Smoke Run

```bash
python scripts/run_lm_stage.py --config configs/lm_stage3_smoke.yaml
```

The checked-in smoke config uses `corpora/smoke_prompts.jsonl`, `tiny_student`,
`emit_logits=true`, short static batches, SGD, and optional gradient clipping.
It is CPU-safe and network-free.

## Data Shape

Each selected prompt becomes one example. Text is byte-tokenized with
`SmokeTokenizer`, EOS/PAD token `0` is appended, sequences are truncated or
padded to `sequence_length + 1`, and batches are shifted:

```text
tokens:         [batch, sequence_length + 1]
input_ids:      tokens[:, :-1]
labels:         tokens[:, 1:]
attention_mask: input_ids != 0
label_mask:     labels != 0
```

The final batch is padded to full `batch_size` so the train step keeps static
shapes.

## Config

Stage 3 config lives under top-level `lm`:

```yaml
lm:
  data:
    prompt_corpus: corpora/smoke_prompts.jsonl
    prompt_split: train
    sequence_length: 16
    batch_size: 2
    tokenizer: smoke
  student:
    architecture: tiny_student
    vocab_size: 512
    hidden_size: 8
    num_layers: 2
    emit_logits: true
  training:
    stage: 3
    max_steps: 2
```

`emit_logits` must be true and `vocab_size` must be at least `257` for the
smoke tokenizer. Optimizer, LR schedule, gradient clipping, checkpoint, and run
tracking sections reuse the distillation dataclasses and CLI conventions.

## Checkpoints And Metrics

Stage 3 saves the same JSON + NPZ checkpoint format as distillation. The
checkpoint manifest records `next_token_ce` in `loss_config` and prompt-corpus
metadata in the existing target metadata slot. Resuming uses additive
`--max-steps`, restores optimizer state when present, and logs `ce_loss`,
learning rate, and gradient metrics to tracking JSONL when enabled.

## Multi-device pmap smoke

Phase 21 adds a skip-safe data-parallel smoke path for Stage 3:

```bash
python scripts/pmap_lm_smoke.py --config configs/lm_stage3_pmap_smoke.yaml
```

This path reuses the same prompt-corpus tokenizer/batching flow, shards the
batch over the device axis, averages gradients with `pmean`, and saves optional
checkpoints from unreplicated first-device state.
