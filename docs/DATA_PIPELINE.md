# Data Pipeline

P23 adds a local tokenized-corpus artifact for Stage 3 CE training.

## What it is

Raw prompt JSONL can now be tokenized once and reused across runs:

```text
prompt JSONL -> tokenizer backend -> manifest.json + shards/*.npz -> Stage 3 batches
```

The default offline path still uses `SmokeTokenizer`. Optional HF/Qwen tokenizers remain lazy and opt-in.

## Artifact layout

```text
<output_dir>/
  manifest.json
  shards/
    shard-00000.npz
    shard-00001.npz
    ...
```

`manifest.json` records:

- `format` / `format_version`
- `created_at`
- source provenance and source hash
- tokenizer metadata
- packing policy (`concat_pack_v1`)
- shard paths plus per-shard hash/count metadata
- total shard / sequence / token counts

Each shard stores static Stage 3 arrays:

- `input_ids: int32[num_sequences, sequence_length]`
- `labels: int32[num_sequences, sequence_length]`
- `attention_mask: int32[num_sequences, sequence_length]`
- `loss_mask: int32[num_sequences, sequence_length]`

## Stage 3 convention

P23 keeps the existing Stage 3 CE convention:

```text
tokens:         [batch, sequence_length + 1]
input_ids:      tokens[:, :-1]
labels:         tokens[:, 1:]
attention_mask: input_ids != pad_token_id
loss_mask:      labels != pad_token_id
```

Packing is deterministic concat-pack with EOS appending and a default stride equal to `sequence_length`.

## Generate a tokenized corpus

```bash
python scripts/tokenize_corpus.py \
  --input corpora/smoke_prompts.jsonl \
  --out artifacts/tokenized_corpora/smoke_stage3 \
  --tokenizer-backend smoke \
  --sequence-length 16 \
  --shard-size-tokens 4096 \
  --overwrite
```

The script prints a compact summary and fails loudly on malformed or empty input.

## Train Stage 3 from a tokenized corpus

```bash
python scripts/run_lm_stage.py \
  --config configs/lm_stage3_tokenized_smoke.yaml \
  --tokenized-corpus artifacts/tokenized_corpora/smoke_stage3
```

Raw prompt JSONL remains supported with `configs/lm_stage3_smoke.yaml`.

## What P23 does not do

P23 is intentionally local and correctness-first. It does **not** add:

- large-scale DCLM/binidx ingestion
- Grain/tf.data streaming
- TPU-first data plumbing
- teacher-target export from tokenized corpora
- sharding / pjit model partitioning
- HF export or lm_eval

Those stay for later phases.
