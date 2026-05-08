# Streaming Data Pipeline Dry-Run

P44 adds a larger/local streaming data pipeline dry-run for QRWKV-XLA.
It proves manifest-backed sharded token data, streaming/chunked iteration,
deterministic cursor resume, mask/boundary validation, and tiny trainer
consumption on CPU.

P44 does **not** prove full-scale dataset throughput or real training quality.

## What P44 proves

- deterministic synthetic corpus generation without network access
- manifest-driven loading from multiple shard files
- chunked batch iteration with `input_ids`, `labels`, `attention_mask`, and `label_mask`
- deterministic order by default, plus optional shuffle with a fixed seed
- resume cursor save/load with exact integer post-resume replay checks
- padding / token-waste accounting
- tiny CPU/local trainer ingestion using the same batch contract as the LM runner

## What P44 does not prove

- full-scale throughput
- real training quality
- cloud or distributed data ingestion
- `pjit` or sharded data pipelines
- Pallas kernels
- WandB integration
- Qwen0.5B-scale target generation

## Manifest schema

`artifacts/data/p44_streaming_dry_run/manifest.json` contains:

- `schema_version: "0.1"`
- `phase: "P44"`
- `created_at_utc`
- `tokenizer`
- `corpus`
  - `num_documents`
  - `num_sequences`
  - `num_tokens`
  - `sequence_length`
  - `shard_tokens`
  - `padded_tokens`
  - `boundary_policy`
- `source`
- `shards[]`
  - `path`
  - `num_sequences`
  - `num_tokens`
  - `dtype`
  - `sha256`
  - `first_sequence_index`
  - `provenance`

## Shard format

Each shard is a compressed NPZ under `shards/` with:

- `input_ids`
- `labels`
- `attention_mask`
- `label_mask`

The iterator consumes prepacked tokenized-corpus rows and does **not** stitch
new sequences across shard boundaries. Token waste from packing remainder and
partial-batch padding is reported.

## Cursor semantics

`resume_cursor.json` stores the next logical sequence position plus the shuffle
mode/seed used for the iterator. Resuming with mismatched shuffle configuration
fails fast.

## Commands

```bash
.venv/bin/python scripts/build_streaming_data_dry_run.py --out artifacts/data/p44_streaming_dry_run --num-documents 1024 --total-tokens 131072 --shard-tokens 32768 --seq-len 128 --overwrite

.venv/bin/python scripts/run_streaming_data_dry_run.py --manifest artifacts/data/p44_streaming_dry_run/manifest.json --batch-size 4 --seq-len 128 --num-batches 32 --out artifacts/data/p44_streaming_dry_run --overwrite

.venv/bin/python scripts/run_streaming_trainer_dry_run.py --manifest artifacts/data/p44_streaming_dry_run/manifest.json --out artifacts/data/p44_streaming_dry_run --overwrite
```

## Reports

The dataset builder writes:

- `manifest.json`
- `P44_DATASET_SUMMARY.md`
- `shards/*.npz`

The streaming dry-run writes:

- `streaming_dry_run_report.json`
- `P44_STREAMING_DRY_RUN_REPORT.md`
- `resume_cursor.json`

The trainer dry-run writes:

- `trainer_dry_run_report.json`
- `P44_TRAINER_DRY_RUN_REPORT.md`

## Caveats

Trainer consumption currently proves CPU/local ingestion using the same batch
contract as the LM runner. It does not yet add a new top-level LM-stage config
source for streaming manifests. That is an intentional scope limit for P44.
