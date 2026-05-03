# Prompt Corpora

Prompt corpora make teacher-export inputs reproducible. Instead of only passing
inline prompt lists, QRWKV-XLA can now point at a checked-in or local JSONL
corpus, filter by split and tags, hash the exact selected records, and record
that prompt-source metadata in the exported target manifest.

## JSONL schema

Each line is one prompt record:

```json
{"id":"smoke_000001","text":"Explain recurrence in one sentence.","split":"train","tags":["smoke","recurrence"],"metadata":{"source":"handwritten"}}
```

Required fields:

- `id`
- `text`

Optional fields:

- `split`
- `tags`
- `metadata`

Canonical split labels:

- `train`
- `validation`
- `test`
- `unspecified`

## Manifest schema

A corpus manifest records stable provenance for a corpus file:

```json
{
  "schema_version": "0.1",
  "corpus_id": "smoke_prompts",
  "description": "Tiny checked-in smoke prompt corpus.",
  "created_by": "qrwkv_xla.prompting.corpus",
  "prompt_count": 8,
  "splits": {"train": 6, "validation": 2},
  "tags": ["code", "generation", "greeting", "jax", "reasoning", "recurrence", "smoke", "teacher", "xla"],
  "sha256": "...",
  "source_path": "corpora/smoke_prompts.jsonl",
  "notes": []
}
```

## Hashing behavior

Corpus hashes are SHA-256 over canonical normalized prompt records in record
order. That means:

- changing prompt text changes the hash
- changing record order changes the hash
- metadata is included in the hash
- filesystem timestamps and other external file metadata are not

Order affecting the hash is intentional for now and is documented in
`docs/DECISIONS.md`.

## CLIs

Inspect a corpus:

```bash
python scripts/inspect_prompt_corpus.py corpora/smoke_prompts.jsonl
python scripts/inspect_prompt_corpus.py corpora/smoke_prompts.jsonl --split train --tag smoke --limit 4 --json
```

Create a manifest:

```bash
python scripts/create_prompt_manifest.py corpora/smoke_prompts.jsonl \
  --out corpora/smoke_prompts.manifest.json \
  --description "Tiny checked-in smoke prompt corpus." \
  --overwrite
```

Assign deterministic splits:

```bash
python scripts/split_prompt_corpus.py corpora/smoke_prompts.jsonl \
  --out /tmp/smoke_split.jsonl \
  --validation-fraction 0.2 \
  --test-fraction 0.0 \
  --seed 123 \
  --overwrite
```

## Teacher export integration

Corpus-backed teacher export configs use:

```yaml
targets:
  prompt_texts: []
  prompt_corpus: corpora/smoke_prompts.jsonl
  prompt_split: train
  prompt_tags: ["smoke"]
  prompt_limit: 4
```

`prompt_corpus` is mutually exclusive with `prompt_texts` and `prompt_file`.
When corpus prompts are used, target manifests record prompt-source metadata such
as corpus ID, corpus hash, selected prompt IDs, split, tags, and limit.

## Smoke corpus

The repo includes a tiny built-in corpus:

- `corpora/smoke_prompts.jsonl`
- `corpora/smoke_prompts.manifest.json`

## Tokenization

Prompt corpora store text, not token IDs. Stage 3 LM loading routes selected
records through the tokenizer registry. The default `smoke` backend is
dependency-free; optional `hf`/`qwen` backends require `.[teacher-hf]` and are
not used by default tests.

## Limitations

- no tokenizer-aware packing yet
- no dataset streaming
- no remote dataset dependency
- no external dataset libraries
- prompt order affects the hash
## Evaluation Prompt Corpora

P15 evaluation prompts use the same JSONL prompt corpus format and manifest
hashing as teacher export prompt corpora. The fixed regression corpus lives at
`corpora/eval_regression_prompts.jsonl` and is intended for repeatable
generation snapshots, not quality benchmarking.
