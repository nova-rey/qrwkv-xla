# Evaluation Harness

P15 adds a lightweight regression evaluation harness for logits-capable
checkpoints.

This evaluates whether a checkpoint can generate reproducibly on fixed prompts,
write inspectable artifacts, pass basic sanity checks, and be compared against
another snapshot. It does not measure model quality, reasoning, alignment,
instruction following, or benchmark performance.

## Fixed Prompts

Evaluation prompts use the same JSONL prompt corpus format as teacher export.
The default corpus is `corpora/eval_regression_prompts.jsonl`, with a checked-in
manifest at `corpora/eval_regression_prompts.manifest.json`.

The default config is `configs/eval_regression_smoke.yaml`.

## Running Evaluation

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/eval_smoke \
  --config configs/eval_regression_smoke.yaml \
  --output-dir eval_outputs/eval_smoke
```

By default, the command exits nonzero only for invalid inputs or generation
failures. Sanity warnings are recorded but do not fail CI. Pass `--strict` to
make sanity failures exit nonzero.

## Output Layout

Evaluation snapshots are generated state and stay under gitignored
`eval_outputs/` by default:

```text
eval_outputs/<eval_id>/
  eval.json
  generations.jsonl
  sanity.json
  comparison.json
```

`eval.json` records checkpoint metadata, prompt corpus identity and hash,
selected prompt ids, generation settings, artifact paths, and explicit
limitations. `generations.jsonl` stores one generation record per prompt.
`sanity.json` stores per-prompt sanity checks and summary counts.

## Sanity Checks

The harness checks:

- generated output is non-empty
- generated output is not dominated by one repeated token
- decoded output is not dominated by smoke-tokenizer unknown markers
- records have the expected serializable shape

These checks only flag obvious degenerate output or artifact problems. Passing
them is not a quality claim.

## Comparing Snapshots

```bash
python scripts/compare_eval_snapshots.py \
  --baseline eval_outputs/eval_a \
  --candidate eval_outputs/eval_b \
  --out eval_outputs/comparison.json
```

Comparison is by `prompt_id`. It reports same, changed, and missing prompts.
It does not judge better or worse output.

## Limitations

- smoke tokenizer only
- greedy decoding only
- no chat templates
- no BLEU, ROUGE, BERTScore, semantic graders, benchmarks, or LLM judges
- no required Hugging Face tokenizer, transformers, datasets, pandas, sklearn,
  or PyTorch dependency
