# Hugging Face Teacher Export

## Purpose

P7 adds the first real teacher exporter backend without changing the default
CPU-only validation path. The fake exporter remains the default for CI and local
validation; the HF backend is optional and opt-in.

## Install

```bash
python -m pip install -e ".[dev,teacher-hf]"
```

## Tiny smoke export

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_hf_tiny.yaml --backend hf
python scripts/inspect_targets.py artifacts/teacher_targets/hf_tiny
```

The tiny smoke config uses `sshleifer/tiny-gpt2`. This is for backend
validation only. Qwen policy labels remain documented, but Qwen is not the
manual smoke default yet.

P29 adds an even smaller checked-in real-HF smoke path for the RADLADS reference
backend:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_tiny_hf_smoke.yaml --backend hf
python scripts/inspect_targets.py artifacts/teacher_targets/tiny_hf_smoke
python scripts/run_distill_stage.py --config configs/distill_stage0_radlads_reference_tiny_hf.yaml --checkpoint-out checkpoints/p29_tiny_hf_first --checkpoint-overwrite
python scripts/run_distill_stage.py --config configs/distill_stage0_radlads_reference_tiny_hf.yaml --resume-from checkpoints/p29_tiny_hf_first --checkpoint-out checkpoints/p29_tiny_hf_resume --checkpoint-overwrite
```

That distill config is intentionally hidden-MSE only. The tiny HF export writes
logits so the target key contract is exercised, but `logits_kl` stays disabled
and the `rwkv7_radlads_reference` student does not consume logits in this smoke.
The backend remains a RADLADS-shaped JAX reference backend, not full RADLADS
parity.

For Qwen-specific policy handling, see `docs/QWEN_EXPORT_POLICY.md`.
`Qwen3.latest` is a local policy label only; resolving it never performs a web
lookup.

The optional HF smoke can also be run through the canonical pipeline harness:

```bash
python scripts/validate_pipeline.py --include-hf
```

This flag is intentionally outside default CI and default local validation.

## Config path resolution

When a teacher export YAML is loaded, relative input paths such as
`targets.prompt_file`, `targets.prompt_corpus`, `targets.tokenized_corpus`, and
`runtime.qwen_policy_path` resolve relative to the YAML file. Explicit
`runtime.output_dir` values in config also resolve relative to the YAML file.

CLI path overrides preserve normal command-line behavior. For example, `--out`,
`--prompt-file`, `--prompt-corpus`, `--tokenized-corpus`, and `--qwen-policy`
remain relative to the current working directory unless an absolute path is
provided.

For cache-only validation, pass `--local-files-only` or set
`teacher.local_files_only: true` in the export config. The flag is forwarded to
both tokenizer and model `from_pretrained` calls and recorded in the target
bundle manifest.

## Prompt handling

Prompts are resolved in this order:

1. repeated `--prompt` CLI values
2. `targets.prompt_texts` from config
3. `--prompt-file` / `targets.prompt_file`
4. `targets.prompt_corpus` JSONL selection
5. built-in tiny defaults

Prompt files are read as one prompt per non-empty line.
Prompt corpora are JSONL and are mutually exclusive with inline and prompt-file
sources. See `docs/PROMPT_CORPORA.md`.

## Hidden state export shape

HF models typically return:

- embedding hidden state
- one hidden state tensor per transformer block

QRWKV target bundles store hidden states as:

- `[batch, num_layers, sequence_length, hidden_size]`

The HF exporter excludes the embedding layer when present, then infers the
actual `hidden_size` and `num_layers` from the model outputs and records those
values in `manifest.json`.

## Logits export

Logits are optional and only written when `include_logits` is enabled.
Hidden states and logits are exported as fp32 NumPy arrays to keep the JAX-side
consumer path simple.

## Manifest behavior

HF export writes the standard QRWKV target bundle layout:

- `manifest.json`
- `shards/shard_*.npz`

Each runtime prompt batch becomes one shard. The exporter ignores
`runtime.num_shards` as a sharding plan for HF export.
Manifests record prompt-source metadata but do not store full prompt texts.

Corpus-backed HF example:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_hf_tiny_corpus.yaml --backend hf
python scripts/inspect_targets.py artifacts/teacher_targets/hf_tiny_corpus
```

Tokenized-corpus HF example:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_hf_tiny.yaml --backend hf --tokenized-corpus artifacts/tokenized_corpora/smoke
python scripts/inspect_targets.py artifacts/teacher_targets/hf_tiny
```

When `targets.tokenized_corpus` or `--tokenized-corpus` is used, the HF
exporter skips tokenizer loading, reuses the corpus `input_ids`,
`attention_mask`, and `loss_mask`, and records the tokenized corpus manifest
metadata in `prompt_source`.

## Why Qwen is not default yet

The backend needs to be proven on a tiny public model before we make larger or
policy-driven teacher runs part of the workflow. That keeps CI, local iteration,
and dependency expectations sane.

## Troubleshooting

### Missing torch / transformers

Install the optional extra:

```bash
python -m pip install -e ".[dev,teacher-hf]"
```

The same extra enables the optional LM tokenizer registry `hf` and `qwen`
backends. Default tests still use `SmokeTokenizer` and do not import
transformers.

### Tokenizer has no pad token

The exporter reuses `eos_token` as `pad_token` when possible. If neither exists,
it raises a clear error.

### Network / cache issues

HF integration is optional. Default tests use stubs and do not require network.
For real export runs, make sure the model is reachable or already cached.

### CPU slowness

CPU export is expected to be slower than a GPU-backed inference path. P7 is
about correctness and artifact compatibility, not performance.
## Manual attention target capture

HF attention target export is manual-only. Use `attention_capture` with either
`auto_qwen` or explicit module names. Default CI does not download models or run
real attention capture.
