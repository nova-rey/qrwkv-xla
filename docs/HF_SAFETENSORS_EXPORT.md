# HF Safetensors Export

P41 adds a tiny, local QRWKV-XLA checkpoint export path that writes a Hugging
Face-style safetensors directory:

- `config.json`
- `model.safetensors`
- `qrwkv_xla_export.json`
- `weight_map.json`

This is an interchange smoke path, not a production Hugging Face model class.
It only proves that a tiny QRWKV-XLA checkpoint can be exported, reloaded by
QRWKV-XLA helpers, and produce identical local outputs.

## Export a checkpoint

```bash
python scripts/export_student_hf_safetensors.py \
  --checkpoint checkpoints/generation_smoke \
  --output-dir artifacts/hf_safetensors_export/generation_smoke \
  --overwrite
```

The source checkpoint must be a normal QRWKV-XLA JSON + NPZ checkpoint with
student config fields for `vocab_size`, `hidden_size`, and `num_layers`.

## Run the smoke

```bash
python scripts/run_export_smoke.py \
  --output-dir artifacts/p41_hf_safetensors_export_smoke \
  --overwrite
```

The smoke creates a deterministic tiny logits-capable checkpoint, exports it,
reloads it from `model.safetensors`, compares hidden states and logits on a
fixed CPU batch, and writes:

- `export_smoke_report.json`
- `P41_EXPORT_SMOKE_REPORT.md`

## Dependency

The helper imports `safetensors` only when export or load is requested. If it is
missing, the command fails with:

```text
safetensors is required for HF/safetensors export. Install with `pip install safetensors`.
```

## Scope Limits

P41 does not add a production `transformers` model class, Qwen-scale export,
sharded or `pjit` export, `lm_eval` integration, Pallas/WKV optimized kernels,
or model quality claims.
