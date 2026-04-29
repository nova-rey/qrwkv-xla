# Artifact Formats

QRWKV-XLA target bundles now use a canonical manifest-plus-shards layout:

```text
artifacts/
  teacher_targets/
    <bundle_id>/
      manifest.json
      shards/
        shard_000000.npz
        shard_000001.npz
```

A bundle directory must contain:
- `manifest.json`
- `shards/`
- at least one `.npz` shard

## Manifest

The manifest is pretty-printed UTF-8 JSON (`indent=2`, `sort_keys=True`) using
the validated `TeacherTargetManifest` contract.

Example manifest:

```json
{
  "schema_version": "0.1",
  "teacher_family": "qwen",
  "teacher_model_id": null,
  "teacher_policy_label": "Qwen3.latest",
  "fallback_policy_label": "Qwen3.0",
  "tokenizer_id": null,
  "sequence_length": 64,
  "hidden_size": 128,
  "num_layers": 2,
  "targets": {
    "input_ids": true,
    "attention_mask": true,
    "hidden_states": true,
    "logits": false,
    "attention_targets": false
  },
  "dtype": "fp32",
  "created_by": "teacher_exporter",
  "notes": []
}
```

## Shards

Each shard is a NumPy `.npz` archive. P1 supports these keys:
- `input_ids`
- `attention_mask`
- `hidden_states`
- `logits` (optional)
- `attention_targets` (optional)

Required keys for a minimal valid shard:
- `input_ids`
- `attention_mask`
- `hidden_states`

### Shape contracts

Required:
- `input_ids`: `[batch, sequence_length]`
- `attention_mask`: `[batch, sequence_length]`
- `hidden_states`: `[batch, num_layers, sequence_length, hidden_size]`

Optional:
- `logits`: `[batch, sequence_length, vocab_size]`
- `attention_targets`: lightly validated in P1; if present, the first two dims
  must match batch and sequence length.

## Fake target workflow

Create a fake bundle:

```bash
PYTHONPATH=src python scripts/create_fake_targets.py --out artifacts/teacher_targets/fake_p1
```

Inspect a fake bundle:

```bash
PYTHONPATH=src python scripts/inspect_targets.py artifacts/teacher_targets/fake_p1
```

## Exporter Integration

Teacher exporters write bundles using the artifact store API. P2 provides a
fake exporter that generates deterministic target bundles without loading a
real model.

```bash
PYTHONPATH=src python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
PYTHONPATH=src python scripts/inspect_targets.py artifacts/teacher_targets/fake_export
```

Distinction:
- `scripts/create_fake_targets.py` = low-level artifact store utility
- `scripts/export_teacher_targets.py` = teacher-export subsystem entrypoint

## Notes

- Real teacher extraction is not implemented yet.
- P1 uses NumPy `.npz` shards because they are simple, CPU-only, inspectable,
  and easy to test.
- Larger-scale formats (for example Zarr, safetensors, or mmap-oriented layouts)
  can be reconsidered later if real exporter scale demands it.
