# TopK Tail TeacherTextbook v0

## Purpose

P119 adds an opt-in compressed teacher target type:

```text
topk_with_tail_v0
```

This is a print-side artifact format for future sparse target consumption. It
does not replace the dense P117 mini textbook path, does not add trainer loss,
does not run training or a real burn, and does not change Pallas or WKV runtime
semantics.

The branch for P119/P120 work is:

```text
p119-p120-cascaded-targets
```

## Shard Arrays

Each compressed shard contains:

```text
input_ids        [N,T]   int
attention_mask   [N,T]   int or bool
top_token_ids    [N,T,K] int32
top_log_probs    [N,T,K] float16 or float32
top_mass         [N,T]   float32
tail_mass        [N,T]   float32
teacher_entropy  [N,T]   float32
```

Dense `logits` are not persisted for `topk_with_tail_v0` shards.

The default `top_k` contract is 256. Small fake-mode tests may pass a smaller
`--top-k` with a matching fake vocab size.

## Builder

`scripts/build_teacher_textbook.py` supports:

```bash
python scripts/build_teacher_textbook.py \
  --teacher-mode fake \
  --output artifacts/p119_topk_tail_smoke \
  --sequence-length 16 \
  --batch-size 2 \
  --max-examples 4 \
  --vocab-size 512 \
  --target-type topk_with_tail_v0 \
  --top-k 256 \
  --top-log-probs-dtype float16
```

Dense remains the default:

```bash
python scripts/build_teacher_textbook.py \
  --teacher-mode fake \
  --output artifacts/p117_teacher_textbook \
  --target-type dense_logits
```

HF mode computes dense logits batch by batch, compresses them in memory, and
writes only the compressed arrays when `--target-type topk_with_tail_v0` is
selected.

## Math

Compression uses stable `log_softmax` over the full dense teacher distribution.
Top-k entries are sorted by descending log probability. `top_mass` is the sum of
the exponentiated retained log probabilities. `tail_mass` is clipped
`1 - top_mass`. `teacher_entropy` is computed from the full distribution before
dense logits are discarded.

## Metadata

`metadata.json` records:

```json
{
  "target_type": "topk_with_tail_v0",
  "target_params": {
    "top_k": "256",
    "top_log_probs_dtype": "float16",
    "top_token_ids_dtype": "int32",
    "top_mass_dtype": "float32",
    "tail_mass_dtype": "float32",
    "teacher_entropy_dtype": "float32"
  }
}
```

`teacher_manifest.json` and `emission_config.json` also record `target_type`,
`top_k`, and top-log-probability dtype fields.

## Validation

The TeacherTargetStore validator checks required arrays, shapes, top-k width,
finite values, ID range, duplicate IDs per position, descending log-prob order,
mass range and approximate mass sum, and finite non-negative entropy.

The TeacherTextbook validation report exposes compressed-target fields:

```text
target_type
top_k
compressed_target_ok
mass_ok
sort_ok
duplicate_ok
```

## Boundaries

P119 explicitly does not add training, trainer loss consumption, a real burn,
Rosetta/Vocab C/tokenizer remapping, Qwen/Gemma scale-up, full HF model behavior
changes, Pallas promotion, or WKV math changes.

Recommended next phase: P120 Sparse Target Loss Consumption on the same
`p119-p120-cascaded-targets` branch. Do not merge into the main P117 dense burn
path until P119/P120 are reviewed together.
