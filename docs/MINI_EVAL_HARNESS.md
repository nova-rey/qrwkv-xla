# Mini Eval Harness

P110 adds a tiny evaluation/reporting smoke for stored target artifacts.

## Purpose

P110 proves the eval/reporting wire exists for tiny artifacts. It is a
scoreboard smoke, not a benchmark suite, quality proof, training loop, or
production evaluation system.

## Inputs

The harness consumes a `TeacherTargetStore`. It can also create a built-in
tiny two-shard synthetic target store for CLI smoke runs.

Primary APIs:

- `run_mini_eval_harness()`: `src/qrwkv_xla/eval/mini_eval.py`
- `write_mini_eval_report()`: `src/qrwkv_xla/eval/mini_eval.py`
- `scripts/run_mini_eval_harness.py`

## Compatibility Requirement

Direct-logit evaluation is gated through the P99 compatibility validator before
metrics are computed. Student backends are selected through the P101 registry,
so vocab contract, architecture, and runtime remain separate choices.

## Metrics

The report includes:

- mean MSE logits loss
- finite loss status
- shard count
- examples evaluated
- tokens evaluated
- logit elements evaluated
- compatibility status and reason
- selected architecture/runtime
- toy top-1 argmax agreement

Top-1 agreement is a tiny artifact sanity metric only; it is not a model
quality claim.

## Report Schema

Reports are JSON-serializable and include:

```json
{
  "phase": "P110",
  "status": "pass",
  "scope": "mini_eval_harness_smoke",
  "mean_mse_loss": 0.123,
  "loss_finite": true,
  "shard_count": 2,
  "examples_evaluated": 4,
  "tokens_evaluated": 12,
  "elements_evaluated": 96,
  "compatibility_status": "compatible",
  "architecture_id": "tiny_debug",
  "runtime": "reference",
  "target_type": "full_logits",
  "vocab_size": 8,
  "top1_agreement": 1.0
}
```

## Command

```bash
python scripts/run_mini_eval_harness.py \
  --output artifacts/p110_mini_eval/mini_eval_report.json \
  --architecture-id tiny_debug
```

If `--target-store` is omitted, the script creates a tiny built-in target store
under the output directory.

## What P110 Proves

P110 proves tiny target artifacts can be evaluated through existing target
loaders, multi-shard iteration, compatibility checks, and registry-selected
student backends.

## What P110 Does Not Prove

P110 does not prove model quality, benchmark performance, training readiness,
lm_eval integration, production eval readiness, Qwen support, tokenizer
remapping, or burn readiness.

## Future Phases

P111 is the next planned checkpoint: Big Burn Readiness Report.
