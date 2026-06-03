# Big Burn Readiness Report

P111 adds a CPU-friendly readiness gate before the first serious compute burn.
It aggregates evidence from prior smoke phases into a structured pass/warn/fail
JSON report.

P111 is a readiness gate, not the burn. A pass does not guarantee P112 success.
A fail means do not proceed to P112 until blockers are resolved.

## Purpose

The report answers whether the repository has enough green lightweight evidence
to justify attempting P112 with a conservative scoped run plan.

It does not answer whether P112 will succeed, whether model quality is good, or
whether production training is ready.

## Readiness Categories

The P111 report includes these categories:

- `core_correctness_fixtures`
- `pallas_opt_in_runtime`
- `teacher_backend_generic_hf`
- `teacher_specimen_swap`
- `vocab_contract_and_compatibility`
- `target_store_multishard`
- `tiny_dataset_pipeline`
- `checkpoint_resume_export`
- `runtime_environment_preflight`
- `mini_eval_harness`
- `student_backend_registry`
- `second_student_backend`

Most categories use stable module/function availability as evidence. The
runtime environment preflight runs in read-only, no-TPU-required mode. The mini
eval harness runs on built-in tiny target artifacts.

## Status Aggregation

Status aggregation is deterministic:

- any failed check makes the report `fail`
- otherwise, any warning makes the report `warn`
- otherwise, the report is `pass`

Recommended next action follows the aggregate status:

- `pass`: proceed to P112 with a conservative scoped run plan
- `warn`: review warnings before P112 and proceed only if accepted
- `fail`: do not proceed to P112 until blockers are resolved

## Report Schema

The JSON report contains:

- `phase`
- `status`
- `scope`
- `checks`
- `blockers`
- `warnings`
- `recommended_next_action`
- `claims_not_made`

Each check contains:

- `name`
- `status`
- `summary`
- `evidence`
- `blockers`
- `warnings`

## CLI Usage

```bash
python scripts/run_big_burn_readiness_report.py \
  --output artifacts/p111_big_burn_readiness/readiness_report.json
```

By default, `pass` and `warn` exit zero, while `fail` exits nonzero. With
`--strict`, `warn` also exits nonzero.

## What P111 Proves

P111 proves that the repo can produce an inspectable readiness report covering
the major burn-readiness surfaces without requiring TPU, GPU, internet, real HF
downloads, Qwen, large datasets, training, benchmarking, pjit/sharding, or
Pallas promotion.

## What P111 Does Not Prove

P111 does not prove training success, model quality, production training
readiness, large-scale performance, distributed training readiness, tokenizer
remapping support, Qwen-specific support, or Pallas default readiness.

## P112 Relationship

P112 is the first serious compute burn. P111 only decides whether the repo
appears ready to attempt it. Proceed to P112 only if the P111 report passes, or
if warnings are explicitly reviewed and accepted.
