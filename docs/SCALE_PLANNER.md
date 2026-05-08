# QRWKV-XLA Scale Planner

P34 adds a conservative planning tool for estimating whether a named
QRWKV-XLA model profile is worth trying on a named hardware budget.

It is a planner, not an oracle.

It helps answer:

- what parameter scale we are talking about,
- which memory components dominate,
- whether a given batch/sequence/training mode looks yes/maybe/no,
- and what smaller batch/sequence plan is safer to try first.

It does **not** prove real hardware fit, TPU readiness, `pjit` viability,
compiler peak-memory behavior, or model quality.

## What it estimates

The planner combines:

- a **model profile**
- a **hardware profile**
- a **training mode**

and produces:

- parameter estimates
- component-level memory estimates
- fit classification (`yes` / `maybe` / `no` / `unknown`)
- warnings and recommendations
- an optional planning-only distill-config skeleton

## What it does not estimate

The planner does not add or prove:

- Pallas kernels
- TPU execution or profiling
- `pjit` or model sharding
- large real training runs
- Qwen-scale export or training
- student HF export
- `lm_eval`
- WandB
- new model math
- full RADLADS parity

## Conservative assumptions

The planner is intentionally cautious.

- CPU/GPU plans reserve **20%** overhead.
- TPU/XLA plans reserve **30%** overhead.
- Activation memory uses a rough `batch * sequence * hidden * layers * dtype * multiplier` estimate.
- Microbatch size, not just global batch size, drives peak activation/recurrent-state estimates.
- Full-vocab logits target memory is modeled explicitly.
- TPU multi-device profiles are treated as **aggregate planning budgets only**.

That last point matters: a TPU plan may look numerically possible while still
being unrunnable today because no sharding path exists yet.

## Built-in profiles

### Model profiles

- `tiny_debug`
- `small_cpu`
- `colab_tpu_smoke`
- `qwen_0_5b_candidate`
- `qwen_1_5b_candidate`
- `qwen_7b_stretch`

Candidate Qwen-scale profiles are documented approximations, not validated
runnable configs.

### Hardware profiles

- `local_cpu_16gb`
- `local_cpu_32gb`
- `local_cpu_64gb`
- `colab_tpu_v2_8`
- `colab_tpu_v3_8`
- `single_l4_24gb`
- `single_a100_40gb`
- `grant_tpu_v5e_8`
- `big_budget_tpu_placeholder`

### Training modes

- `smoke_hidden_sgd`
- `smoke_hidden_logits_sgd`
- `local_hidden_adamw`
- `tpu_hidden_bf16_adamw`
- `tpu_hidden_logits_bf16_adamw`
- `scale_hidden_only_bf16_adamw`
- `scale_sampled_logits_bf16_adamw_placeholder`

## Estimated parameter surfaces

The parameter estimator follows the current `rwkv7_qwen_reference` surface and
reports:

- `embedding_params`
- `per_layer_params`
- `mlp_params`
- `attention_or_time_mix_params`
- `lm_head_params`
- `total_params`

This is a QRWKV-XLA reference-backend planning estimate, not a claim of exact
RADLADS or Hugging Face checkpoint parity.

## Estimated memory components

The memory estimator reports:

- `weights`
- `gradients`
- `optimizer_state`
- `activations_estimate`
- `wkv_recurrent_state`
- `shift_state`
- `teacher_hidden_targets_per_batch`
- `teacher_logits_targets_per_batch`
- `input_ids_and_masks`
- `checkpoint_size_estimate`
- `xla_overhead_reserve`

For Adam/AdamW, detailed optimizer subcomponents are also included.

## Fit classes

The fit classifier uses conservative thresholds against the hardware profile's
usable memory budget:

- `yes`: estimated total <= 60%
- `maybe`: estimated total <= 85%
- `no`: estimated total > 85%
- `unknown`: insufficient usable-memory data

Treat `yes` as “reasonable to try,” not “guaranteed to work.”
Treat `maybe` as “borderline and worth a smaller first attempt.”
Treat `no` as “do not trust vibes here.”

## CLI examples

Basic local plan:

```bash
.venv/bin/python scripts/plan_model_scale.py \
  --model-profile tiny_debug \
  --hardware-profile local_cpu_16gb \
  --training-mode smoke_hidden_sgd \
  --sequence-length 32 \
  --batch-size 1 \
  --output artifacts/p34_scale_plans/tiny_debug_local_cpu.yaml
```

Auto-plan example:

```bash
.venv/bin/python scripts/plan_model_scale.py \
  --model-profile qwen_0_5b_candidate \
  --hardware-profile colab_tpu_v3_8 \
  --training-mode tpu_hidden_bf16_adamw \
  --target-sequence-length 1024 \
  --auto \
  --output artifacts/p34_scale_plans/qwen_0_5b_colab_auto.yaml
```

Optional planning-only distill skeleton:

```bash
.venv/bin/python scripts/plan_model_scale.py \
  --model-profile tiny_debug \
  --hardware-profile local_cpu_16gb \
  --training-mode smoke_hidden_sgd \
  --sequence-length 32 \
  --batch-size 1 \
  --emit-distill-config artifacts/p34_scale_plans/tiny_debug_distill.yaml
```

## Auto mode behavior

Auto mode only adjusts conservative runtime knobs:

- batch size: `8 -> 4 -> 2 -> 1`
- sequence length: `target -> target/2 -> target/4 -> minimum`
- grad accumulation may increase to preserve effective larger batch intent

It does **not** silently change architecture.

If logits memory dominates, the planner recommends disabling logits KL or using
future sampled-logits work instead of pretending full logits are cheap.

## Checked-in examples

Checked-in example plans live under `docs/examples/scale_plans/`:

- `tiny_debug_on_local_cpu.yaml`
- `small_cpu_on_local_cpu_32gb.yaml`
- `qwen_0_5b_on_colab_tpu_v3_hidden_only.yaml`
- `qwen_1_5b_on_grant_tpu_v5e_hidden_only.yaml`
- `qwen_7b_stretch_big_budget_placeholder.yaml`

## P39 tiny TPU smoke profile

P39 adds `p39_tiny_hf_qwen_rope_smoke`, a deliberately tiny
`rwkv7_qwen_reference` profile shaped for `sshleifer/tiny-gpt2` target bundles.
It uses hidden size 2, one head, and one KV head so the RoPE head size is 2 and
valid. The matching hardware profile is `kaggle_tpu_v5e_8`.

The generated P39 plan files live under
`artifacts/p39_planner_tpu_smoke/`. They are still planning artifacts; P39 only
validates execution for the tiny smoke path driven by
`scripts/run_planner_tpu_smoke.py`.
