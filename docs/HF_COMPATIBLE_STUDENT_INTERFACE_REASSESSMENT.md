# P114 HF-Compatible Student Interface Reassessment

P114 defines what "HF-compatible student" should mean for QRWKV-XLA / Radjax
before the first serious burn. It is a reassessment and contract-spec phase,
not a Hugging Face implementation phase.

## Executive Summary

QRWKV-XLA should treat HF compatibility as layered. The near-term target is a
stable local student artifact and load contract, with a small forward adapter as
the next likely implementation step. Full `transformers.PreTrainedModel`
support, generation, and lm_eval integration should not block P117.

P116 should revise the burn config so the first serious burn produces or expects
an HF-style student artifact. P117 should remain a matched-vocab,
compatibility-gated burn, not a universal HF-native eval claim.

## Why HF Compatibility Moved Forward

P113 found that RADLADS2/FLA/hfattnconv-style work benefits from HF-compatible
surfaces because model configs, local loading, generation/eval loops, and
artifact inspection stay familiar while attention or recurrent internals change.
That reduces testing/eval friction and makes failures easier to compare against
teacher-family baselines.

QRWKV-XLA should avoid overbuilding:

- no required `transformers` dependency yet
- no `PreTrainedModel` subclass yet
- no generation API yet
- no lm_eval integration yet
- no tokenizer remapping
- no Qwen- or GPT-2-specific student path

## Current QRWKV-XLA Student Surface

`StudentBackend` currently exposes:

- `init_params(key)`
- `init_state(batch_size, **kwargs)`
- `forward_full(params, input_ids, attention_mask, initial_state, **kwargs)`
- `forward_step(params, input_ids, state, attention_mask, **kwargs)`
- `export_state(state)`
- `import_state(payload, template=None)`
- `logits(output)`

`StudentRuntime` currently exposes the WKV execution boundary:

- runtime name
- normalized WKV runtime
- `step(state, k, v, decay)`
- `sequence(initial_state, k_seq, v_seq, decay_seq)`

Checkpoint/export rehearsal currently saves and reloads a simple checkpoint and
can export a tiny student through the existing HF/safetensors interchange path:

- `config.json`
- `model.safetensors`
- `qrwkv_xla_export.json`
- `weight_map.json`

That export is intentionally an interchange artifact. It explicitly does not
provide a production Hugging Face model class.

Mini eval expects a `TeacherTargetStore`, a registry-selected student backend,
a matching `VocabContract`, and direct logits produced through the backend.

The P112 burn harness expects config, readiness status, dry-run output paths,
checkpoint/eval evidence, and launch commands. It does not require a
Transformers-native student.

## Compatibility Levels

Level 0 - Artifact compatibility:
stable config plus weight/checkpoint/export layout. Predictable local files. No
`transformers` dependency required. QRWKV-XLA partially has this through the
HF/safetensors interchange export.

Level 1 - Loader compatibility:
stable load-from-disk contract for student config plus weights. QRWKV-XLA
partially has this through `load_hf_safetensors_export()`, but it should be
made explicit as a student artifact contract before P117 planning.

Level 2 - Forward compatibility:
adapter/wrapper exposes `input_ids` and optional `attention_mask` to logits.
QRWKV-XLA has the internal backend operation, but not a named HF-style forward
adapter contract.

Level 3 - Generation compatibility:
generate-like loop or generation adapter. This can wait until after the first
burn target is selected.

Level 4 - Eval compatibility:
lm_eval or similar tooling can call the model with minimal custom glue.
QRWKV-XLA has toy lm-eval-style scoring over exported students, but official
lm_eval integration remains deferred.

Level 5 - Full HF integration:
`PretrainedConfig` / `PreTrainedModel` style support. This should be later and
only if P117/P118 evidence says the maintenance cost is worthwhile.

Near-term target: Levels 0-1, with Level 2 scoped next.

Mid-term target: Levels 3-4 after the burn target and artifact contract are
stable.

Later target: Level 5 only if it materially improves evaluation, adoption, or
checkpoint interchange.

## Minimum Viable HF-Style Student Artifact Contract

Proposed contract:

```text
hf_student_artifact/
  student_config.json
  vocab_contract.json
  params.npz or model.safetensors
  export_report.json
```

Minimum fields:

- `schema_version`
- `created_by`
- `architecture_id`
- `runtime`
- `student_config`
- `vocab_contract`
- `weights_format`
- `weights_path`
- `checkpoint_source`
- `forward_contract`
- `claims_not_made`

Forward contract expectations:

- input: integer `input_ids` shaped `[batch, time]`
- optional input: `attention_mask` shaped `[batch, time]`
- output: logits shaped `[batch, time, vocab_size]`
- vocab size equals the artifact `VocabContract`
- no tokenizer remapping
- no hidden cross-vocab projection

Claims not made:

- no full HF-native model class
- no generation support
- no official lm_eval support
- no model quality claim
- no production serving claim

This is a proposed contract for P116/P119-level implementation planning. P114
does not implement it.

## P117 Burn Impact

- P117 should not require full HF integration.
- P116 should include an HF-style artifact/export expectation in revised burn
  config.
- Matched vocab contract remains required.
- Direct-logit eval remains compatibility-gated.
- Full generation/lm_eval/HF native integration can wait.
- Level 0/1 should be prioritized before or during burn planning.
- Level 2 should be the next implementation target if P115/P116 identify high
  risk in evaluating burn outputs without a forward adapter.
- Level 3-5 can wait until after P117/P118 unless P115 finds they are essential
  for the selected burn target.

## What Stays Out of Scope

P114 does not include:

- full Hugging Face wrapper
- required `transformers` dependency
- `PreTrainedModel` subclass
- generation API
- official lm_eval integration
- training
- real burn execution
- tokenizer remapping
- Qwen-specific or GPT-2-specific student code
- FLA dependency
- KVM implementation
- checkpoint format changes
- TeacherTargetStore layout changes
- StudentBackend or StudentRuntime behavior changes

## Decision Table

| Topic | Decision | Rationale | Phase Impact |
| --- | --- | --- | --- |
| Level 0 artifact compatibility | promote | existing HF/safetensors interchange is close but should become a named contract | P116/P119 |
| Level 1 loader compatibility | promote | burn outputs need reproducible local reload semantics | P116/P119 |
| Level 2 forward compatibility | adapt | internal backend can already map input ids to logits, but needs an HF-style adapter contract | P119/P120 |
| Level 3 generation compatibility | defer | useful, but not required to prove matched-vocab burn mechanics | post-P117 |
| Level 4 eval compatibility | defer | toy eval exists; official lm_eval should wait until artifact/forward contracts stabilize | post-P117/P121 |
| Level 5 full HF integration | defer | high maintenance cost and not required for first burn | later only if useful |
| P117 dependency | keep narrow | P117 should require matched vocab and artifact evidence, not full HF-native integration | P115/P116/P117 |
| P116 burn config impact | adapt | revised config should name expected HF-style artifact/export outputs | P116 |
| transformers dependency | avoid | keep baseline dependency-free from Transformers on student side | no dependency added |
| lm_eval integration | defer | avoid benchmark/eval scope creep before first burn | P121 candidate |
| tokenizer remapping | avoid | violates current burn clarity and architecture law | future Vocab C arc only |

## Recommended Future Phases

P115 - Architecture Similarity / First Burn Target Decision:
decide whether the first burn target should remain QRWKV/RWKV-style or pivot
toward an FLA/KVM-like student family.

P116 - Revised Burn Config + Launch Plan:
add expected HF-style artifact/export outputs to the burn config and make the
P117 launch criteria explicit.

P119 - Minimal HF-Style Student Artifact Contract:
implement the Level 0/1 artifact and loader contract if P116 keeps it in scope.

P120 - HF-Style Forward Wrapper Smoke:
add a small Level 2 adapter smoke for `input_ids` / `attention_mask` -> logits.

P121 - Eval Adapter / lm_eval Recon Smoke:
investigate official lm_eval glue only after artifact and forward contracts are
stable.

P122 - Generation Adapter Smoke:
add generate-like behavior only after the selected student family has a stable
forward interface.

## Claims Not Made

P114 does not claim HF integration is complete. It does not add a full HF
wrapper, `transformers` dependency, generation, lm_eval, tokenizer remapping,
training, real burn execution, Qwen-specific code, GPT-2-specific architecture
assumptions, FLA dependency, KVM implementation, model behavior changes,
runtime semantic changes, WKV math changes, or Pallas promotion.
