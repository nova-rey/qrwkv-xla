# P115 First Burn Target Decision

## Executive Summary

P115 selects the first serious burn target before revising the launch plan.
The selected target is the existing `current_qrwkv` / RWKV-family student path
under a matched vocab contract, reference runtime by default, Pallas opt-in
only, and an HF-shaped Level 0/1 artifact expectation.

This is an architecture-similarity decision, not a training phase. It does not
start the real burn, implement KVM, add FLA, add Vocab C, change WKV math,
change runtime semantics, or add full Hugging Face integration.

## Decision

P117 should target:

- student family: `current_qrwkv` / RWKV-family
- vocab: matched teacher/student vocab contract required
- runtime: reference runtime by default
- Pallas: opt-in only
- artifact: HF-shaped Level 0/1 artifact and loader boundary expected

P116 should turn this decision into a revised burn config and launch plan.

## Candidate Targets

The evaluated candidates were:

- `current_qrwkv` / RWKV-family
- `tiny_debug`
- KVM-inspired student
- FLA/hybrid-compatible student
- Vocab C / cross-vocab target

## Architecture Similarity Analysis

The first serious burn should minimize architecture uncertainty. The current
QRWKV/RWKV-family path is already represented by the student backend registry,
current config selection, WKV runtime split, tiny offline consumption, overfit
rehearsals, checkpoint/resume/export rehearsal, mini eval, readiness report,
and burn dry-run harness. That makes it the only candidate with enough local
scaffold to support an honest first compute burn.

`tiny_debug` is valuable as a deterministic socket/control backend, but it is
not a serious architecture target. KVM and FLA-style designs are promising
research directions, but neither is implemented as a QRWKV-XLA student backend.
Using either as P117 target would turn the first burn into an architecture
implementation phase. Vocab C / cross-vocab work is a separate vocabulary
research track and would obscure whether failures come from recurrent student
training or vocab translation.

## Vocab Contract Requirement

P117 remains direct-logit and matched-vocab. The teacher target store and the
student artifact must agree on vocab size and vocab identity through the
existing `VocabContract` gate.

P115 does not add tokenizer remapping, hidden projection heads, vocab A to vocab
B mapping, or Vocab C support. Any cross-vocab target belongs to a later arc
with its own explicit success criteria.

## HF-Shaped Artifact Boundary

P117 should expect a Level 0/1 HF-shaped student artifact boundary:

- stable local config and metadata files
- stable vocab contract file
- stable weights/checkpoint export location
- loader semantics sufficient for local reload and inspection
- explicit claims not made

Full HF-native integration is not required for P117. P115 does not add a
`transformers` dependency, `PreTrainedModel` subclass, generation API, or
official lm_eval integration.

## Runtime Decision

The reference runtime remains the default for P117. Pallas remains opt-in only.
Pallas smoke evidence is useful runway evidence, but P117 should not depend on
Pallas promotion or production Pallas readiness.

## P117 Success Claim Boundary

A successful P117 can claim only that the selected matched-vocab
`current_qrwkv` / RWKV-family path completed the configured burn and produced
the configured evidence.

It must not claim:

- KVM viability
- FLA parity
- Vocab C or cross-vocab success
- full HF-native model compatibility
- generation quality
- benchmark quality
- production Pallas readiness
- universal recurrent architecture success

## Risks and Stop Conditions

Main risks:

- matched vocab mismatch between teacher targets and student artifact
- burn config drift from the selected `current_qrwkv` family
- artifact output not reloadable under the Level 0/1 boundary
- runtime accidentally switching to Pallas by default
- treating tiny_debug smoke success as serious architecture evidence
- interpreting a first burn pass as KVM, FLA, or Vocab C evidence

Stop conditions:

- compatibility validator rejects the teacher/student vocab contract
- selected student architecture is not `current_qrwkv` / RWKV-family
- real burn launch would require tokenizer remapping or cross-vocab projection
- config requires Pallas as default
- artifact contract cannot identify config, vocab, weights, and claims
- launch plan hard-codes a teacher model ID before P116 freezes it

## P116 Requirements

P116 should produce the revised burn config and launch plan for the selected
target. It should:

- name `current_qrwkv` / RWKV-family as the student target
- require matched vocab compatibility before launch
- preserve reference runtime as default and Pallas as opt-in
- specify Level 0/1 HF-shaped artifact outputs
- define dry-run and real-run launch commands
- keep real burn execution manually confirmed
- freeze the concrete teacher/model specimen only in P116, not P115
- define the P117 evidence bundle and failure reporting shape

## Future Directions Deferred

KVM remains a future StudentBackend/research candidate. FLA remains a design
reference and possible later backend direction, not an immediate dependency.
Vocab C / cross-vocab support remains deferred to a separate vocabulary arc.
Full HF-native integration, generation, official lm_eval, broader training
loops, pjit/sharding, and production Pallas work remain outside P115.

## Decision Table

| Candidate | Decision | Rationale | Impact |
| --- | --- | --- | --- |
| `current_qrwkv` / RWKV-family | select | only serious candidate with existing QRWKV-XLA backend, runtime, target, checkpoint, eval, readiness, and burn harness scaffold | P116 should plan P117 around this family |
| `tiny_debug` | retain as control only | useful deterministic socket backend, not a real serious student architecture | keep for smoke/control validation |
| KVM-inspired | defer | no implemented backend and would turn P117 into architecture research | future StudentBackend track |
| FLA/hybrid-compatible | defer as design reference | useful research context, but no dependency/backend should be introduced before first burn | future architecture/design track |
| Vocab C / cross-vocab | reject for P117, defer | would mix vocab translation risk with first burn training risk | future vocabulary arc only |

## Claims Not Made

P115 does not claim that:

- the real burn has started or passed
- KVM is implemented or validated
- FLA is implemented, vendored, or required
- Vocab C or tokenizer remapping exists
- full Hugging Face student integration exists
- Pallas is production-ready or default
- the selected RWKV-family path is a universal best architecture
- model quality, generation quality, or benchmark quality has been proven
