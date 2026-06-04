# P113 Research Intake

P113 begins the post-P112 research alignment arc. It pauses real burn execution
long enough to inspect new RADLADS2 / FLA / hfattnconv / KVM context and revise
what the first serious burn should prove.

P113 does not run the burn, train, vendor external repos, add FLA, implement
KVM, implement Vocab C, implement 3Tier, or change model/runtime behavior.

## Source Inventory

Available uploaded sources:

- `distillation-fla-main.zip`: available; 50 non-directory files detected by
  `scripts/inventory_research_sources.py`; includes README, configs,
  checkpoint config templates, `eval/harness.py`, Liger/Lolcats model code, and
  training code.
- `hfattnconv-main.zip`: available; 32 non-directory files detected; includes
  README, Qwen/Llama/Gemma configs, distillation configs, RWKV6/RWKV7 attention
  replacement code, CUDA kernels, train script, and lm-eval script.
- `2605.09877.pdf`: available; PDF metadata identifies it as "Key-Value Means:
  Transformers with Expandable Block-Recurrent Compressed Memory", arXiv
  `2605.09877v3`, by Daniel Goldstein and Eugene Cheah.

Unavailable in this upload bundle:

- `vocab c braindump.docx`: not present. P113 indexes Vocab C from the prompt as
  a future cross-vocab track only.
- `3Tier_QRWKV_XLA_Memory_Scaffold_Spec.md`: not present. P113 indexes 3Tier
  from the prompt as future memory-scaffold work only.

No external source tree was extracted into or vendored under `qrwkv-xla`.

## Executive Summary

P112 remains useful as a launchpad and guardrail, but the real burn should not
be launched immediately. The new material reframes the next work: first clarify
HF-compatible student interfaces, then decide a burn target based on
architecture similarity, then revise the burn config before spending compute.

The current QRWKV-XLA architecture already has several aligned boundaries:
teacher/vocab/student/runtime separation, compatibility gates, target stores,
student backend registry, readiness reporting, and a guarded burn harness. The
mis-prioritization risk is treating the first burn as a universal
transformer-to-recurrent proof before deciding whether the student interface and
architecture family are close enough to the intended teacher.

Recommended arc:

- P114: HF-Compatible Student Interface Reassessment
- P115: Architecture Similarity / First Burn Target Decision
- P116: Revised Burn Config + Launch Plan
- P117: First Serious Compute Burn
- P118: Burn Result Analysis / Next-Arc Decision

## distillation-fla Findings

The `distillation-fla` repo presents Liger: linearizing large language models to
gated recurrent structures. Its README describes a workflow that starts from a
pretrained base model directory, edits the original model config into a Liger
config, runs linearization training, and evaluates through lm-evaluation-harness.

Observed patterns:

- Purpose: convert/linearize transformer LLMs into gated recurrent/linear
  attention variants rather than simply train a new recurrent student from
  scratch.
- Teacher/student assumptions: it starts from an existing HF checkpoint and
  preserves model-family config shape through subclasses such as
  `LigerQwen2GLAConfig`, `LigerMistralGLAConfig`, and related Liger/Lolcats
  configs.
- Config representation: YAML experiment configs point at model directories,
  HF tokenizer/model settings, optimizer settings, and trainable Q/K/V LoRA
  switches; checkpoint config JSON files encode model variants.
- Evaluation: `eval/harness.py` registers an HF/lm-eval compatible model wrapper
  and expects lm-evaluation-harness.
- Dependency shape: tightly tied to PyTorch, CUDA, FLA/Triton kernels,
  Transformers, PEFT, datasets, and lm-evaluation-harness.
- Useful concept: architecture-family-aware conversion matters. The repo keeps
  model-family-specific HF config surfaces while swapping attention mechanisms.
- Avoid copying literally: do not add FLA/Triton/PyTorch runtime dependency to
  QRWKV-XLA, do not vendor its source, and do not make lm_eval a baseline
  requirement.

Implication for QRWKV-XLA: borrow the interface lesson, not the dependency
stack. HF-compatible student interfaces probably need earlier attention because
the first burn target should be shaped by what can be loaded, exported,
configured, and evaluated like an HF causal LM.

## hfattnconv Findings

The `hfattnconv` repo is a compact HF attention-conversion/distillation
prototype. It contains configs for Qwen2.5, Llama 3.2, and Gemma 2, plus
distillation configs and attention replacement code.

Observed patterns:

- Families/configs: Qwen2.5 0.5B/7B/32B variants, Llama-3.2-3B, and
  Gemma-2-2B appear in configs.
- Conversion method: model-specific attention class dictionaries are patched so
  HF models can run with replacement RWKV-style attention modules.
- Distillation path: the train script uses HF `Trainer`, streamed datasets,
  teacher logits, KL-style distillation, and attention-distillation stages.
- Export/eval: `convert_to_safetensors.py`, `generate.py`, and
  `run_lm_eval.py` are present; eval uses lm-evaluation-harness with an HF
  model wrapper.
- Model-family specificity: the Qwen-focused config class and attention patch
  paths show that exact HF architecture surfaces matter. The generic idea is
  portable, but the actual wiring is per-family.
- Historical/reference-only pieces: CUDA WKV kernels, direct attention class
  monkey-patching, and PyTorch/HF training code should not be imported into
  QRWKV-XLA.

Implication for QRWKV-XLA: hfattnconv reinforces that first-burn claims must be
architecture-family-specific. The first real burn should not claim universal
recursive distillation. It should pick a matched-vocab, well-defined teacher
family and a student interface whose HF compatibility story is explicit.

## KVM Findings

The KVM paper available in the upload is `2605.09877.pdf`, titled "Key-Value
Means: Transformers with Expandable Block-Recurrent Compressed Memory".

P113 treats KVM as a strategic design input, not an implementation target.
Based on the prompt's paper-specific claims and the PDF metadata, the relevant
takeaways are:

- Core mechanism: KVM keeps a compressed key-value memory that can be fixed-size
  or grow over time.
- Bridge role: it connects transformer KV-cache behavior to linear
  RNN-style memory, which is directly relevant to transformer-to-recurrent
  distillation.
- Implementation signal: KVM can be implemented with standard operations and
  does not inherently require custom kernels.
- Hybrid route: KVM can appear in hybrid form with LRNN layers.
- Architecture-similarity signal: because KVM keeps Q/K/V and block/sliding
  window attention behavior relatively close to transformers, it may be a more
  natural distillation target than a recurrent architecture whose interface is
  too far from the teacher.

P113 decision: KVM should become a future StudentBackend/research candidate,
not part of the immediate first burn. It should influence P115 target selection
criteria, especially architecture similarity and memory-interface compatibility.

## Vocab C Future Track

The Vocab C source file was unavailable in this upload bundle. From the P113
prompt, Vocab C is clearly related to the warning that separate-vocab
distillation is its own research topic.

P113 stance:

- First burn uses a matched vocab contract only.
- Vocab C is deferred to a separate cross-vocab research arc.
- It should require explicit design phases for tokenizer contracts, target
  representation, adapter/projection semantics, compatibility validation,
  evaluation, and failure modes.
- It should not be folded into P117 because that would confound architecture
  similarity, vocab mapping, and training behavior in one burn.

## 3Tier Future Track

The 3Tier memory scaffold source file was unavailable in this upload bundle.
From the P113 prompt, 3Tier is future memory-scaffold work.

P113 stance:

- 3Tier should not affect the first serious distillation burn.
- It should remain outside the current burn path until a recurrent student
  exists and has evidence from matched-vocab distillation.
- It may become relevant after P118 when deciding whether to add broader memory
  systems, compression layers, or multi-tier inference scaffolds.

## Comparison to QRWKV-XLA / Radjax

Already aligned:

- Teacher picks the vocab contract.
- Student is born with that contract.
- Student architecture is selected separately.
- Runtime is selected separately.
- HF teacher paths are optional/cache-local by default.
- StudentBackend registry and second backend smoke exist.
- Target stores, checkpoint rehearsal, mini eval, readiness report, and guarded
  launch harness exist.

Potentially mis-prioritized:

- HF-compatible student interface may need to move earlier than full burn
  execution.
- Architecture similarity may need to become a formal gate before spending
  serious compute.
- FLA/hybrid-distillation ideas should be considered as design references, not
  imported dependencies.
- The first burn should be framed as a matched-vocab, architecture-family-
  specific experiment, not a universal proof.

## Decision Table

| Topic | Decision | Rationale | Phase Impact |
| --- | --- | --- | --- |
| HF-compatible student interface | promote | distillation-fla and hfattnconv both preserve HF-like config/load/eval surfaces | P114 |
| FLA dependency | avoid | FLA/Triton/PyTorch stack conflicts with QRWKV-XLA baseline constraints | P113 records no dependency |
| FLA conceptual alignment | borrow | gated linear attention and HF-compatible model config patterns are useful references | P114/P115 |
| hfattnconv historical reference | keep | shows model-family-specific conversion and attention replacement lessons | P115 |
| KVM future backend | defer | promising memory architecture but not ready for immediate burn implementation | future StudentBackend arc |
| Vocab C | defer | separate-vocab distillation is a separate research problem | post-P118 cross-vocab arc |
| 3Tier | defer | future memory scaffold, not first-burn target | post-P118 memory arc |
| P112 real burn timing | pause | launchpad is valid but target/config should be revised first | P113-P116 before P117 |
| P117 first serious burn target | adapt | use matched vocab and architecture-family-specific framing | P115/P116/P117 |

## First Burn Implications

- P112 launchpad remains valid.
- Do not run the real burn immediately after P112.
- Use P113-P116 to revise the target/config.
- First burn should use matched vocab contract.
- First burn should be framed as architecture-family-specific, not universal
  recursive distillation proof.
- P114 should reassess whether the student interface should expose HF-compatible
  config/load/eval surfaces before the burn.
- P115 should choose the first-burn architecture target using architecture
  similarity, implementation risk, and evaluation interpretability.
- P116 should update the P112 burn config and launch plan after those decisions.

## Recommended P114-P118 Roadmap

P114 - HF-Compatible Student Interface Reassessment:
evaluate whether the student side needs explicit HF-compatible config, load,
save, generation, and eval surfaces before real burn execution.

P115 - Architecture Similarity / First Burn Target Decision:
compare current QRWKV/RWKV-style target, FLA-like gated linear attention, and
KVM-like memory candidates as first-burn targets.

P116 - Revised Burn Config + Launch Plan:
update the P112 dry-run/real-run config, readiness assumptions, target store
requirements, evaluation framing, and manual command sequence.

P117 - First Serious Compute Burn:
run the serious burn only after P114-P116 resolve interface and target
selection.

P118 - Burn Result Analysis / Next-Arc Decision:
analyze the burn result and decide whether to continue QRWKV, pivot toward
FLA/KVM-style backends, or open a separate cross-vocab/memory arc.

## Claims Not Made

P113 does not claim:

- the real burn succeeded
- model quality is proven
- FLA is integrated
- KVM is implemented
- Vocab C is implemented
- 3Tier is implemented
- separate-vocab distillation is supported
- Pallas is production/default
- QRWKV-XLA should be renamed
- external source code has been vendored
