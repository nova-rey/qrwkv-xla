# RADLADS Same-Run Update Ingredient Trace

## why P67 exists
P67 replaces mixed-lineage update-boundary comparisons with a same-run,
same-fixture, same-parameter ingredient trace for RADLADS, QRWKV off, and
QRWKV experimental balance-state mode.

## why P66 was directional but not definitive
P66 was useful for direction, but it mixed artifact lineages, so it could not
establish courtroom-clean ingredient parity.

## same-run methodology
- keep fixture manifest and parameter payload fixed
- trace RADLADS, QRWKV off, and QRWKV experimental in one run family
- reject mismatched same_run_group_id / fixture_id / parameter_id

## stage dependency order
input_to_attention → pre_attention_norm → raw_k → raw_v → v_after_v_first_mix
→ a → b → k_k → k_a → kk → k_for_update → v_for_update → ab → vk
→ update_outer_product → prev_state → decay_value → decayed_state
→ balance_composite_term → state_after_from_formula → state_after_live
→ state_after_exported

## decay/log_w precondition
P67 checks decay/log_w first. If that fails, downstream ingredient claims are
invalid.

## ingredient comparison results
The comparison report identifies the first differing ingredient by dependency
order and records whether experimental balance-state mode is closer to
RADLADS.

## first differing ingredient
Use `FIRST_DIFFERING_INGREDIENT.md` for the source-backed gap summary and
`P67_DECISION.md` for the next phase recommendation.

## whether experimental balance-state helps
P67 only reports whether experimental mode is closer on the captured
same-run trace; it does not promote the mode by default.

## P68 recommendation
P67 must end in exactly one bounded recommendation: targeted source fix,
compatibility hardening, residual gate, or Pallas-with-caveat.

## kernel readiness
Kernel-ready is `yes` only when same-run validity, decay/log_w parity, and the
critical update ingredients are all acceptable.

## known caveats
P67 does not implement Pallas.
P67 does not prove training throughput.
P67 does not prove model quality.
P67 does not promote experimental balance_state mode by default.
P67 only establishes a same-run ingredient-level RADLADS-vs-QRWKV comparison
on tiny/local fixtures.
