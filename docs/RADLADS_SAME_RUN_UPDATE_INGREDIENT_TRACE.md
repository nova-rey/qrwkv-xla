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

## P68 live same-run trace contract
P68 adds `src/qrwkv_xla/parity/radlads_live_same_run_trace.py` and
`scripts/run_live_same_run_update_trace.py` for strict-live trace generation
under `artifacts/p68_live_same_run_trace/`. P68 rows carry
`same_run_group_id`, deterministic `fixture_id`, deterministic `parameter_id`,
fixture context, source stage, capture kind, shape, dtype, inline array, and
summary fields.

P68 does not use old P66/P67 rows as the source of truth. If a true live
RADLADS hook is unavailable, P68 emits unavailable rows for critical stages,
marks `same_run_valid: false`, and recommends only targeted live RADLADS trace
hook completion. It does not recommend math fixes, promotion, Pallas work, or a
residual-impact gate while strict-live validity or decay/log_w preconditions
fail.

## P73 balance-state lane mapping
P73 extends the live same-run report with explicit balance-state lane
classification. RADLADS and QRWKV off are the `balance_state_terms` lane, where
`k_k` and `k_a` are active. QRWKV experimental is the
`direct_balance_state` lane, where `radlads_balance_state=True` bypasses
`k_k`/`k_a` and computes `kk` directly from `k`.

Direct-lane `k_k` and `k_a` rows are `not_applicable` with
`not_active_in_lane` reasons, not missing hooks. The reports now separate
`first_overall_non_applicable_stage` from
`first_comparable_differing_stage` and write
`P73_BALANCE_STATE_LANE_MAP.md` plus `balance_state_lane_map.json` under
`artifacts/p68_live_same_run_trace/`.

P73 is reporting/source mapping only. It does not change recurrence math,
tolerances, dtype policy, Pallas/kernel code, RADLADS upstream/vendor code, or
default experimental `balance_state` behavior.

## P74 direct-balance-state lane
P74 adds the missing RADLADS `direct_balance_state` lane by running the same
local reference capture path with `radlads_balance_state=True`. The existing
RADLADS `balance_state_terms` lane is preserved, and trace keys include
`balance_state_lane` so same-context terms/direct RADLADS rows do not collide.

Reports now compare `radlads_terms` vs `qrwkv_off_terms` separately from
`radlads_direct` vs `qrwkv_experimental_direct`. The direct lane still marks
`k_k` and `k_a` as `not_applicable`; no copied or synthetic direct-lane rows
are used.

P74 writes `P74_DIRECT_BALANCE_LANE_REPORT.md`,
`direct_balance_lane_comparison.json`, and `P74_FIX_NOTE.md` under
`artifacts/p68_live_same_run_trace/`. It does not change recurrence math,
balance-prep math, tolerances, dtype policy, Pallas/kernel code, RADLADS
upstream/vendor code, or default experimental `balance_state` behavior.
