# RADLADS Balance-State Three-Way Parity

## Why P66 exists

P66 checks whether QRWKV-XLA experimental balance-state mode moves the update
boundary closer to RADLADS, instead of only changing the recurrence path.

## P65 result summary

P65 proved the balance-state hook wires in cleanly, stays finite, and keeps
default/off behavior unchanged. It did not prove RADLADS parity improvement.

## River/Cooper framing

The hook wires cleanly, but RADLADS parity was not proven by P65 alone.

## Three-way comparison methodology

P66 compares:

- RADLADS P64 composite-hook rows
- QRWKV off-mode P65 artifacts
- QRWKV experimental balance-state P65 artifacts

The comparison is local, CPU-only, and provenance-locked to checked-in
artifacts.

## Off vs experimental vs RADLADS results

The generated report records a comparison-first view over the following stages:

- `state_before`
- `decay_value`
- `decayed_state`
- `k_for_update`
- `v_for_update`
- `update_outer_product`
- `balance_state_term`
- `composite_update_term`
- `final_update_term`
- `state_after`

## Update-boundary finding

Experimental mode improves the update boundary on the checked-in fixtures, but
the comparison still stops short of a same-run RADLADS proof.

## Whether experimental mode helps

Yes, experimentally closer on the checked-in tiny/local fixtures.

## What remains unresolved

- k/v boundary surfaces remain incomplete in the direct RADLADS trace family.
- The result is still a diagnostic comparison, not a promotion gate.

## Recommendation for P67

`P67 promote/harden balance-state compatibility path`

## Kernel readiness

Kernel-ready remains `no` until the remaining boundary gap is either removed or
explicitly shown to be non-blocking.
