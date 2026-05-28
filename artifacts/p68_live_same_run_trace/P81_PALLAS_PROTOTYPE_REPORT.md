# P81 Pallas Prototype Report

## Runtime Selector
- default runtime: `reference`
- allowed runtimes: `['reference', 'pallas']`
- reference default preserved: `True`
- CLI/config path: `--wkv-runtime reference|pallas`, `wkv_runtime`

## Pallas Path
- pallas requested: `True`
- pallas available: `True`
- pallas effective runtime: `pallas`
- fallback used: `False`
- fallback reason: `None`

## Prototype Probe
- prototype_status: `pass`
- prototype_scope: `minimal_pallas_wkv_execution_probe`
- kernel_parity_claimed: `True`
- parity compatibility note: `P81 compatibility report retained; current parity status is reported in P83/P84 artifacts.`

## Previous Gate Preservation
- P78/P79/P80 readiness: `not_rerun_by_p81_probe`
- fixture alias lineage: `preserved_from_p80_reference_path`
- covered fixture family: `preserved for reference path`

## Decision
- recommended_next_phase: `P86 fused/scan Pallas WKV kernel scaffold`
