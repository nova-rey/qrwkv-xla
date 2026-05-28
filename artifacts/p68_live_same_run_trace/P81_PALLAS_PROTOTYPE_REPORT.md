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
- reason parity not claimed: `P81 only establishes the opt-in runtime/probe path; no reference-vs-Pallas numerical comparison ran.`

## Previous Gate Preservation
- P78/P79/P80 readiness: `not_rerun_by_p81_probe`
- fixture alias lineage: `preserved_from_p80_reference_path`
- covered fixture family: `preserved for reference path`

## Decision
- recommended_next_phase: `P84 broader Pallas WKV shape/dtype parity`
