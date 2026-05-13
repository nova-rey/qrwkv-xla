# RADLADS Clean Parameter Loader

P54 repairs RADLADS-side clean payload loading and output export after P53 showed the shared deterministic-finite payload could run in QRWKV-XLA but RADLADS needed a local compatibility layer.

P54 does not modify the RADLADS repository.
P54 does not vendor RADLADS into QRWKV-XLA.
P54 treats RADLADS as an external reference source loaded from `--radlads-repo`.
P54 does not implement Pallas.
P54 does not prove training throughput.
P54 does not prove model quality.

## What it does
- Loads the clean deterministic payload from P52/P53.
- Resolves RADLADS parameter names against the live tiny model.
- Applies safe local shims for layer-wise payload surfaces.
- Records exact, transposed, reshaped, defaulted, excluded, and unsupported rows.
- Exports RADLADS and QRWKV tiny outputs for comparison.

## What it does not do
- No vendoring.
- No upstream RADLADS edits.
- No fake parity.
- No Pallas work.

## Current outcomes
- 4 gate surfaces needed deterministic adaptation.
- 34 payload-only surfaces are excluded as not needed for the tiny case.
- The live tiny model can now load and run the clean payload locally.

## Usage
- Audit: `python scripts/audit_radlads_clean_payload_loader.py`
- Export: `python scripts/export_radlads_clean_payload_outputs.py`
- Compare: `python scripts/compare_radlads_qrwkv_head_to_head.py --radlads-outputs ... --qrwkv-outputs ...`
