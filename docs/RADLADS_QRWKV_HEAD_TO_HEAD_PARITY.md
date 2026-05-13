# RADLADS QRWKV Head-to-Head Parity

P53 adds a tiny head-to-head fixture path for comparing live RADLADS outputs
against QRWKV-XLA replay outputs using the P52 `deterministic_finite` parameter
payload path as the source of truth.

The default artifact directory is:

```bash
artifacts/p53_radlads_qrwkv_head_to_head/
```

Generate fixtures:

```bash
.venv/bin/python scripts/generate_radlads_qrwkv_head_to_head_fixtures.py --overwrite
```

Compare generated outputs:

```bash
.venv/bin/python scripts/compare_radlads_qrwkv_head_to_head.py
```

The generator first writes `fixtures_clean/` by calling the existing P52 clean
fixture path with `--init-policy deterministic_finite` and seed `5353`. QRWKV
outputs are then produced through the existing P50/P52 RADLADS replay importer.
RADLADS outputs are attempted only by loading the local RADLADS source runtime
and applying the same clean payload to the live tiny model.

Reports include:

- `manifest.json`
- `head_to_head_comparison_report.json`
- `P53_RESULTS.md`
- `P53_SURFACE_COMPARISON.md`
- `qrwkv_import/parameter_import_report.json`
- `fixtures_clean/manifest.json`
- paired `radlads_outputs/*.npz` and `qrwkv_outputs/*.npz` when live RADLADS is
  available

If live RADLADS cannot import or execute the clean payload, the manifest records
the exact blocker under `radlads.blocker`, marks RADLADS cases unsupported, and
the comparison report returns `overall_status=unsupported`. This is intentional:
P53 must not fabricate RADLADS outputs from QRWKV current behavior.

If `tiny_no_mask` final surfaces diverge beyond the tiny diagnostic threshold,
the comparison writes `tiny_no_mask_intermediate_trace.json` with the minimal
case input context needed for follow-up debugging.

P53 is CPU/offline-friendly fixture plumbing only. It does not add Pallas
kernels, TPU optimization, real training, Qwen-scale export, HF
`PreTrainedModel` support, multi-host sharding, or tolerance loosening.
