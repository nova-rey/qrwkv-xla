# P124 Tome 100 Generation Result

P124 produced and validated 100-example TeacherTextbook artifacts from
WikiText-103 using the tiny real HF teacher specimen `sshleifer/tiny-gpt2`.

Input:

- JSONL: `~/qrwkv_artifacts/p124_inputs/wikitext103_100.jsonl`
- Examples: 100
- Source: `Salesforce/wikitext:wikitext-103-raw-v1:train`
- Recorded license field: `CC BY-SA`

Artifacts:

- Dense tome:
  `~/qrwkv_artifacts/p124_tomes/p124_tiny_gpt2_wikitext103_100_dense`
- Dense validation: `status=pass blockers=0 warnings=0`
- Dense size: approximately `1.2G`
- Cascaded tome:
  `~/qrwkv_artifacts/p124_tomes/p124_tiny_gpt2_wikitext103_100_cascaded`
- Cascaded validation: `status=pass blockers=0 warnings=0`
- Cascaded size: approximately `13M`

Conclusion:

P124 successfully produced and validated both dense and
`cascaded_soft_labels_v1` 100-example tomes for this tiny teacher/textbook
specimen. The cascaded artifact is much smaller than the dense-logit artifact
for this specimen.

This is an artifact-generation and validation result only. It does not prove
model quality, production training readiness, distributed training readiness,
Qwen support, tokenizer remapping, or large-scale performance.
