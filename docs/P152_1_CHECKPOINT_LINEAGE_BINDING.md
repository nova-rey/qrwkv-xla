# P152.1 Checkpoint Lineage Binding

P152.1 closes the identity gap between the P151 training report and P152
held-out evaluation. P151 now records stable hashes for the shared initial
checkpoint, both canonical final checkpoint metadata files and parameter
archives, path-aware parameter-tree fingerprints, student contracts, step
budgets, training artifact, source-example set, and software commit.

P152 recomputes those values for the supplied checkpoint directories and writes
`checkpoint_lineage_validation.json` before loading held-out records. Any
metadata, parameter, fingerprint, role, step, student, artifact, source-set, or
shared-initialization mismatch blocks evaluation and winner reporting.

Checkpoint bundle hashes depend only on `checkpoint.json` and `params.npz`
contents, so moving an intact checkpoint directory does not alter its identity.
Parameter fingerprints include canonical tree paths, shapes, dtypes, and raw
array bytes.

Source JSONL rows require explicit `example_id` values. Joining is performed by
ID and then reordered to artifact order before hashing. Legacy text-only files
are accepted only with `--allow-legacy-positional-source-join`, which records
reduced confidence and `publication_grade_lineage: false`.

This phase changes provenance only. It does not alter training objectives,
held-out metrics, bootstrap logic, winner direction, corridor definitions, or
model architecture.
