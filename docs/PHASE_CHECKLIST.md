# Phase Checklist

## Phase 0 — Foundation

### P0A — Docs and skeleton

- [x] Create project skeleton
- [x] Add pyproject / lint / pytest setup
- [x] Add docs directory
- [x] Add architecture doc
- [x] Add roadmap
- [x] Add decisions log
- [x] Add risk register
- [x] Add artifact format doc
- [x] Add testing strategy
- [x] Add TPU notes
- [x] Add snapshot
- [x] Add append-only Bible
- [x] Add Nyx agent entrypoint
- [x] Add placeholder configs
- [x] Add placeholder scripts
- [x] Add import/layout tests
- [x] Run tests
- [x] Ensure lint/format compliance

### P0.5 — Normalize foundation and add contracts

- [x] Normalize scaffold files into readable multiline content
- [x] Update workflow docs to Nyx-primary / Codex-subagent framing
- [x] Add typed config schema dataclasses
- [x] Add config loading and validation
- [x] Add teacher target manifest schema
- [x] Add manifest validation and round-trip helpers
- [x] Update artifact contract docs to match implemented schema
- [x] Update smoke scripts to load configs
- [x] Add config loading tests
- [x] Add target manifest tests
- [x] Run full validation stack

## P1 — Target Artifact Store Foundation

- [x] Add artifact layout helpers
- [x] Add shard read/write helpers
- [x] Add bundle read/write/inspect/validate helpers
- [x] Add fake target generation script
- [x] Add target inspection script
- [x] Add shard tests
- [x] Add bundle tests
- [x] Update artifact docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation commands

## P2 — Teacher Exporter Interface + Fake Export Pipeline

- [x] Add teacher export config dataclasses
- [x] Add teacher export config loader
- [x] Add export request/result contracts
- [x] Add TeacherExporter protocol
- [x] Add fake deterministic exporter
- [x] Add exporter registry
- [x] Add export_teacher_targets.py CLI
- [x] Update teacher_export_stub.yaml
- [x] Add fake exporter tests
- [x] Add CLI tests
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation commands
