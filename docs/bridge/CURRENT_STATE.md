# CURRENT_STATE

## Project

QRWKV-XLA

## Current Phase / Checkpoint

Bridge0 — Nova ↔ Nyx bridge protocol setup.

## Current Goal

Install lightweight repo-local bridge files so Nova/chat can hand Nyx scoped task packets and Nyx can return compact execution reports.

## Known Constraints

- Keep agent context intake small.
- Prefer scoped task packets over broad repo scans.
- Use `NEXT_ACTION.md` as the active task instruction.
- Use `NYX_REPORT.md` as the return packet to Nova/chat.

## Relevant Canonical Files

- `NOVA_AGENT_ENTRYPOINT.yaml` (not present in this repo at Bridge0 setup time)
- `docs/CSC_SNAPSHOT.yaml` (not present in this repo at Bridge0 setup time)
- `docs/PHASE_CHECKLIST.md`
- `docs/DEV_WORKFLOW.md`
- `docs/bridge/NEXT_ACTION.md`
- `docs/bridge/NYX_REPORT.md`

## Notes

This file should stay compact. It is not a project history log.
