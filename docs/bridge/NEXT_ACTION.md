# NEXT_ACTION

## Task ID

Bridge0

## Objective

Install the Nova ↔ Nyx bridge protocol files and helper scripts.

## Scope

Allowed to create:

- `docs/bridge/README.md`
- `docs/bridge/CURRENT_STATE.md`
- `docs/bridge/NEXT_ACTION.md`
- `docs/bridge/NYX_REPORT.md`
- `docs/bridge/BLOCKERS.md`
- `docs/bridge/COMMAND_LOG.md`
- `scripts/nyx/status.sh`
- `scripts/nyx/validate.sh`

Allowed to minimally edit if present:

- `NOVA_AGENT_ENTRYPOINT.yaml`
- `docs/PHASE_CHECKLIST.md`
- `docs/DEV_WORKFLOW.md`
- `docs/CSC_SNAPSHOT.yaml`

Do not edit production code.

Do not edit tests.

Do not overwrite `docs/CSC_BIBLE.md`.

## Required Context

Read only these files first if present:

- `NOVA_AGENT_ENTRYPOINT.yaml`
- `docs/CSC_SNAPSHOT.yaml`
- `docs/PHASE_CHECKLIST.md`
- `docs/DEV_WORKFLOW.md`

Do not scan the full repo before reading these.

## Implementation Notes

Set up the bridge folder and helper scripts exactly as described in the implementation spec.

Update project entrypoint/workflow docs with a short note that future Nyx tasks should start by reading:

- `NOVA_AGENT_ENTRYPOINT.yaml`
- `docs/CSC_SNAPSHOT.yaml`
- `docs/PHASE_CHECKLIST.md`
- `docs/bridge/CURRENT_STATE.md`
- `docs/bridge/NEXT_ACTION.md`

Future Nyx tasks should end by updating:

- `docs/bridge/NYX_REPORT.md`
- `docs/bridge/BLOCKERS.md` when needed
- `docs/bridge/COMMAND_LOG.md`

## Validation Commands

Run what is available and appropriate:

```bash
bash scripts/nyx/status.sh
bash scripts/nyx/validate.sh
```

If Python tooling is available, optionally run:

```bash
ruff check .
black --check .
pytest -q
```

Do not install new dependencies for this setup task unless the repo already documents that setup path.

## Done Criteria

- [ ] `docs/bridge/` exists with all required bridge files.
- [ ] `scripts/nyx/status.sh` exists and is executable.
- [ ] `scripts/nyx/validate.sh` exists and is executable.
- [ ] Existing entrypoint/workflow docs mention the bridge protocol.
- [ ] No production code changed.
- [ ] `NYX_REPORT.md` updated with the result.
- [ ] `COMMAND_LOG.md` updated with commands run.
- [ ] `BLOCKERS.md` updated if any blockers exist.

## Report Requirements

Update `docs/bridge/NYX_REPORT.md` with:

- summary
- files changed
- commands run
- validation results
- blockers
- weirdness/risk
- suggested next action
