# Nova ↔ Nyx Bridge Protocol

This folder is the shared handoff layer between Nova/chat, Nyx, Codex, and CI.

The goal is to keep agent context small, structured, and reusable.

Nova/chat owns planning and task scope.
Nyx owns orchestration, command execution, validation, and reporting.
Codex owns implementation edits when invoked by Nyx.
CI/tests provide objective validation.

## Bridge Files

### `CURRENT_STATE.md`

A compact summary of the current project state relevant to the next task.

This should not duplicate the full project history. Keep it short and current.

### `NEXT_ACTION.md`

The active task packet from Nova/chat to Nyx.

Nyx should treat this as the primary instruction file for the current checkpoint or setup task.

### `NYX_REPORT.md`

Nyx's completion report.

This is the main file the user should bring back to Nova/chat after Nyx finishes.

### `BLOCKERS.md`

Any unresolved blockers, ambiguity, missing dependencies, failing commands, or external actions needed.

Leave as "None currently known" when empty.

### `COMMAND_LOG.md`

A concise log of commands Nyx ran and their results.

Do not paste massive logs. Summarize long output and point to artifacts if needed.

## Role Contract

### Nova/chat

Nova/chat is responsible for:

- planning
- task scope
- constraints
- done criteria
- writing or updating `NEXT_ACTION.md`
- reviewing `NYX_REPORT.md`

### Nyx

Nyx is responsible for:

- reading the bridge files first
- keeping context intake small
- creating focused Codex prompts when code edits are needed
- invoking Codex only with scoped tasks
- running validation commands
- updating `NYX_REPORT.md`
- updating `BLOCKERS.md` when needed
- updating `COMMAND_LOG.md`

### Codex

Codex is responsible for implementation edits when invoked by Nyx.

Codex should receive a focused task, allowed files, forbidden files, validation commands, and done criteria.

### CI/tests

CI and local test commands provide objective validation.

## Hard Rules

Nyx must not recursively inspect the whole repo unless the task explicitly requires it.

Nyx must not summarize the entire project before beginning work.

Nyx must not broaden task scope without explicit instruction.

Nyx must not overwrite `docs/CSC_BIBLE.md`; append only if logging there is necessary.

Nyx must maintain Ruff + Black compliance where applicable.

Nyx must prefer small, reversible changes.

Nyx must report:

- files changed
- commands run
- test results
- blockers
- weirdness or risk
- suggested next action

## Normal Task Loop

1. Read `NOVA_AGENT_ENTRYPOINT.yaml`, `docs/CSC_SNAPSHOT.yaml`, `docs/PHASE_CHECKLIST.md`, and the files in `docs/bridge/`.
2. Read `docs/bridge/NEXT_ACTION.md`.
3. Summarize the task internally.
4. If implementation is needed, create a focused Codex task.
5. Invoke Codex only with that focused task.
6. Run validation commands.
7. Update `docs/bridge/NYX_REPORT.md`.
8. Update `docs/bridge/BLOCKERS.md` if needed.
9. Update `docs/bridge/COMMAND_LOG.md`.
