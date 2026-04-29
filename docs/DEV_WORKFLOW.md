# Development Workflow

QRWKV-XLA now uses Nyx as the primary implementation agent.

## Roles

### Nova
- design
- architecture
- specs
- review guidance

### Nyx
- primary implementation agent
- repo-level decisions within the current spec
- file structure consistency
- docs updates
- ensuring tests pass
- deciding when to delegate mechanical tasks

### Codex
- sub-agent used by Nyx for mechanical edits
- targeted code generation
- test writing
- lint/format cleanup
- import/path issue cleanup

## Default workflow

The default operating model is whole-implementation specs executed by Nyx,
optionally with Codex as a sub-agent for mechanical work.

The older Prompt A / B / C split is not the default workflow anymore, though it
can still be resurrected when it is useful for a phase plan.

## Checkpoint discipline

Every phase/checkpoint should preserve:
- small reversible changes
- lint clean
- tests passing
- docs updated
- snapshot updated
- Bible appended, never overwritten

## Historical log policy

Only append new phase/update notes to `docs/QRWKV_BIBLE.md`. Do not rewrite
earlier history.

## Current commit themes

- `P0A: scaffold QRWKV-XLA foundation docs`
- `P0.5: normalize scaffold and add config artifact contracts`
