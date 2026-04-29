# Development Workflow

QRWKV-XLA follows the normal Nova / Nyx / Codex rhythm.

## Roles

### Nova
- design
- architecture
- specs
- prompt writing
- review guidance

### Nyx / Codex
- mechanical implementation
- repo edits
- tests
- lint
- scaffolding
- CI preparation

## Default phase structure

### Prompt A
Design, docs, scaffolding, and minimal or no functional code.

### Prompt B
Main implementation.

### Prompt C
CI, polish, cleanup, and docs sync.

## Checkpoint discipline

Every phase/checkpoint should preserve:
- small reversible changes
- lint clean
- tests passing
- docs updated
- snapshot updated
- Bible appended, never overwritten

## Historical log policy

Only append new phase/update notes to `docs/QRWKV_BIBLE.md`. Do not rewrite earlier history.

## Current commit theme

`P0A: scaffold QRWKV-XLA foundation docs`
