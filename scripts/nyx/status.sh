#!/usr/bin/env bash
set -euo pipefail

echo "== git branch =="
git branch --show-current || true

echo
echo "== git status =="
git status --short || true

echo
echo "== latest commit =="
git log -1 --oneline || true

if [[ -f NOVA_AGENT_ENTRYPOINT.yaml ]]; then
 echo
 echo "== NOVA_AGENT_ENTRYPOINT.yaml =="
 sed -n '1,180p' NOVA_AGENT_ENTRYPOINT.yaml
fi

if [[ -f docs/CSC_SNAPSHOT.yaml ]]; then
 echo
 echo "== docs/CSC_SNAPSHOT.yaml =="
 sed -n '1,180p' docs/CSC_SNAPSHOT.yaml
fi

if [[ -f docs/PHASE_CHECKLIST.md ]]; then
 echo
 echo "== docs/PHASE_CHECKLIST.md =="
 sed -n '1,220p' docs/PHASE_CHECKLIST.md
fi

if [[ -f docs/bridge/CURRENT_STATE.md ]]; then
 echo
 echo "== docs/bridge/CURRENT_STATE.md =="
 sed -n '1,220p' docs/bridge/CURRENT_STATE.md
fi

if [[ -f docs/bridge/NEXT_ACTION.md ]]; then
 echo
 echo "== docs/bridge/NEXT_ACTION.md =="
 sed -n '1,260p' docs/bridge/NEXT_ACTION.md
fi
