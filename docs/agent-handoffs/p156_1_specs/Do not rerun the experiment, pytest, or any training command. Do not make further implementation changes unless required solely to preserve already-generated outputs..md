Do not rerun the experiment, pytest, or any training command. Do not make further implementation changes unless required solely to preserve already-generated outputs.

Work only in the current local workspace.

1. Determine whether the previously launched pytest/full regression command finished, was interrupted, or failed. Inspect existing terminal output, process state, logs, shell history, and files only. Do not rerun it.

2. Locate every output produced by the P156/P156.1 work, including:
   - quality-per-byte reports
   - byte-controlled, step-controlled, and time-controlled reports
   - matrix-state and reuse-plan files
   - budget subset manifests
   - paired comparisons
   - CPU/backend receipts
   - test logs or summaries
   - any experiment result containing `exemplar_only_better`
   - checkpoints or other generated artifacts

3. Write a concise `docs/agent-handoffs/p156_1_recovery_report.md` stating:
   - exact command(s) that ran, if recoverable
   - whether the full requested CPU matrix actually ran
   - whether it completed
   - test-suite final status, or the last confirmed percentage if incomplete
   - all failures, distinguishing pre-existing failures from P156.1 failures
   - whether `exemplar_only_better` came from a unit test, fixture smoke, integration run, or full scientific matrix
   - exact paths and sizes of all generated outputs
   - which results are scientifically valid, diagnostic-only, incomplete, or unknown

4. Preserve and commit all source changes, tests, specs, small text/JSON/JSONL/Markdown reports, manifests, and logs produced by this task.

5. Do not commit virtual environments, caches, model weights, checkpoints, large binary artifacts, or duplicated generated data. Instead, list every omitted file with its absolute path, size, SHA-256 hash, and reason in:
   `docs/agent-handoffs/p156_1_large_artifacts_manifest.json`

6. Run only lightweight read-only validation needed to confirm files are parseable, such as JSON parsing and `git diff --check`. Do not run pytest or training.

7. Review the staged diff for secrets, machine-specific credentials, and accidental large files.

8. Commit everything on a new branch named:
   `p156-1-budget-controls-and-results`

   Use commit message:
   `Implement P156.1 budget controls and preserve results`

9. Push that branch to the existing GitHub remote.

10. At the end, report only:
    - branch name
    - commit SHA
    - GitHub commit or branch URL
    - whether the prior test completed
    - classification of `exemplar_only_better`
    - paths of any large outputs that could not be pushed

Do not begin any expensive work. If pushing requires authentication or approval, stop at that exact point with the commit safely created locally.