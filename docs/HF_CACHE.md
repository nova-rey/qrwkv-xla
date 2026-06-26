# Hugging Face Cache

Project scripts do not set `TRANSFORMERS_CACHE`. That variable is deprecated in
recent Hugging Face tooling and may still warn if it is present in an operator
shell.

For local real-teacher work, prefer:

```bash
export HF_HOME="$HOME/radjax_cache/huggingface"
```

Keep `--local-files-only` enabled for smoke gates unless a spec explicitly allows
downloads. To preserve an existing cache without redownloading, point `HF_HOME`
at the directory that already contains the Hugging Face cache contents, and do
not set both cache variables in the same shell.
