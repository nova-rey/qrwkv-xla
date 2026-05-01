from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from qrwkv_xla.generation import (
    GenerationRecord,
    SmokeTokenizer,
    greedy_generate,
    load_student_from_checkpoint,
    write_generation_jsonl,
    write_generation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal greedy generation from a QRWKV-XLA checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-dir")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    loaded = load_student_from_checkpoint(args.checkpoint)
    tokenizer = SmokeTokenizer(
        vocab_size=args.vocab_size or int(loaded.manifest.student_config["vocab_size"])
    )
    prompts = _collect_prompts(args.prompt, args.prompt_file)
    records: list[GenerationRecord] = []

    for index, prompt_text in enumerate(prompts):
        prompt_token_ids = tokenizer.encode(prompt_text)
        result = greedy_generate(
            student=loaded.student,
            params=loaded.params,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded_text = tokenizer.decode(result.full_token_ids)
        records.append(
            GenerationRecord(
                prompt_id=f"prompt_{index:06d}",
                prompt_text=prompt_text,
                prompt_token_ids=result.prompt_token_ids,
                generated_token_ids=result.generated_token_ids,
                full_token_ids=result.full_token_ids,
                decoded_text=decoded_text,
                metadata={
                    "checkpoint_dir": str(loaded.checkpoint_dir),
                    "max_new_tokens": args.max_new_tokens,
                    "tokenizer": "smoke",
                },
            )
        )

    summary = {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "prompt_count": len(records),
        "passed": all(record.generated_token_ids for record in records),
        "max_new_tokens": args.max_new_tokens,
        "tokenizer": "smoke",
        "vocab_size": tokenizer.vocab_size,
        "limitations": [
            "smoke tokenizer only",
            "greedy decoding only",
            "no quality benchmark",
        ],
    }
    if args.output_dir:
        output_dir = Path(args.output_dir)
        write_generation_jsonl(records, output_dir / "generations.jsonl")
        write_generation_summary(summary, output_dir / "summary.json")
        summary["output_dir"] = str(output_dir)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "generations": [asdict(record) for record in records],
                },
                sort_keys=True,
            )
        )
    else:
        for record in records:
            print(record.decoded_text)
        if args.output_dir:
            print(f"output_dir: {args.output_dir}")


def _collect_prompts(prompts: list[str], prompt_file: str | None) -> list[str]:
    values = list(prompts)
    if prompt_file is not None:
        path = Path(prompt_file)
        values.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not values:
        values.append("Hello from QRWKV-XLA")
    return values


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
