from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.export import load_hf_safetensors_export


@dataclass(frozen=True)
class ToyContinuationExample:
    id: str
    context_token_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ContinuationScore:
    example_id: str
    loglikelihood: float
    num_tokens_scored: int
    greedy_match: bool


@dataclass(frozen=True)
class ToyEvalResult:
    export_dir: Path
    task_path: Path
    num_examples: int
    num_tokens_scored: int
    total_loglikelihood: float
    mean_loglikelihood: float
    mean_neg_loglikelihood: float
    perplexity: float
    greedy_accuracy: float
    scores: tuple[ContinuationScore, ...]
    official_lm_eval: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "phase": "P42",
            "mode": "lm_eval_style_toy_harness",
            "export_dir": str(self.export_dir),
            "task_path": str(self.task_path),
            "num_examples": self.num_examples,
            "num_tokens_scored": self.num_tokens_scored,
            "total_loglikelihood": self.total_loglikelihood,
            "mean_loglikelihood": self.mean_loglikelihood,
            "mean_neg_loglikelihood": self.mean_neg_loglikelihood,
            "perplexity": self.perplexity,
            "greedy_accuracy": self.greedy_accuracy,
            "scores": [
                {
                    "example_id": score.example_id,
                    "loglikelihood": score.loglikelihood,
                    "num_tokens_scored": score.num_tokens_scored,
                    "greedy_match": score.greedy_match,
                }
                for score in self.scores
            ],
            "official_lm_eval": self.official_lm_eval,
            "limitations": [
                "lm_eval-style toy continuation scoring only",
                "token-id fixture only; no production tokenizer integration",
                "tiny CPU/local exported-student smoke only",
                "no benchmark or model-quality claim",
            ],
        }


class ExportedStudentEvalAdapter:
    def __init__(self, export_dir: str | Path) -> None:
        self.export_dir = Path(export_dir)
        self._loaded = load_hf_safetensors_export(self.export_dir)
        student_config = self._loaded.metadata.get("student_config", {})
        if not bool(student_config.get("emit_logits", False)):
            raise ValueError("exported-student eval requires emit_logits=true")
        self.vocab_size = int(student_config["vocab_size"])

    def loglikelihood(self, example: ToyContinuationExample) -> ContinuationScore:
        self._validate_example(example)
        token_ids = example.context_token_ids + example.continuation_token_ids
        input_ids = jnp.asarray([token_ids], dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        cpu_devices = jax.devices("cpu")
        with jax.default_device(cpu_devices[0]):
            output = self._loaded.student.apply(
                self._loaded.params,
                input_ids,
                attention_mask,
            )
            if output.logits is None:
                raise ValueError("exported-student eval requires logits")
            logits = jnp.asarray(output.logits)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

        context_len = len(example.context_token_ids)
        labels = np.asarray(example.continuation_token_ids, dtype=np.int64)
        positions = np.arange(
            context_len - 1,
            context_len - 1 + len(example.continuation_token_ids),
            dtype=np.int64,
        )
        log_probs_np = np.asarray(log_probs[0])
        token_log_probs = log_probs_np[positions, labels]
        greedy_ids = np.asarray(logits[0, positions, :]).argmax(axis=-1)
        loglikelihood = float(np.sum(token_log_probs))
        if not math.isfinite(loglikelihood):
            raise ValueError(f"non-finite loglikelihood for example {example.id!r}")
        return ContinuationScore(
            example_id=example.id,
            loglikelihood=loglikelihood,
            num_tokens_scored=len(example.continuation_token_ids),
            greedy_match=bool(np.array_equal(greedy_ids, labels)),
        )

    def _validate_example(self, example: ToyContinuationExample) -> None:
        if not example.id:
            raise ValueError("toy eval example id must be non-empty")
        if not example.context_token_ids:
            raise ValueError(f"example {example.id!r} has empty context_token_ids")
        if not example.continuation_token_ids:
            raise ValueError(f"example {example.id!r} has empty continuation_token_ids")
        for field_name, token_ids in (
            ("context_token_ids", example.context_token_ids),
            ("continuation_token_ids", example.continuation_token_ids),
        ):
            for token_id in token_ids:
                if token_id < 0 or token_id >= self.vocab_size:
                    raise ValueError(
                        f"example {example.id!r} {field_name} token {token_id} "
                        f"is outside exported vocab_size {self.vocab_size}"
                    )


def load_toy_continuation_task(path: str | Path) -> tuple[ToyContinuationExample, ...]:
    task_path = Path(path)
    if not task_path.is_file():
        raise FileNotFoundError(f"missing toy eval task file: {task_path}")
    examples: list[ToyContinuationExample] = []
    lines = task_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{task_path}:{line_number} must contain a JSON object")
        examples.append(
            ToyContinuationExample(
                id=_required_str(raw, "id", task_path, line_number),
                context_token_ids=_token_tuple(
                    raw,
                    "context_token_ids",
                    task_path,
                    line_number,
                ),
                continuation_token_ids=_token_tuple(
                    raw,
                    "continuation_token_ids",
                    task_path,
                    line_number,
                ),
            )
        )
    if not examples:
        raise ValueError(f"toy eval task file has no examples: {task_path}")
    return tuple(examples)


def run_toy_exported_student_eval(
    *,
    export_dir: str | Path,
    task_path: str | Path,
) -> ToyEvalResult:
    adapter = ExportedStudentEvalAdapter(export_dir)
    examples = load_toy_continuation_task(task_path)
    scores = tuple(adapter.loglikelihood(example) for example in examples)
    num_tokens = sum(score.num_tokens_scored for score in scores)
    if num_tokens <= 0:
        raise ValueError("toy eval scored zero continuation tokens")
    total_loglikelihood = float(sum(score.loglikelihood for score in scores))
    mean_loglikelihood = total_loglikelihood / num_tokens
    mean_neg_loglikelihood = -mean_loglikelihood
    perplexity = float(math.exp(mean_neg_loglikelihood))
    greedy_accuracy = sum(score.greedy_match for score in scores) / len(scores)
    for name, value in (
        ("total_loglikelihood", total_loglikelihood),
        ("mean_loglikelihood", mean_loglikelihood),
        ("mean_neg_loglikelihood", mean_neg_loglikelihood),
        ("perplexity", perplexity),
        ("greedy_accuracy", greedy_accuracy),
    ):
        if not math.isfinite(value):
            raise ValueError(f"toy eval produced non-finite {name}: {value}")
    return ToyEvalResult(
        export_dir=Path(export_dir),
        task_path=Path(task_path),
        num_examples=len(scores),
        num_tokens_scored=num_tokens,
        total_loglikelihood=total_loglikelihood,
        mean_loglikelihood=mean_loglikelihood,
        mean_neg_loglikelihood=mean_neg_loglikelihood,
        perplexity=perplexity,
        greedy_accuracy=greedy_accuracy,
        scores=scores,
        official_lm_eval={
            "integrated": False,
            "reason": (
                "official lm_eval is deferred; P42 keeps default tests offline and "
                "uses a tiny local lm_eval-style harness"
            ),
        },
    )


def _required_str(
    raw: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_number} field {key!r} must be a string")
    return value


def _token_tuple(
    raw: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{line_number} field {key!r} must be a non-empty list")
    tokens: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(
                f"{path}:{line_number} field {key!r} must contain integer tokens"
            )
        tokens.append(item)
    return tuple(tokens)
