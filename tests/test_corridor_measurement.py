from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qrwkv_xla.fingerprint import (
    CorridorMeasurementConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    corridor_distance,
    detect_corridor_entries,
    run_corridor_measurement,
    run_tiny_real_teacher_fingerprint_capture,
    write_fingerprint_provenance,
)
from qrwkv_xla.teachers import HFTeacherBackend


def test_corridor_distance_inside_and_outside_bounds() -> None:
    assert corridor_distance(0.5, 0.0, 1.0) == (0.0, 0.0)
    assert corridor_distance(-0.25, 0.0, 1.0) == (0.25, 0.25)
    assert corridor_distance(1.5, 0.0, 1.0) == (0.5, 0.5)
    raw, normalized = corridor_distance(2.0, 1.0, 1.0)
    assert raw == 1.0
    assert math.isfinite(normalized)


def test_entry_detection_distinguishes_transient_and_stable() -> None:
    transient = [
        {"optimizer_step": 0, "inside_all_rate": 0.2},
        {"optimizer_step": 1, "inside_all_rate": 0.95},
        {"optimizer_step": 2, "inside_all_rate": 0.4},
        {"optimizer_step": 3, "inside_all_rate": 1.0},
    ]
    result = detect_corridor_entries(
        transient,
        threshold=0.95,
        stable_entry_evals=2,
    )
    assert result["first_threshold_entry_step"] == 1
    assert result["first_strict_entry_step"] == 3
    assert result["first_stable_entry_step"] is None

    stable = [
        *transient,
        {"optimizer_step": 4, "inside_all_rate": 0.98},
    ]
    result = detect_corridor_entries(stable, threshold=0.95, stable_entry_evals=2)
    assert result["first_stable_entry_step"] == 4
    assert result["stable_entry_achieved"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("eval_every", 0, "eval_every must be > 0"),
        ("stable_entry_evals", 0, "stable_entry_evals must be >= 1"),
        (
            "corridor_entry_threshold",
            1.1,
            "corridor_entry_threshold must be within",
        ),
    ),
)
def test_invalid_measurement_config_fails(
    tmp_path: Path,
    field: str,
    value: int | float,
    message: str,
) -> None:
    values = {field: value}
    config = CorridorMeasurementConfig(
        fingerprint_artifact=tmp_path / "train",
        held_out_fingerprint_artifact=tmp_path / "held",
        source_texts=tmp_path / "source.jsonl",
        output_dir=tmp_path / "output",
        **values,
    )
    with pytest.raises(ValueError, match=message):
        run_corridor_measurement(config)


def test_corridor_measurement_integration(tmp_path: Path) -> None:
    train_source = _source_file(
        tmp_path / "train.jsonl",
        "p153-train",
        (
            "Aster circuits map a narrow winter signal.",
            "Copper vectors cross a quiet theorem.",
            "Lantern matrices preserve a stable rhythm.",
            "Quartz recurrences follow a distant pulse.",
        ),
    )
    held_source = _source_file(
        tmp_path / "held.jsonl",
        "p153-held",
        (
            "Violet signals turn beneath the observatory.",
            "Seven tensors cross an unfamiliar harbor.",
            "Silver states remember a hidden sequence.",
            "Amber kernels trace a separate path.",
        ),
    )
    backend = _fake_backend(16)
    train_artifact = _capture(
        tmp_path / "train_artifact",
        train_source,
        "p153-train",
        backend,
    )
    held_artifact = _capture(
        tmp_path / "held_artifact",
        held_source,
        "p153-held",
        backend,
    )
    write_fingerprint_provenance(
        train_artifact,
        source_file=train_source,
        artifact_role="training",
    )
    write_fingerprint_provenance(
        held_artifact,
        source_file=held_source,
        artifact_role="held_out_evaluation",
    )
    result = run_corridor_measurement(
        CorridorMeasurementConfig(
            fingerprint_artifact=train_artifact,
            held_out_fingerprint_artifact=held_artifact,
            source_texts=train_source,
            output_dir=tmp_path / "p153",
            steps=3,
            eval_every=1,
            checkpoint_every=2,
            batch_size=2,
            optimizer="sgd",
            learning_rate=0.01,
            corridor_entry_threshold=0.0,
            stable_entry_evals=1,
            overwrite=True,
        )
    )
    report = _json(result.report_path)
    trajectory = [
        json.loads(line)
        for line in result.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.status == "pass"
    assert result.completed_steps == 3
    assert report["phase"] == "P153"
    assert report["training_cycle"] == "corridor_only"
    assert report["exemplar_training_enabled"] is False
    assert report["evaluation_steps"] == [0, 1, 2, 3]
    assert len(trajectory) == 4
    assert trajectory[0]["optimizer_step"] == 0
    assert trajectory[-1]["parameter_delta_from_initial"] > 0.0
    assert all(math.isfinite(point["held_out_corridor_loss"]) for point in trajectory)
    assert all(
        math.isfinite(point["mean_distance_outside_corridor"]) for point in trajectory
    )
    assert report["resource_accounting"]["total_record_visits"] == 6
    assert report["resource_accounting"]["artifact_bytes_logically_consumed"] > 0
    assert report["wall_clock"]["total_wall_clock_seconds"] >= 0.0
    assert report["general_quality_claim_made"] is False
    assert report["quality_per_byte_claim_made"] is False
    assert (result.output_dir / "checkpoints/step_000000/checkpoint.json").is_file()
    assert (result.output_dir / "checkpoints/step_000002/checkpoint.json").is_file()
    assert (result.output_dir / "checkpoints/final/checkpoint.json").is_file()
    for name in (
        "corridor_measurement_report.json",
        "corridor_measurement_summary.md",
        "corridor_trajectory.jsonl",
        "corridor_efficiency_metrics.json",
        "corridor_entry_receipt.json",
        "resource_accounting.json",
        "checkpoint_lineage_validation.json",
    ):
        assert (result.output_dir / name).is_file()


def _source_file(path: Path, prefix: str, texts: tuple[str, ...]) -> Path:
    path.write_text(
        "".join(
            json.dumps({"example_id": f"{prefix}-{index:06d}", "text": text}) + "\n"
            for index, text in enumerate(texts)
        ),
        encoding="utf-8",
    )
    return path


def _capture(
    output: Path,
    source: Path,
    prefix: str,
    backend: HFTeacherBackend,
) -> Path:
    return run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=output,
            texts_path=source,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            max_exemplars=16,
            example_id_prefix=prefix,
            overwrite=True,
        ),
        backend=backend,
    ).output_dir


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_backend(vocab_size: int) -> HFTeacherBackend:
    return HFTeacherBackend(
        "local/fake-real-teacher",
        tokenizer=_FakeTokenizer(vocab_size),
        model=_FakeCausalLM(vocab_size),
        prompts=("placeholder",),
    )


class _FakeTokenizer:
    name_or_path = "local/fake-tokenizer"
    eos_token = "<eos>"
    pad_token = None
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = 0
    unk_token_id = 2

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, prompts: list[str], **kwargs) -> dict[str, np.ndarray]:
        length = int(kwargs["max_length"])
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            rows.append(
                [
                    3 + (digest[offset] % (self.vocab_size - 3))
                    for offset in range(length)
                ]
            )
        ids = np.asarray(rows, dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


class _FakeCausalLM:
    def __init__(self, vocab_size: int) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            model_type="fake_causal_lm",
        )

    def eval(self) -> _FakeCausalLM:
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        ids = np.asarray(input_ids)
        vocab = np.arange(self.config.vocab_size, dtype=np.float32)
        logits = -np.square(vocab[None, None, :] - ids[..., None]).astype(np.float32)
        return SimpleNamespace(logits=logits)
