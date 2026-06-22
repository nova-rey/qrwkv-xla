from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrwkv_xla.fingerprint import (
    FingerprintTrainedBaselineConfig,
    TinyRealTeacherFingerprintCaptureConfig,
    masked_causal_lm_loss,
    parameter_fingerprint,
    run_fingerprint_trained_baseline_comparison,
    run_tiny_real_teacher_fingerprint_capture,
)
from qrwkv_xla.teachers import HFTeacherBackend

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ROOT / "tests/fixtures/fingerprint_capture_real_teacher/tiny_texts.jsonl"
SCRIPT = ROOT / "scripts/run_fingerprint_trained_baseline_comparison.py"


def test_causal_lm_loss_is_finite_non_negative_and_masks_padding() -> None:
    logits = jnp.asarray([[[0.0, 3.0], [3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]])
    input_ids = jnp.asarray([[0, 1, 0, 1]])
    full = masked_causal_lm_loss(logits, input_ids, jnp.ones_like(input_ids))
    padded = masked_causal_lm_loss(
        logits.at[:, 2, :].set(jnp.asarray([[-100.0, 100.0]])),
        input_ids,
        jnp.asarray([[1, 1, 1, 0]]),
    )

    assert np.isfinite(float(full))
    assert float(full) >= 0.0
    assert float(padded) < 0.1


def test_parameter_fingerprint_is_stable_and_sensitive() -> None:
    params = {"a": jnp.asarray([1.0, 2.0]), "b": jnp.asarray([[3.0]])}
    clone = jax.tree_util.tree_map(lambda value: jnp.array(value), params)
    changed = {**clone, "b": clone["b"] + 1.0}

    assert parameter_fingerprint(params) == parameter_fingerprint(clone)
    assert parameter_fingerprint(params) != parameter_fingerprint(changed)


def test_trained_baseline_and_corridor_complete_matched_steps(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    result = run_fingerprint_trained_baseline_comparison(_config(tmp_path, artifact))
    report = _json(result.report_path)

    assert result.status == "pass"
    assert report["fairness"]["comparison_valid"] is True
    assert report["fairness"]["same_initial_parameter_fingerprint"] is True
    assert report["fairness"]["same_source_example_ids"] is True
    assert report["fairness"]["same_completed_steps"] is True
    assert report["baseline"]["optimizer_steps_completed"] == 3
    assert report["fingerprint"]["optimizer_steps_completed"] == 3
    assert report["baseline"]["params_changed"] is True
    assert report["fingerprint"]["params_changed"] is True
    assert report["baseline"]["checkpoint_written"] is True
    assert report["fingerprint"]["checkpoint_written"] is True
    assert report["claims"]["winner_declared"] is False
    assert report["claims"]["held_out_claim_made"] is False
    assert result.metrics_path.is_file()
    assert result.summary_path.is_file()
    assert (result.output_dir / "baseline/checkpoints/final/checkpoint.json").is_file()
    assert (result.output_dir / "baseline/metrics.json").is_file()
    assert (result.output_dir / "baseline/run_report.json").is_file()
    assert (
        result.output_dir / "fingerprint/checkpoints/final/checkpoint.json"
    ).is_file()
    assert (result.output_dir / "fingerprint/metrics.json").is_file()
    assert (
        result.output_dir / "fingerprint/fingerprint_corridor_report.json"
    ).is_file()


def test_mismatched_source_ids_fail_before_training(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    source = tmp_path / "wrong.jsonl"
    source.write_text('{"example_id":"wrong","text":"x"}\n', encoding="utf-8")
    config = FingerprintTrainedBaselineConfig(
        fingerprint_artifact=artifact,
        source_texts=source,
        output_dir=tmp_path / "out",
        steps=1,
        batch_size=1,
        optimizer="sgd",
        learning_rate=0.01,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="source example IDs do not align"):
        run_fingerprint_trained_baseline_comparison(config)


@pytest.mark.parametrize("field", ["steps", "batch_size"])
def test_zero_budget_fails_clearly(tmp_path: Path, field: str) -> None:
    values = {"steps": 1, "batch_size": 1, field: 0}
    config = FingerprintTrainedBaselineConfig(
        fingerprint_artifact=tmp_path / "artifact",
        source_texts=TEXTS,
        output_dir=tmp_path / "out",
        **values,
    )
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        run_fingerprint_trained_baseline_comparison(config)


def test_cli_smoke(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fingerprint-artifact",
            str(artifact),
            "--source-texts",
            str(TEXTS),
            "--output-dir",
            str(output),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--optimizer",
            "sgd",
            "--learning-rate",
            "0.01",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert "winner_declared=false" in completed.stdout
    assert (output / "trained_baseline_comparison_report.json").is_file()


def _config(tmp_path: Path, artifact: Path) -> FingerprintTrainedBaselineConfig:
    return FingerprintTrainedBaselineConfig(
        fingerprint_artifact=artifact,
        source_texts=TEXTS,
        output_dir=tmp_path / "p151",
        steps=3,
        batch_size=2,
        optimizer="sgd",
        learning_rate=0.01,
        overwrite=True,
    )


def _artifact(tmp_path: Path) -> Path:
    result = run_tiny_real_teacher_fingerprint_capture(
        TinyRealTeacherFingerprintCaptureConfig(
            output_dir=tmp_path / "artifact",
            texts_path=TEXTS,
            sequence_length=4,
            max_examples=4,
            max_target_positions=16,
            max_exemplars=4,
            overwrite=True,
        ),
        backend=_fake_backend(16),
    )
    return result.output_dir


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
        masks = []
        for row, prompt in enumerate(prompts):
            used = min(max(1, len(prompt.split())), length)
            ids = [((row + offset + 3) % self.vocab_size) for offset in range(used)]
            rows.append(ids + [self.pad_token_id] * (length - used))
            masks.append([1] * used + [0] * (length - used))
        return {"input_ids": np.asarray(rows), "attention_mask": np.asarray(masks)}


class _FakeCausalLM:
    def __init__(self, vocab_size: int) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocab_size, model_type="fake_causal_lm"
        )

    def eval(self) -> _FakeCausalLM:
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        ids = np.asarray(input_ids)
        vocab = np.arange(self.config.vocab_size, dtype=np.float32)
        logits = -np.square(vocab[None, None, :] - ids[..., None]).astype(np.float32)
        return SimpleNamespace(logits=logits)
