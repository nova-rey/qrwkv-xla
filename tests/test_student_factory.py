from __future__ import annotations

import pytest

from qrwkv_xla.students import (
    RWKV7RADLADSReferenceStudent,
    RWKV7ReferenceStudent,
    TinyStudent,
    create_student,
)


def test_create_student_builds_tiny_student() -> None:
    student = create_student(
        "tiny_student",
        vocab_size=11,
        hidden_size=7,
        num_layers=2,
    )

    assert isinstance(student, TinyStudent)
    assert student.config.vocab_size == 11
    assert student.config.hidden_size == 7
    assert student.config.num_layers == 2


def test_create_student_builds_rwkv7_reference_student() -> None:
    student = create_student(
        "rwkv7_reference",
        vocab_size=11,
        hidden_size=7,
        num_layers=2,
    )

    assert isinstance(student, RWKV7ReferenceStudent)
    assert student.config.vocab_size == 11
    assert student.config.hidden_size == 7
    assert student.config.num_layers == 2


def test_create_student_builds_rwkv7_radlads_reference_student() -> None:
    student = create_student(
        "rwkv7_radlads_reference",
        vocab_size=12,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
    )

    assert isinstance(student, RWKV7RADLADSReferenceStudent)
    assert student.config.vocab_size == 12
    assert student.config.hidden_size == 8
    assert student.config.num_layers == 2
    assert student.config.num_heads == 2


def test_create_student_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown student architecture"):
        create_student("unknown", vocab_size=11, hidden_size=7, num_layers=2)
