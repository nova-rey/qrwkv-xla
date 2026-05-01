"""Student model interfaces for QRWKV-XLA."""

from qrwkv_xla.students.base import StudentModel, StudentOutput
from qrwkv_xla.students.factory import create_student
from qrwkv_xla.students.lm_head import (
    apply_lm_head,
    apply_tied_lm_head,
    init_lm_head_params,
)
from qrwkv_xla.students.rwkv7_reference import (
    RWKV7ReferenceConfig,
    RWKV7ReferenceStudent,
    rwkv7_reference_layer,
)
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig

__all__ = [
    "RWKV7ReferenceConfig",
    "RWKV7ReferenceStudent",
    "StudentModel",
    "StudentOutput",
    "TinyStudent",
    "TinyStudentConfig",
    "apply_lm_head",
    "apply_tied_lm_head",
    "create_student",
    "init_lm_head_params",
    "rwkv7_reference_layer",
]
