"""Student model interfaces for QRWKV-XLA."""

from qrwkv_xla.students.base import StudentModel, StudentOutput
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig

__all__ = [
    "StudentModel",
    "StudentOutput",
    "TinyStudent",
    "TinyStudentConfig",
]
