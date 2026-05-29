"""Teacher backend boundaries for QRWKV-XLA."""

from qrwkv_xla.teachers.backend import TeacherBackend
from qrwkv_xla.teachers.emission import emit_teacher_target_store
from qrwkv_xla.teachers.hf import HFTeacherBackend, HFTeacherUnavailable
from qrwkv_xla.teachers.synthetic import SyntheticTeacherBackend

__all__ = [
    "HFTeacherBackend",
    "HFTeacherUnavailable",
    "SyntheticTeacherBackend",
    "TeacherBackend",
    "emit_teacher_target_store",
]
