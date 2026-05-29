"""Teacher backend boundaries for QRWKV-XLA."""

from qrwkv_xla.teachers.backend import TeacherBackend
from qrwkv_xla.teachers.emission import emit_teacher_target_store
from qrwkv_xla.teachers.hf import HFTeacherBackend, HFTeacherUnavailable
from qrwkv_xla.teachers.hf_specimen_smoke import (
    DEFAULT_HF_SPECIMEN_MODEL_ID,
    HFTeacherSpecimenSmokeResult,
    run_hf_teacher_specimen_smoke,
)
from qrwkv_xla.teachers.synthetic import SyntheticTeacherBackend

__all__ = [
    "DEFAULT_HF_SPECIMEN_MODEL_ID",
    "HFTeacherBackend",
    "HFTeacherSpecimenSmokeResult",
    "HFTeacherUnavailable",
    "SyntheticTeacherBackend",
    "TeacherBackend",
    "emit_teacher_target_store",
    "run_hf_teacher_specimen_smoke",
]
