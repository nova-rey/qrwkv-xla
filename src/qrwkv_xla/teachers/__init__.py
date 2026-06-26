"""Teacher backend boundaries for QRWKV-XLA."""

from qrwkv_xla.teachers.backend import TeacherBackend
from qrwkv_xla.teachers.emission import emit_teacher_target_store
from qrwkv_xla.teachers.hf import (
    HFCompactTeacherTargets,
    HFTeacherBackend,
    HFTeacherUnavailable,
)
from qrwkv_xla.teachers.hf_specimen_smoke import (
    DEFAULT_HF_SPECIMEN_MODEL_ID,
    HFTeacherSpecimenConfig,
    HFTeacherSpecimenSmokeResult,
    HFTeacherSpecimenSwapReport,
    run_hf_teacher_specimen_smoke,
    run_hf_teacher_specimen_swap_smoke,
)
from qrwkv_xla.teachers.synthetic import SyntheticTeacherBackend

__all__ = [
    "DEFAULT_HF_SPECIMEN_MODEL_ID",
    "HFTeacherBackend",
    "HFCompactTeacherTargets",
    "HFTeacherSpecimenConfig",
    "HFTeacherSpecimenSmokeResult",
    "HFTeacherSpecimenSwapReport",
    "HFTeacherUnavailable",
    "SyntheticTeacherBackend",
    "TeacherBackend",
    "emit_teacher_target_store",
    "run_hf_teacher_specimen_smoke",
    "run_hf_teacher_specimen_swap_smoke",
]
