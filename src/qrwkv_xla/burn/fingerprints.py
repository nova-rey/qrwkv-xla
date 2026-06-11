from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def fingerprint_pytree(tree: Any) -> str:
    sha = hashlib.sha256()
    _update_fingerprint(sha, tree, path=())
    return sha.hexdigest()


def fingerprint_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _update_fingerprint(
    sha: Any,
    value: Any,
    *,
    path: tuple[str, ...],
) -> None:
    sha.update("/".join(path).encode("utf-8"))
    sha.update(b"\0")
    if isinstance(value, Mapping):
        sha.update(b"dict")
        for key in sorted(value, key=lambda item: str(item)):
            _update_fingerprint(sha, value[key], path=(*path, str(key)))
        return
    if isinstance(value, tuple):
        sha.update(b"tuple")
        for index, item in enumerate(value):
            _update_fingerprint(sha, item, path=(*path, str(index)))
        return
    if _is_list_like(value):
        sha.update(b"list")
        for index, item in enumerate(value):
            _update_fingerprint(sha, item, path=(*path, str(index)))
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        arr = np.asarray(value)
    else:
        arr = np.asarray(value)
    sha.update(str(arr.shape).encode("utf-8"))
    sha.update(str(arr.dtype).encode("utf-8"))
    sha.update(np.ascontiguousarray(arr).tobytes())


def _is_list_like(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, tuple))
