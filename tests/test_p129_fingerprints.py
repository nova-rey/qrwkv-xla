from __future__ import annotations

import numpy as np

from qrwkv_xla.burn import fingerprint_pytree


def test_same_pytree_gives_same_fingerprint() -> None:
    tree = {"b": np.asarray([1, 2], dtype=np.int32), "a": {"x": 1.5}}

    assert fingerprint_pytree(tree) == fingerprint_pytree(tree)


def test_dict_key_order_is_deterministic() -> None:
    first = {"b": np.asarray([2], dtype=np.int32), "a": np.asarray([1])}
    second = {"a": np.asarray([1]), "b": np.asarray([2], dtype=np.int32)}

    assert fingerprint_pytree(first) == fingerprint_pytree(second)


def test_different_values_change_fingerprint() -> None:
    first = {"x": np.asarray([1, 2, 3], dtype=np.int32)}
    second = {"x": np.asarray([1, 2, 4], dtype=np.int32)}

    assert fingerprint_pytree(first) != fingerprint_pytree(second)


def test_different_shape_changes_fingerprint() -> None:
    first = {"x": np.asarray([1, 2, 3], dtype=np.int32)}
    second = {"x": np.asarray([[1, 2, 3]], dtype=np.int32)}

    assert fingerprint_pytree(first) != fingerprint_pytree(second)


def test_different_dtype_changes_fingerprint() -> None:
    first = {"x": np.asarray([1, 2, 3], dtype=np.int32)}
    second = {"x": np.asarray([1, 2, 3], dtype=np.int64)}

    assert fingerprint_pytree(first) != fingerprint_pytree(second)


def test_nested_lists_and_tuples_are_deterministic() -> None:
    first = {"x": [np.asarray([1.0]), (2, {"y": "z"})]}
    second = {"x": [np.asarray([1.0]), (2, {"y": "z"})]}

    assert fingerprint_pytree(first) == fingerprint_pytree(second)
