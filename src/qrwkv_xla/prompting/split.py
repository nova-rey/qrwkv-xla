from __future__ import annotations

import random
from dataclasses import replace

from qrwkv_xla.prompting.corpus import PromptCorpus


def assign_splits(
    corpus: PromptCorpus,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> PromptCorpus:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be >= 0 and < 1")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    count = len(corpus.records)
    indices = list(range(count))
    random.Random(seed).shuffle(indices)

    validation_count = int(count * validation_fraction)
    if validation_fraction > 0 and count > 1:
        validation_count = max(1, validation_count)

    test_count = int(count * test_fraction)
    if test_fraction > 0 and count > 2:
        test_count = max(1, test_count)

    while validation_count + test_count >= count and count > 0:
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break

    validation_indices = set(indices[:validation_count])
    test_indices = set(indices[validation_count : validation_count + test_count])

    records = []
    for index, record in enumerate(corpus.records):
        if index in validation_indices:
            split = "validation"
        elif index in test_indices:
            split = "test"
        else:
            split = "train"
        records.append(replace(record, split=split))

    return PromptCorpus(
        corpus_id=corpus.corpus_id,
        records=tuple(records),
        source_path=corpus.source_path,
    )
