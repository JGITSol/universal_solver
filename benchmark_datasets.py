"""Utility helpers for loading HuggingFace math benchmark datasets.

The helpers in this module wrap ``datasets.load_dataset`` but provide
type-safe handling for the many container types that the HuggingFace
``datasets`` package may return (``Dataset``, ``DatasetDict``,
``IterableDataset`` …). Pylance reports attribute access issues whenever we
directly call methods such as ``select`` on a union of these container types.
To keep the experience smooth inside VS Code we normalise every dataset into
an in-memory list of dictionaries. This also gives us predictable behaviour
for sampling without depending on optional dataset methods.
"""

from __future__ import annotations

import random
from itertools import islice
from typing import Dict, Iterable, List, cast

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

# Supported datasets and their configs
BENCHMARK_DATASETS = {
    "math": {"hf_id": "hendrycks/math", "splits": ["test"]},
    "gsm8k": {"hf_id": "gsm8k", "splits": ["test"]},
    "mathqa": {"hf_id": "math_qa", "splits": ["test"]},
    "asdiv": {"hf_id": "asdiv", "splits": ["test"]},
    "svamp": {"hf_id": "svamp", "splits": ["test"]},
    "aqua_rat": {"hf_id": "aqua_rat", "splits": ["test"]},
    "minif2f": {"hf_id": "minif2f", "splits": ["test"]},
}


def list_benchmark_datasets():
    """Return a list of available math benchmark datasets."""
    return list(BENCHMARK_DATASETS.keys())


def load_benchmark_dataset(
    name: str,
    split: str = "test",
    sample_size: int | None = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Load a benchmark dataset and return it as a list of samples.

    HuggingFace datasets can materialise as several container types. We map
    them into a concrete list so that downstream code receives a predictable
    ``List[Dict[str, object]]``. The optional ``sample_size`` argument applies
    a deterministic sample using the provided ``seed`` without relying on the
    dataset's ``shuffle``/``select`` helpers (which are not available for all
    dataset variants).
    """
    if name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    config = BENCHMARK_DATASETS[name]
    dataset_raw = load_dataset(config["hf_id"], split=split)

    # Normalise to an iterable of samples for any HuggingFace container type.
    if isinstance(dataset_raw, (DatasetDict, IterableDatasetDict)):
        dataset_iterable = dataset_raw.get(split)
        if dataset_iterable is None:
            dataset_values = list(dataset_raw.values())
            dataset_iterable = dataset_values[0] if dataset_values else []
    else:
        dataset_iterable = dataset_raw

    samples: List[Dict[str, object]]
    if isinstance(dataset_iterable, Dataset):
        # ``Dataset`` already provides len and deterministic shuffling.
        if sample_size is not None:
            bounded = min(sample_size, len(dataset_iterable))
            dataset_iterable = dataset_iterable.shuffle(seed=seed).select(
                range(bounded)
            )
        samples = [cast(Dict[str, object], rec) for rec in dataset_iterable]
    elif isinstance(dataset_iterable, IterableDataset):
        # Streaming datasets are consumed lazily; take a prefix and optionally
        # shuffle it deterministically.
        if sample_size is None:
            samples = [cast(Dict[str, object], rec) for rec in dataset_iterable]
        else:
            prefix = list(islice(dataset_iterable, sample_size))
            random.Random(seed).shuffle(prefix)
            samples = [cast(Dict[str, object], rec) for rec in prefix[:sample_size]]
    else:
        # Already a concrete sequence (typically ``list``).
        iterable = cast(Iterable[Dict[str, object]], dataset_iterable)
        samples = [cast(Dict[str, object], rec) for rec in iterable]
        if sample_size is not None:
            random.Random(seed).shuffle(samples)
            samples = samples[:sample_size]

    return samples


def get_problem_and_answer(example, dataset_name):
    """Standardize access to problem and answer fields for each dataset."""
    if dataset_name == "math":
        return example["problem"], example["solution"]
    elif dataset_name == "gsm8k":
        return example["question"], example["answer"]
    elif dataset_name == "mathqa":
        return example["Problem"], example["Rationale"]
    elif dataset_name == "asdiv":
        return example["question"], example["answer"]
    elif dataset_name == "svamp":
        return example["Body"], example["Answer"]
    elif dataset_name == "aqua_rat":
        return example["question"], example["correct"]
    elif dataset_name == "minif2f":
        return example["problem"], example["solution"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    # Example: list datasets and sample 3 problems from each
    for ds_name in list_benchmark_datasets():
        print(f"\nDataset: {ds_name}")
        ds = load_benchmark_dataset(ds_name, sample_size=3)
        for ex in ds:
            prob, ans = get_problem_and_answer(ex, ds_name)
            print(f"Problem: {prob}\nAnswer: {ans}\n---")
