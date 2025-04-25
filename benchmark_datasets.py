"""
benchmark_datasets.py

Industry-standard math datasets loader for benchmarking advanced math solvers (April 2025).
"""
from datasets import load_dataset
import random

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

def load_benchmark_dataset(name, split="test", sample_size=None, seed=42):
    """Load a benchmark dataset (optionally sample a subset)."""
    if name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    config = BENCHMARK_DATASETS[name]
    ds = load_dataset(config["hf_id"], split=split)
    if sample_size is not None:
        ds = ds.shuffle(seed=seed).select(range(sample_size))
    return ds

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
