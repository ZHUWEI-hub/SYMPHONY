"""Utility functions for code generation experiments."""

import os
import gzip
import json
import openai
import jsonlines
from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")


def make_printv(verbose: bool):
    """Create a verbose print function.
    
    Args:
        verbose: Whether to enable verbose printing
        
    Returns:
        Print function that respects verbose flag
    """
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v


def read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries from the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a JSONL file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def write_jsonl(path: str, data: List[dict], append: bool = False):
    """Write data to a JSONL file.
    
    Args:
        path: Path to the output file
        data: List of dictionaries to write
        append: Whether to append to existing file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl_gz(path: str) -> List[dict]:
    """Read a gzipped JSONL file.
    
    Args:
        path: Path to the .jsonl.gz file
        
    Returns:
        List of dictionaries from the file
        
    Raises:
        ValueError: If file is not a .jsonl.gz file
    """
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data


def enumerate_resume(dataset, results_path):
    """Enumerate dataset items, resuming from previous progress.
    
    If results_path exists, skip items that have been processed before.
    
    Args:
        dataset: List of dataset items
        results_path: Path to results file
        
    Yields:
        Tuple of (index, item) for unprocessed items
    """
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1

        for i, item in enumerate(dataset):
            if i < count:
                continue
            yield i, item


def resume_success_count(dataset) -> int:
    """Count the number of solved items in a dataset.
    
    Args:
        dataset: List of dataset items
        
    Returns:
        Number of items marked as solved
    """
    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count

