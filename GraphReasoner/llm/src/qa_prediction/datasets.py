import os
import json
from typing import Callable, Iterable, List, Any


class DatasetList(list):
    def map(self, func: Callable, remove_columns=None, num_proc: int = 1, **kwargs):
        # simple single-process map
        return DatasetList([func(x) for x in self])

    def filter(self, func: Callable, num_proc: int = 1):
        return DatasetList([x for x in self if func(x)])

    def to_json(self, path: str, orient='records', lines=True):
        with open(path, 'w', encoding='utf-8') as f:
            for item in self:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(path: str) -> Iterable[Any]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_dataset(input_path: str, split: str = 'test') -> DatasetList:
    """Load dataset from a directory containing <split>.json or from a direct JSONL file.

    Args:
        input_path: directory path or file path.
        split: split name (train/dev/test).

    Returns:
        DatasetList of samples (list-like with map/filter).
    """
    if os.path.isdir(input_path):
        path = os.path.join(input_path, f"{split}.json")
    else:
        # if user passed a file, use it directly
        path = input_path if input_path.endswith('.json') or input_path.endswith('.jsonl') else os.path.join(input_path, f"{split}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data = list(load_jsonl(path))
    return DatasetList(data)
