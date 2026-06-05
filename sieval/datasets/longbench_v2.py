"""LongBench v2 dataset loader (503 long-context MCQ).

Source = `zai-org/LongBench-v2` (the current HF home of THUDM/LongBench-v2).
Single `data.json`, 503 rows, contexts 8k–2M words.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset

LongBenchV2DatasetSample = TypedDict(
    "LongBenchV2DatasetSample",
    {
        "_id": str,
        "domain": str,
        "sub_domain": str,
        "difficulty": str,  # easy | hard
        "length": str,  # short | medium | long
        "question": str,
        "choice_A": str,
        "choice_B": str,
        "choice_C": str,
        "choice_D": str,
        "answer": str,  # gold letter A-D
        "context": str,
    },
)


@sieval_dataset(
    name="longbench_v2",
    display_name="LongBench v2",
    description=(
        "LongBench v2 — 503 long-context multiple-choice questions (8k–2M words)."
    ),
    source="hf:zai-org/LongBench-v2",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "multiple-choice", "long-context"),
    license="Apache-2.0",
)
class LongBenchV2Dataset(Dataset[LongBenchV2DatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, split="train", **kwargs)
        dataset = ensure_dataset(dataset)
        return HFDatasetDict({"train": dataset, "test": dataset})
