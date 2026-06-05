"""IFBench dataset loader (allenai/IFBench test set, 300 prompts).

Source = official HF mirror `allenai/IFBench_test` (revision
2e8a48de45ff3bf41242f927254ca81b59ca3ae2), verified byte-identical (300 rows,
same prompts + instruction_id_lists) to the in-repo data/IFBench_test.jsonl at
the vendored-verifier commit 1091c4c3de6c1f6ed12c012ed68f11ea450b0117. The HF
mirror is used (not the raw github jsonl) because the url: downloader mismatches
gzip Content-Length on raw.githubusercontent; the hf handler decompresses cleanly.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

from typing import Any, TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset


class IFBenchDatasetSample(TypedDict):
    key: str
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


@sieval_dataset(
    name="ifbench",
    display_name="IFBench",
    description=(
        "IFBench — 300 prompts with 58 new out-of-domain verifiable constraints."
    ),
    source="hf:allenai/IFBench_test",
    categories=(Category(Level1Category.LANGUAGE, "InstructionFollowing"),),
    tags=("english", "open-ended"),
    license="Apache-2.0",
)
class IFBenchDataset(Dataset[IFBenchDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = load_dataset(name_or_path, split="train", **kwargs)
        dataset = ensure_dataset(dataset)
        return HFDatasetDict({"train": dataset, "test": dataset})
