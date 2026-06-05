"""OpenBookQA dataset loader (closed-book `main` config).

The `main` config carries no `fact` field, so evaluation is closed-book by
construction. Raw fields are preserved verbatim; presentation into the
simple-evals MCQ template happens in the task's preprocess.

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
from sieval.core.utils.hf import apply_eval_split, ensure_dataset_dict


class OpenBookQADatasetSample(TypedDict):
    id: str
    question_stem: str
    choices: dict
    answerKey: str


@sieval_dataset(
    name="openbookqa",
    display_name="OpenBookQA",
    description=(
        "OpenBookQA — elementary-science MCQ (main config, closed-book, 4 options)."
    ),
    source="hf:allenai/openbookqa",
    categories=(Category(Level1Category.KNOWLEDGE, "STEM"),),
    tags=("english", "multiple-choice"),
    license="Apache-2.0",
)
class OpenBookQADataset(Dataset[OpenBookQADatasetSample]):
    @override
    def load(
        self,
        name_or_path: str,
        eval_split: str | None = None,
        **kwargs,
    ) -> HFDatasetDict:
        # "main" config is closed-book (no `fact` field); "additional" adds facts.
        dataset = load_dataset(name_or_path, "main", **kwargs)
        dataset = ensure_dataset_dict(dataset)
        dataset = apply_eval_split(dataset, eval_split)
        return dataset
