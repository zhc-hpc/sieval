from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict


class GSM8KDatasetSample(TypedDict):
    question: str
    answer: str  # raw; final line is "#### <number>" (gold extracted via split('####'))


@sieval_dataset(
    name="gsm8k",
    display_name="GSM8K",
    description="Grade School Math 8K — grade-school math word problems (test=1319).",
    source="hf:openai/gsm8k",
    categories=(Category(Level1Category.MATHEMATICS, "ElementaryMath"),),
    tags=("english", "open-ended"),
    license="MIT",
)
class GSM8KDataset(Dataset[GSM8KDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # config "main"; keep raw answer (gold = answer.split('####')[-1]).
        dataset = load_dataset(name_or_path, name="main", **kwargs)
        return ensure_dataset_dict(dataset)
