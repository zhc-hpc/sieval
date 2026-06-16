"""AIME 2025 0-shot 生成式 —— 【n_repeats 变体，不动原 aime_2025_0shot_gen】

目的：在 PD 分离后端上，把"请求级 n=10"换成"dataset.repeat(10) + 每请求 n=1"，
即用 N 条独立 n=1 请求等效 n=10，避开 sglang PD 对 n>1(一个 bootstrap_room 多序列)
的崩溃 bug（详见 /mnt/pqing/pdtest/PD_n_gt1_crash.md）。

打分口径与原 aime 等价：原 report 的 pass@1 = Σ_question (correct/n_samples)/total；
这里每个 repeat 是独立 1 样本行，pass@1 = correct/1，汇总 = 总对/总行 = 同样的平均正确率。
prompt / 答案抽取(ANSWER_PATTERN) / 判分(math_verify) 全部与原 aime 逐字一致。
"""
import re
from typing import TypedDict, override

from loguru import logger
from math_verify import parse, verify
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.simple_evals.common import ANSWER_PATTERN
from sieval.community.simple_evals.math_eval import QUERY_TEMPLATE
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import AIME2025DatasetSample


class Feedback(TypedDict):
    correct: bool
    answer: str


@sieval_task(
    name="aime_2025_0shot_gen_rep",
    display_name="AIME 2025 (0-shot, generative, n_repeats)",
    description=(
        "AIME 2025 0-shot；用 dataset.repeat 把每题复制成多条独立 n=1 请求(等效 n)，避开 PD n>1 崩溃。"
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py",
        notes="与 aime_2025_0shot_gen 逐字相同的 prompt/抽取/判分；仅把 n=10 改为 dataset.repeat + n=1。",
    ),
)
class AIME2025RepeatZeroShotGenTask(
    Task[
        AIME2025DatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[str | None],
        list[Feedback],
        dict[str, float],
    ],
):
    def __init__(self, dataset, model, name: str | None = None, n_repeats: int = 10):
        # 每题复制 n_repeats 行 → 每行将作为独立的 n=1 请求被 runner 并发跑。
        expanded = dataset.repeat(n_repeats) if n_repeats > 1 else dataset
        super().__init__(dataset=expanded, model=model, name=name)
        self._n_repeats = n_repeats

    @override
    async def preprocess(self, raw, ctx):
        return [
            {"role": "user", "content": QUERY_TEMPLATE.format(problem=raw["question"])},
        ]

    @override
    async def infer(self, pre, ctx):
        # 关键：每条请求 n=1（独立 bootstrap_room），不走 PD 的 n>1 崩溃路径。
        return await self.model.agenerate(pre, n=1)

    @override
    async def postprocess(self, inf, ctx):
        res = []
        for choice in inf.texts:
            match = re.search(ANSWER_PATTERN, choice)
            res.append(match.group(1).strip() if match else None)
        return res

    @override
    async def feedback(self, post, ctx):
        feedbacks: list[Feedback] = []
        ground_truth = ctx.raw_sample["answer"]
        for pred in post:
            if pred is None:
                feedbacks.append({"correct": False, "answer": ground_truth})
                continue
            try:
                correct = verify(parse(f"${pred}$"), parse(f"${ground_truth}$"))
            except Exception as e:  # noqa: BLE001
                logger.warning("Feedback failed for sample {}: {}", ctx.sample_id, e)
                correct = False
            feedbacks.append({"correct": correct, "answer": ground_truth})
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        # 每行(一次重复)是独立 1 样本；汇总平均正确率 = pass@1，与原 aime 口径一致。
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}
        correct = 0
        for f in finals:
            fb = f.feedback_result
            correct += sum(1 for x in fb if x["correct"])
        # n=1/行 → len(fb)==1，correct 即每行 0/1；除以 total 得平均正确率。
        score = correct * 100 / total
        return {"score": score, "fails": len(fails), "pass@1": score, "n_repeats": self._n_repeats}
