"""HumanEval 0-shot 生成式 —— 【n_repeats 变体，不动原 human_eval_0shot_gen】

用 dataset.repeat(n_repeats) + 每请求 n=1 等效请求级 n，避开 sglang PD 对 n>1 的崩溃
（详见 /mnt/pqing/pdtest/PD_n_gt1_crash.md）。prompt/抽取/沙箱判分与原 task 逐字一致；
打分=平均 pass 率(pass@1)，与原 task 口径等价。需 code-evaluator 沙箱(:11451)。
"""
import os
import re
import time
from typing import TypedDict, override

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.simple_evals.humaneval_eval import QUERY_TEMPLATE
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import HumanEvalDatasetSample


class ResourceMetrics(TypedDict):
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float


class Feedback(TypedDict):
    correct: bool
    msg: str
    metrics: ResourceMetrics | None


@sieval_task(
    name="human_eval_0shot_gen_rep",
    display_name="HumanEval (0-shot, generative, n_repeats)",
    description="HumanEval 0-shot；dataset.repeat 把每题复制成独立 n=1 请求，避开 PD n>1 崩溃。",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/humaneval_eval.py",
        notes="与 human_eval_0shot_gen 逐字相同；仅 n=10 改为 dataset.repeat + n=1。",
    ),
)
class HumanEvalRepeatZeroShotGenTask(
    Task[
        HumanEvalDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[str],
        list[Feedback],
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        n_repeats: int = 10,
        max_concurrency: int = 4,
        timeout: float = 5.0,
    ):
        expanded = dataset.repeat(n_repeats) if n_repeats > 1 else dataset
        super().__init__(dataset=expanded, model=model, name=name)
        self._n_repeats = n_repeats
        self._max_concurrency = max_concurrency
        self._timeout = timeout
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def preprocess(self, raw, ctx):
        return [
            {"role": "user", "content": QUERY_TEMPLATE.format(prompt=raw["prompt"])}
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=1)

    @override
    async def postprocess(self, inf, ctx):
        res: list[str] = []
        for choice in inf.texts:
            pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = pattern.findall(choice)
            extracted_answer = matches[0] if len(matches) >= 1 else choice
            extracted_answer = extracted_answer[
                extracted_answer.find(":\n    ") + 2 :
            ]  # remove signature
            res.append(extracted_answer)
        return res

    @override
    async def feedback(self, post, ctx):
        feedbacks = [
            {"correct": False, "msg": "Not evaluated", "metrics": None}
            for _ in range(len(post))
        ]
        for idx, pred in enumerate(post):
            check_program = (
                ctx.raw_sample["prompt"]
                + pred
                + "\n"
                + ctx.raw_sample["test"]
                + "\n"
                + f"check({ctx.raw_sample['entry_point']})"
            )
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "human-eval",
                        "code": check_program,
                    },
                    timeout=self._timeout + 2,
                )
                resp.raise_for_status()
                res = resp.json()
                feedbacks[idx] = {
                    "correct": res["status"],
                    "msg": res["msg"],
                    "metrics": res["data"],
                }
            except Exception as e:
                logger.warning(
                    "Evaluation error for sample {}: [{}] {}", idx, type(e).__name__, e
                )
                raise e
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        # 每行 1 样本；平均 pass 率 = pass@1，与原 task 口径一致。
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}
        correct = 0
        timeouts = 0
        for f in finals:
            fb = f.feedback_result
            correct += sum(1 for x in fb if x["correct"])
            timeouts += sum(1 for x in fb if "timeout" in x["msg"].lower())
        score = correct * 100 / total
        return {"score": score, "fails": len(fails), "timeouts": timeouts,
                "pass@1": score, "n_repeats": self._n_repeats}

    @override
    async def shutdown(self):
        await self._http_client.aclose()
