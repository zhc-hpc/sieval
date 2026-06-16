"""LiveCodeBench Code Generation 0-shot —— 【n_repeats 变体，不动原 task】

用 dataset.repeat(n_repeats) + 每请求 n=1 等效请求级 n，避开 sglang PD 对 n>1 的崩溃
（详见 /mnt/pqing/pdtest/PD_n_gt1_crash.md）。prompt/extract_code/沙箱判分与原 task 逐字一致；
打分=平均 pass 率(pass@1)，与原 task 口径等价。需 code-evaluator 沙箱(:11451)。
"""
import base64
import json
import os
import pickle
import time
import zlib
from typing import TypedDict, override

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.livecodebench.prompts.code_generation import (
    PromptConstants,
    get_generic_question_template_answer,
)
from sieval.community.livecodebench.utils.extraction_utils import extract_code
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import LiveCodeBenchDatasetSample


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
    name="livecodebench_code_generation_0shot_gen_rep",
    display_name="LiveCodeBench Code Generation (0-shot, n_repeats)",
    description="LCB code-gen 0-shot；dataset.repeat 把每题复制成独立 n=1 请求，避开 PD n>1 崩溃。",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="livecodebench",
        url="https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/prompts/code_generation.py",
        notes="与 livecodebench_code_generation_0shot_gen 逐字相同；仅 n=10 改为 dataset.repeat + n=1。",
    ),
)
class LiveCodeBenchCodeGenerationRepeatZeroShotGenTask(
    Task[
        LiveCodeBenchDatasetSample,
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
        cot: bool = False,
        n_repeats: int = 10,
        max_concurrency: int = 4,
        timeout: float = 6.0,
    ):
        expanded = dataset.repeat(n_repeats) if n_repeats > 1 else dataset
        super().__init__(dataset=expanded, model=model, name=name)
        self._cot = cot
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
        question = {
            "question_content": raw["question_content"],
            "starter_code": raw["starter_code"],
        }
        prompt = get_generic_question_template_answer(question, self._cot)
        return [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": prompt},
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=1)

    @override
    async def postprocess(self, inf, ctx):
        res: list[str] = []
        for choice in inf.texts:
            res.append(extract_code(choice))
        return res

    @override
    async def feedback(self, post, ctx):
        public_test_cases = json.loads(ctx.raw_sample["public_test_cases"])
        private_test_cases = ctx.raw_sample["private_test_cases"]
        try:
            private_test_cases = json.loads(ctx.raw_sample["private_test_cases"])
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(private_test_cases.encode("utf-8"))
                    )
                )
            )
        metadata = json.loads(ctx.raw_sample["metadata"])

        feedbacks = [
            {"correct": False, "msg": "Not evaluated", "metrics": None}
            for _ in range(len(post))
        ]
        cases = public_test_cases + private_test_cases
        inputs = [t["input"] for t in cases]
        outputs = [t["output"] for t in cases]
        fn_name = metadata.get("func_name", None)

        for idx, pred in enumerate(post):
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{idx}-{time.perf_counter_ns()}",
                        "source": "livecodebench",
                        "code": pred,
                        "test": {
                            "inputs": inputs,
                            "outputs": outputs,
                            "fn_name": fn_name,
                        },
                    },
                    timeout=self._timeout + len(inputs) * 2 + 2,
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
