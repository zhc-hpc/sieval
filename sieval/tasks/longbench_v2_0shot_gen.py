"""LongBench v2 — 0-shot generative long-context MCQ, official scoring.

Prompt templates, middle-out truncation, and answer extraction are vendored
verbatim from THUDM/LongBench (v2) @ 2e00731 (see sieval/community/longbench_v2/).
Default cot=True replicates the official w/ CoT 2-turn flow (turn-1 reasons over
the doc via 0shot_cot.txt; turn-2 extracts the letter via the doc-less
0shot_cot_ans.txt) — the column the leaderboard reports for reasoning models.
Headline = overall accuracy; also broken down by difficulty and length
(result.py convention).

Truncation uses the SERVED MODEL's tokenizer (official does the same): the
filled prompt is cut middle-out to `max_len` tokens before inference.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

import os
from collections import defaultdict
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from sieval.community.longbench_v2.eval import (
    build_0shot_prompt,
    build_cot_ans_prompt,
    build_cot_prompt,
    extract_answer,
    truncate_middle,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import LongBenchV2DatasetSample

# Default tokenizer = the served DeepSeek-V4-Flash weights (override via task args
# `tokenizer_path` to match whatever model is being evaluated).
_DEFAULT_TOKENIZER = "/models/preset/deepseek-ai/DeepSeek-V4-Flash/v1.0"


class Feedback(TypedDict):
    correct: bool
    difficulty: str
    length: str
    pred: str


@sieval_task(
    name="longbench_v2_0shot_gen",
    display_name="LongBench v2 (0-shot, generative)",
    description="LongBench v2 — 503 long-context MCQ; accuracy by difficulty/length.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "multiple-choice", "long-context"),
    deps_group="longbench_v2",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="THUDM/LongBench",
        url=(
            "https://github.com/THUDM/LongBench/blob/"
            "2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py"
        ),
        notes=(
            "0-shot prompt (prompts/0shot.txt) + middle-out truncation + "
            "extract_answer vendored verbatim @ 2e00731. Headline = overall "
            "accuracy (result.py); broken down by difficulty and length."
        ),
    ),
)
class LongBenchV2ZeroShotGenTask(
    Task[
        LongBenchV2DatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        tokenizer_path: str = _DEFAULT_TOKENIZER,
        max_len: int = 120000,
        cot: bool = True,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self.max_len = max_len
        # cot=True replicates the official w/ CoT 2-turn flow (the column the
        # leaderboard reports for reasoning models); cot=False = no-CoT single turn.
        self.cot = cot
        # tokenizer for middle-out truncation (official truncates by model tokens).
        # Fall back to loading tokenizer.json directly when AutoConfig can't parse
        # a too-new architecture (e.g. DeepSeek-V4-Flash's `deepseek_v4` config).
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json")
            )

    @override
    async def preprocess(self, raw, ctx):
        # turn-1 prompt: w/ CoT uses 0shot_cot.txt, no-CoT uses 0shot.txt
        builder = build_cot_prompt if self.cot else build_0shot_prompt
        prompt = truncate_middle(builder(raw), self.tokenizer, self.max_len)
        return [{"role": "user", "content": prompt}]

    @override
    async def infer(self, pre, ctx):
        out = await self.model.agenerate(pre)
        if not self.cot:
            return out
        # w/ CoT turn-2: feed turn-1 response as $COT$ into the (doc-less)
        # answer-extraction prompt and re-query (official pred.py --cot flow).
        cot_text = out.texts[0]
        ans_prompt = build_cot_ans_prompt(ctx.raw_sample, cot_text)
        ans_prompt = truncate_middle(ans_prompt, self.tokenizer, self.max_len)
        return await self.model.agenerate([{"role": "user", "content": ans_prompt}])

    @override
    async def postprocess(self, inf, ctx):
        pred = extract_answer(inf.texts[0])  # extract from final (turn-2) content
        return pred or ""

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        return True, {
            "correct": post == raw["answer"],
            "difficulty": raw["difficulty"],
            "length": raw["length"],
            "pred": post,
        }

    @override
    async def report(self, finals, fails):
        by_diff = defaultdict(lambda: {"correct": 0, "total": 0})
        by_len = defaultdict(lambda: {"correct": 0, "total": 0})
        correct_num = 0
        for ctx in finals:
            fb = ctx.feedback_result
            ok = fb["correct"]
            correct_num += int(ok)
            by_diff[fb["difficulty"]]["correct"] += int(ok)
            by_diff[fb["difficulty"]]["total"] += 1
            by_len[fb["length"]]["correct"] += int(ok)
            by_len[fb["length"]]["total"] += 1

        results = {"score": 100 * correct_num / len(finals) if finals else 0.0}
        for bucket, metrics in {**by_diff, **by_len}.items():
            results[f"score_{bucket}"] = (
                100 * metrics["correct"] / metrics["total"]
                if metrics["total"]
                else 0.0
            )
        results["fails"] = len(fails)
        return results
