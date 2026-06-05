from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.lm_eval_harness.gsm8k import (
    build_prompt,
    extract_flexible,
    extract_flexible_robust,
    extract_strict,
    gold_target,
    is_correct,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import GSM8KDatasetSample


class Feedback(TypedDict):
    strict_correct: bool
    flexible_correct: bool
    robust_correct: bool
    pred_strict: str
    pred_flexible: str
    pred_robust: str
    gold: str


@sieval_task(
    name="gsm8k_8shot_gen",
    display_name="GSM8K (8-shot CoT, generative)",
    description="Grade School Math 8K, 8-shot CoT; protocol aligned to lm-eval gsm8k_cot.",
    eval_mode=EvalMode.GEN,
    n_shot=8,
    tags=("english", "math"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/"
            "b733598568fde3afa0e008892f8e7bb18cd472b6/lm_eval/tasks/gsm8k/gsm8k-cot.yaml"
        ),
        notes=(
            "8-shot CoT exemplars + Q:/A: prompt, strict + flexible regex extraction, "
            "and exact_match scoring vendored verbatim from lm-eval gsm8k_cot."
        ),
    ),
)
class GSM8KFewShotGenTask(
    Task[
        GSM8KDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[dict[str, str]],
        list[Feedback],
        dict[str, float],
    ]
):
    """GSM8K 8-shot chain-of-thought, aligned to lm-eval `gsm8k_cot`.

    Headline metric = robust-extract pass@1. The lm-eval strict/flexible regex filters
    assume the model continues the few-shot "The answer is X." style; reasoning models
    (e.g. Qwen3-Thinking) instead emit `\\boxed{...}` / "Final Answer: N", which those
    regexes mis-extract (grab the trailing "$$") — a measurement artifact, not a
    capability gap. `extract_flexible_robust` adds a boxed/final priority tier on top of
    the verbatim lm-eval flexible filter (boxed -> final -> lm-eval fallback). The pure
    lm-eval `flexible_extract_acc` and `strict_match_acc` are still reported for
    comparability to standard gsm8k_cot. Sampling uses the model-recommended temp/top_p
    from the leaderboard YAML (our baseline standard), which deviates from lm-eval's
    greedy `do_sample=false` — documented in the cell's alignment.md.
    """

    def __init__(self, dataset, model, name: str | None = None, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        return [{"role": "user", "content": build_prompt(raw["question"])}]

    @override
    async def infer(self, pre, ctx):
        # NOTE: lm-eval gsm8k_cot uses stop=["Q:", ...]. We DROP the "Q:" stop for this
        # thinking model: its reasoning quotes the few-shot exemplars (which contain
        # "Q:"), so a "Q:" stop fires mid-reasoning and truncates output (empty content).
        # The chat model emits its answer then stops at EOS on its own. Documented
        # deviation; prompt/exemplars/extraction/scoring stay aligned to lm-eval.
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Apply both lm-eval filters to the FULL generation (reasoning + content):
        # lm-eval extracts from the whole model output; for this reasoning model the
        # final answer may land in reasoning_content rather than content.
        res: list[dict[str, str]] = []
        for i, content in enumerate(inf.texts):
            reasoning = inf.reasoning_texts[i] if i < len(inf.reasoning_texts) else ""
            full = (reasoning or "") + "\n" + (content or "")
            res.append(
                {
                    "strict": extract_strict(full),
                    "flexible": extract_flexible(full),
                    "robust": extract_flexible_robust(full),
                }
            )
        return res

    @override
    async def feedback(self, post, ctx):
        gold = gold_target(ctx.raw_sample["answer"])
        feedbacks: list[Feedback] = []
        for p in post:
            feedbacks.append(
                {
                    "strict_correct": is_correct(p["strict"], gold),
                    "flexible_correct": is_correct(p["flexible"], gold),
                    "robust_correct": is_correct(p["robust"], gold),
                    "pred_strict": p["strict"],
                    "pred_flexible": p["flexible"],
                    "pred_robust": p["robust"],
                    "gold": gold,
                }
            )
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}
        strict_total = 0.0
        flex_total = 0.0
        robust_total = 0.0
        for f in finals:
            fbs = f.feedback_result
            n = len(fbs)
            strict_total += sum(1 for x in fbs if x["strict_correct"]) / n
            flex_total += sum(1 for x in fbs if x["flexible_correct"]) / n
            robust_total += sum(1 for x in fbs if x["robust_correct"]) / n
        flexible_acc = flex_total * 100 / total
        strict_acc = strict_total * 100 / total
        robust_acc = robust_total * 100 / total
        return {
            "score": robust_acc,  # headline = robust-extract (boxed/final aware; see task docstring)
            "fails": len(fails),
            "robust_extract_acc": robust_acc,
            "flexible_extract_acc": flexible_acc,  # pure lm-eval, for gsm8k_cot comparability
            "strict_match_acc": strict_acc,
        }
