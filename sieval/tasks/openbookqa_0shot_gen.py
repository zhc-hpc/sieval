"""OpenBookQA — 0-shot generative MCQ, aligned to simple-evals MMLU extraction.

prompt template / answer extraction / scoring reuse the vendored simple-evals
MCQ protocol (`community/simple_evals/common.py`), identical to `mmlu_0shot_gen`.

Deviation (documented in baseline alignment.md): OpenBookQA's canonical protocol
is loglikelihood multiple-choice (lm-eval); a thinking chat model served over a
generation endpoint cannot score loglikelihoods, and we keep one consistent MCQ
protocol across our MMLU/GPQA/MMLU-Pro/OpenBookQA cells.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

import re
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.simple_evals.common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    QUERY_TEMPLATE_MULTICHOICE,
    normalize_extracted_answer,
    normalize_response,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import OpenBookQADatasetSample


class Feedback(TypedDict):
    correct: bool
    answer: str


def _letters(raw: OpenBookQADatasetSample) -> tuple[list[str], str]:
    """Return (option_texts_in_ABCD_order, gold_letter).

    `main` config uses label=["A","B","C","D"] in option order; we map robustly
    so a digit-labelled variant still resolves the gold to the canonical letter.
    """
    texts = list(raw["choices"]["text"])
    labels = [str(lab).strip() for lab in raw["choices"]["label"]]
    answer_key = str(raw["answerKey"]).strip()
    try:
        gold_idx = labels.index(answer_key)
    except ValueError:
        gold_idx = int(answer_key) if answer_key.isdigit() else 0
    gold_letter = chr(65 + gold_idx) if 0 <= gold_idx < 26 else ""
    return texts, gold_letter


@sieval_task(
    name="openbookqa_0shot_gen",
    display_name="OpenBookQA (0-shot, generative)",
    description="OpenBookQA — elementary-science MCQ, closed-book, generative.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "multiple-choice"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url=(
            "https://github.com/openai/simple-evals/blob/"
            "ee3b0318d8d1d9d72755a4120879be65f7c07e9e/common.py"
        ),
        notes=(
            "Closed-book 0-shot generative MCQ; prompt template + letter "
            "extraction aligned with simple-evals MMLU. OpenBookQA's canonical "
            "protocol is loglikelihood-MC (lm-eval); generative chosen for "
            "thinking chat models and one consistent MCQ protocol across cells."
        ),
    ),
)
class OpenBookQAZeroShotGenTask(
    Task[
        OpenBookQADatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        texts, _ = _letters(raw)
        opts = (texts + ["", "", "", ""])[:4]
        data = {
            "Question": raw["question_stem"],
            "A": opts[0],
            "B": opts[1],
            "C": opts[2],
            "D": opts[3],
        }
        return [
            {"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE.format(**data)},
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        response_text = normalize_response(inf.texts[0])  # n=1, only one choice
        extracted_answer = ""
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response_text)
            if match:
                extracted_answer = normalize_extracted_answer(match.group(1))
                break
        return extracted_answer

    @override
    async def feedback(self, post, ctx):
        _, gold_letter = _letters(ctx.raw_sample)
        return True, {
            "correct": post == gold_letter,
            "answer": gold_letter,
        }

    @override
    async def report(self, finals, fails):
        correct_num = sum(1 for ctx in finals if ctx.feedback_result["correct"])
        score = 100 * correct_num / len(finals) if finals else 0.0
        return {"score": score, "fails": len(fails)}
