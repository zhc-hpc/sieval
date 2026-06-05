"""IFBench — 0-shot generative instruction following, official strict/loose scoring.

Verifiers + strict/loose prompt-level/instruction-level scoring are vendored
verbatim from allenai/IFBench @ 1091c4c (see sieval/community/ifbench/).
Headline score = loose prompt-level accuracy (the metric the IFBench paper reports).

AI-Generated Code - Opus 4.8 (Anthropic)
"""

from collections import defaultdict
from typing import override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.ifbench.evaluation_lib import (
    InputExample,
    OutputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import IFBenchDatasetSample


@sieval_task(
    name="ifbench_0shot_gen",
    display_name="IFBench (0-shot, generative)",
    description=(
        "IFBench — 300 prompts with 58 new out-of-domain verifiable constraints."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="ifbench",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="allenai/IFBench",
        url=(
            "https://github.com/allenai/IFBench/blob/"
            "1091c4c3de6c1f6ed12c012ed68f11ea450b0117/evaluation_lib.py"
        ),
        notes=(
            "instructions + instructions_registry + instructions_util + "
            "evaluation_lib vendored verbatim from allenai/IFBench @ 1091c4c "
            "(only flat imports rewritten to package-relative). Headline = "
            "loose prompt-level accuracy (the metric the IFBench paper reports)."
        ),
    ),
)
class IFBenchZeroShotGenTask(
    Task[
        IFBenchDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        str,
        dict[str, float],
    ]
):
    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    @override
    async def preprocess(self, raw, ctx):
        return [{"role": "user", "content": raw["prompt"]}]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        return inf.texts[0]  # n=1; verify against final content (not reasoning)

    @override
    async def feedback(self, post, ctx):
        return True, post  # pass response through to report-stage verification

    @override
    async def report(self, finals, fails):
        inputs = [
            InputExample(
                key=f.raw_sample["key"],
                instruction_id_list=f.raw_sample["instruction_id_list"],
                prompt=f.raw_sample["prompt"],
                kwargs=self._clean_kwargs(f.raw_sample["kwargs"]),
            )
            for f in finals
        ]
        prompt_to_response = {f.raw_sample["prompt"]: f.feedback_result for f in finals}
        results = {"fails": len(fails)}
        for func, grade in [
            (test_instruction_following_strict, "strict"),
            (test_instruction_following_loose, "loose"),
        ]:
            outputs = [func(inp, prompt_to_response) for inp in inputs]
            follow_all_instructions = [o.follow_all_instructions for o in outputs]
            accuracy = sum(follow_all_instructions) / len(outputs) if outputs else 0.0
            results[f"{grade}_accuracy"] = accuracy * 100

            report = self._get_report(outputs)
            results[f"{grade}_prompt_level_accuracy"] = (
                report.get("prompt-level", 0.0) * 100
            )
            results[f"{grade}_instruction_level_accuracy"] = (
                report.get("instruction-level", 0.0) * 100
            )
        # headline = loose prompt-level accuracy (the metric the IFBench paper
        # reports; strict + instruction-level kept in report for reference)
        results["score"] = results["loose_prompt_level_accuracy"]
        return results

    def _clean_kwargs(self, kwargs):
        # avoid hf datasets underlying Arrow sparse struct problem
        return [{k: v for k, v in d.items() if v is not None} for d in kwargs]

    def _get_report(self, outputs: list[OutputExample]) -> dict[str, float]:
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        tier0_total = defaultdict(int)
        tier0_correct = defaultdict(int)
        tier1_total = defaultdict(int)
        tier1_correct = defaultdict(int)

        for example in outputs:
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

            for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list, strict=True
            ):
                tier0 = instruction_id.split(":")[0]
                tier0_total[tier0] += 1
                if followed_or_not:
                    tier0_correct[tier0] += 1
                tier1_total[instruction_id] += 1
                if followed_or_not:
                    tier1_correct[instruction_id] += 1

        return {
            "prompt-level": prompt_correct / prompt_total if prompt_total else 0.0,
            "instruction-level": (
                instruction_correct / instruction_total if instruction_total else 0.0
            ),
        }
