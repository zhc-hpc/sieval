"""LongBench v2 prompt / extraction / truncation, vendored from THUDM/LongBench
@ 2e00731 (repo-root `pred.py` + `prompts/0shot.txt`). See package __init__.

Verbatim: TEMPLATE_0SHOT (prompts/0shot.txt), extract_answer, the prompt-fill
and middle-out token truncation from pred.py:query_llm. Only repackaged as
importable functions (no logic change).
"""

import re

# verbatim from prompts/0shot.txt
TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


# verbatim from prompts/0shot_cot.txt (turn-1 of the w/ CoT flow)
TEMPLATE_0SHOT_COT = """Please read the following text and answer the questions below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let’s think step by step:"""

# verbatim from prompts/0shot_cot_ans.txt (turn-2; NO $DOC$ — text omitted)
TEMPLATE_0SHOT_COT_ANS = """Please read the following text and answer the questions below.

The text is too long and omitted here.

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let’s think step by step: $COT$

Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""


def _fill(template: str, item: dict) -> str:
    return (
        template.replace("$DOC$", item["context"].strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def build_0shot_prompt(item: dict) -> str:
    """Fill the no-CoT 0-shot template (verbatim substitution from pred.py)."""
    return _fill(TEMPLATE_0SHOT, item)


def build_cot_prompt(item: dict) -> str:
    """Turn-1 of the w/ CoT flow: 0shot_cot.txt filled (includes the doc)."""
    return _fill(TEMPLATE_0SHOT_COT, item)


def build_cot_ans_prompt(item: dict, cot: str) -> str:
    """Turn-2 of the w/ CoT flow: 0shot_cot_ans.txt (no doc) + the turn-1 CoT.

    Mirrors pred.py: $DOC$ is still .replace()'d (a no-op — the template omits
    the text) and $COT$ is the turn-1 response.
    """
    return _fill(TEMPLATE_0SHOT_COT_ANS, item).replace("$COT$", cot)


def truncate_middle(prompt: str, tokenizer, max_len: int) -> str:
    """Middle-out truncation, verbatim from pred.py:query_llm (HF-tokenizer branch).

    Keeps the first ``max_len//2`` and last ``max_len//2`` tokens.
    """
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt


def extract_answer(response: str):
    """Verbatim from pred.py:extract_answer. Returns 'A'-'D' or None."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None
