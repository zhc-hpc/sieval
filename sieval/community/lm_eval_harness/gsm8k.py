"""GSM8K 8-shot CoT protocol, vendored verbatim from EleutherAI lm-evaluation-harness.

Source: lm_eval/tasks/gsm8k/gsm8k-cot.yaml + lm_eval/filters/extraction.py
Repo:   https://github.com/EleutherAI/lm-evaluation-harness  (Apache-2.0)
Pinned: commit b733598568fde3afa0e008892f8e7bb18cd472b6
Task:   gsm8k_cot (num_fewshot=8, metadata.version 3.0)

Everything alignment-critical is reproduced exactly so our numbers are comparable
to the standard `gsm8k_cot` protocol:
  - the 8 fixed CoT exemplars (each solution ends "The answer is X."),
  - the `Q: {question}\nA:` template + fewshot/target delimiters,
  - the strict-match + flexible-extract regex filters (RegexFilter semantics),
  - the gold target = answer.split("####")[-1].strip(),
  - exact_match scoring with regexes_to_ignore + ignore_case.
"""

import re

# --- fewshot_config.samples (verbatim) -------------------------------------
GSM8K_COT_FEWSHOT_SAMPLES: list[tuple[str, str]] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more "
        "were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
        "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and "
        "dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
        "then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now "
        "in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers "
        "were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer "
        "is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end of "
        "wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
        "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The "
        "answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does "
        "she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 "
        "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]

# lm-eval defaults: target_delimiter=" ", fewshot_delimiter="\n\n"
_TARGET_DELIM = " "
_FEWSHOT_DELIM = "\n\n"


def _doc_to_text(question: str) -> str:
    # doc_to_text: 'Q: {{question}}\nA:'
    return f"Q: {question}\nA:"


# Prebuilt 8-shot prefix (fixed for every question), reproducing lm-eval assembly:
#   fewshot_delimiter.join(doc_to_text(ex) + target_delimiter + target  for ex)
GSM8K_COT_FEWSHOT_PREFIX: str = _FEWSHOT_DELIM.join(
    _doc_to_text(q) + _TARGET_DELIM + t for q, t in GSM8K_COT_FEWSHOT_SAMPLES
)


def build_prompt(question: str) -> str:
    """Full 8-shot CoT prompt: fixed prefix + fewshot_delimiter + the query doc."""
    return GSM8K_COT_FEWSHOT_PREFIX + _FEWSHOT_DELIM + _doc_to_text(question)


# generation_kwargs.until (stop sequences)
STOP_SEQUENCES: tuple[str, ...] = ("Q:", "</s>", "<|im_end|>")

# --- filters (RegexFilter, verbatim patterns) ------------------------------
STRICT_PATTERN = r"The answer is (\-?[0-9\.\,]+)."
FLEXIBLE_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
_STRICT_RE = re.compile(STRICT_PATTERN)
_FLEXIBLE_RE = re.compile(FLEXIBLE_PATTERN)
_FALLBACK = "[invalid]"


def _regex_filter(text: str, regex: re.Pattern, group_select: int) -> str:
    """Replicate lm-eval RegexFilter.apply (extraction.py @ pinned commit)."""
    if not isinstance(text, str):
        text = ""
    matches = regex.findall(text)
    if not matches:
        return _FALLBACK
    match = matches[group_select]
    if isinstance(match, tuple):
        nonempty = [m for m in match if m]
        match = nonempty[0] if nonempty else _FALLBACK
    return match.strip()


def extract_strict(text: str) -> str:
    """strict-match filter: regex `The answer is (...)`, group_select 0, take_first."""
    return _regex_filter(text, _STRICT_RE, 0)


def extract_flexible(text: str) -> str:
    """flexible-extract filter: last number in text (group_select -1), take_first."""
    return _regex_filter(text, _FLEXIBLE_RE, -1)


def gold_target(answer: str) -> str:
    """doc_to_target: answer.split('####')[-1].strip()."""
    return answer.split("####")[-1].strip()


# --- exact_match metric (regexes_to_ignore + ignore_case) ------------------
# metric_list: exact_match, ignore_case=true, ignore_punctuation=false,
#   regexes_to_ignore: [',', '\$', '(?s).*#### ', '\.$']
_REGEXES_TO_IGNORE = [r",", r"\$", r"(?s).*#### ", r"\.$"]


def _em_normalize(s: str) -> str:
    for rgx in _REGEXES_TO_IGNORE:
        s = re.sub(rgx, "", s)
    return s.lower()  # ignore_case=true; ignore_punctuation=false


def is_correct(prediction: str, gold: str) -> bool:
    """exact_match after regexes_to_ignore + lowercasing, on both pred and gold."""
    return _em_normalize(prediction) == _em_normalize(gold)


# --- reasoning-model robust extraction (deviation from lm-eval; documented) --------
# Everything ABOVE this line is vendored verbatim from lm-eval gsm8k_cot and is left
# untouched. Reasoning/instruct models (e.g. Qwen3-Thinking) frequently ignore the
# few-shot "The answer is X." convention and emit the final answer as LaTeX
# `\boxed{...}` or "Final Answer: N". lm-eval's flexible regex then grabs the trailing
# "$$" or a mid-step number -> large undercount (measured: 82.34 vs true ~96.3 on Qwen).
#
# `extract_flexible_robust` layers a priority tier ON TOP of the verbatim filter:
#   1. last `\boxed{...}` number, else
#   2. "Final Answer: N" / "the answer is N", else
#   3. fall back to lm-eval `extract_flexible` (verbatim) -- identical for non-marker text.
# It only overrides outputs that actually carry a boxed/final marker. Validated on the
# Qwen GSM8K run (1319 samples) vs lm-eval flexible: 184 fixes, 0 regressions.
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(
    r"(?:final answer|the answer is)[:\s\*]*\\?\\?[\$]*\s*\{?(-?[0-9][0-9,]*\.?[0-9]*)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")


def _last_number(s: str) -> str:
    nums = _NUM_RE.findall(s)
    return nums[-1] if nums else ""


def extract_flexible_robust(text: str) -> str:
    """Reasoning-model-aware flexible extraction (boxed/final priority, lm-eval fallback)."""
    if not isinstance(text, str):
        return _FALLBACK
    boxed = _BOXED_RE.findall(text)
    if boxed:
        num = _last_number(boxed[-1])
        if num:
            return num
    final = _FINAL_RE.findall(text)
    if final:
        return final[-1]
    return extract_flexible(text)
