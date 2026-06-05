"""GSM8K protocol unit tests — verify extraction/scoring align with lm-eval gsm8k_cot.

Reference: lm_eval/tasks/gsm8k/gsm8k-cot.yaml + filters/extraction.py
@ commit b733598568fde3afa0e008892f8e7bb18cd472b6.
"""

from sieval.community.lm_eval_harness import gsm8k as g


def test_prompt_has_8_shots_and_query():
    prompt = g.build_prompt("What is 2 + 2?")
    # 8 fixed exemplars all present, each ending "The answer is X."
    assert prompt.count("The answer is") == 8
    assert "There are 15 trees in the grove" in prompt  # first exemplar
    assert "Olivia has $23" in prompt  # last exemplar
    # query appended in Q:/A: form, ending with "A:" (model continues)
    assert prompt.rstrip().endswith("Q: What is 2 + 2?\nA:")


def test_strict_extract():
    # strict filter: regex `The answer is (...)`, first match
    assert g.extract_strict("...so 21 - 15 = 6. The answer is 6.") == "6"
    assert g.extract_strict("blah The answer is 1,234.") == "1,234"
    assert g.extract_strict("negative The answer is -8.") == "-8"
    # no "The answer is" marker -> fallback
    assert g.extract_strict("the result is 42 dollars") == "[invalid]"


def test_flexible_extract_last_number():
    # flexible filter: last number in the text (group_select -1)
    assert g.extract_flexible("I think 12 then 8 maybe 39 total") == "39"
    assert g.extract_flexible("costs $1,234 total") == "$1,234"
    assert g.extract_flexible("no digits here") == "[invalid]"


def test_gold_target_strips_hash():
    assert g.gold_target("Natalia sold ...\n#### 72") == "72"
    assert g.gold_target("work\n#### 1,234") == "1,234"


def test_is_correct_em_normalization():
    # exact_match after regexes_to_ignore (',', '$', up-to-####, trailing '.') + lowercase
    assert g.is_correct("1,234", "1234") is True  # comma ignored
    assert g.is_correct("8.", "8") is True  # trailing period ignored
    assert g.is_correct("$8", "8") is True  # dollar ignored
    assert g.is_correct("72", "72") is True
    assert g.is_correct("9", "8") is False
    assert g.is_correct("[invalid]", "8") is False  # fallback never matches


def test_end_to_end_extract_then_score():
    # model emits the gsm8k_cot answer phrasing; gold from dataset answer field
    completion = "She has 23 - 15 = 8 dollars left. The answer is 8."
    gold = g.gold_target("Olivia ... 23 - 15 is 8.\n#### 8")
    assert g.is_correct(g.extract_strict(completion), gold) is True
    assert g.is_correct(g.extract_flexible(completion), gold) is True
