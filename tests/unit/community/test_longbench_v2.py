"""LongBench v2 unit tests — extraction/prompt/truncation vs THUDM/LongBench @ 2e00731.

extract_answer cases mirror pred.py:extract_answer; truncate_middle mirrors the
middle-out logic in pred.py:query_llm.
"""

from sieval.community.longbench_v2.eval import (
    build_0shot_prompt,
    build_cot_ans_prompt,
    build_cot_prompt,
    extract_answer,
    truncate_middle,
)

_ITEM = {
    "context": "Some long document.",
    "question": "What is X?",
    "choice_A": "a",
    "choice_B": "b",
    "choice_C": "c",
    "choice_D": "d",
}


def test_extract_answer_parenthesized():
    assert extract_answer("The correct answer is (B)") == "B"
    assert extract_answer("Reasoning...\nThe correct answer is (D).") == "D"


def test_extract_answer_fallback_and_star_strip():
    assert extract_answer("The correct answer is C") == "C"  # no-paren fallback
    assert extract_answer("**The correct answer is (A)**") == "A"  # '*' stripped


def test_extract_answer_none():
    assert extract_answer("I am not sure about this one.") is None
    assert extract_answer("The answer: B") is None  # wrong phrasing -> None


def test_build_0shot_prompt():
    item = {
        "context": "  Some long document.  ",
        "question": " What is X? ",
        "choice_A": " a ",
        "choice_B": " b ",
        "choice_C": " c ",
        "choice_D": " d ",
    }
    p = build_0shot_prompt(item)
    assert "Some long document." in p
    assert "What is X?" in p
    assert "(A) a" in p and "(D) d" in p
    assert 'The correct answer is (insert answer here)' in p


class _FakeTok:
    # char-level tokenizer: encode -> list of chars, decode -> join
    def encode(self, text):
        return list(text)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


def test_truncate_middle_keeps_head_and_tail():
    tok = _FakeTok()
    prompt = "H" * 100 + "M" * 100 + "T" * 100  # 300 chars
    out = truncate_middle(prompt, tok, max_len=100)
    # keeps first 50 + last 50 tokens, middle dropped
    assert len(out) == 100
    assert out.startswith("H" * 50)
    assert out.endswith("T" * 50)
    assert "M" not in out


def test_truncate_middle_noop_when_short():
    tok = _FakeTok()
    prompt = "short prompt"
    assert truncate_middle(prompt, tok, max_len=1000) == prompt


def test_build_cot_prompt_turn1_has_doc_and_step_by_step():
    p = build_cot_prompt(_ITEM)
    assert "Some long document." in p  # turn-1 includes the doc
    assert "(A) a" in p and "(D) d" in p
    assert p.rstrip().endswith("Let’s think step by step:")  # curly apostrophe


def test_build_cot_ans_prompt_turn2_omits_doc_and_injects_cot():
    cot = "First I considered A, then B... so the answer is c."
    p = build_cot_ans_prompt(_ITEM, cot)
    assert "Some long document." not in p  # turn-2 omits the doc
    assert "The text is too long and omitted here." in p
    assert cot in p  # $COT$ injected
    assert "What is X?" in p and "(C) c" in p
    assert 'The correct answer is (insert answer here)' in p
