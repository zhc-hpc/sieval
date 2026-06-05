"""OpenBookQA task unit tests — option mapping + simple-evals MCQ protocol.

Verifies closed-book choice→ABCD mapping, gold resolution, and that the
vendored simple-evals letter extraction (shared with mmlu_0shot_gen) applies.
"""

from types import SimpleNamespace

import pytest

from sieval.tasks.openbookqa_0shot_gen import OpenBookQAZeroShotGenTask, _letters

pytestmark = pytest.mark.anyio


def _sample(answer_key="B", labels=("A", "B", "C", "D")):
    return {
        "id": "8-1",
        "question_stem": "What is the main source of energy for Earth?",
        "choices": {
            "text": ["the moon", "the sun", "the wind", "the soil"],
            "label": list(labels),
        },
        "answerKey": answer_key,
    }


def test_letters_letter_labels():
    texts, gold = _letters(_sample(answer_key="B"))
    assert texts == ["the moon", "the sun", "the wind", "the soil"]
    assert gold == "B"


def test_letters_digit_labels_map_to_canonical():
    # some OpenBookQA variants use numeric labels; gold must resolve by position
    texts, gold = _letters(_sample(answer_key="3", labels=("1", "2", "3", "4")))
    assert texts[2] == "the wind"
    assert gold == "C"  # answerKey "3" is the 3rd option -> C


async def test_preprocess_builds_closed_book_mcq():
    msgs = await OpenBookQAZeroShotGenTask.preprocess(None, _sample(), None)
    content = msgs[0]["content"]
    assert msgs[0]["role"] == "user"
    # options rendered in order with simple-evals MCQ scaffolding
    assert "A) the moon" in content
    assert "B) the sun" in content
    assert "D) the soil" in content
    assert "Answer: $LETTER" in content
    # closed-book: no supporting fact leaked into the prompt
    assert "fact" not in content.lower()


async def test_postprocess_extracts_letter():
    inf = SimpleNamespace(texts=["Reasoning about energy.\nAnswer: B"])
    assert await OpenBookQAZeroShotGenTask.postprocess(None, inf, None) == "B"


async def test_postprocess_no_marker_returns_empty():
    inf = SimpleNamespace(texts=["I think it is the sun but won't say."])
    assert await OpenBookQAZeroShotGenTask.postprocess(None, inf, None) == ""


async def test_feedback_scores_against_gold():
    ctx = SimpleNamespace(raw_sample=_sample(answer_key="B"))
    ok, fb = await OpenBookQAZeroShotGenTask.feedback(None, "B", ctx)
    assert ok is True and fb["correct"] is True and fb["answer"] == "B"
    _, fb_wrong = await OpenBookQAZeroShotGenTask.feedback(None, "A", ctx)
    assert fb_wrong["correct"] is False
