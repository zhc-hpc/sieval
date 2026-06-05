"""IFBench verifier unit tests — ported verbatim from allenai/IFBench instructions_test.py

Reference: https://github.com/allenai/IFBench/blob/
1091c4c3de6c1f6ed12c012ed68f11ea450b0117/instructions_test.py
Confirms the vendored constraint verifiers behave identically to upstream.
"""

from sieval.community.ifbench import instructions
from sieval.community.ifbench import instructions_registry as registry

TEST_SHORT_MESSAGE = "\n    This message has five words."
TEST_MEDIUM_MESSAGE = "\n    This message has exactly ten words in the entire text."
TEST_LONG_MESSAGE = """
    This message has many more words than the previous messages because
    we need to test the upper bounds of our word count range checker with
    a sufficiently long piece of text."""


def test_registry_has_58_ood_constraints():
    # IFBench introduces 58 new out-of-domain verifiable constraints
    assert len(registry.INSTRUCTION_DICT) == 58
    assert "count:keywords_multiple" in registry.INSTRUCTION_DICT
    assert "words:palindrome" in registry.INSTRUCTION_DICT


def test_word_count_range_checker():
    instruction = instructions.WordCountRangeChecker("count:word_count_range")
    instruction.build_description(min_words=5, max_words=5)
    assert instruction.check_following(TEST_SHORT_MESSAGE) is True
    instruction.build_description(min_words=10, max_words=10)
    assert instruction.check_following(TEST_MEDIUM_MESSAGE) is True
    instruction.build_description(min_words=20, max_words=20)
    assert instruction.check_following(TEST_LONG_MESSAGE) is False


def test_unique_word_count_checker():
    instruction = instructions.UniqueWordCountChecker("count:unique_word_count")
    instruction.build_description(N=5)
    assert instruction.check_following("\n    This message has five unique words.") is True


def test_stop_word_ratio_checker():
    instruction = instructions.StopWordPercentageChecker("ratio:stop_words")
    instruction.build_description(percentage=50)
    assert (
        instruction.check_following("\n    This message has a high stop word ratio.")
        is True
    )


def test_ngram_overlap_checker():
    ref = "This is the test."
    instruction = instructions.NGramOverlapChecker("ratio:overlap")
    instruction.build_description(reference_text=ref, percentage=100)
    assert instruction.check_following("This is the test.") is True


def test_numbers_count_checker():
    instruction = instructions.NumbersCountChecker("count:numbers")
    instruction.build_description(N=3)
    assert instruction.check_following("This is 1 number. This is not 10 numbers. It is 3.") is True
    assert instruction.check_following("Decimals like 3.14 should only count as one number 2.") is False
    assert instruction.check_following("This is more than 3 numbers: 1 2 3.") is False
    assert instruction.check_following("1 2 3") is True
    instruction.build_description(N=1)
    assert instruction.check_following("This is one number: 100,000") is True


def test_end_to_end_strict_loose_scoring():
    # one prompt with one verifiable constraint -> evaluation_lib strict/loose
    from sieval.community.ifbench.evaluation_lib import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    prompt = "Write something using exactly five words."
    inp = InputExample(
        key=0,
        instruction_id_list=["count:word_count_range"],
        prompt=prompt,
        kwargs=[{"min_words": 5, "max_words": 5}],
    )
    good = {prompt: "This message has five words"}
    out = test_instruction_following_strict(inp, good)
    assert out.follow_all_instructions is True
    bad = {prompt: "This message clearly has far too many words to satisfy the constraint"}
    out_bad = test_instruction_following_strict(inp, bad)
    assert out_bad.follow_all_instructions is False
    # loose is an upper bound: the good response still passes
    assert test_instruction_following_loose(inp, good).follow_all_instructions is True
