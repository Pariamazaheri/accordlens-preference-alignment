import pytest

from accordlens.data import encode_response, parse_hh, preference_pair
from accordlens.judging import (
    normalize_label,
    parse_verdict,
    summarize,
    swap_verdict,
    wilson,
)


def test_hh_full_history_preserved():
    text = (
        "\n\nHuman: First\n\nAssistant: Earlier\n\nHuman: Follow-up\n\nAssistant: Final"
    )
    messages, response = parse_hh(text)
    assert len(messages) == 3 and messages[1]["content"] == "Earlier"
    assert response == "Final"


def test_escaped_newlines():
    assert parse_hh(r"\n\nHuman: Hi\n\nAssistant: Hello")[1] == "Hello"


def test_mismatched_prompts_rejected():
    with pytest.raises(ValueError):
        preference_pair(
            "\n\nHuman: One\n\nAssistant: Yes", "\n\nHuman: Two\n\nAssistant: No"
        )


@pytest.mark.parametrize(
    "raw", ["A because it is better", "Answer A", "A B", "tie-ish", "", "a"]
)
def test_invalid_judgments_not_ties(raw):
    assert parse_verdict(raw) == "invalid"


@pytest.mark.parametrize(
    "raw,expected", [("A", "A"), ("B", "B"), ("TIE", "tie"), (" A\n", "A")]
)
def test_strict_verdict(raw, expected):
    assert parse_verdict(raw) == expected


def test_label_normalization():
    assert normalize_label("tie (inconsistent)") == "tie"
    with pytest.raises(ValueError):
        normalize_label("unknown")


def test_swap_involution():
    for v in ["A", "B", "tie", "invalid"]:
        assert swap_verdict(swap_verdict(v)) == v


def test_invalid_counts_against_agreement_and_not_swap_consistency():
    rows = [
        {"label": "A", "verdict": "A", "swapped_verdict": "B"},
        {"label": "tie", "verdict": "invalid", "swapped_verdict": "invalid"},
    ]
    s = summarize(rows)
    assert s["agreement"] == 0.5 and s["valid_rate"] == 0.5
    assert s["swap_consistency"] == 1 and s["swap_valid_pairs"] == 1


def test_wilson_bounds():
    assert wilson(0, 10)[1] > 0
    assert wilson(10, 10)[0] < 1
    with pytest.raises(ValueError):
        wilson(1, 0)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2] if add_generation_prompt else [1, 2, 3, 4]


def test_complete_response_mask():
    ids, mask = encode_response(
        FakeTokenizer(), [{"role": "user", "content": "Hi"}], "Hello", 4
    )
    assert ids.tolist() == [[1, 2, 3, 4]] and mask.tolist() == [[0, 0, 1, 1]]
    with pytest.raises(ValueError):
        encode_response(FakeTokenizer(), [], "Hello", 3)
