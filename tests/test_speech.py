"""Text normalisation for the speech model.

The reported symptom: "95.6%" was narrated as "ninety-five" — pause — "six percent".
Kokoro's grapheme-to-phoneme stage reads any period as a sentence end.
"""

import pytest

from app import speech
from app.tts import chunk_text


class TestNumbers:
    def test_a_decimal_no_longer_carries_a_sentence_ending_period(self):
        """The reported bug: 95.6% read as two sentences."""
        assert speech.normalize("It reaches 95.6% accuracy.") == (
            "It reaches 95 point 6 percent accuracy."
        )

    def test_a_dotted_section_number_expands_at_every_point(self):
        assert speech.normalize("See Section 3.2.1 below.") == (
            "See Section 3 point 2 point 1 below."
        )

    def test_thousands_separators_are_removed(self):
        assert speech.normalize("It holds 4,500,000 pairs.") == "It holds 4500000 pairs."

    def test_a_numeric_range_is_spoken_as_a_range(self):
        assert speech.normalize("Sweep 3-5 layers.") == "Sweep 3 to 5 layers."

    def test_a_model_suffix_does_not_become_a_minus_sign(self):
        assert speech.normalize("We use GPT-3 here.") == "We use GPT 3 here."

    def test_a_real_sentence_end_survives(self):
        assert speech.normalize("It works. The next part follows.") == (
            "It works. The next part follows."
        )


class TestAbbreviations:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Some layers, e.g. attention, are fast.", "Some layers, for example attention, are fast."),
            ("The output, i.e. the logits, is used.", "The output, that is the logits, is used."),
            ("This follows Vaswani et al. in spirit.", "This follows Vaswani and colleagues in spirit."),
            ("We compare LSTM vs. Transformer now.", "We compare LSTM versus Transformer now."),
            ("As Fig. 3 shows, loss drops.", "As Figure 3 shows, loss drops."),
            ("We optimise Eq. 2 directly.", "We optimise Equation 2 directly."),
            ("See Tab. 4 for details.", "See Table 4 for details."),
        ],
    )
    def test_expansion(self, text, expected):
        assert speech.normalize(text) == expected

    def test_a_sentence_final_abbreviation_keeps_its_stop(self):
        """"...et al. The model..." really is two sentences; dropping the period would
        run them together."""
        assert speech.normalize("We follow Vaswani et al. The model is deep.") == (
            "We follow Vaswani and colleagues. The model is deep."
        )

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Models, e.g. BERT, are large.", "Models, for example BERT, are large."),
            ("We compare LSTM vs. Transformer now.", "We compare LSTM versus Transformer now."),
        ],
    )
    def test_a_following_capital_is_a_model_name_not_a_new_sentence(self, text, expected):
        """"e.g. BERT" and "vs. Transformer" look sentence-final to a naive capital
        check, but the capital is the term being introduced."""
        assert speech.normalize(text) == expected

    def test_a_mid_sentence_abbreviation_does_not_gain_a_stop(self):
        """Regression: re.IGNORECASE also made the [A-Z] lookahead case-insensitive, so
        every mid-sentence abbreviation gained the exact period this exists to remove."""
        assert "." not in speech.normalize("Some layers, e.g. attention, are fast.")[:-1]


class TestSymbolsAndGreek:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("It is 88 ± 4 overall.", "It is 88 plus or minus 4 overall."),
            ("The input is 224 × 224 pixels.", "The input is 224 by 224 pixels."),
            ("We set α and β today.", "We set alpha and beta today."),
            ("Loss is ≈ 4 now.", "Loss is approximately 4 now."),
            ("It is ≤ 5 always.", "It is less than or equal to 5 always."),
            ("Scale by λ here.", "Scale by lambda here."),
        ],
    )
    def test_replacement(self, text, expected):
        assert speech.normalize(text) == expected


class TestNormalizeIsSafe:
    def test_normalising_twice_changes_nothing(self):
        text = "Fig. 3 shows 95.6% ± 0.4 for α, e.g. the base model."
        assert speech.normalize(speech.normalize(text)) == speech.normalize(text)

    def test_empty_text_is_untouched(self):
        assert speech.normalize("") == ""

    def test_ordinary_prose_is_untouched(self):
        text = "The encoder is composed of a stack of identical layers."
        assert speech.normalize(text) == text

    def test_no_double_spaces_are_introduced(self):
        assert "  " not in speech.normalize("It is 88 ± 4 and 224 × 224 for α.")


class TestChunkingInteraction:
    def test_an_abbreviation_no_longer_splits_a_sentence_in_two(self):
        """chunk_text splits on '. ', and render() inserts a 0.35s gap between chunks --
        so "e.g. attention" was cut in half and got a pause of our own making on top of
        Kokoro's."""
        assert len(chunk_text("Some layers, e.g. attention, run in parallel.")) == 1

    def test_a_decimal_does_not_split_a_chunk(self):
        assert len(chunk_text("It reaches 95.6% accuracy on the benchmark.")) == 1

    def test_genuine_sentences_still_split_when_over_the_limit(self):
        text = " ".join(f"This is sentence number {i} here." for i in range(40))
        assert len(chunk_text(text, max_chars=200)) > 1
