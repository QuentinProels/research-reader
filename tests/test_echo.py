"""Distinguishing the listener's voice from the narration leaking into the microphone.

Hands-free listening means the mic is open while the narration plays, and the narration
is speech, so voice activity detection fires on it. The transcript gives ground truth
that no acoustic canceller has: the exact words being spoken at that moment.
"""

import pytest

from app import echo

NARRATION = (
    "The Transformer follows this overall architecture using stacked self attention "
    "and point wise fully connected layers for both the encoder and decoder. The encoder "
    "is composed of a stack of six identical layers with residual connections."
)


class TestEchoIsCaught:
    @pytest.mark.parametrize(
        "heard",
        [
            "the transformer follows this overall architecture using stacked self attention",
            "encoder is composed of a stack of six identical layers",
            # speech to text on a degraded echo is approximate, not exact
            "the transformer follows this overall architecture using stack self attention layers",
        ],
    )
    def test_narration_coming_back_through_the_mic_is_rejected(self, heard):
        assert echo.is_echo(heard, NARRATION)


class TestRealSpeechSurvives:
    @pytest.mark.parametrize(
        "heard",
        [
            "why does that reduce memory",
            "go back two minutes",
            "repeat that section",
            "what did that mean about the residual connections",
            "can you explain why they chose six layers instead of twelve",
        ],
    )
    def test_the_listener_is_never_suppressed(self, heard):
        """A false positive here silently swallows what the user said, which is the
        worst outcome available: they get no answer and no indication why."""
        assert not echo.is_echo(heard, NARRATION)

    def test_a_question_quoting_the_paper_still_gets_through(self):
        """Quoting a phrase back is exactly how people ask about it."""
        assert not echo.is_echo(
            "what does residual connections mean here and why six layers", NARRATION
        )

    def test_short_commands_are_never_judged_echo(self):
        """Too few words to score reliably, and these are the commands most likely to be
        spoken over playing audio."""
        assert not echo.is_echo("pause", NARRATION)
        assert not echo.is_echo("go back", NARRATION)


class TestScoring:
    def test_identical_text_scores_one(self):
        assert echo.overlap(NARRATION, NARRATION) == 1.0

    def test_unrelated_text_scores_low(self):
        assert echo.overlap("what is the weather like today", NARRATION) < 0.3

    def test_empty_input_scores_zero(self):
        assert echo.overlap("", NARRATION) == 0.0
        assert echo.overlap("the and of", NARRATION) == 0.0

    def test_narration_window_selects_by_timestamp(self):
        chunks = [
            {"text": "far in the past", "start_s": 10.0},
            {"text": "right about here", "start_s": 100.0},
            {"text": "far in the future", "start_s": 400.0},
        ]
        assert echo.narration_near(chunks, 100.0) == "right about here"
