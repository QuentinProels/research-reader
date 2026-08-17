"""Classifying what a listener said into a command or a question.

The digit forms here are not hypothetical. Measured against this project's own STT,
"go back ninety seconds" is transcribed "go back 90 seconds" and "show me figure three"
as "show me figure 3", so both forms must work.
"""

import pytest

from app.commands import (
    DEFAULT_BACK_SECONDS,
    DEFAULT_FEW_MINUTES,
    DEFAULT_MOMENT_SECONDS,
    parse,
)


class TestRewindAndSkip:
    @pytest.mark.parametrize(
        "said,seconds",
        [
            ("go back two minutes", 120),
            ("go back 2 minutes", 120),
            ("go back 90 seconds", 90),
            ("go back ninety seconds", 90),
            ("back thirty seconds", 30),
            ("rewind one minute", 60),
            ("jump back 45 seconds", 45),
            ("go back a minute and a half", 90),
        ],
    )
    def test_rewind_amounts(self, said, seconds):
        command = parse(said)
        assert command.kind == "back"
        assert command.seconds == seconds

    @pytest.mark.parametrize(
        "said,seconds",
        [
            ("skip ahead one minute", 60),
            ("go forward 30 seconds", 30),
            ("fast forward two minutes", 120),
        ],
    )
    def test_skip_amounts(self, said, seconds):
        command = parse(said)
        assert command.kind == "forward"
        assert command.seconds == seconds

    @pytest.mark.parametrize(
        "said,seconds",
        [
            ("go back a bit", DEFAULT_BACK_SECONDS),
            ("rewind a little", DEFAULT_BACK_SECONDS),
            ("go back a few minutes", DEFAULT_FEW_MINUTES),
            ("go back a moment", DEFAULT_MOMENT_SECONDS),
        ],
    )
    def test_vague_amounts_use_the_tunable_defaults(self, said, seconds):
        assert parse(said).seconds == seconds


class TestSectionNavigation:
    @pytest.mark.parametrize(
        "said",
        [
            "repeat that section",
            "say that again",
            "read that again",
            "start this section over",
            "replay that",
        ],
    )
    def test_repeat(self, said):
        assert parse(said).kind == "repeat_section"

    @pytest.mark.parametrize("said", ["next section", "skip this section", "move on"])
    def test_next(self, said):
        assert parse(said).kind == "next_section"

    @pytest.mark.parametrize(
        "said", ["previous section", "go back a section", "last section"]
    )
    def test_previous(self, said):
        assert parse(said).kind == "previous_section"

    def test_a_time_amount_is_what_separates_a_rewind_from_a_section_jump(self):
        """"go back a section" and "go back two minutes" open identically."""
        assert parse("go back a section").kind == "previous_section"
        assert parse("go back two minutes").kind == "back"


class TestFigures:
    @pytest.mark.parametrize(
        "said,label",
        [
            ("show me figure three", "Figure 3"),
            ("show me figure 3", "Figure 3"),
            ("pull up table 2", "Table 2"),
            ("bring up fig 4", "Figure 4"),
            ("go to figure one", "Figure 1"),
        ],
    )
    def test_figure_requests(self, said, label):
        command = parse(said)
        assert command.kind == "show_figure"
        assert command.figure == label

    def test_the_label_matches_what_the_parser_produces(self):
        """The label is used to look a figure up, so it has to match pdf._caption_label."""
        from app.pdf import _caption_label

        assert parse("show me figure 3").figure == _caption_label("Figure 3: Something.")


class TestTransport:
    @pytest.mark.parametrize("said", ["pause", "stop", "hold on", "wait"])
    def test_pause(self, said):
        assert parse(said).kind == "pause"

    @pytest.mark.parametrize("said", ["resume", "continue", "keep going", "carry on"])
    def test_resume(self, said):
        assert parse(said).kind == "resume"


class TestQuestionsAreTheSafeDefault:
    @pytest.mark.parametrize(
        "said",
        [
            "why did they go back to the original architecture",
            "what happens in the next section",
            "how does low rank adaptation reduce memory",
            "what is a KL divergence term",
            "can you explain that again in simpler terms",
            "why would you skip the second stage",
        ],
    )
    def test_questions_are_never_mistaken_for_commands(self, said):
        """Seeking the audio somewhere the listener did not ask for loses their place,
        which is far worse than answering something that was meant as a command."""
        assert parse(said).kind == "question"

    @pytest.mark.parametrize("said", ["", "   ", "\n"])
    def test_empty_input_is_a_question_not_a_crash(self, said):
        assert parse(said).kind == "question"

    def test_unrecognised_speech_becomes_a_question(self):
        assert parse("mumble something indistinct here").kind == "question"


class TestSpeechToTextArtefacts:
    @pytest.mark.parametrize(
        "said,kind",
        [
            ("um go back two minutes", "back"),
            ("hey repeat that section", "repeat_section"),
            ("okay next section", "next_section"),
            ("go back two minutes please", "back"),
        ],
    )
    def test_filler_and_politeness_are_ignored(self, said, kind):
        assert parse(said).kind == kind

    def test_the_original_transcript_is_preserved(self):
        assert parse("um go back two minutes").transcript == "um go back two minutes"
