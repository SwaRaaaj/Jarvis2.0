"""SCRIBE — transcription accuracy."""

import pytest

from agents.scribe import ScribeAgent, Transcript
from agents.retina import RetinaAgent
from conftest import FakeLLM, FakeTelemetry, FakeVision, el


def alts(*pairs):
    """Builds a SpeechRecognition show_all=True style ranked guess list."""
    return [{"transcript": t, "confidence": c} for t, c in pairs]


@pytest.fixture
def scribe():
    vision = FakeVision([
        el("Alice Johnson", "ListItemControl", 300, 220),
        el("Send", "ButtonControl", 900, 800),
        el("Subscribe", "ButtonControl", 700, 500),
    ])
    retina = RetinaAgent(vision=vision, telemetry=FakeTelemetry("Instagram"), llm=FakeLLM())
    return ScribeAgent(FakeLLM(), retina=retina)


# ======================================================================
# Re-ranking against what is really on screen
# ======================================================================


def test_an_alternative_matching_the_screen_beats_the_top_guess(scribe):
    """The recogniser ranks with no idea what you were looking at. A contact name it ranked second
    is far more likely than a nonsense phrase it ranked first."""
    result = scribe.transcribe(alts(
        ("open the chat of a list johnson", 0.71),
        ("open the chat of alice johnson", 0.62),
    ))
    assert "alice johnson" in result.text.lower()
    assert result.source in ("reranked", "repaired")
    assert scribe.reranked == 1


def test_the_top_guess_is_kept_when_nothing_beats_it(scribe):
    result = scribe.transcribe(alts(("open chrome", 0.95), ("open chrom", 0.4)))
    assert result.text == "open chrome"
    assert scribe.reranked == 0


def test_screen_context_beats_a_higher_asr_confidence(scribe):
    """Confidence alone is not enough — the point is to overturn the recogniser's ranking when
    real context disagrees with it."""
    result = scribe.transcribe(alts(
        ("click subscribed", 0.90),
        ("click subscribe", 0.55),
    ))
    assert "subscribe" in result.text.lower()


def test_no_alternatives_returns_empty(scribe):
    assert scribe.transcribe([]).text == ""
    assert scribe.transcribe(None).text == ""


def test_a_plain_string_still_gets_repaired(scribe):
    """Callers with only one guess must still benefit."""
    result = scribe.transcribe("open crome")
    assert "chrome" in result.text.lower()


# ======================================================================
# Token repair
# ======================================================================


@pytest.mark.parametrize("heard,expected", [
    ("open crome", "chrome"),
    ("open notepd", "notepad"),
    ("open instagam", "instagram"),
    ("open youtub", "youtube"),
])
def test_mis_heard_app_names_are_repaired(scribe, heard, expected):
    assert expected in scribe.transcribe(heard).text.lower()


def test_ordinary_words_are_not_rewritten_into_commands(scribe):
    """An over-eager corrector that turns English into command words is worse than none."""
    for phrase in ("what is the weather today", "tell me a joke about cats",
                   "how long until the meeting"):
        result = scribe.transcribe(phrase)
        assert result.text.lower() == phrase, f"{phrase!r} was altered to {result.text!r}"


def test_short_words_are_never_repaired(scribe):
    """Three-letter words are too easy to fuzzy-match onto something wrong."""
    result = scribe.transcribe("go to bed")
    assert result.text == "go to bed"


def test_corrections_are_reported(scribe):
    result = scribe.transcribe("open crome")
    assert result.corrections
    assert result.changed
    assert result.as_dict()["corrections"]


# ======================================================================
# Vocabulary
# ======================================================================


def test_vocabulary_includes_live_screen_text(scribe):
    vocab = [v.lower() for v in scribe.vocabulary()]
    assert "alice johnson" in vocab, "on-screen names are the highest-value vocabulary there is"
    assert "chrome" in vocab
    assert "jarvis" in vocab


def test_vocabulary_survives_a_broken_screen():
    class Broken(FakeVision):
        def find_visible_ui_elements(self, **kwargs):
            raise RuntimeError("UIA down")

    retina = RetinaAgent(vision=Broken(), telemetry=FakeTelemetry(), llm=FakeLLM())
    agent = ScribeAgent(FakeLLM(), retina=retina)
    assert "chrome" in [v.lower() for v in agent.vocabulary()]


def test_works_with_no_retina_at_all():
    agent = ScribeAgent(FakeLLM())
    result = agent.transcribe(alts(("open crome", 0.8)))
    assert "chrome" in result.text.lower()


def test_stats_report_correction_rate(scribe):
    scribe.transcribe("open crome")
    scribe.transcribe(alts(("open chrome", 0.99)))
    stats = scribe.stats()
    assert stats["transcriptions"] == 2
    assert stats["token_repairs"] == 1
