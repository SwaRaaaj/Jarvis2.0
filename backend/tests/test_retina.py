"""RETINA — perception caching and query-aware observation."""

import time

import pytest

from agents.retina import RetinaAgent
from agents.base import rank_elements, ScreenElement
from conftest import FakeLLM, FakeTelemetry, FakeVision, el


@pytest.fixture
def retina():
    vision = FakeVision([
        el("Send", "ButtonControl", 900, 800),
        el("Arundhati Sharma", "ListItemControl", 300, 220),
        el("Search", "EditControl", 300, 130),
    ])
    return RetinaAgent(vision=vision, telemetry=FakeTelemetry("Instagram"), llm=FakeLLM())


def test_unchanged_screen_is_not_re_walked(retina):
    """The monolith walked the UI Automation tree on every step of every task. With
    SetGlobalSearchTimeout(1.5) per control that is the dominant latency cost, and re-deriving an
    identical tree is pure waste."""
    retina.snapshot()
    for _ in range(10):
        time.sleep(0.13)  # past MIN_RECHECK_INTERVAL, so the digest is genuinely consulted
        retina.snapshot()

    assert retina.requests == 11
    assert retina.walks == 1, "a static screen must be walked exactly once"
    assert retina.stats()["walks_avoided"] == 10


def test_changed_screen_is_re_walked(retina):
    retina.snapshot()
    assert retina.walks == 1

    time.sleep(0.13)
    retina.vision.change_screen(elements=[el("New Button", "ButtonControl", 10, 10)])
    snap = retina.snapshot()

    assert retina.walks == 2
    assert snap.names() == ["New Button"]


def test_invalidate_forces_a_fresh_walk(retina):
    retina.snapshot()
    retina.invalidate()
    retina.snapshot()
    assert retina.walks == 2


def test_window_change_alone_forces_a_walk(retina):
    retina.snapshot()
    time.sleep(0.13)
    retina.telemetry.title = "Chrome"
    retina.snapshot()
    assert retina.walks == 2, "same pixels but a different foreground window is a real change"


def test_duplicate_elements_are_collapsed():
    """UI Automation reports the same logical control at several tree depths. Deduplicating keeps
    the prompt budget for real controls."""
    duplicated = [el("Send", "ButtonControl", 900, 800)] * 5 + [el("Cancel", "ButtonControl", 800, 800)]
    retina = RetinaAgent(vision=FakeVision(duplicated), telemetry=FakeTelemetry(), llm=FakeLLM())
    snap = retina.snapshot()
    assert snap.names() == ["Send", "Cancel"]


def test_observation_is_ranked_by_the_query():
    """The old renderer sorted by screen position and cut at [:25]. Here the caller's query decides
    which elements survive a tight budget, so the relevant one is always shown."""
    filler = [el(f"Item {i}", "ListItemControl", 100, i * 20) for i in range(40)]
    target = el("Arundhati Sharma", "ListItemControl", 100, 2000)
    retina = RetinaAgent(vision=FakeVision(filler + [target]), telemetry=FakeTelemetry(), llm=FakeLLM())

    observation = retina.observation(query="open the chat of Arundhati", budget=5)
    assert "Arundhati Sharma" in observation
    assert "+36 more elements" in observation


def test_observation_without_a_query_prefers_interactive_controls():
    elements = [
        el("Some label", "TextControl", 100, 10),
        el("Click me", "ButtonControl", 100, 500),
    ]
    retina = RetinaAgent(vision=FakeVision(elements), telemetry=FakeTelemetry(), llm=FakeLLM())
    observation = retina.observation(budget=2)
    assert observation.index("Click me") < observation.index("Some label")


def test_should_use_vision_only_when_the_tree_is_hopeless(retina):
    snap = retina.snapshot()
    assert retina.should_use_vision("click Send", snap) is False
    assert retina.should_use_vision("click the red record dot", snap) is True

    empty = RetinaAgent(vision=FakeVision([]), telemetry=FakeTelemetry(), llm=FakeLLM())
    assert empty.should_use_vision("anything", empty.snapshot()) is True


def test_vision_answers_are_cached_per_frame(retina):
    retina.snapshot()
    first = retina.ask("what is on screen?")
    second = retina.ask("what is on screen?")

    assert first == second
    assert retina.vision_calls == 1
    assert retina.vision_cache_hits == 1
    assert len(retina.vision.vision_questions) == 1, "the local model is the slowest thing here"


def test_wait_for_change_returns_early_when_the_screen_settles(retina):
    retina.snapshot()
    started = time.time()

    import threading

    threading.Timer(0.2, lambda: retina.vision.change_screen()).start()
    changed = retina.wait_for_change(timeout=3.0)

    assert changed is True
    assert time.time() - started < 1.0, "a fast page must not cost a blind full-length sleep"


def test_wait_for_change_times_out_when_nothing_happens(retina):
    retina.snapshot()
    assert retina.wait_for_change(timeout=0.4) is False


def test_capture_failure_degrades_instead_of_crashing():
    class BrokenVision(FakeVision):
        def capture_screen_pil(self):
            raise RuntimeError("no display")

        def find_visible_ui_elements(self, **kwargs):
            raise RuntimeError("UIA unavailable")

    retina = RetinaAgent(vision=BrokenVision(), telemetry=FakeTelemetry(), llm=FakeLLM())
    snap = retina.snapshot()
    assert snap.elements == []
    assert snap.frame_hash == ""


def test_rank_elements_is_stable_and_deterministic():
    elements = [ScreenElement.from_dict(el(n, "ButtonControl", 10, i))
                for i, n in enumerate(["Save", "Save As", "Cancel"])]
    assert [e.name for e in rank_elements(elements, "save")][0] == "Save"
    assert rank_elements(elements, "save") == rank_elements(elements, "save")
