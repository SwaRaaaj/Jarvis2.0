"""The live screen feed (RETINA) and the ambient observer (VIGIL)."""

import time

import pytest

from agents.retina import FPS_ACTIVE, NOISE_FLOOR, RetinaAgent
from agents.vigil import CHANGE_THRESHOLD, MIN_INTERVAL, Observation, VigilAgent
from conftest import FakeLLM, FakeTelemetry, FakeVision, el


@pytest.fixture
def retina():
    vision = FakeVision([el("Send", "ButtonControl", 900, 800)])
    return RetinaAgent(vision=vision, telemetry=FakeTelemetry("Instagram"), llm=FakeLLM())


@pytest.fixture
def feed(retina):
    """A running feed, always stopped afterwards so a failing test can't leak a thread."""
    retina.start_feed()
    yield retina
    retina.stop_feed()


# ======================================================================
# One capture, many consumers
# ======================================================================


def test_single_grab_serves_digest_magnitude_and_jpeg(retina):
    """These were three separate full-screen grabs: RETINA's digest check, the dashboard
    broadcaster, and the desktop HUD thumbnail, all running twice a second."""
    frame = retina.capture_frame(want_jpeg=True)
    assert frame.digest
    assert frame.jpeg_b64 and frame.jpeg_b64.startswith("data:image/jpeg;base64,")
    assert retina.vision.captures == 1, "one grab must produce everything downstream needs"


def test_jpeg_is_not_encoded_when_nobody_is_watching(retina):
    """The desktop HUD runs without the web dashboard open most of the time; encoding a JPEG for
    a subscriber that doesn't exist is pure waste."""
    retina.capture_frame(want_jpeg=False)
    assert retina.frames_encoded == 0

    retina.capture_frame(want_jpeg=True)
    assert retina.frames_encoded == 1


def test_subscribers_only_hear_about_changed_frames(feed):
    seen = []
    feed.subscribe(seen.append)
    time.sleep(0.6)
    quiet = len(seen)

    feed.vision.change_screen()
    time.sleep(0.6)

    assert len(seen) > quiet, "a real change must reach subscribers"
    assert all(f.changed for f in seen), "an unchanged screen must never be published"


def test_unsubscribe_stops_delivery(feed):
    seen = []
    cancel = feed.subscribe(seen.append)
    feed.vision.change_screen()
    time.sleep(0.5)
    cancel()
    count = len(seen)

    feed.vision.change_screen()
    time.sleep(0.5)
    assert len(seen) == count


def test_feed_frame_is_reused_instead_of_grabbing_again(feed):
    time.sleep(0.4)
    before = feed.grabs_saved
    for _ in range(5):
        feed.frame_digest()
    assert feed.grabs_saved > before, "a running feed must serve the digest, not re-grab"


def test_digest_never_reuses_a_stale_frame_without_a_feed(retina):
    """Without a producer continuously refreshing frames there is no guarantee the cached frame
    reflects the screen now — serving it would make a changed screen look unchanged, which is
    exactly what this cache exists to detect."""
    retina.capture_frame()
    first = retina.frame_digest()
    retina.vision.change_screen()
    assert retina.frame_digest() != first


# ======================================================================
# Change magnitude
# ======================================================================


def test_identical_frames_have_zero_magnitude(retina):
    retina.capture_frame()
    frame = retina.capture_frame()
    assert frame.magnitude == 0.0
    assert frame.changed is False


def test_a_real_change_exceeds_the_noise_floor(retina):
    retina.capture_frame()
    retina.vision.change_screen()
    frame = retina.capture_frame()
    assert frame.magnitude > NOISE_FLOOR
    assert frame.changed is True


def test_magnitude_is_proportional_to_how_much_changed():
    a = bytes([0] * 256)
    tiny = bytes([0] * 255 + [255])
    total = bytes([255] * 256)

    small = RetinaAgent.change_magnitude(a, tiny)
    large = RetinaAgent.change_magnitude(a, total)
    assert 0 < small < large
    assert large == pytest.approx(1.0)


def test_magnitude_handles_missing_frames():
    assert RetinaAgent.change_magnitude(None, None) == 0.0
    assert RetinaAgent.change_magnitude(None, bytes([1] * 256)) == 1.0


# ======================================================================
# Adaptive rate
# ======================================================================


def test_active_screen_is_sampled_faster_than_idle(retina):
    """A screen mid-animation deserves smooth frames; a static one does not deserve two full
    encodes a second forever."""
    retina.start_feed()
    try:
        # Keep the screen moving so the feed stays at the active rate.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            retina.vision.change_screen()
            time.sleep(0.05)
        active = retina.frames_captured
        assert active >= FPS_ACTIVE * 0.6, "an active screen should sample near FPS_ACTIVE"
    finally:
        retina.stop_feed()


def test_static_screen_suppresses_almost_every_frame(feed):
    """The old broadcaster pushed a fresh base64 JPEG every 500 ms whether or not anything had
    moved — tens of KB per tick of pure waste on an idle machine."""
    time.sleep(1.2)
    stats = feed.stats()
    assert stats["frames_captured"] > 1
    # The very first frame has no predecessor to compare against, so it counts as new information
    # and is published once. Every frame after it on a static screen must be suppressed.
    assert stats["frames_published"] <= 1
    assert stats["frames_suppressed"] >= stats["frames_captured"] - 1


def test_feed_lifecycle_is_idempotent(retina):
    retina.start_feed()
    retina.start_feed()          # second call must not spawn a second thread
    assert retina.feed_running
    retina.stop_feed()
    assert not retina.feed_running
    retina.stop_feed()           # stopping twice must not raise


def test_capture_failure_does_not_kill_the_feed():
    class Broken(FakeVision):
        def capture_screen_pil(self):
            raise RuntimeError("display lost")

    retina = RetinaAgent(vision=Broken(), telemetry=FakeTelemetry(), llm=FakeLLM())
    retina.start_feed()
    try:
        time.sleep(0.5)
        assert retina.feed_running, "a capture error must not take the feed thread down"
    finally:
        retina.stop_feed()


# ======================================================================
# VIGIL — the ambient observer
# ======================================================================


@pytest.fixture
def vigil(retina):
    agent = VigilAgent(retina, FakeLLM())
    yield agent
    agent.stop()


def test_ambient_observation_makes_screen_questions_instant(vigil, retina):
    """The whole point: a screen question used to cost 5-19s of cold vision inference every single
    time, even on a screen that had been sitting untouched for minutes."""
    retina.capture_frame()
    vigil._observe()

    view = vigil.current_view()
    assert view is not None
    assert view.text == retina.vision.vision_answer
    assert vigil.observations == 1


def test_a_stale_description_is_dropped_not_served(vigil, retina):
    retina.capture_frame()
    vigil._observe()
    assert vigil.current_view() is not None

    # The user has navigated somewhere else; describing the old screen would be worse than
    # admitting we don't know.
    retina.vision.change_screen()
    retina.capture_frame()
    assert vigil.current_view() is None
    assert vigil.stale_misses == 1


def test_expired_descriptions_are_dropped(vigil, retina):
    retina.capture_frame()
    vigil._observe()
    assert vigil.current_view(max_age=60) is not None

    # Age it explicitly rather than sleeping: Windows' time.time() resolution is coarse enough
    # that two calls in the same tick can report an age of exactly 0.0.
    vigil._current.observed_at -= 120.0
    assert vigil.current_view(max_age=90) is None
    assert vigil.stale_misses == 1


def test_vigil_does_not_look_while_a_task_is_running(vigil, retina):
    """The vision model is single-threaded and slow. Ambient curiosity must never compete with an
    order the user actually gave."""
    retina.start_feed()
    try:
        vigil.pause()
        assert vigil._should_observe() is False
        assert vigil.skipped_busy >= 1
    finally:
        retina.stop_feed()


def test_vigil_waits_for_the_screen_to_settle(vigil, retina):
    """Describing a half-painted page wastes the call, so it waits for stillness."""
    retina.capture_frame()
    retina.vision.change_screen()
    retina.capture_frame()          # change just happened -> not settled
    assert vigil._should_observe() is False


def test_vigil_is_rate_limited(vigil, retina):
    retina.capture_frame()
    vigil._observe()
    retina.vision.change_screen()
    retina.capture_frame()
    # Even a big change cannot trigger a second look inside MIN_INTERVAL.
    assert vigil._should_observe() is False
    assert MIN_INTERVAL > 0


def test_repeated_vision_failure_backs_off(retina):
    class Blind(FakeVision):
        def ask_vision(self, question, **kwargs):
            return None

    retina.vision = Blind()
    agent = VigilAgent(retina, FakeLLM())
    for _ in range(4):
        agent._last_attempt_at = 0.0
        agent._observe()
    assert agent._backoff_until > time.time(), "a dead vision daemon must not be retried forever"
    assert agent.observations == 0


def test_vigil_keeps_a_short_timeline(vigil, retina):
    for _ in range(3):
        retina.vision.change_screen()
        retina.capture_frame()
        vigil._last_attempt_at = 0.0
        vigil._observe()
    assert len(vigil.recent()) == 3


def test_vigil_start_stop_is_clean(vigil):
    vigil.start()
    assert vigil.running
    vigil.stop()
    assert not vigil.running


def test_stats_report_the_instant_answer_rate(vigil, retina):
    retina.capture_frame()
    vigil._observe()
    vigil.current_view()
    stats = vigil.stats()
    assert stats["observations"] == 1
    assert stats["instant_answers"] == 1
    assert stats["current"]["text"]
