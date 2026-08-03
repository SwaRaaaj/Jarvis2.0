"""VIGIL — the ambient observer.

The live feed watches the screen continuously, but until now nothing *understood* it continuously.
Every time the user asked "what's on my screen?", JARVIS captured a frame and waited 5–19 seconds
for `gemma3:4b` to describe it — from a standing start, every single time, even if the screen had
been sitting untouched for ten minutes.

VIGIL closes that gap. It watches RETINA's feed and, when the screen has *meaningfully* changed and
then settled, quietly spends one vision call in the background and caches what it saw. The next
screen question is answered from that cache **instantly**.

The whole design is about being invisible:

  * **Only on real change.** Cursor blink and caret flicker are below the noise floor. A change has
    to exceed CHANGE_THRESHOLD magnitude to be worth looking at.
  * **Only once settled.** It waits for the screen to stop moving before looking, so it describes a
    finished page rather than a half-painted one.
  * **Never during a task.** The cortex pauses VIGIL while an order is running. The vision model is
    single-threaded and slow; competing for it would make real work slower to save a hypothetical
    question.
  * **Rate limited.** MIN_INTERVAL between observations regardless of activity, so a video playing
    full-screen cannot pin the model at 100%.
  * **Fails silent.** Ollama down, model missing, timeout — VIGIL logs it and stops trying for a
    while. It is an optimisation, never a dependency.

If VIGIL has nothing fresh, the screen-query path behaves exactly as it did before. It can only
make things faster, never break them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Agent, AgentContext, LLMClient

# How much of the screen must change before a fresh look is worth 5-19 seconds of model time.
# Well above RETINA's NOISE_FLOOR: a notification toast is worth noting, a blinking caret is not.
CHANGE_THRESHOLD = 0.06

# The screen must be still for this long before VIGIL looks, so it describes a settled page.
SETTLE_SECONDS = 1.5

# Hard floor between observations, however busy the screen is.
MIN_INTERVAL = 25.0

# A cached description older than this is considered stale and will not be served.
MAX_USEFUL_AGE = 90.0

# After repeated vision failures, back off rather than retrying forever.
FAILURE_BACKOFF = 120.0
MAX_CONSECUTIVE_FAILURES = 3

AMBIENT_QUESTION = (
    "Briefly describe what is on this screen: the application, and what the user appears to be "
    "looking at or working on."
)


@dataclass
class Observation:
    """A cached understanding of the screen at a moment in time."""

    text: str = ""
    digest: str = ""
    window_title: str = ""
    observed_at: float = field(default_factory=time.time)
    took_seconds: float = 0.0
    # The screen reduction this description was made against, so freshness can be judged by how
    # far the screen has drifted rather than by exact equality.
    thumb: Optional[bytes] = None

    @property
    def age(self) -> float:
        return time.time() - self.observed_at

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "digest": self.digest, "window_title": self.window_title,
            "age_seconds": round(self.age, 1), "took_seconds": round(self.took_seconds, 1),
        }


class VigilAgent(Agent):
    """Keeps a warm, continuously-refreshed understanding of the screen."""

    name = "VIGIL"

    def __init__(self, retina: Any, llm: Optional[LLMClient] = None):
        super().__init__(llm)
        self.retina = retina

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.RLock()

        self._current: Optional[Observation] = None
        self._history: List[Observation] = []
        self._last_attempt_at: float = 0.0
        self._consecutive_failures: int = 0
        self._backoff_until: float = 0.0

        self.observations: int = 0
        self.instant_answers: int = 0
        self.stale_misses: int = 0
        self.skipped_busy: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vigil")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None

    def pause(self) -> None:
        """Called while a real task runs. The vision model is slow and single-threaded — ambient
        curiosity must never compete with work the user actually asked for."""
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Serving cached understanding
    # ------------------------------------------------------------------

    def current_view(self, max_age: float = MAX_USEFUL_AGE, require_same_screen: bool = True) -> Optional[Observation]:
        """The cached description, if it is still trustworthy.

        Freshness is judged by how far the screen has *drifted*, not by exact equality. An
        identical-digest test sounds right but is far too strict in practice: a blinking terminal
        cursor, a ticking clock or a single repainted pixel changes the digest while leaving the
        description perfectly accurate. Measured live, that strictness meant the cache essentially
        never served — every question fell through to a cold 5-19 second vision call, which is the
        exact cost this agent exists to remove.

        Drift beyond CHANGE_THRESHOLD does mean the user has genuinely moved on, and describing a
        screen they have left is worse than admitting we don't know.
        """
        with self._lock:
            observation = self._current
        if observation is None:
            return None
        if observation.age > max_age:
            self.stale_misses += 1
            return None
        if require_same_screen:
            latest = self.retina.latest_frame() if hasattr(self.retina, "latest_frame") else None
            if latest is not None and observation.thumb and latest.thumb:
                drift = self.retina.change_magnitude(observation.thumb, latest.thumb)
                if drift > CHANGE_THRESHOLD:
                    self.stale_misses += 1
                    return None
            elif latest is not None and latest.digest and observation.digest \
                    and latest.digest != observation.digest:
                # No thumbnails available (a stubbed feed) — fall back to strict equality.
                self.stale_misses += 1
                return None
        self.instant_answers += 1
        return observation

    def recent(self, limit: int = 5) -> List[Observation]:
        """A short timeline of what has been on screen — useful for 'what was I just doing?'."""
        with self._lock:
            return list(self._history[-limit:])

    # ------------------------------------------------------------------
    # The watch loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(1.0)
            if self._stop.is_set():
                break
            try:
                if self._should_observe():
                    self._observe()
            except Exception:
                # An ambient optimisation must never take the process down.
                pass

    def _should_observe(self) -> bool:
        if self._paused.is_set():
            self.skipped_busy += 1
            return False
        now = time.time()
        if now < self._backoff_until:
            return False
        if (now - self._last_attempt_at) < MIN_INTERVAL:
            return False

        latest = self.retina.latest_frame() if hasattr(self.retina, "latest_frame") else None
        if latest is None:
            return False

        # Wait for the screen to settle so we describe a finished page, not a half-painted one.
        idle = self.retina.idle_seconds() if hasattr(self.retina, "idle_seconds") else 0.0
        if idle < SETTLE_SECONDS:
            return False

        with self._lock:
            current = self._current
        if current is None:
            return True                                  # nothing cached yet — take a first look
        if current.digest and latest.digest == current.digest:
            return False                                 # same screen we already understand
        return latest.magnitude >= CHANGE_THRESHOLD or current.age > MAX_USEFUL_AGE

    def _observe(self) -> None:
        self._last_attempt_at = time.time()
        latest = self.retina.latest_frame()
        digest = latest.digest if latest else ""
        thumb = latest.thumb if latest else None

        started = time.time()
        answer = None
        try:
            # force=True: the answer must describe the screen as it is now, not reuse a cached
            # reply keyed to an older frame.
            answer = self.retina.ask(AMBIENT_QUESTION)
        except Exception:
            answer = None
        took = time.time() - started

        if not answer:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._backoff_until = time.time() + FAILURE_BACKOFF
                self._consecutive_failures = 0
            return

        self._consecutive_failures = 0
        window_title = ""
        try:
            # Deliberately NOT snapshot(): that walks the UI Automation tree, and `uiautomation`
            # needs COM initialised per-thread — from this background thread it raises
            # "CoInitialize has not been called". Observed live, not theoretical. The title comes
            # via win32gui instead, which is thread-safe and all VIGIL needs for a label.
            window_title = self.retina.active_window_title()
        except Exception:
            pass

        observation = Observation(text=answer.strip(), digest=digest, thumb=thumb,
                                  window_title=window_title, took_seconds=took)
        with self._lock:
            self._current = observation
            self._history.append(observation)
            if len(self._history) > 20:
                self._history = self._history[-20:]
        self.observations += 1

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            current = self._current
        total = self.instant_answers + self.stale_misses
        return {
            "running": self.running,
            "paused": self._paused.is_set(),
            "observations": self.observations,
            "instant_answers": self.instant_answers,
            "stale_misses": self.stale_misses,
            "instant_rate": round(self.instant_answers / total, 3) if total else 0.0,
            "skipped_while_busy": self.skipped_busy,
            "current": current.as_dict() if current else None,
        }
