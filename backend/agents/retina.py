"""RETINA — the perception agent.

Owns every read of the screen: capture, the UI Automation tree, change detection, caching, and the
decision about when a vision call is worth its latency.

The problem it solves is the single biggest latency item in the old design. `_build_observation`
ran a full `find_visible_ui_elements(max_depth=8)` tree walk on *every step of every task*, and
with `SetGlobalSearchTimeout(1.5)` applied per control, a slow or unresponsive provider (Electron
and Chromium windows are the usual offenders) could stall the whole run for seconds — repeatedly,
re-deriving a tree that hadn't changed since the last step.

RETINA walks the tree only when the screen has actually changed. Change detection is a 16x16
grayscale frame digest costing well under a millisecond, compared against the previous frame plus
the foreground window handle. On a stable screen every subsequent request is a cache hit.

It also fixes the observation quality problem: the old renderer sorted elements by screen position
and hard-truncated to `[:25]`, so the correct element was frequently never shown to the model at
all. RETINA keeps the full element list and lets the *caller's query* decide which ones survive
into the prompt (see ScreenSnapshot.describe_for_prompt).
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (
    Agent,
    AgentContext,
    LLMClient,
    ScreenElement,
    ScreenSnapshot,
    rank_elements,
)

# How long a snapshot may be reused without even checking the frame digest. Below this the screen
# physically cannot have changed in a way the user could have reacted to, so re-checking is waste.
MIN_RECHECK_INTERVAL = 0.12

# A snapshot older than this is always re-derived, digest or not, as a guard against a stuck
# capture pipeline silently serving a frozen view of the world.
MAX_SNAPSHOT_AGE = 8.0


class RetinaAgent(Agent):
    """Cached, change-driven screen perception."""

    name = "RETINA"

    def __init__(
        self,
        vision: Any = None,
        telemetry: Any = None,
        llm: Optional[LLMClient] = None,
        max_depth: int = 8,
    ):
        super().__init__(llm)
        self.vision = vision
        self.telemetry = telemetry
        self.max_depth = max_depth

        self._lock = threading.RLock()
        self._snapshot: Optional[ScreenSnapshot] = None
        self._last_digest: str = ""
        self._last_window: str = ""
        self._dirty: bool = False
        self._vision_cache: Dict[Tuple[str, str], str] = {}

        # Instrumentation — these are the numbers that justify the agent existing.
        self.walks: int = 0            # actual UI Automation tree walks performed
        self.requests: int = 0         # snapshot() calls served
        self.cache_hits: int = 0
        self.vision_calls: int = 0
        self.vision_cache_hits: int = 0

    # ------------------------------------------------------------------
    # Frame digest — the cheap "did anything change?" test
    # ------------------------------------------------------------------

    def frame_digest(self) -> str:
        """A 16x16 grayscale digest of the current screen.

        Deliberately lossy: it must flag real UI changes (a menu opening, a page loading) while
        ignoring cursor blink and antialiasing noise. Downscaling to 256 pixels does exactly that,
        and costs orders of magnitude less than a UI Automation walk.
        """
        if self.vision is None:
            return ""
        try:
            img = self.vision.capture_screen_pil()
            if img is None:
                return ""
            small = img.convert("L").resize((16, 16))
            return hashlib.md5(small.tobytes()).hexdigest()
        except Exception:
            return ""

    def _active_window(self) -> Dict[str, Any]:
        if self.telemetry is None:
            return {"title": "", "app": ""}
        try:
            return self.telemetry.get_active_window() or {}
        except Exception:
            return {"title": "", "app": ""}

    def _walk(self) -> List[ScreenElement]:
        self.walks += 1
        if self.vision is None:
            return []
        try:
            raw = self.vision.find_visible_ui_elements(max_depth=self.max_depth) or []
        except Exception:
            return []
        seen: set = set()
        elements: List[ScreenElement] = []
        for d in raw:
            try:
                el = ScreenElement.from_dict(d)
            except Exception:
                continue
            if not el.name or len(el.name) < 2:
                continue
            # UI Automation routinely reports the same logical control at several tree depths with
            # identical geometry. Deduplicating here keeps the prompt budget for real controls.
            key = (el.name, el.x, el.y)
            if key in seen:
                continue
            seen.add(key)
            elements.append(el)
        return elements

    # ------------------------------------------------------------------
    # The main entry point
    # ------------------------------------------------------------------

    def snapshot(
        self, query: str = "", force: bool = False, ctx: Optional[AgentContext] = None
    ) -> ScreenSnapshot:
        """Returns the current screen state, re-walking the tree only if it actually changed."""
        with self._lock:
            self.requests += 1
            now = time.time()
            cached = self._snapshot

            if cached is not None and not force and not self._dirty:
                if now - cached.captured_at < MIN_RECHECK_INTERVAL:
                    self.cache_hits += 1
                    return cached
                if now - cached.captured_at < MAX_SNAPSHOT_AGE:
                    digest = self.frame_digest()
                    window = self._active_window()
                    title = str(window.get("title") or "")
                    if digest and digest == self._last_digest and title == self._last_window:
                        self.cache_hits += 1
                        cached.captured_at = now  # still current; refresh the clock, not the data
                        return cached

            window = self._active_window()
            elements = self._walk()
            snap = ScreenSnapshot(
                window_title=str(window.get("title") or ""),
                window_app=str(window.get("app") or ""),
                elements=elements,
                frame_hash=self.frame_digest(),
                captured_at=time.time(),
            )
            self._snapshot = snap
            self._last_digest = snap.frame_hash
            self._last_window = snap.window_title
            self._dirty = False
            if ctx is not None:
                self._emit(ctx, "status", f"perceived {len(elements)} controls in \"{snap.window_title[:40]}\"")
            return snap

    def invalidate(self) -> None:
        """Called after any action that touched the screen, so the next read re-derives.

        `_last_digest` is deliberately preserved: it is the digest of the screen *before* the
        action, which is exactly the baseline wait_for_change needs. Clearing it here made every
        wait_for_change compare the post-action screen against itself and burn the full timeout.
        """
        with self._lock:
            self._snapshot = None
            self._dirty = True

    def wait_for_change(self, timeout: float = 3.0, poll: float = 0.15) -> bool:
        """Blocks until the screen visibly changes, or the timeout expires.

        This replaces the model's habit of calling `wait(seconds)` with a guessed duration — it
        returns the instant the UI actually settles, so a fast page costs 150ms instead of a
        blind 2-second sleep, and a slow one is still waited out properly.
        """
        baseline = self._last_digest or self.frame_digest()
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            time.sleep(poll)
            current = self.frame_digest()
            if current and current != baseline:
                self.invalidate()
                return True
        return False

    # ------------------------------------------------------------------
    # Vision arbitration
    # ------------------------------------------------------------------

    def should_use_vision(self, query: str, snapshot: Optional[ScreenSnapshot] = None) -> bool:
        """Decides whether the accessibility tree is hopeless enough to justify a vision call.

        gemma3:4b runs locally with a 45-second ceiling, so it is by far the most expensive thing
        JARVIS can do. The old code reached for it whenever a name lookup missed; most of those
        misses were the `[:25]` truncation hiding the element rather than the tree lacking it.
        """
        snap = snapshot or self._snapshot
        if snap is None:
            return True
        if not snap.elements:
            return True
        if not query:
            return False
        ranked = rank_elements(snap.elements, query)
        if not ranked:
            return True
        from .base import text_score, keywords

        terms = keywords(query)
        if not terms:
            return False
        best = max((max((text_score(t, e.name) for t in terms), default=0.0) for e in ranked[:20]), default=0.0)
        return best < 0.42

    def ask(self, question: str, ctx: Optional[AgentContext] = None) -> Optional[str]:
        """Visual question answering, cached per (frame digest, question).

        The cache matters more than it looks: within a single task the same screen is frequently
        asked about twice (once to decide, once to describe in the reply), and each miss is a
        multi-second local model call.
        """
        if self.vision is None:
            return None
        digest = self._last_digest or self.frame_digest()
        key = (digest, question.strip().lower())
        if digest and key in self._vision_cache:
            self.vision_cache_hits += 1
            return self._vision_cache[key]
        self.vision_calls += 1
        self._emit(ctx, "status", "looking at the screen")
        try:
            answer = self.vision.ask_vision(question)
        except Exception as e:  # noqa: BLE001
            self._emit(ctx, "status", f"vision unavailable: {e}")
            return None
        if answer and digest:
            self._vision_cache[key] = answer
        return answer

    def locate(self, description: str, ctx: Optional[AgentContext] = None) -> Optional[Tuple[int, int]]:
        """Grid-grounded visual location of a target, for ANCHOR's vision fallback."""
        if self.vision is None:
            return None
        self.vision_calls += 1
        try:
            return self.vision.locate_via_vision(description)
        except Exception as e:  # noqa: BLE001
            self._emit(ctx, "status", f"vision locate failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def observation(self, query: str = "", budget: int = 28, ctx: Optional[AgentContext] = None) -> str:
        return self.snapshot(query=query, ctx=ctx).describe_for_prompt(query=query, budget=budget)

    def stats(self) -> Dict[str, Any]:
        hit_rate = (self.cache_hits / self.requests) if self.requests else 0.0
        return {
            "snapshot_requests": self.requests,
            "tree_walks": self.walks,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(hit_rate, 3),
            "walks_avoided": max(0, self.requests - self.walks),
            "vision_calls": self.vision_calls,
            "vision_cache_hits": self.vision_cache_hits,
        }
