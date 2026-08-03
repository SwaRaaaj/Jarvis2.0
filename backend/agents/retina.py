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

import base64
import hashlib
import io
import threading
import time
from dataclasses import dataclass, field
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

# Adaptive feed rates. A screen mid-animation deserves smooth frames; a screen nobody is touching
# does not deserve two full JPEG encodes a second forever.
FPS_ACTIVE = 5.0
FPS_IDLE = 0.5
IDLE_AFTER = 3.0          # seconds without meaningful change before dropping to the idle rate

# Below this the frame differs only by cursor blink, caret flicker or JPEG noise.
NOISE_FLOOR = 0.012


@dataclass
class Frame:
    """One tick of the live screen feed."""

    seq: int = 0
    digest: str = ""
    magnitude: float = 0.0        # 0..1, how much changed since the previous frame
    captured_at: float = field(default_factory=time.time)
    jpeg_b64: Optional[str] = None
    # The 16x16 reduction this frame's digest was built from. Kept so consumers can measure *how
    # far* the screen has drifted from an earlier frame, not just whether it differs at all.
    thumb: Optional[bytes] = None

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    @property
    def changed(self) -> bool:
        return self.magnitude > NOISE_FLOOR


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

        # --- live feed state ---------------------------------------------------------
        self._feed_thread: Optional[threading.Thread] = None
        self._feed_stop = threading.Event()
        self._latest: Optional[Frame] = None
        self._prev_thumb: Optional[bytes] = None
        self._subscribers: List[Callable[[Frame], None]] = []
        self._seq: int = 0
        self._last_change_at: float = 0.0
        self._want_jpeg: bool = False

        # Instrumentation — these are the numbers that justify the agent existing.
        self.walks: int = 0            # actual UI Automation tree walks performed
        self.requests: int = 0         # snapshot() calls served
        self.cache_hits: int = 0
        self.vision_calls: int = 0
        self.vision_cache_hits: int = 0
        self.frames_captured: int = 0
        self.frames_encoded: int = 0   # JPEG encodes actually performed
        self.frames_published: int = 0 # frames handed to subscribers (changed frames only)
        self.grabs_saved: int = 0      # duplicate mss grabs avoided by serving the feed's frame

    # ------------------------------------------------------------------
    # Frame digest — the cheap "did anything change?" test
    # ------------------------------------------------------------------

    @staticmethod
    def _thumb_bytes(img) -> Optional[bytes]:
        """16x16 grayscale reduction — the basis of both the digest and the change magnitude.

        Deliberately lossy: it must flag real UI changes (a menu opening, a page loading) while
        ignoring cursor blink and antialiasing noise. 256 pixels does exactly that, and costs
        orders of magnitude less than a UI Automation walk.
        """
        try:
            return img.convert("L").resize((16, 16)).tobytes()
        except Exception:
            return None

    @staticmethod
    def change_magnitude(prev: Optional[bytes], cur: Optional[bytes]) -> float:
        """Mean absolute difference between two thumbnails, normalised to 0..1.

        The digest alone is a boolean — same or different — which cannot distinguish a caret blink
        from a whole new window. Magnitude is what lets the feed pick a frame rate and lets VIGIL
        decide whether a change is worth spending a multi-second vision call on.
        """
        if not prev or not cur or len(prev) != len(cur):
            return 1.0 if cur else 0.0
        total = 0
        for a, b in zip(prev, cur):
            total += a - b if a > b else b - a
        return total / (len(cur) * 255.0)

    def capture_frame(self, want_jpeg: bool = False, quality: int = 40, scale: float = 0.35) -> Frame:
        """One screen grab that produces everything downstream needs at once.

        Previously the digest check, the dashboard stream and the HUD thumbnail each did their own
        independent `mss` grab of the full screen every cycle. This is the single producer they all
        now read from.
        """
        if self.vision is None:
            return Frame(seq=self._seq, digest="", magnitude=0.0)
        try:
            img = self.vision.capture_screen_pil()
        except Exception:
            img = None
        if img is None:
            return Frame(seq=self._seq, digest=self._last_digest, magnitude=0.0)

        self.frames_captured += 1
        thumb = self._thumb_bytes(img)
        digest = hashlib.md5(thumb).hexdigest() if thumb else ""
        magnitude = self.change_magnitude(self._prev_thumb, thumb)

        jpeg = None
        if want_jpeg:
            # Encoding is skipped entirely when nobody is watching, which is the common case for
            # the desktop HUD running without the web dashboard open.
            try:
                frame_img = img
                if scale != 1.0:
                    frame_img = img.resize((int(img.width * scale), int(img.height * scale)))
                buf = io.BytesIO()
                frame_img.save(buf, format="JPEG", quality=quality)
                jpeg = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                self.frames_encoded += 1
            except Exception:
                jpeg = None

        with self._lock:
            self._prev_thumb = thumb
            self._seq += 1
            frame = Frame(seq=self._seq, digest=digest, magnitude=magnitude, jpeg_b64=jpeg, thumb=thumb)
            self._latest = frame
            if frame.changed:
                self._last_change_at = frame.captured_at
        return frame

    def frame_digest(self) -> str:
        """Current screen digest.

        Reuses the feed's most recent frame instead of taking another full-screen grab — that
        duplicate grab was happening on every single snapshot() call.

        Reuse is gated on the feed actually running. Without a producer continuously refreshing
        frames there is no guarantee `_latest` reflects the screen *now*, and serving a stale
        digest would make a changed screen look unchanged — exactly the failure this cache is
        supposed to detect.
        """
        if self.feed_running:
            with self._lock:
                latest = self._latest
            max_age = 1.5 / FPS_ACTIVE
            if latest is not None and latest.digest and latest.age < max_age:
                self.grabs_saved += 1
                return latest.digest
        return self.capture_frame().digest

    def _active_window(self) -> Dict[str, Any]:
        window: Dict[str, Any] = {"title": "", "app": ""}
        if self.telemetry is not None:
            try:
                window = self.telemetry.get_active_window() or window
            except Exception:
                pass
        # When JARVIS's own HUD holds focus — which it does every time you type or speak to it —
        # the foreground window is JARVIS, not the app you are talking about. ScreenVision looks
        # past itself to pick the real target; the reported title has to agree with the element
        # tree, or SENTINEL verifies against a window nobody is looking at.
        if self.vision is not None and hasattr(self.vision, "get_target_window"):
            try:
                hwnd, title = self.vision.get_target_window()
                if title:
                    window = dict(window)
                    window["title"] = title
            except Exception:
                pass
        return window

    def active_window_title(self) -> str:
        """The foreground window title, without walking the UI Automation tree.

        This goes through win32gui, which works from any thread. The full tree walk does not:
        `uiautomation` needs COM initialised per-thread and raises "CoInitialize has not been
        called" from a bare background thread. Background consumers that only want a label must
        use this rather than snapshot().
        """
        return str(self._active_window().get("title") or "")

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

    # ------------------------------------------------------------------
    # The live feed: one producer, many consumers
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[Frame], None]) -> Callable[[], None]:
        """Registers a consumer of changed frames. Returns an unsubscribe function.

        Consumers are notified only when a frame actually differs from the last one, so a static
        screen costs subscribers nothing at all.
        """
        with self._lock:
            self._subscribers.append(callback)
            self._want_jpeg = True
        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)
                self._want_jpeg = bool(self._subscribers)
        return unsubscribe

    def start_feed(self, quality: int = 40, scale: float = 0.35) -> None:
        """Starts the single background capture loop.

        Rate adapts to what the screen is doing: FPS_ACTIVE while things are moving, dropping to
        FPS_IDLE once nothing has meaningfully changed for IDLE_AFTER seconds. A user watching an
        animation gets smooth frames; a user staring at a static editor costs almost nothing.
        """
        if self._feed_thread is not None and self._feed_thread.is_alive():
            return
        self._feed_stop.clear()

        def loop() -> None:
            while not self._feed_stop.is_set():
                started = time.time()
                try:
                    with self._lock:
                        want = self._want_jpeg
                        subs = list(self._subscribers)
                    frame = self.capture_frame(want_jpeg=want, quality=quality, scale=scale)

                    if frame.changed:
                        # A changed frame invalidates the cached UIA tree for free — the next
                        # snapshot() re-walks without needing its own detection pass.
                        self.invalidate()
                        self.frames_published += 1
                        for cb in subs:
                            try:
                                cb(frame)
                            except Exception:
                                pass
                except Exception:
                    pass

                idle = (time.time() - self._last_change_at) > IDLE_AFTER
                interval = 1.0 / (FPS_IDLE if idle else FPS_ACTIVE)
                self._feed_stop.wait(max(0.0, interval - (time.time() - started)))

        self._feed_thread = threading.Thread(target=loop, daemon=True, name="retina-feed")
        self._feed_thread.start()

    def stop_feed(self) -> None:
        self._feed_stop.set()
        thread = self._feed_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._feed_thread = None

    @property
    def feed_running(self) -> bool:
        return self._feed_thread is not None and self._feed_thread.is_alive()

    def latest_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def idle_seconds(self) -> float:
        """How long the screen has been visually still. VIGIL uses this to pick its moment."""
        if not self._last_change_at:
            return 0.0
        return time.time() - self._last_change_at

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
        publish_rate = (self.frames_published / self.frames_captured) if self.frames_captured else 0.0
        return {
            "snapshot_requests": self.requests,
            "tree_walks": self.walks,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(hit_rate, 3),
            "walks_avoided": max(0, self.requests - self.walks),
            "vision_calls": self.vision_calls,
            "vision_cache_hits": self.vision_cache_hits,
            "feed_running": self.feed_running,
            "frames_captured": self.frames_captured,
            "frames_encoded": self.frames_encoded,
            "frames_published": self.frames_published,
            "frames_suppressed": max(0, self.frames_captured - self.frames_published),
            "publish_rate": round(publish_rate, 3),
            "grabs_saved": self.grabs_saved,
            "idle_seconds": round(self.idle_seconds(), 1),
        }
