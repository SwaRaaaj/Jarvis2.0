"""EARS — the voice gate.

The desktop HUD's listener loop sends *every* successfully transcribed utterance straight to
`send_command`. There is no wake word and no filter, so any conversation happening near the
microphone becomes a real desktop action: clicks, launches, closed tabs. It also calls
`adjust_for_ambient_noise` on every single loop iteration, which blocks the microphone for 0.3s
each time for a measurement that is stable for minutes.

EARS is the cheapest agent here and prevents the most wasted work, because the work it prevents is
an entire pipeline run — plan, ground, act, verify — triggered by speech that was never aimed at
JARVIS.

It decides in pure Python, with no model call and no network:

    * was JARVIS addressed?          wake word, or an unambiguous imperative in always-on mode
    * is this even a command?        filler, fragments and back-channel noise are dropped
    * did we just do this?           identical transcripts inside a debounce window are dropped
    * is recalibration due?          time-based, not every-iteration
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import Agent, normalize

WAKE_WORDS = ("jarvis", "hey jarvis", "ok jarvis", "okay jarvis", "yo jarvis")

# Utterances that are noise even when perfectly transcribed.
_FILLER = {
    "", "uh", "um", "hmm", "hm", "ah", "oh", "eh", "mm", "mhm", "yeah", "yep", "yes", "no", "nope",
    "ok", "okay", "right", "sure", "well", "so", "like", "you know", "i mean", "what", "huh",
    "the", "and", "but", "a", "i", "it", "that", "this", "thanks", "thank you",
}

# Verbs that make an utterance an instruction rather than a remark. In always-on mode an utterance
# must open with one of these to be treated as addressed to JARVIS.
_IMPERATIVES = (
    "open", "close", "click", "launch", "start", "run", "play", "pause", "stop", "type", "write",
    "send", "message", "search", "google", "find", "show", "tell", "switch", "go", "scroll",
    "press", "select", "read", "check", "what", "who", "when", "where", "how", "which", "can you",
    "could you", "please", "turn", "set", "make", "take", "bring", "pull", "fire", "mute", "unmute",
    "minimize", "maximize", "screenshot", "copy", "paste", "save", "delete", "refresh", "reload",
)

# Speech clearly aimed at another person, not the assistant.
_THIRD_PARTY = re.compile(
    r"\b(?:he|she|they|we) (?:said|says|told|thinks?)\b|"
    r"^\s*(?:mom|dad|bro|dude|guys|everyone|hello everyone)\b",
    re.IGNORECASE,
)


@dataclass
class GateDecision:
    dispatch: bool
    reason: str
    cleaned: str = ""
    had_wake_word: bool = False


class EarsAgent(Agent):
    """Decides whether heard speech should become a command."""

    name = "EARS"

    def __init__(
        self,
        require_wake_word: bool = False,
        debounce_seconds: float = 2.5,
        min_words: int = 1,
        recalibrate_every: float = 60.0,
    ):
        super().__init__(None)
        self.require_wake_word = require_wake_word
        self.debounce_seconds = debounce_seconds
        self.min_words = min_words
        self.recalibrate_every = recalibrate_every

        self._last_transcript: str = ""
        self._last_dispatch_at: float = 0.0
        self._last_calibration_at: float = 0.0

        self.heard: int = 0
        self.dispatched: int = 0
        self.dropped: Dict[str, int] = {}

    # ------------------------------------------------------------------

    def gate(self, transcript: str) -> GateDecision:
        """The whole decision, in one pure function."""
        self.heard += 1
        raw = (transcript or "").strip()
        low = normalize(raw)

        if not low:
            return self._drop("empty transcript")

        had_wake = any(re.search(rf"\b{re.escape(w)}\b", low) for w in WAKE_WORDS)
        stripped = re.sub(r"\b(?:hey |ok |okay |yo )?jarvis\b", " ", low).strip()
        stripped = re.sub(r"\s+", " ", stripped).strip(" ,.!?")

        if stripped in _FILLER:
            return self._drop("filler", cleaned=stripped, had_wake_word=had_wake)

        words = stripped.split()
        if len(words) < self.min_words:
            return self._drop("too short to be a command", cleaned=stripped, had_wake_word=had_wake)

        # The wake word on its own is an attention call, not an order.
        if had_wake and not stripped:
            return self._drop("wake word with no command", had_wake_word=True)

        if _THIRD_PARTY.search(raw):
            return self._drop("appears to be addressed to someone else", cleaned=stripped)

        if self.require_wake_word and not had_wake:
            return self._drop("no wake word", cleaned=stripped)

        if not had_wake and not self._looks_like_command(stripped):
            return self._drop("not phrased as an instruction", cleaned=stripped)

        # Speech recognition often returns the same phrase twice as a phrase boundary is re-cut.
        # Acting on both means doing the action twice, which for a click is a real double-action.
        now = time.time()
        if stripped == self._last_transcript and (now - self._last_dispatch_at) < self.debounce_seconds:
            return self._drop("duplicate of the previous utterance", cleaned=stripped, had_wake_word=had_wake)

        self._last_transcript = stripped
        self._last_dispatch_at = now
        self.dispatched += 1
        return GateDecision(dispatch=True, reason="addressed to JARVIS", cleaned=stripped,
                            had_wake_word=had_wake)

    @staticmethod
    def _looks_like_command(text: str) -> bool:
        return any(re.match(rf"^{re.escape(v)}\b", text) for v in _IMPERATIVES)

    def _drop(self, reason: str, cleaned: str = "", had_wake_word: bool = False) -> GateDecision:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        return GateDecision(dispatch=False, reason=reason, cleaned=cleaned, had_wake_word=had_wake_word)

    # ------------------------------------------------------------------
    # Listener-loop helpers
    # ------------------------------------------------------------------

    def calibration_due(self) -> bool:
        """Ambient noise is stable for minutes. Re-measuring it every loop iteration blocks the
        microphone for 0.3s each time and buys nothing."""
        return (time.time() - self._last_calibration_at) >= self.recalibrate_every

    def note_calibration(self) -> None:
        self._last_calibration_at = time.time()

    def stats(self) -> Dict[str, Any]:
        return {
            "heard": self.heard,
            "dispatched": self.dispatched,
            "dropped": dict(self.dropped),
            "drop_rate": round(1 - (self.dispatched / self.heard), 3) if self.heard else 0.0,
        }
