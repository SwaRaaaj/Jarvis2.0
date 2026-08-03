"""SCRIBE — transcription accuracy.

Speech recognition returns a *ranked guess list*, and the old listener threw all of it away:
`recognize_google(audio)` hands back only the single top string. That top guess is chosen with no
idea what you were looking at, which is why domain words come back mangled — contact names, app
names and the wake word itself are exactly the vocabulary a generic model is worst at.

SCRIBE keeps the alternatives and re-ranks them against what is actually true right now:

    * the names of every control currently on screen  (RETINA)
    * every app, website and platform JARVIS can open (the alias tables)
    * commands the user has actually given before     (SCHOLAR)
    * the wake word and the action verbs

So if the screen shows a conversation with "Alice Johnson" and the recogniser's alternatives are
["a list johnson", "alice johnson", "at least johnson"], the one that matches something really on
screen wins — even when it was not the recogniser's own first choice.

It also repairs single tokens inside an otherwise-good transcript ("open crome" -> "open chrome"),
which is the most common failure of all and costs nothing to fix.

Everything here is deterministic string work: no model call, no network, and it degrades to
returning the recogniser's original top guess if nothing better is found.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import Agent, AgentContext, LLMClient, normalize, text_score

try:
    from os_automation import APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES
except Exception:  # pragma: no cover
    APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES = {}, {}, {}

# A single word is only swapped when the replacement is clearly better than what was heard. Too
# low and "open the door" becomes "open the word"; too high and nothing is ever repaired.
TOKEN_REPAIR_FLOOR = 0.72

# How much an alternative may be boosted for matching real on-screen text. Large enough to
# overturn the recogniser's own ranking, which is the entire point.
CONTEXT_BOOST = 0.55

# Command words worth protecting from mishearing, because getting them wrong changes the action.
ACTION_VOCAB = (
    "open", "close", "click", "type", "send", "search", "play", "pause", "stop", "scroll",
    "switch", "launch", "start", "minimize", "maximize", "select", "copy", "paste", "delete",
    "next", "previous", "back", "forward", "refresh", "save", "mute", "volume", "screenshot",
    "jarvis", "message", "reply", "read", "show", "tell", "find", "google", "youtube",
)


@dataclass
class Transcript:
    """A finished transcription decision."""

    text: str = ""
    original: str = ""
    confidence: float = 0.0
    source: str = "asr"           # asr | reranked | repaired
    corrections: List[Tuple[str, str]] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return normalize(self.text) != normalize(self.original)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "original": self.original,
            "confidence": round(self.confidence, 3), "source": self.source,
            "corrections": [f"{a} -> {b}" for a, b in self.corrections],
        }


class ScribeAgent(Agent):
    """Turns a ranked ASR guess list into the most plausible thing the user actually said."""

    name = "SCRIBE"

    def __init__(self, llm: Optional[LLMClient] = None, retina: Any = None, scholar: Any = None):
        super().__init__(llm)
        self.retina = retina
        self.scholar = scholar
        self.transcriptions: int = 0
        self.reranked: int = 0
        self.repaired: int = 0

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def vocabulary(self, include_screen: bool = True) -> List[str]:
        """Everything it is currently plausible for the user to have said.

        Rebuilt per utterance rather than cached, because the most valuable half of it — the text
        actually on screen — changes constantly, and stale vocabulary is worse than none.
        """
        vocab: List[str] = list(ACTION_VOCAB)
        vocab += list(APP_ALIASES.keys()) + list(WEBSITE_ALIASES.keys()) + list(SOCIAL_INBOX_URLS.keys())

        if include_screen and self.retina is not None:
            try:
                snapshot = self.retina.snapshot()
                for element in snapshot.elements[:120]:
                    name = (element.name or "").strip()
                    if 2 < len(name) < 48:
                        vocab.append(name)
                title = (snapshot.window_title or "").strip()
                if title:
                    vocab.append(title[:48])
            except Exception:
                pass

        if self.scholar is not None:
            try:
                for rule in list(self.scholar._cache.keys())[:50]:
                    vocab.append(rule)
            except Exception:
                pass

        seen, unique = set(), []
        for phrase in vocab:
            key = normalize(phrase)
            if key and key not in seen:
                seen.add(key)
                unique.append(phrase)
        return unique

    # ------------------------------------------------------------------
    # The main entry point
    # ------------------------------------------------------------------

    def transcribe(
        self,
        alternatives: Sequence[Any],
        ctx: Optional[AgentContext] = None,
        include_screen: bool = True,
    ) -> Transcript:
        """Picks the best transcription from a ranked alternative list and repairs its tokens.

        `alternatives` accepts what SpeechRecognition's `show_all=True` produces — a list of
        {"transcript": ..., "confidence": ...} dicts — or plain strings, so a caller with only one
        guess still benefits from token repair.
        """
        self.transcriptions += 1
        parsed = self._parse_alternatives(alternatives)
        if not parsed:
            return Transcript()

        top_text, top_conf = parsed[0]
        vocab = self.vocabulary(include_screen=include_screen)

        # --- 1. re-rank the alternatives against real context ---------------------------
        best_text, best_score, best_conf = top_text, self._context_score(top_text, vocab) + top_conf, top_conf
        for text, conf in parsed[1:6]:
            score = self._context_score(text, vocab) + conf
            if score > best_score:
                best_text, best_score, best_conf = text, score, conf

        source = "asr"
        if normalize(best_text) != normalize(top_text):
            self.reranked += 1
            source = "reranked"
            self._emit(ctx, "status", f"heard '{top_text}' but '{best_text}' matches the screen")

        # --- 2. repair individual mis-heard tokens --------------------------------------
        repaired, corrections = self._repair_tokens(best_text, vocab)
        if corrections:
            self.repaired += 1
            source = "repaired" if source == "asr" else source
            self._emit(ctx, "status", "corrected " + ", ".join(f"{a}->{b}" for a, b in corrections))

        return Transcript(
            text=repaired, original=top_text, confidence=max(best_conf, 0.0), source=source,
            corrections=corrections, alternatives=[t for t, _ in parsed[:5]],
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_alternatives(alternatives: Sequence[Any]) -> List[Tuple[str, float]]:
        """Normalises SpeechRecognition's several shapes into (text, confidence) pairs."""
        if isinstance(alternatives, str):
            return [(alternatives.strip(), 0.0)] if alternatives.strip() else []
        if isinstance(alternatives, dict):
            alternatives = alternatives.get("alternative") or []

        parsed: List[Tuple[str, float]] = []
        for item in alternatives or []:
            if isinstance(item, str):
                text, conf = item, 0.0
            elif isinstance(item, dict):
                text = str(item.get("transcript") or item.get("text") or "")
                try:
                    conf = float(item.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
            else:
                continue
            text = text.strip()
            if text:
                parsed.append((text, conf))
        return parsed

    @staticmethod
    def _context_score(text: str, vocab: Iterable[str]) -> float:
        """How much of this transcription corresponds to something really present right now."""
        tokens = [t for t in normalize(text).split() if len(t) > 2]
        if not tokens:
            return 0.0
        vocab_norm = [normalize(v) for v in vocab]
        hits = 0.0
        for token in tokens:
            best = 0.0
            for phrase in vocab_norm:
                if token == phrase or token in phrase.split():
                    best = 1.0
                    break
                score = text_score(token, phrase)
                if score > best:
                    best = score
            if best >= 0.72:
                hits += 1.0
        return CONTEXT_BOOST * (hits / len(tokens))

    @staticmethod
    def _repair_tokens(text: str, vocab: Iterable[str]) -> Tuple[str, List[Tuple[str, str]]]:
        """Fixes single mis-heard words against the vocabulary ("crome" -> "chrome").

        Only single words are considered, and only when the replacement is a clear improvement —
        an over-eager corrector that rewrites ordinary English into command words would be far
        worse than leaving the transcript alone.
        """
        single_words = sorted({v for v in vocab if v and " " not in v and len(v) > 3},
                              key=len)
        if not single_words:
            return text, []
        lowered = [w.lower() for w in single_words]

        corrections: List[Tuple[str, str]] = []
        out_tokens: List[str] = []
        for raw in text.split():
            token = raw.strip(".,!?;:")
            low = token.lower()
            if len(low) < 4 or low in lowered:
                out_tokens.append(raw)
                continue
            match = difflib.get_close_matches(low, lowered, n=1, cutoff=TOKEN_REPAIR_FLOOR)
            if match and match[0] != low:
                replacement = single_words[lowered.index(match[0])]
                corrections.append((token, replacement))
                out_tokens.append(raw.replace(token, replacement))
            else:
                out_tokens.append(raw)
        return " ".join(out_tokens), corrections

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "transcriptions": self.transcriptions,
            "reranked": self.reranked,
            "token_repairs": self.repaired,
            "correction_rate": round((self.reranked + self.repaired) / self.transcriptions, 3)
            if self.transcriptions else 0.0,
        }
