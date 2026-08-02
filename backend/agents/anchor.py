"""ANCHOR — the screen-order grounding agent.

This is the agent that makes JARVIS *follow the order that was given*, against the screen that is
actually in front of it. Everything else in the cortex can be right and the run still fails if the
click lands on the wrong thing, so ANCHOR runs before every screen-touching action.

It answers three questions:

  1. WHAT DID THE ORDER NAME?      extract_referent("open the chat of Alice") -> "Alice"
  2. WHERE IS THAT ON SCREEN?      ground(referent, snapshot)  -> GroundedTarget(element, coords)
  3. DID WE ACTUALLY HIT IT?       verify_on_target(referent, result) -> (ok, reason)

The monolith had all three, badly, as three unrelated patches:

  * `_extract_named_target` looked for `\\b[A-Z][a-z]{2,}\\b` and subtracted a hand-listed verb set.
    That finds nothing at all in "open the chat of alice" (voice transcription is frequently
    lowercase), and finds "Chrome" in "Close Chrome" only by luck of capitalisation.
  * target matching was `named_target.lower() not in matched_name.lower()` — a substring test with
    no notion of confidence, ordinals, or near-misses.
  * the `[:25]` element truncation in the observation builder meant the correct element was often
    never even shown to the model, so grounding failed for reasons no amount of prompting fixes.

ANCHOR resolves the overwhelming majority of orders with pure string scoring and **zero model
calls**. It spends one cheap 8B call only when the top candidates are genuinely too close to call,
and one local vision call only when the accessibility tree has nothing plausible at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .base import (
    FAST_MODEL,
    Agent,
    AgentContext,
    GroundedTarget,
    LLMClient,
    ScreenElement,
    ScreenSnapshot,
    keywords,
    normalize,
    rank_elements,
    text_score,
)

# Confidence gates. Tuned so that the common, unambiguous case never spends a model call, and the
# genuinely ambiguous case never blind-clicks.
CONFIDENT = 0.72          # act immediately
PLAUSIBLE = 0.42          # act only if clearly ahead of the runner-up
DECISIVE_GAP = 0.14       # how far ahead "clearly ahead" is
VISION_FLOOR = 0.30       # below this the accessibility tree is not worth trusting at all

# Verbs that introduce a target rather than being part of one.
_ACTION_VERBS = (
    "open", "click", "double click", "doubleclick", "right click", "press", "push", "tap", "hit",
    "select", "choose", "show", "go to", "goto", "launch", "start", "run", "switch to", "switch",
    "focus on", "focus", "bring up", "find", "locate", "look for", "search for", "type in",
    "close", "quit", "exit", "minimise", "minimize", "maximise", "maximize", "scroll to",
    "navigate to", "head to", "pull up", "fire up", "boot up", "play", "check",
)

# Nouns that describe the *kind* of control rather than naming a specific one. Mapping them to UI
# Automation control types lets "the third tab" and "that button" narrow the candidate pool without
# a model call.
_TYPE_HINTS: Dict[str, str] = {
    "button": "Button", "btn": "Button",
    "tab": "TabItem",
    "link": "Hyperlink", "hyperlink": "Hyperlink",
    "menu": "MenuItem", "menu item": "MenuItem", "option": "MenuItem",
    "field": "Edit", "box": "Edit", "input": "Edit", "textbox": "Edit", "text box": "Edit",
    "bar": "Edit", "search bar": "Edit", "address bar": "Edit",
    "checkbox": "CheckBox", "check box": "CheckBox",
    "dropdown": "ComboBox", "combo": "ComboBox", "combobox": "ComboBox",
    "row": "ListItem", "item": "ListItem", "entry": "ListItem", "result": "ListItem",
    "chat": "ListItem", "conversation": "ListItem", "thread": "ListItem", "message": "ListItem",
    "contact": "ListItem", "profile": "ListItem", "video": "ListItem", "song": "ListItem",
    "track": "ListItem", "post": "ListItem", "file": "ListItem", "folder": "ListItem",
    "icon": "Button", "avatar": "Button", "picture": "Image", "image": "Image",
    "window": "Window", "panel": "Pane",
}

_ORDINAL_WORDS: Dict[str, int] = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9, "tenth": 10, "10th": 10, "last": -1, "top": 1, "bottom": -1,
}

# Possessive / relational framings that wrap a real target: "the chat OF Bob", "Bob's chat".
_RELATIONAL = re.compile(
    r"\b(?:chat|conversation|thread|dm|message|messages|profile|page|window|tab|folder|file)\s+"
    r"(?:of|with|for|from|to)\s+(.+)$",
    re.IGNORECASE,
)
_POSSESSIVE = re.compile(r"^(.+?)'s\s+(?:chat|conversation|thread|dm|messages?|profile|page)\b", re.IGNORECASE)

# Interrogative phrasing. "is Send visible on screen" is a question about the screen, not an order
# to click Send — grounding it would turn every question into an action.
#
# The old engine guarded this with a flat leading-word regex that also rejected "could you open
# main.py", a perfectly ordinary polite command. The distinction that actually matters is whether a
# real action verb follows: a polite modal plus an action verb is a request, not a question.
_INTERROGATIVE = re.compile(
    r"^\s*(why|what|whats|what's|who|whose|how|when|where|which|is|are|was|were|do|does|did|"
    r"has|have|should|am)\b",
    re.IGNORECASE,
)
_POLITE_MODAL = re.compile(r"^\s*(can|could|would|will|may|please)\b", re.IGNORECASE)

# Anaphora — orders that point at something established earlier rather than naming it.
_ANAPHORA = re.compile(
    r"^(?:that|this|it|them|those|these|the same|again|do it again|same thing)\b|"
    r"\b(?:that one|this one|the other one|the same one)\b",
    re.IGNORECASE,
)

# Words that can only ever be pointing back at something already established, never naming a new
# target. If nothing but these survives cleanup, the order is anaphoric and history must supply the
# real referent.
_ANAPHORIC_TOKENS = {
    "it", "this", "that", "them", "those", "these", "one", "thing", "again", "same", "other",
    "current", "active", "do",
}

# Verbs whose quoted span is the *content being written*, not the thing to click. Without this,
# "send 'hello there' to Bob" grounds on the message text and clicks whatever on screen happens to
# say "hello there" — instead of finding Bob.
_SPEECH_VERBS = (
    "send", "say", "type", "write", "tell", "reply", "respond", "text", "message", "dm", "post",
    "comment", "ask",
)

# Function words that can never be a target on their own. Stripped repeatedly, because removing a
# type noun from the middle of a phrase routinely leaves an orphaned article behind
# ("the search bar" -> strip "search bar" -> "the").
_DETERMINERS = {"the", "a", "an", "my", "this", "that", "me", "us", "our", "some", "any", "on", "at", "of", "to"}


def _strip_determiners(text: str) -> str:
    tokens = [t for t in re.split(r"\s+", (text or "").strip()) if t]
    while tokens and tokens[0].lower().strip(",.!?'\"") in _DETERMINERS:
        tokens.pop(0)
    while tokens and tokens[-1].lower().strip(",.!?'\"") in _DETERMINERS:
        tokens.pop()
    return " ".join(tokens).strip(" ,.!?")


def _only_type_noun(text: str, type_hint: str) -> bool:
    """True when the referent is nothing but its control-type noun plus filler ("this tab")."""
    if not text or not type_hint:
        return False
    residual = [t for t in normalize(text).split() if t not in _DETERMINERS]
    return bool(residual) and all(_TYPE_HINTS.get(t) == type_hint for t in residual)


@dataclass
class Referent:
    """The parsed 'thing the order is about'."""

    text: str = ""
    # The referent with its control-type noun left in. "Saved Tab Groups" is a real control name
    # that merely contains the word "tab"; stripping it to "Saved Groups" and filtering the screen
    # down to TabItems hides the very element being asked for. ground() falls back to this when the
    # type-hinted interpretation finds nothing convincing.
    full_text: str = ""
    ordinal: Optional[int] = None
    type_hint: str = ""
    is_anaphoric: bool = False
    is_quoted: bool = False
    is_question: bool = False
    payload: str = ""
    raw: str = ""

    @property
    def empty(self) -> bool:
        if self.is_question:
            return True
        return not self.text and self.ordinal is None and not self.type_hint

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "ordinal": self.ordinal, "type_hint": self.type_hint,
            "is_anaphoric": self.is_anaphoric, "is_quoted": self.is_quoted,
            "payload": self.payload,
        }

    def describe(self) -> str:
        bits = []
        if self.ordinal:
            bits.append(f"#{self.ordinal}")
        if self.text:
            bits.append(f'"{self.text}"')
        if self.type_hint:
            bits.append(f"({self.type_hint})")
        if not bits and self.is_anaphoric:
            return "(the thing referred to earlier)"
        return " ".join(bits) or "(nothing named)"


class AnchorAgent(Agent):
    """Binds orders to on-screen reality."""

    name = "ANCHOR"

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(llm)
        self.groundings: int = 0
        self.model_disambiguations: int = 0
        self.vision_escalations: int = 0

    # ==============================================================================
    # 1. WHAT DID THE ORDER NAME?
    # ==============================================================================

    def extract_referent(self, order: str, history: Optional[List[Dict[str, str]]] = None) -> Referent:
        """Pulls the target out of an order, case-insensitively and without relying on the user
        capitalising proper nouns (voice transcription usually doesn't)."""
        raw = order or ""
        cmd = raw.strip()

        # The wake word is never semantic content. This project's own folder is literally named
        # "jarvis", so every VS Code window title in this repo contains it — without stripping,
        # saying "...Jarvis?" fuzzy-matches that window and triggers a blind click on it. That was
        # an observed bug in the monolith, preserved here as a hard rule.
        cmd = re.sub(r"\bjarvis\b", " ", cmd, flags=re.IGNORECASE)
        cmd = re.sub(r"\bboss\b", " ", cmd, flags=re.IGNORECASE)
        cmd = re.sub(r"\s+", " ", cmd).strip(" ,.!?")

        anaphoric_marker = bool(_ANAPHORA.search(cmd))

        # A polite modal is not part of the target: "could you please open main.py" is an order.
        # Stripping it first also lets the verb test below see the real verb.
        had_modal = bool(_POLITE_MODAL.match(cmd))
        body = _POLITE_MODAL.sub("", cmd, count=1).strip()
        body = re.sub(r"^\s*(you|u)\b\s*", "", body, flags=re.IGNORECASE).strip()
        body = re.sub(r"^\s*(please|kindly|just)\b\s*", "", body, flags=re.IGNORECASE).strip()

        has_action_verb = any(
            re.match(rf"^{re.escape(v)}\b", body, flags=re.IGNORECASE)
            for v in _ACTION_VERBS + _SPEECH_VERBS
        )
        # A question about the screen must never become a click on it.
        #
        # The deciding factor is whether a real action verb follows, which is what separates
        # "could you open main.py" (an order) from "can you see my screen" (a question) — both open
        # with a modal, so the modal alone settles nothing.
        #
        # Anaphoric orders are exempt: "do it again" opens with "do", an interrogative word in
        # "does it work" but plainly imperative here. The anaphora pattern is anchored, so it
        # separates the two without needing a parts-of-speech tagger.
        if not has_action_verb and not anaphoric_marker and (
            had_modal or _INTERROGATIVE.match(body) or cmd.rstrip().endswith("?")
        ):
            return Referent(is_question=True, raw=raw)

        # Strip a leading action verb so "open settings" grounds on "settings", not "open settings".
        leading_verb = ""
        for verb in sorted(_ACTION_VERBS + _SPEECH_VERBS, key=len, reverse=True):
            m = re.match(rf"^{re.escape(verb)}\b\s*(.*)$", body, flags=re.IGNORECASE)
            if m:
                leading_verb = verb.lower()
                body = m.group(1).strip()
                break

        # A quoted span is an exact literal the user chose deliberately. For a writing verb it is
        # the message *content*; for everything else it is the target's exact label.
        payload = ""
        quoted = re.search(r"[\"'‘“]([^\"'’”]{2,})[\"'’”]", body)
        if quoted:
            if leading_verb in _SPEECH_VERBS:
                payload = quoted.group(1).strip()
                body = (body[: quoted.start()] + " " + body[quoted.end():]).strip(" ,.!?")
            else:
                return Referent(text=quoted.group(1).strip(), is_quoted=True, raw=raw,
                                type_hint=self._sniff_type_hint(cmd))

        body = re.sub(r"\s+(please|now|for me|thanks|thank you|will you|would you)\s*$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"^(?:on|at|the|a|an|this|that|my)\s+", "", body, flags=re.IGNORECASE).strip(" ,.!?")

        ordinal = self._sniff_ordinal(body)
        type_hint = self._sniff_type_hint(body)

        # "the chat of Bob" / "Bob's chat" -> Bob. Done before generic cleanup because the relational
        # noun ("chat") is a type hint, not the name to match.
        rel = _RELATIONAL.search(body)
        if rel:
            body = rel.group(1).strip(" ,.!?")
        else:
            poss = _POSSESSIVE.match(body)
            if poss:
                body = poss.group(1).strip(" ,.!?")
            elif payload or leading_verb in _SPEECH_VERBS:
                # "send ... to Bob" / "message Bob" — the recipient after the preposition is the
                # on-screen target; anything before it was the content.
                tail = re.search(r"\b(?:to|for|at)\s+(.+)$", body, flags=re.IGNORECASE)
                said = re.search(r"^(.+?)\s+(?:saying|that says|and say|to say|telling (?:him|her|them))\s+(.+)$",
                                 body, flags=re.IGNORECASE)
                if said:
                    # "message Alice saying hey" — recipient before the verb, content after.
                    body, payload = said.group(1).strip(" ,.!?"), payload or said.group(2).strip(" ,.!?")
                elif tail:
                    body = tail.group(1).strip(" ,.!?")
                elif not payload and body:
                    # "type hello world" with no quotes and no recipient: the remainder is content,
                    # and there is no screen target to bind at all.
                    payload, body = body, ""

        body = self._strip_ordinal_words(body)
        full_body = _strip_determiners(body)
        body = self._strip_type_nouns(body, type_hint, allow_empty=ordinal is not None or _only_type_noun(body, type_hint))
        body = _strip_determiners(body)

        # If nothing but pointing-words survives ("do it again", "close this"), there is no name to
        # match and history has to supply the referent.
        residual = [t for t in normalize(body).split() if t]
        if residual and all(t in _ANAPHORIC_TOKENS for t in residual):
            body, anaphoric_marker = "", True

        return Referent(text=body, full_text=full_body, ordinal=ordinal, type_hint=type_hint,
                        payload=payload, is_anaphoric=anaphoric_marker and not body, raw=raw)

    @staticmethod
    def _strip_type_nouns(text: str, type_hint: str, allow_empty: bool = False) -> str:
        """Removes the control-type noun once it has been captured as a hint, so "the Send button"
        grounds on "Send" — the element's actual accessible name is almost never "Send button".

        `allow_empty` permits stripping down to nothing, which is correct when something else
        already identifies the target: an ordinal ("the 3rd video" -> #3 of the ListItems) or the
        type noun being the entire referent ("close this tab" -> the active TabItem).
        """
        if not text or not type_hint:
            return text
        stripped = text
        for phrase, mapped in sorted(_TYPE_HINTS.items(), key=lambda kv: len(kv[0]), reverse=True):
            if mapped != type_hint:
                continue
            candidate = re.sub(rf"\b{re.escape(phrase)}\b", " ", stripped, flags=re.IGNORECASE)
            candidate = re.sub(r"\s+", " ", candidate).strip(" ,.!?")
            if candidate or allow_empty:
                stripped = candidate
        return stripped

    @staticmethod
    def _sniff_ordinal(text: str) -> Optional[int]:
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        for word, val in _ORDINAL_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
                return val
        return None

    @staticmethod
    def _strip_ordinal_words(text: str) -> str:
        text = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\b", " ", text, flags=re.IGNORECASE)
        for word in _ORDINAL_WORDS:
            text = re.sub(rf"\b{re.escape(word)}\b", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" ,.!?")

    @staticmethod
    def _sniff_type_hint(text: str) -> str:
        low = normalize(text)
        for phrase in sorted(_TYPE_HINTS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", low):
                return _TYPE_HINTS[phrase]
        return ""

    # ==============================================================================
    # 2. WHERE IS IT ON SCREEN?
    # ==============================================================================

    def score_candidates(
        self, referent: Referent, snapshot: ScreenSnapshot
    ) -> List[Tuple[float, ScreenElement]]:
        """Scores every on-screen element against the referent. Pure, deterministic, no I/O —
        which is what makes ANCHOR's behaviour assertable in tests rather than vibes."""
        pool: List[ScreenElement] = [e for e in snapshot.elements if e.name and len(e.name) >= 2]

        if referent.type_hint:
            typed = [e for e in pool if referent.type_hint.lower() in e.type.lower()]
            # Only narrow if the hint actually matches something; a wrong guess about the control
            # type must never be able to hide the correct element entirely.
            if typed:
                pool = typed

        scored: List[Tuple[float, ScreenElement]] = []
        for e in pool:
            if referent.text:
                score = text_score(referent.text, e.name)
            else:
                # No name given (pure ordinal/type order) — every type-matching element is an equal
                # candidate and the ordinal alone decides.
                score = 0.5 if referent.type_hint else 0.0

            if score <= 0:
                continue
            if e.is_interactive:
                score += 0.06
            if referent.type_hint and referent.type_hint.lower() in e.type.lower():
                score += 0.08
            # A screen-filling control is usually a container that merely *contains* the match
            # rather than the thing to click. Scoped to non-interactive types: a large Pane or
            # Window is structural, but a large Button or ListItem is a genuine target, and
            # penalising those loses real matches like the pane named "Chrome Legacy Window" that a
            # user can legitimately ask for by name.
            if not e.is_interactive and e.area > 0 and e.width > 1200 and e.height > 700:
                score -= 0.12
            scored.append((min(score, 1.0), e))

        scored.sort(key=lambda t: (-t[0], t[1].top, t[1].left))
        return scored

    def ground(
        self,
        referent: Referent,
        snapshot: ScreenSnapshot,
        ctx: Optional[AgentContext] = None,
        vision_locate: Optional[Callable[[str], Optional[Tuple[int, int]]]] = None,
        allow_model: bool = True,
    ) -> GroundedTarget:
        """Resolves a referent to a concrete on-screen element."""
        self.groundings += 1
        label = referent.describe()

        if referent.empty:
            return GroundedTarget(referent=label, method="none", reason="the order named no on-screen target")

        scored = self.score_candidates(referent, snapshot)

        # Retry without the control-type interpretation when it found nothing convincing.
        #
        # A type noun inside a real control name ("Saved Tab Groups", "Chrome Legacy Window") gets
        # read as a descriptor, which both strips it out of the target text and narrows the pool to
        # that control type — hiding the exact element being asked for. Measured against a live
        # Chrome window, this single case accounted for every grounding miss.
        # Always scored, not just on low confidence: "click Chrome Legacy Window" reaches CONFIDENT
        # on the *wrong* element (a button named "Chrome") once "window" has been stripped away, so
        # a confidence gate would never fire. Comparing both readings and keeping the stronger one
        # costs microseconds of pure string work and no model call.
        if referent.type_hint and referent.full_text and referent.full_text != referent.text:
            plain = Referent(text=referent.full_text, ordinal=referent.ordinal,
                             payload=referent.payload, raw=referent.raw)
            alt = self.score_candidates(plain, snapshot)
            if alt and (not scored or alt[0][0] > scored[0][0]):
                scored = alt
                referent = plain

        # --- ordinal selection: "the 3rd profile" is positional, not a similarity question -------
        if referent.ordinal is not None:
            pool = [e for _, e in scored] if scored else list(snapshot.elements)
            pool = sorted(pool, key=lambda e: (e.top, e.left))
            if not pool:
                return GroundedTarget(referent=label, method="none", reason="nothing on screen to count")
            idx = len(pool) - 1 if referent.ordinal == -1 else min(referent.ordinal - 1, len(pool) - 1)
            idx = max(0, idx)
            chosen = pool[idx]
            self._emit(ctx, "ground", f"{label} -> ordinal pick '{chosen.name}'", element=chosen.as_dict())
            return GroundedTarget(
                referent=label, element=chosen, confidence=0.78, method="ordinal",
                reason=f"picked #{referent.ordinal} of {len(pool)} matching controls in reading order",
                alternatives=[e for e in pool[:5] if e is not chosen],
            )

        if not scored:
            return self._vision_fallback(referent, label, ctx, vision_locate,
                                         "no element name resembled the target")

        top_score, top_el = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        gap = top_score - runner_up
        alternatives = [e for _, e in scored[1:6]]

        # --- confident: act now, no model call -------------------------------------------------
        if top_score >= CONFIDENT and (gap >= DECISIVE_GAP or len(scored) == 1):
            method = "exact" if top_score >= 0.97 else "strong"
            self._emit(ctx, "ground", f"{label} -> '{top_el.name}' ({method}, {top_score:.2f})",
                       element=top_el.as_dict())
            return GroundedTarget(referent=label, element=top_el, confidence=top_score, method=method,
                                  reason=f"name match {top_score:.2f}, clear of runner-up by {gap:.2f}",
                                  alternatives=alternatives)

        # --- plausible and clearly ahead: still no model call ----------------------------------
        if top_score >= PLAUSIBLE and gap >= DECISIVE_GAP:
            self._emit(ctx, "ground", f"{label} -> '{top_el.name}' (lead, {top_score:.2f})",
                       element=top_el.as_dict())
            return GroundedTarget(referent=label, element=top_el, confidence=top_score, method="strong",
                                  reason=f"best of {len(scored)} candidates, ahead by {gap:.2f}",
                                  alternatives=alternatives)

        # --- genuinely ambiguous: one cheap call over a shortlist, not the whole tree -----------
        if allow_model and top_score >= VISION_FLOOR and len(scored) > 1:
            picked = self._disambiguate(referent, [e for _, e in scored[:8]], snapshot, ctx)
            if picked is not None:
                return GroundedTarget(referent=label, element=picked, confidence=max(top_score, 0.66),
                                      method="disambiguated",
                                      reason="top candidates were too close to call; picked by model from shortlist",
                                      alternatives=alternatives)

        # --- accessibility tree is not trustworthy here: spend the vision call ------------------
        if top_score < CONFIDENT:
            vt = self._vision_fallback(referent, label, ctx, vision_locate,
                                       f"best tree match was only {top_score:.2f}")
            if vt.resolved:
                return vt

        if top_score >= PLAUSIBLE:
            return GroundedTarget(referent=label, element=top_el, confidence=top_score, method="weak",
                                  reason=f"low-confidence match ({top_score:.2f}) — verify after acting",
                                  alternatives=alternatives)

        return GroundedTarget(referent=label, method="none", confidence=top_score,
                              reason=f"nothing on screen convincingly matches {label}",
                              alternatives=alternatives)

    def ground_order(
        self,
        order: str,
        snapshot: ScreenSnapshot,
        ctx: Optional[AgentContext] = None,
        vision_locate: Optional[Callable[[str], Optional[Tuple[int, int]]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        allow_model: bool = True,
    ) -> GroundedTarget:
        """extract + ground in one call — the entry point HANDS uses."""
        referent = self.extract_referent(order, history)
        if referent.is_anaphoric and history:
            resolved = self.resolve_anaphora(history)
            if resolved:
                referent = Referent(text=resolved, type_hint=referent.type_hint,
                                    ordinal=referent.ordinal, raw=order)
                self._emit(ctx, "ground", f"resolved '{order}' to earlier target '{resolved}'")
        return self.ground(referent, snapshot, ctx, vision_locate, allow_model)

    def _disambiguate(
        self, referent: Referent, shortlist: List[ScreenElement],
        snapshot: ScreenSnapshot, ctx: Optional[AgentContext],
    ) -> Optional[ScreenElement]:
        """One 8B call over <=8 candidates. Costs a fraction of the monolith's approach of shipping
        the whole element dump plus 21 tool schemas to a 70B model on every step."""
        self.model_disambiguations += 1
        options = "\n".join(f"{i + 1}. [{e.short_type}] {e.name}" for i, e in enumerate(shortlist))
        system = (
            "You match a user's spoken target to one of the numbered on-screen controls. "
            'Reply ONLY as {"choice": <number>, "confidence": <0-1>} — or {"choice": 0} if none of '
            "them is what the user meant."
        )
        user = (
            f'The user asked for: {referent.describe()}\n'
            f'Full order: "{referent.raw}"\n'
            f'Active window: "{snapshot.window_title}"\n\n'
            f"Candidates:\n{options}"
        )
        data, error = self.llm.json_call(system, user, agent=self.name, model=FAST_MODEL,
                                         timeout=12, max_tokens=80)
        if error or not data:
            self._emit(ctx, "status", f"disambiguation unavailable ({error}) — using best string match")
            return None
        try:
            choice = int(data.get("choice", 0))
        except (TypeError, ValueError):
            return None
        if 1 <= choice <= len(shortlist):
            picked = shortlist[choice - 1]
            self._emit(ctx, "ground", f"{referent.describe()} -> '{picked.name}' (model pick)",
                       element=picked.as_dict())
            return picked
        return None

    def _vision_fallback(
        self, referent: Referent, label: str, ctx: Optional[AgentContext],
        vision_locate: Optional[Callable[[str], Optional[Tuple[int, int]]]], why: str,
    ) -> GroundedTarget:
        if vision_locate is None or not referent.text:
            return GroundedTarget(referent=label, method="none", reason=why)
        self.vision_escalations += 1
        self._emit(ctx, "status", f"can't place {label} from the accessibility tree — looking with vision")
        coords = None
        try:
            coords = vision_locate(referent.text)
        except Exception as e:  # noqa: BLE001 — a vision failure must degrade, never crash the run
            self._emit(ctx, "status", f"vision lookup failed: {e}")
        if not coords:
            return GroundedTarget(referent=label, method="none", reason=f"{why}; vision could not see it either")
        el = ScreenElement(name=referent.text, type="VisionTarget", x=int(coords[0]), y=int(coords[1]))
        return GroundedTarget(referent=label, element=el, confidence=0.60, method="vision",
                              reason=f"{why}; located visually at {coords}")

    # ==============================================================================
    # 3. DID WE ACTUALLY HIT IT?  (the scope guard)
    # ==============================================================================

    def verify_on_target(self, referent: Referent, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks that what the action actually touched is what the order named.

        This is the guard that stops the classic failure the monolith kept hitting: told to "open
        the chat of Alice", it clicked a generic "Instagram Messages" tab, the click returned
        success, and the run declared victory. A successful click is not evidence of a *correct*
        click.
        """
        if referent.empty or not referent.text:
            return True, "no specific target was named, nothing to mismatch"
        if result.get("status") != "success":
            return False, "the action itself did not succeed"

        touched = str(result.get("matched_name") or result.get("title") or result.get("app") or "").strip()
        if not touched:
            # Coordinate clicks and keystrokes report no name. There is nothing to compare, so this
            # is explicitly *not* treated as a match — it is "unknown", and the caller should lean
            # on SENTINEL's screen-diff evidence instead of ANCHOR's name check.
            return True, "action reported no target name — deferring to screen verification"

        score = text_score(referent.text, touched)
        if score >= PLAUSIBLE:
            return True, f"'{touched}' matches the requested {referent.describe()} ({score:.2f})"
        return False, (
            f"the order named {referent.describe()} but the action landed on '{touched}' "
            f"(match {score:.2f}) — that is very likely not the thing that was asked for"
        )

    def scope_report(self, order: str, referent: Referent, result: Dict[str, Any]) -> str:
        """Human-readable correction fed back to the executor when the scope guard trips, so the
        next step is a genuine course-correction rather than a repeat of the same wrong click."""
        ok, reason = self.verify_on_target(referent, result)
        if ok:
            return ""
        return (
            f"SCOPE WARNING: {reason}. The order was: \"{order}\". Do not report this as done. "
            f"Look for '{referent.text}' specifically on the current screen — it may only have "
            f"become visible after that last action."
        )

    # ==============================================================================
    # Follow-up resolution — makes "do it again" / "the other one" work at all
    # ==============================================================================

    def resolve_anaphora(self, history: Sequence[Dict[str, str]]) -> str:
        """Recovers the most recent concrete target from conversation history.

        The monolith appended to `self.history` in seven places and read it in exactly zero, so
        every follow-up order arrived with no context whatsoever.
        """
        for turn in reversed(list(history)[-6:]):
            for field_name in ("target", "bot", "user"):
                text = str(turn.get(field_name) or "")
                if not text:
                    continue
                quoted = re.search(r"'([^']{2,60})'", text)
                if quoted:
                    return quoted.group(1)
            user_text = str(turn.get("user") or "")
            if user_text and not _ANAPHORA.search(user_text):
                ref = self.extract_referent(user_text)
                if ref.text:
                    return ref.text
        return ""

    # ==============================================================================
    # Convenience used by TRIAGE's fast path
    # ==============================================================================

    def quick_match(self, order: str, snapshot: ScreenSnapshot) -> Optional[ScreenElement]:
        """Zero-LLM, zero-vision resolution or nothing. Used to answer trivially-grounded orders
        without entering the execution loop at all."""
        target = self.ground_order(order, snapshot, allow_model=False)
        if target.resolved and target.confidence >= CONFIDENT and target.method != "vision":
            return target.element
        return None
