"""Shared contracts, scoring helpers and the pooled Groq client used by every agent.

Everything here is deliberately free of Windows/screen/network dependencies at import time so the
whole agent layer can be unit-tested on any machine with fake snapshots and a fake LLM.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------------
# Model tiers
# ----------------------------------------------------------------------------------
# The monolith used one 70B model for everything, including questions like "is this small talk?"
# that an 8B model answers just as correctly for a fraction of the latency and token budget. Groq's
# per-minute token bucket is a real constraint this project already fights (see the 429 retry-after
# handling below), so routing cheap questions to the cheap model is a direct throughput win.

FAST_MODEL = "llama-3.1-8b-instant"       # classification, disambiguation, yes/no judgments
SMART_MODEL = "llama-3.3-70b-versatile"   # planning, tool selection, prose
VISION_MODEL = "gemma3:4b"                # local via Ollama; Groq has no vision model on this account

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def load_local_env_file(path: Optional[str] = None) -> Dict[str, str]:
    """Reads backend/.env. A newly-set persistent Windows env var only reaches processes started
    after it was set, so an already-running terminal can serve a stale environment indefinitely;
    reading the file sidesteps that entirely. (Same rationale as the original ollama_engine helper,
    kept here so the agent layer doesn't have to import the monolith.)"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        pass
    return values


# ----------------------------------------------------------------------------------
# Screen contracts
# ----------------------------------------------------------------------------------

INTERACTIVE_TYPES = (
    "Button", "ListItem", "TabItem", "MenuItem", "Edit", "Hyperlink", "CheckBox",
    "ComboBox", "RadioButton", "TreeItem", "SplitButton", "Slider", "Document",
)


@dataclass(frozen=True)
class ScreenElement:
    """One named, visible control from the UI Automation tree."""

    name: str
    type: str
    x: int
    y: int
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def is_interactive(self) -> bool:
        return any(t in self.type for t in INTERACTIVE_TYPES)

    @property
    def short_type(self) -> str:
        return self.type.replace("Control", "")

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "type": self.type, "x": self.x, "y": self.y,
            "left": self.left, "top": self.top, "width": self.width, "height": self.height,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScreenElement":
        return ScreenElement(
            name=str(d.get("name", "")).strip(),
            type=str(d.get("type", "")),
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            left=int(d.get("left", 0)),
            top=int(d.get("top", 0)),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )

    def describe(self) -> str:
        return f'[{self.short_type}] "{self.name[:60]}" at ({self.x},{self.y})'


@dataclass
class ScreenSnapshot:
    """An immutable-ish view of the screen at one instant.

    RETINA produces these; ANCHOR, HANDS and SENTINEL consume them. Holding the *whole* element
    list (rather than the monolith's hard `[:25]` truncation) matters: truncating by screen
    position alone made element 26+ invisible to the model, which is a real cause of spurious
    "no element found" errors that then escalated to a slow vision call for no reason.
    """

    window_title: str = ""
    window_app: str = ""
    elements: List[ScreenElement] = field(default_factory=list)
    frame_hash: str = ""
    captured_at: float = field(default_factory=time.time)
    stale: bool = False

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def interactive(self) -> List[ScreenElement]:
        return [e for e in self.elements if e.is_interactive]

    def names(self) -> List[str]:
        return [e.name for e in self.elements]

    def find_all(self, needle: str) -> List[ScreenElement]:
        n = needle.lower().strip()
        if not n:
            return []
        return [e for e in self.elements if n in e.name.lower()]

    def contains_text(self, needle: str) -> bool:
        return bool(self.find_all(needle))

    def signature(self) -> str:
        """Cheap structural fingerprint used by SENTINEL to tell "the screen changed" from
        "the screen is identical", without diffing every element pair."""
        parts = [self.window_title, str(len(self.elements))]
        parts += sorted(e.name for e in self.elements[:40])
        return "|".join(parts)

    def describe_for_prompt(self, query: str = "", budget: int = 28) -> str:
        """Renders the screen for a model prompt, selecting which elements survive the budget by
        *relevance to the query* rather than by raw screen position.

        This is the fix for the monolith's `sorted(...)[:25]`: if the user said "open the chat of
        Arundhati" and Arundhati's row was the 40th element in reading order, the old renderer
        simply never showed it to the model, which then reported that no such element existed.
        """
        ranked = rank_elements(self.elements, query)[:budget]
        lines = [
            f'Active window: "{self.window_title}" (process: {self.window_app})',
            "",
            "Visible UI elements (type, name, center x,y):",
        ]
        if not ranked:
            lines.append("(no named UI elements detected — use vision instead)")
        for e in ranked:
            lines.append("- " + e.describe())
        hidden = len(self.elements) - len(ranked)
        if hidden > 0:
            lines.append(f"(+{hidden} more elements not shown; ask for a specific name to find them)")
        return "\n".join(lines)


# ----------------------------------------------------------------------------------
# Text matching — shared by ANCHOR (target binding) and RETINA (query-aware ranking)
# ----------------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "to", "on", "in", "for", "my", "me", "please", "now", "it", "this",
    "that", "and", "then", "with", "into", "at", "is", "are", "just", "can", "you", "your",
    "jarvis", "boss", "there", "here", "go", "open", "click", "press", "select", "show",
}


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keywords(text: str, keep_stopwords: bool = False) -> List[str]:
    words = normalize(text).split()
    if keep_stopwords:
        return words
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _acronym(text: str) -> str:
    words = [w for w in normalize(text).split() if w]
    return "".join(w[0] for w in words) if len(words) > 1 else ""


def text_score(needle: str, haystack: str) -> float:
    """Similarity of a referent to an element name, in [0, 1].

    Deliberately rule-based rather than embedding-based: it runs in microseconds, is fully
    deterministic (so a test can assert an exact ranking), and needs no model call. ANCHOR only
    escalates to a model when this scorer says the top candidates are too close to call.
    """
    n, h = normalize(needle), normalize(haystack)
    if not n or not h:
        return 0.0
    if n == h:
        return 1.0
    n_tokens = n.split()
    h_tokens = h.split()

    # Whole-phrase containment: strongest signal short of an exact match. Scaled by how much of the
    # element name the match accounts for, so "Bob" matching the label "Bob" beats "Bob" matching
    # "Bobsleigh Championship Highlights 2019".
    if n in h:
        coverage = len(n) / len(h)
        return 0.72 + 0.26 * coverage
    if h in n:
        return 0.68 + 0.20 * (len(h) / len(n))

    if _acronym(h) and _acronym(h) == n.replace(" ", ""):
        return 0.80

    n_set, h_set = set(n_tokens), set(h_tokens)
    overlap = n_set & h_set
    if overlap:
        precision = len(overlap) / len(n_set)
        recall = len(overlap) / len(h_set)
        f1 = 2 * precision * recall / (precision + recall)
        return 0.30 + 0.38 * f1

    # Prefix / typo tolerance on the strongest single token pair, e.g. "instagr" -> "Instagram".
    best_partial = 0.0
    for nt in n_tokens:
        for ht in h_tokens:
            if len(nt) >= 4 and (ht.startswith(nt) or nt.startswith(ht)):
                best_partial = max(best_partial, 0.34 + 0.10 * (min(len(nt), len(ht)) / max(len(nt), len(ht))))
    return best_partial


def rank_elements(elements: Sequence[ScreenElement], query: str = "") -> List[ScreenElement]:
    """Orders elements by usefulness: query relevance first, then interactivity, then reading order.

    With an empty query this degrades to "interactive controls first, in reading order", which is
    the monolith's old behaviour — so RETINA is never worse than what it replaces.
    """
    terms = keywords(query)

    def relevance(e: ScreenElement) -> float:
        if not terms:
            return 0.0
        return max((text_score(t, e.name) for t in terms), default=0.0)

    scored: List[Tuple[float, int, int, int, ScreenElement]] = []
    for e in elements:
        if not e.name or len(e.name) < 2:
            continue
        scored.append((-relevance(e), 0 if e.is_interactive else 1, e.top, e.left, e))
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return [t[4] for t in scored]


# ----------------------------------------------------------------------------------
# Agent-level contracts
# ----------------------------------------------------------------------------------


@dataclass
class GroundedTarget:
    """ANCHOR's answer to "what on screen does this order actually refer to?"."""

    referent: str
    element: Optional[ScreenElement] = None
    confidence: float = 0.0
    method: str = "none"            # exact | strong | disambiguated | vision | coordinate | none
    reason: str = ""
    alternatives: List[ScreenElement] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.element is not None

    @property
    def coords(self) -> Optional[Tuple[int, int]]:
        return (self.element.x, self.element.y) if self.element else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "referent": self.referent,
            "element": self.element.as_dict() if self.element else None,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "reason": self.reason,
            "alternatives": [e.name for e in self.alternatives[:5]],
        }


@dataclass
class Intent:
    """TRIAGE's classification of an utterance."""

    kind: str                      # chat | screen_query | deterministic | single_action | multi_step | clarify
    slots: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "rule"           # rule | learned | model | fallback
    raw: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "slots": self.slots,
            "confidence": round(self.confidence, 3), "source": self.source,
        }


@dataclass
class PlanStep:
    """One clause of the user's order, with the evidence that would prove it done."""

    description: str
    target: str = ""               # the referent ANCHOR must bind, verbatim from the order
    tool_hint: str = ""
    success_criteria: str = ""
    lane: str = "hands"            # hands | pathfinder | perception
    done: bool = False
    attempts: int = 0
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description, "target": self.target, "tool_hint": self.tool_hint,
            "success_criteria": self.success_criteria, "lane": self.lane,
            "done": self.done, "attempts": self.attempts, "evidence": self.evidence,
        }


@dataclass
class Plan:
    """ARCHITECT's decomposition of an order into an explicit, verifiable checklist."""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    source: str = "model"          # model | rule | single

    @property
    def pending(self) -> List[PlanStep]:
        return [s for s in self.steps if not s.done]

    @property
    def complete(self) -> bool:
        return bool(self.steps) and all(s.done for s in self.steps)

    def current(self) -> Optional[PlanStep]:
        return self.pending[0] if self.pending else None

    def budget(self) -> int:
        """Step budget derived from the real plan length instead of the monolith's regex clause
        count (`clauses * 2 + 2`), which mis-sized any order that didn't use its comma/"then"
        vocabulary."""
        return max(3, min(12, len(self.steps) * 2 + 2))

    def as_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "source": self.source, "steps": [s.as_dict() for s in self.steps]}


@dataclass
class Verdict:
    """SENTINEL's judgment on whether a step actually happened."""

    done: bool
    confidence: float = 0.0
    evidence: str = ""
    method: str = "screen"         # screen | tool | model | assumed
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "done": self.done, "confidence": round(self.confidence, 3),
            "evidence": self.evidence, "method": self.method, "reason": self.reason,
        }


@dataclass
class AgentEvent:
    """A trace entry. CORTEX converts these into the {"type": ...} dicts the UI already consumes,
    so the agent layer never has to know about the transport."""

    agent: str
    kind: str                      # status | thought | tool_exec | ground | verify | plan | response
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


@dataclass
class AgentContext:
    """Everything an agent may need, passed explicitly so nothing reaches for a global."""

    order: str = ""                                     # the user's literal words
    history: List[Dict[str, str]] = field(default_factory=list)
    plan: Optional[Plan] = None
    snapshot: Optional[ScreenSnapshot] = None
    trace: List[AgentEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    model: str = SMART_MODEL

    def emit(self, agent: str, kind: str, message: str = "", /, **data: Any) -> AgentEvent:
        # Positional-only: trace payloads are built by splatting agent dataclasses, several of which
        # legitimately carry their own "kind" or "message" field. Without this, `**intent.as_dict()`
        # collides with the parameter name and raises TypeError at runtime.
        ev = AgentEvent(agent=agent, kind=kind, message=message, data=data)
        self.trace.append(ev)
        return ev

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


# ----------------------------------------------------------------------------------
# LLM client
# ----------------------------------------------------------------------------------


class LLMClient:
    """Pooled Groq chat client shared by every agent.

    Beyond deduplicating the retry/429 logic that used to live inside the monolith, this counts
    calls and tokens per task. That counter is not decoration: "how many model round-trips did
    this order cost?" is the metric this whole refactor exists to reduce, and the tests assert on
    it directly.
    """

    def __init__(self, api_key: Optional[str] = None, session: Any = None):
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY") or load_local_env_file().get("GROQ_API_KEY", "")
        self.api_key = api_key or ""
        self._session = session
        self.calls: int = 0
        self.calls_by_agent: Dict[str, int] = {}
        self.last_error: Optional[str] = None

    # -- accounting ---------------------------------------------------------------

    def reset_counters(self) -> None:
        self.calls = 0
        self.calls_by_agent = {}

    def _count(self, agent: str) -> None:
        self.calls += 1
        self.calls_by_agent[agent] = self.calls_by_agent.get(agent, 0) + 1

    # -- transport ----------------------------------------------------------------

    def _post(self, payload: Dict[str, Any], timeout: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        import requests  # imported lazily so the module imports cleanly in a bare test env

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        poster = self._session.post if self._session is not None else requests.post
        last_err = None
        for attempt in range(3):
            try:
                resp = poster(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"], None
                if resp.status_code == 429:
                    if attempt < 2:
                        retry_after = resp.headers.get("retry-after")
                        try:
                            wait_s = min(float(retry_after), 10.0) if retry_after else 3.0
                        except (TypeError, ValueError):
                            wait_s = 3.0
                        time.sleep(wait_s)
                        continue
                    last_err = "I've hit Groq's rate limit, Boss — give it a moment and try again."
                    break
                try:
                    body = resp.json().get("error", {}).get("message", resp.text[:200])
                except Exception:
                    body = resp.text[:200]
                last_err = f"Groq API error (HTTP {resp.status_code}): {body}"
            except requests.exceptions.ConnectionError:
                last_err = "I can't reach Groq's API right now, Boss — check your internet connection."
            except requests.exceptions.Timeout:
                last_err = "Groq's API is taking too long to respond, Boss."
            except Exception as e:  # noqa: BLE001 - surfaced to the user, never swallowed
                last_err = f"Execution error: {e}"
            if attempt < 2:
                time.sleep(1.0)
        return None, last_err

    # -- public API ---------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        agent: str = "unknown",
        model: str = SMART_MODEL,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        timeout: int = 30,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.api_key:
            return None, "I don't have a Groq API key configured, Boss — set GROQ_API_KEY."
        payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        self._count(agent)
        message, error = self._post(payload, timeout)
        self.last_error = error
        return message, error

    def json_call(
        self,
        system: str,
        user: str,
        *,
        agent: str = "unknown",
        model: str = FAST_MODEL,
        temperature: float = 0.0,
        timeout: int = 20,
        max_tokens: Optional[int] = 400,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Structured-output convenience wrapper — the shape most agents here need."""
        message, error = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            agent=agent, model=model, temperature=temperature, json_mode=True,
            timeout=timeout, max_tokens=max_tokens,
        )
        if error or not message:
            return None, error or "no response"
        return parse_json_object(message.get("content") or ""), None

    def text_call(
        self,
        system: str,
        user: str,
        *,
        agent: str = "unknown",
        model: str = FAST_MODEL,
        temperature: float = 0.3,
        timeout: int = 20,
        max_tokens: int = 120,
    ) -> Tuple[Optional[str], Optional[str]]:
        message, error = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            agent=agent, model=model, temperature=temperature, timeout=timeout, max_tokens=max_tokens,
        )
        if error or not message:
            return None, error or "no response"
        return (message.get("content") or "").strip(), None


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Tolerant JSON extraction — models occasionally wrap JSON in prose or a ```json fence even
    when asked not to, and losing an otherwise-good answer to that is a wasted round-trip."""
    if not raw:
        return {}
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass
    start, depth = -1, 0
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(raw[start:i + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    start = -1
    return {}


class Agent:
    """Minimal base: a name for tracing and a shared LLM client."""

    name: str = "agent"

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def _emit(self, ctx: Optional[AgentContext], kind: str, message: str = "", /, **data: Any) -> None:
        if ctx is not None:
            ctx.emit(self.name, kind, message, **data)
