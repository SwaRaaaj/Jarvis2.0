"""TRIAGE — the intent router.

Every order used to fall through a ladder of hand-written regexes and, if none matched, into the
full ReAct loop: 21 tool schemas, a tree walk per step, and a second model call after every action
just to ask whether the job was done. A 3-action task cost roughly six 70B round-trips.

The regexes were the real problem. `^(?:open|launch|start|run)\\s+` matches "open chrome" but not
"fire up chrome"; the small-talk list matches "hello" but not "yo". Every phrasing the author
didn't think of took the most expensive path available.

TRIAGE keeps the deterministic layer — it is genuinely free and genuinely correct for the cases it
covers — but backs it with one ~100ms 8B classification instead of a cliff. Nothing falls into the
expensive loop by accident any more; it goes there because TRIAGE decided it belongs there.

Order of resolution, cheapest first:
    1. learned rules (SCHOLAR) ...... 0 model calls, 0 network
    2. deterministic patterns ....... 0 model calls
    3. 8B classification ............ 1 cheap model call
    4. conservative fallback ........ multi_step, i.e. "let the planner look at it"
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .base import FAST_MODEL, Agent, AgentContext, Intent, LLMClient, normalize

try:  # keeps the agent layer importable in a bare test environment
    from os_automation import APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES
except Exception:  # pragma: no cover - exercised only when pyautogui/pywin32 are absent
    APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES = {}, {}, {}


CONVERSATIONAL = [
    r"^(hi|hello|hey|yo|sup|hiya)\b", r"\bhow are you\b", r"\bwho are you\b", r"\bwhat can you do\b",
    r"\bwhat are you\b", r"^(thanks|thank you|ty|cheers)\b", r"^good\s?(morning|afternoon|evening|night)\b",
    r"^(bye|goodbye|see you|see ya)\b", r"\bare you (there|online|awake|listening|ok)\b",
    r"^(nice|cool|great|awesome|perfect|nevermind|never mind)\b", r"\byour name\b",
]

SCREEN_QUERY = [
    r"\bcan you see (my |the )?screen\b", r"\bwhat can you see\b",
    r"\bwhat'?s (currently )?on (my |the )?(current )?screen\b",
    r"\bwhat is (currently )?on (my |the )?(current )?screen\b",
    r"\bdescribe (my |the )?screen\b", r"\bwhat do you see\b", r"\bread (my |the )?screen\b",
    r"\bwhat'?s (this|that|it) (say|showing)\b", r"\bwhat am i looking at\b",
    r"\bwhat'?s open\b", r"\bwhich (window|app|tab) is (open|active)\b",
]

TIME_QUERY = [
    "what time is it", "what's the time", "current time", "what day is it", "what's the date",
    "today's date", "what is the date", "what is the time", "time right now", "what day is today",
]

# Multi-clause markers. Present in the monolith too, but there they *also* sized the step budget;
# here they only hint that a plan is needed, and ARCHITECT decides the real shape.
COMPOUND_MARKERS = (" then ", " and then ", " after that", ";", " next ", " followed by ", " also ")

LAUNCH_RE = re.compile(
    r"^(?:open|launch|start|run|fire up|boot up|pull up|bring up|get me|load)\s+(.+)$", re.IGNORECASE
)
SWITCH_RE = re.compile(r"^(?:switch to|switch|go to|focus(?: on)?|bring up|jump to)\s+(.+)$", re.IGNORECASE)
SEARCH_RE = re.compile(r"^(?:search|google|look up|find)\s+(?:for\s+)?(.+?)(?:\s+on\s+(google|youtube))?$", re.IGNORECASE)
YOUTUBE_RE = re.compile(r"^(?:play|search)\s+(.+?)\s+on\s+youtube$", re.IGNORECASE)


class TriageAgent(Agent):
    """Classifies an order into the cheapest lane that can correctly handle it."""

    name = "TRIAGE"

    def __init__(self, llm: Optional[LLMClient] = None, rule_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None):
        super().__init__(llm)
        # Injected by SCHOLAR at wiring time; returns a previously-learned (command -> action) hit.
        self.rule_lookup = rule_lookup
        self.by_source: Dict[str, int] = {}

    # ------------------------------------------------------------------

    def classify(self, order: str, ctx: Optional[AgentContext] = None, allow_model: bool = True) -> Intent:
        intent = self._classify_inner(order, ctx, allow_model)
        self.by_source[intent.source] = self.by_source.get(intent.source, 0) + 1
        self._emit(ctx, "status", f"intent: {intent.kind} (via {intent.source})", **intent.as_dict())
        return intent

    def _classify_inner(self, order: str, ctx: Optional[AgentContext], allow_model: bool) -> Intent:
        raw = (order or "").strip()
        if not raw:
            return Intent(kind="clarify", confidence=1.0, source="rule", raw=raw,
                          slots={"question": "I didn't catch that, Boss — say it again?"})

        # The wake word is addressing, never content. Critically, this repo's own directory is
        # named "jarvis", so leaving it in makes every window title in this project a fuzzy match.
        cmd = re.sub(r"\bjarvis\b", " ", raw, flags=re.IGNORECASE)
        cmd = re.sub(r"\s+", " ", cmd).strip(" ,.!?")
        low = cmd.lower()

        # --- 1. learned shortcuts (SCHOLAR) -------------------------------------------------
        if self.rule_lookup:
            try:
                hit = self.rule_lookup(cmd)
            except Exception:
                hit = None
            if hit:
                return Intent(kind="deterministic", confidence=0.95, source="learned", raw=raw,
                              slots={"action": hit.get("tool"), "args": hit.get("args", {}),
                                     "learned_from": hit.get("trigger", "")})

        # --- 2. deterministic patterns ------------------------------------------------------
        if any(p in low for p in TIME_QUERY):
            return Intent(kind="deterministic", confidence=1.0, source="rule", raw=raw,
                          slots={"action": "get_time_date", "args": {}})

        if any(re.search(p, low) for p in SCREEN_QUERY):
            return Intent(kind="screen_query", confidence=0.95, source="rule", raw=raw,
                          slots={"question": cmd})

        is_compound = any(m in f" {low} " for m in COMPOUND_MARKERS) or low.count(",") >= 2
        if not is_compound:
            if any(re.search(p, low) for p in CONVERSATIONAL):
                return Intent(kind="chat", confidence=0.9, source="rule", raw=raw)

            deterministic = self._match_deterministic(low)
            if deterministic:
                return Intent(kind="deterministic", confidence=0.95, source="rule", raw=raw,
                              slots=deterministic)

        # --- 3. one cheap classification ----------------------------------------------------
        if allow_model:
            modelled = self._classify_via_model(cmd, ctx)
            if modelled is not None:
                return modelled

        # --- 4. conservative fallback -------------------------------------------------------
        # When in doubt, plan. A plan for a single-step order costs one extra cheap call; guessing
        # "single action" for a genuinely multi-step order silently drops half the user's request,
        # which is far worse.
        return Intent(kind="multi_step" if is_compound else "single_action",
                      confidence=0.4, source="fallback", raw=raw)

    # ------------------------------------------------------------------

    def _match_deterministic(self, low: str) -> Optional[Dict[str, Any]]:
        """Known single-action orders with exactly one correct execution and no ambiguity."""
        yt = YOUTUBE_RE.match(low)
        if yt:
            return {"action": "search_youtube", "args": {"query": yt.group(1).strip()}}

        m = LAUNCH_RE.match(low)
        if m:
            target = self._clean_target(m.group(1))
            if target in APP_ALIASES:
                return {"action": "launch_app", "args": {"app_name": target}}
            if target in SOCIAL_INBOX_URLS:
                return {"action": "open_social_inbox", "args": {"platform": target}}
            if target in WEBSITE_ALIASES:
                return {"action": "open_url", "args": {"url": WEBSITE_ALIASES[target]}}

        m = SWITCH_RE.match(low)
        if m:
            target = self._clean_target(m.group(1))
            if target:
                # Not fully deterministic — the window may not be open — so the executor is told to
                # fall back to launching. Encoded as a slot rather than lost in a regex fallthrough.
                return {"action": "switch_window", "args": {"title": target},
                        "fallback": {"action": "launch_app", "args": {"app_name": target}}}

        m = SEARCH_RE.match(low)
        if m and m.group(1):
            engine = (m.group(2) or "google").lower()
            query = m.group(1).strip()
            # "find the send button" is a screen action, not a web search.
            if not re.search(r"\b(button|tab|link|icon|menu|field|row|chat|on screen)\b", query):
                action = "search_youtube" if engine == "youtube" else "search_google"
                return {"action": action, "args": {"query": query}}
        return None

    @staticmethod
    def _clean_target(text: str) -> str:
        target = text.strip(" ?.!,")
        target = re.sub(r"^(the|a|an|my)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(please|now|for me|window|app|application)$", "", target, flags=re.IGNORECASE).strip()
        return normalize(target)

    # ------------------------------------------------------------------

    _SYSTEM = (
        "You route desktop-assistant orders to the cheapest handler that can do them correctly. "
        "Reply ONLY with JSON: {\"kind\": ..., \"confidence\": 0-1, \"reason\": \"...\"}\n"
        "kind must be exactly one of:\n"
        "  chat         - small talk or a question about the assistant itself; no screen action.\n"
        "  screen_query - the user is asking what is visible on screen; answer by looking, don't act.\n"
        "  single_action- one physical action completes it (one click, one launch, one keystroke).\n"
        "  multi_step   - two or more distinct actions are required, or the target must be found first.\n"
        "  clarify      - genuinely too vague to act on without asking a question back.\n"
        "Prefer multi_step over single_action when unsure. Prefer clarify only if truly unactionable."
    )

    def _classify_via_model(self, cmd: str, ctx: Optional[AgentContext]) -> Optional[Intent]:
        data, error = self.llm.json_call(
            self._SYSTEM, f'Order: "{cmd}"', agent=self.name, model=FAST_MODEL,
            temperature=0.0, timeout=12, max_tokens=120,
        )
        if error or not data:
            return None
        kind = str(data.get("kind", "")).strip().lower()
        if kind not in {"chat", "screen_query", "single_action", "multi_step", "clarify"}:
            return None
        try:
            confidence = float(data.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        slots: Dict[str, Any] = {"reason": str(data.get("reason", ""))[:200]}
        if kind == "screen_query":
            slots["question"] = cmd
        if kind == "clarify":
            slots["question"] = str(data.get("reason") or "What exactly would you like me to do, Boss?")
        return Intent(kind=kind, slots=slots, confidence=confidence, source="model", raw=cmd)

    def stats(self) -> Dict[str, Any]:
        total = sum(self.by_source.values())
        free = self.by_source.get("rule", 0) + self.by_source.get("learned", 0)
        return {
            "classified": total,
            "by_source": dict(self.by_source),
            "zero_cost_rate": round(free / total, 3) if total else 0.0,
        }
