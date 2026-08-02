"""ARCHITECT — the planner.

The old engine did no planning at all. It decided one action at a time, and sized its step budget
with `clauses * 2 + 2` where `clauses` came from splitting on commas and the literal words "then"
and "next". An order phrased any other way ("go to instagram and message alice hey") counted as
one clause and got a 4-step budget for a job that needs five actions.

Worse, with no plan there was nothing to check completion *against*. The monolith asked a 70B model
"is this done?" after every action, passing it only the active window title as evidence — an
expensive question asked from a position of near-total ignorance.

ARCHITECT spends one call up front to turn the order into an explicit checklist, where every step
carries the exact referent ANCHOR must bind and the observable condition SENTINEL must confirm.
That one call pays for itself immediately:

  * the step budget reflects the real shape of the work rather than punctuation
  * completion is checked per step against stated criteria, usually with no model call at all
  * scope discipline becomes structural — the plan enumerates what was asked, so there is nothing
    for the executor to invent and nothing for it to quietly drop
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import SMART_MODEL, Agent, AgentContext, LLMClient, Plan, PlanStep, parse_json_object

# Verbs that mean "get somewhere" rather than "manipulate what is already here". Used to assign a
# lane so each executor only ever sees its own small tool set.
_NAV_VERBS = (
    "open", "launch", "start", "run", "go to", "navigate", "switch", "focus", "fire up", "boot up",
    "pull up", "visit", "browse", "search", "google", "play",
)
_LOOK_VERBS = ("look", "see", "read", "check", "describe", "tell me what", "what is on", "find out")

_CLAUSE_SPLIT = re.compile(
    r",\s*(?:and\s+)?then\s+|\s+and\s+then\s+|\s*;\s*|\s+then\s+|\s+after that,?\s+|"
    r"\s+next,?\s+|\s+followed by\s+|\s+and also\s+",
    re.IGNORECASE,
)


class ArchitectAgent(Agent):
    """Decomposes an order into a verifiable checklist."""

    name = "ARCHITECT"

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(llm)
        self.plans_built: int = 0
        self.model_plans: int = 0

    # ------------------------------------------------------------------

    def plan(
        self,
        order: str,
        ctx: Optional[AgentContext] = None,
        screen_context: str = "",
        allow_model: bool = True,
        model: str = SMART_MODEL,
    ) -> Plan:
        self.plans_built += 1
        if allow_model:
            built = self._plan_via_model(order, screen_context, ctx, model)
            if built is not None and built.steps:
                self.model_plans += 1
                self._emit(ctx, "plan", f"{len(built.steps)}-step plan", plan=built.as_dict())
                return built
        fallback = self.rule_plan(order)
        self._emit(ctx, "plan", f"{len(fallback.steps)}-step plan (rules)", plan=fallback.as_dict())
        return fallback

    def single_step_plan(self, order: str, target: str = "", lane: str = "hands") -> Plan:
        """For orders TRIAGE already knows are one action — skips the planning call entirely."""
        self.plans_built += 1
        return Plan(
            goal=order,
            source="single",
            steps=[PlanStep(description=order, target=target, lane=lane,
                            success_criteria="the requested action reports success and the screen reflects it")],
        )

    # ------------------------------------------------------------------

    def rule_plan(self, order: str) -> Plan:
        """Deterministic decomposition used when the model is unavailable or declines to answer.

        Strictly better than the monolith's clause counter: it produces actual steps with lanes and
        targets rather than a number, so the rest of the pipeline behaves identically whether or
        not the planning call succeeded.
        """
        clauses = [c.strip(" ,.!?") for c in _CLAUSE_SPLIT.split(order or "") if c.strip(" ,.!?")]
        if not clauses:
            clauses = [order.strip()] if order.strip() else []
        steps: List[PlanStep] = []
        for clause in clauses:
            steps.append(PlanStep(
                description=clause,
                target=self._guess_target(clause),
                lane=self._guess_lane(clause),
                success_criteria=f"'{clause}' has visibly taken effect on screen",
            ))
        return Plan(goal=order, steps=steps, source="rule")

    @staticmethod
    def _guess_lane(clause: str) -> str:
        low = clause.lower().strip()
        if any(low.startswith(v) or f" {v} " in f" {low} " for v in _LOOK_VERBS):
            return "perception"
        if any(low.startswith(v) for v in _NAV_VERBS):
            return "pathfinder"
        return "hands"

    @staticmethod
    def _guess_target(clause: str) -> str:
        # ANCHOR owns real referent extraction; this only needs to be good enough to seed it, and
        # ANCHOR re-parses the clause anyway. Kept deliberately trivial to avoid two competing
        # implementations of the same logic drifting apart.
        return clause.strip()

    # ------------------------------------------------------------------

    _SYSTEM = """You break a desktop-automation order into the minimum sequence of concrete actions.

Reply ONLY with JSON:
{"steps": [{"description": "...", "target": "...", "lane": "pathfinder|hands|perception",
            "tool_hint": "...", "success_criteria": "..."}]}

Rules:
- One step per distinct physical action. Do not pad. Do not add steps the order did not ask for.
- "target" is the exact on-screen thing this step acts on, copied verbatim from the user's words
  (a contact name, a button label, an app name). Empty string if the step targets nothing specific.
- "lane": pathfinder = launching apps / opening sites / switching windows.
          hands = clicking, typing, scrolling, keyboard shortcuts on what is already on screen.
          perception = looking at or reading the screen to answer something.
- "success_criteria" is an observable condition someone could check by looking at the screen.
  Not "the click succeeded" but "the conversation with Bob is open and its message box is visible".
- Opening an app or site is ONE step; finding something inside it is a SEPARATE step.
- Never invent recipients, messages, or destinations that are not in the order."""

    def _plan_via_model(
        self, order: str, screen_context: str, ctx: Optional[AgentContext], model: str
    ) -> Optional[Plan]:
        user = f'Order: "{order}"'
        if screen_context:
            user += f"\n\nWhat is on screen right now:\n{screen_context}"
        message, error = self.llm.chat(
            [{"role": "system", "content": self._SYSTEM}, {"role": "user", "content": user}],
            agent=self.name, model=model, temperature=0.0, json_mode=True, timeout=25, max_tokens=700,
        )
        if error or not message:
            self._emit(ctx, "status", f"planning unavailable ({error}) — using rule decomposition")
            return None
        data = parse_json_object(message.get("content") or "")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None

        steps: List[PlanStep] = []
        for item in raw_steps[:10]:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            lane = str(item.get("lane") or "").strip().lower()
            if lane not in {"pathfinder", "hands", "perception"}:
                lane = self._guess_lane(description)
            steps.append(PlanStep(
                description=description,
                target=str(item.get("target") or "").strip(),
                tool_hint=str(item.get("tool_hint") or "").strip(),
                lane=lane,
                success_criteria=str(item.get("success_criteria") or "").strip()
                or f"'{description}' has visibly taken effect",
            ))
        return Plan(goal=order, steps=steps, source="model") if steps else None

    def stats(self) -> Dict[str, Any]:
        return {"plans_built": self.plans_built, "model_plans": self.model_plans,
                "rule_plans": self.plans_built - self.model_plans}
