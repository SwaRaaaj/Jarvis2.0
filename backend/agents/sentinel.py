"""SENTINEL — the verification agent.

Answers one question after every action: did that actually accomplish the step?

The monolith asked a 70B model, once per terminal action, and gave it almost nothing to reason
from — the action log plus the active window *title*. That is an expensive round-trip spent on a
question the screen usually answers for free. It also ran three competing completion heuristics
that could contradict each other: a single-action auto-complete short-circuit, a
`terminal_action_count >= clause_count` gate, and the model calling `speak_final`.

SENTINEL replaces all three with one ladder, cheapest rung first:

    1. the tool itself failed                     -> not done            (no model call)
    2. ANCHOR says the action went off-target     -> not done            (no model call)
    3. the step's success criteria are now
       literally visible on screen                -> done                (no model call)
    4. nothing on screen changed at all           -> not done            (no model call)
    5. genuinely ambiguous                        -> one 8B call

Rungs 1-4 cover the large majority of real steps, and each one is backed by concrete screen
evidence rather than a model's recollection of what it just did.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import (
    FAST_MODEL,
    Agent,
    AgentContext,
    LLMClient,
    PlanStep,
    ScreenSnapshot,
    Verdict,
    keywords,
    normalize,
    text_score,
)

# Tools whose success is self-evident from their own return value — a keystroke either went to the
# OS or it didn't, and there is no meaningful screen assertion to make beyond "something changed".
_SELF_EVIDENT = {"get_time_date", "scroll", "move_mouse", "key_combo", "wait"}

# Tools that must produce a visible change to count. A click that changes nothing on screen almost
# certainly hit dead space, even though pyautogui reports success either way.
_MUST_CHANGE_SCREEN = {"click_coordinate", "click_element", "type_text", "close_tab", "close_window",
                       "close_all_tabs", "launch_app", "open_url", "open_social_inbox",
                       "switch_window", "search_google", "search_youtube"}


class SentinelAgent(Agent):
    """Screen-evidence-first verification."""

    name = "SENTINEL"

    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(llm)
        self.checks: int = 0
        self.free_checks: int = 0
        self.model_checks: int = 0

    # ------------------------------------------------------------------

    def verify(
        self,
        step: PlanStep,
        result: Dict[str, Any],
        before: Optional[ScreenSnapshot],
        after: Optional[ScreenSnapshot],
        ctx: Optional[AgentContext] = None,
        allow_model: bool = True,
    ) -> Verdict:
        self.checks += 1
        verdict = self._verify_inner(step, result, before, after, ctx, allow_model)
        if verdict.method != "model":
            self.free_checks += 1
        self._emit(ctx, "verify", f"{'done' if verdict.done else 'not done'}: {verdict.evidence}",
                   **verdict.as_dict())
        return verdict

    def _verify_inner(
        self, step: PlanStep, result: Dict[str, Any],
        before: Optional[ScreenSnapshot], after: Optional[ScreenSnapshot],
        ctx: Optional[AgentContext], allow_model: bool,
    ) -> Verdict:
        tool = str(result.get("tool") or "")

        # --- rung 1: the action itself failed ------------------------------------------------
        if result.get("status") != "success":
            return Verdict(done=False, confidence=0.95, method="tool",
                           evidence=str(result.get("message") or "the action reported failure"),
                           reason="tool error")

        # --- rung 2: ANCHOR's scope guard tripped --------------------------------------------
        if result.get("on_target") is False:
            return Verdict(done=False, confidence=0.9, method="screen",
                           evidence=str(result.get("scope_reason") or "the action landed on the wrong target"),
                           reason="off-target")

        # --- rung 3: is the success condition literally on screen now? -----------------------
        expectations = self._expectations(step)
        if after is not None and expectations:
            appeared = self._newly_visible(expectations, before, after)
            if appeared:
                return Verdict(done=True, confidence=0.88, method="screen",
                               evidence=f"'{appeared}' is now visible on screen",
                               reason="success criteria observed")

        # Navigation lands when the window title reflects the destination.
        if after is not None and step.target and tool in {"launch_app", "switch_window", "open_url",
                                                          "open_social_inbox", "search_google", "search_youtube"}:
            if text_score(step.target, after.window_title) >= 0.42:
                return Verdict(done=True, confidence=0.85, method="screen",
                               evidence=f'the active window is now "{after.window_title[:60]}"',
                               reason="destination reached")

        # --- self-evident tools ---------------------------------------------------------------
        if tool in _SELF_EVIDENT:
            return Verdict(done=True, confidence=0.75, method="tool",
                           evidence=f"{tool} completed", reason="self-evident action")

        # --- rung 4: nothing changed at all ---------------------------------------------------
        if before is not None and after is not None and tool in _MUST_CHANGE_SCREEN:
            if before.signature() == after.signature():
                return Verdict(done=False, confidence=0.7, method="screen",
                               evidence="the screen looks identical to before the action",
                               reason="no observable effect")

        # A grounded, on-target click that visibly changed the screen is strong evidence on its own.
        grounding = result.get("grounding") or {}
        if grounding.get("element") and result.get("on_target") is not False:
            confidence = float(grounding.get("confidence") or 0.0)
            if confidence >= 0.72 and before is not None and after is not None \
                    and before.signature() != after.signature():
                name = (grounding.get("element") or {}).get("name", "the target")
                return Verdict(done=True, confidence=0.8, method="screen",
                               evidence=f"clicked '{name}' and the screen responded",
                               reason="grounded click with visible effect")

        # --- rung 5: genuinely ambiguous ------------------------------------------------------
        if allow_model:
            return self._verify_via_model(step, result, after, ctx)
        return Verdict(done=False, confidence=0.4, method="assumed",
                       evidence="could not confirm from the screen", reason="inconclusive")

    # ------------------------------------------------------------------

    @staticmethod
    def _expectations(step: PlanStep) -> List[str]:
        """Concrete strings whose appearance would prove the step happened."""
        out: List[str] = []
        if step.target:
            out.append(step.target)
        for phrase in re.findall(r"[\"'‘“]([^\"'’”]{2,60})[\"'’”]", step.success_criteria or ""):
            out.append(phrase)
        return [o for o in {o.strip() for o in out} if len(o.strip()) >= 2]

    @staticmethod
    def _newly_visible(
        expectations: List[str], before: Optional[ScreenSnapshot], after: ScreenSnapshot
    ) -> str:
        """Returns an expectation that is visible now and wasn't before.

        The "and wasn't before" half matters: on Instagram the word "Instagram" is in the window
        title the whole time, so its mere presence proves nothing about whether the step just
        succeeded. Only a *change* is evidence.
        """
        for expectation in expectations:
            terms = keywords(expectation) or [normalize(expectation)]
            for term in terms:
                if len(term) < 3:
                    continue
                now = term in normalize(after.window_title) or any(term in normalize(e.name) for e in after.elements)
                if not now:
                    continue
                if before is None:
                    return expectation
                was = term in normalize(before.window_title) or any(term in normalize(e.name) for e in before.elements)
                if not was:
                    return expectation
        return ""

    _SYSTEM = (
        "You judge whether one step of a desktop automation task is now complete, using only the "
        "evidence given. Be strict: if the evidence does not show the step's stated success "
        'condition, it is not complete. Reply ONLY with '
        '{"done": true|false, "evidence": "one short factual sentence", "confidence": 0-1}.'
    )

    def _verify_via_model(
        self, step: PlanStep, result: Dict[str, Any], after: Optional[ScreenSnapshot],
        ctx: Optional[AgentContext],
    ) -> Verdict:
        self.model_checks += 1
        screen = after.describe_for_prompt(query=step.target or step.description, budget=15) if after else "(screen unavailable)"
        matched = result.get("matched_name")
        hit_note = f" (hit '{matched}')" if matched else ""
        user = (
            f"Step: {step.description}\n"
            f"Success condition: {step.success_criteria or '(none stated)'}\n"
            f"Action performed: {result.get('tool')} -> {result.get('status')}{hit_note}\n\n"
            f"Screen now:\n{screen}"
        )
        data, error = self.llm.json_call(self._SYSTEM, user, agent=self.name, model=FAST_MODEL,
                                         temperature=0.0, timeout=15, max_tokens=150)
        if error or not data:
            # Failing open here would let an unverified step count as done. Failing closed at worst
            # costs one more attempt, so that is the safe direction.
            return Verdict(done=False, confidence=0.3, method="assumed",
                           evidence="verification was unavailable", reason=error or "no response")
        try:
            confidence = float(data.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        return Verdict(done=bool(data.get("done")), confidence=confidence, method="model",
                       evidence=str(data.get("evidence") or "")[:200], reason="model judgment")

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "checks": self.checks,
            "free_checks": self.free_checks,
            "model_checks": self.model_checks,
            "free_rate": round(self.free_checks / self.checks, 3) if self.checks else 0.0,
        }
