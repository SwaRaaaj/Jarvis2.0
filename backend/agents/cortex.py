"""CORTEX — the orchestrator.

Wires the agents into one pipeline and speaks the event protocol the existing UIs already consume,
so `main.py`, `jarvis_desktop.py` and the React dashboard need no changes to their event handling.

The route an order takes:

    EARS ......... (upstream, in the listener loop) was this even addressed to JARVIS?
        |
    TRIAGE ....... which lane? learned rule > deterministic > cheap classification
        |
        +-- chat ............ NARRATOR replies, no screen touched
        +-- screen_query .... RETINA looks, NARRATOR phrases the answer
        +-- deterministic ... PATHFINDER acts directly, no planning, no tool schemas
        +-- clarify ......... ask instead of guessing
        |
    ARCHITECT .... decompose into a checklist with per-step success criteria
        |
    for each step, until done or out of budget:
        RETINA ....... snapshot (cached; re-walks the tree only if the screen changed)
        ANCHOR ....... bind the step's target to a real element  <-- keeps JARVIS on-order
        HANDS /
        PATHFINDER ... perform exactly one action
        SENTINEL ..... did it work? (screen evidence first, model only if ambiguous)
        |
    NARRATOR ..... compose the spoken reply and the detailed report
    SCHOLAR ...... record the outcome so a repeat of this order is free next time

Two invariants carried over from the monolith's hard-won lessons, kept because they cost nothing
when the model behaves and save real damage when it doesn't:
  * a wall-clock ceiling on the whole task
  * a per-tool call cap, so a confused loop cannot execute the same real side effect indefinitely
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from .base import (
    FAST_MODEL,
    SMART_MODEL,
    AgentContext,
    LLMClient,
    Plan,
    PlanStep,
    ScreenSnapshot,
)
from .anchor import AnchorAgent
from .architect import ArchitectAgent
from .ears import EarsAgent
from .hands import HandsAgent
from .narrator import NarratorAgent, speechify
from .pathfinder import PathfinderAgent
from .retina import RetinaAgent
from .scholar import ScholarAgent
from .sentinel import SentinelAgent
from .triage import TriageAgent
from .vigil import VigilAgent

MAX_WALL_SECONDS = 90
MAX_CALLS_PER_TOOL = 3
MAX_STEP_ATTEMPTS = 3


class Cortex:
    """The multi-agent replacement for the monolithic ReAct loop."""

    def __init__(
        self,
        memory: Any = None,
        vision: Any = None,
        telemetry: Any = None,
        os_api: Any = None,
        llm: Optional[LLMClient] = None,
        address: str = "Boss",
    ):
        self.llm = llm or LLMClient()
        self.memory = memory

        self.scholar = ScholarAgent(memory=memory, llm=self.llm)
        self.retina = RetinaAgent(vision=vision, telemetry=telemetry, llm=self.llm)
        self.anchor = AnchorAgent(self.llm)
        self.triage = TriageAgent(self.llm, rule_lookup=self.scholar.lookup)
        self.architect = ArchitectAgent(self.llm)
        self.pathfinder = PathfinderAgent(self.llm, os_api=os_api)
        self.hands = HandsAgent(self.llm, anchor=self.anchor, os_api=os_api)
        self.sentinel = SentinelAgent(self.llm)
        self.narrator = NarratorAgent(self.llm, address=address)
        self.ears = EarsAgent()
        self.vigil = VigilAgent(self.retina, self.llm)

        self.history: List[Dict[str, str]] = []
        self._stop = threading.Event()
        self.last_run: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def cancel(self) -> None:
        self._stop.set()

    def mine_history(self) -> Dict[str, Any]:
        """Backfills learned rules from the existing execution log. Safe to call at startup."""
        return self.scholar.mine()

    def start_ambient(self, quality: int = 40, scale: float = 0.35) -> None:
        """Starts the live screen feed and the ambient observer.

        Opt-in rather than automatic: it spawns two background threads and periodically uses the
        local vision model, which a short-lived process or a test run has no use for.
        """
        self.retina.start_feed(quality=quality, scale=scale)
        self.vigil.start()

    def stop_ambient(self) -> None:
        self.vigil.stop()
        self.retina.stop_feed()

    # ------------------------------------------------------------------
    # Main entry point — yields the same event dicts the UIs already handle
    # ------------------------------------------------------------------

    def run(self, order: str, model: str = SMART_MODEL) -> Generator[Dict[str, Any], None, None]:
        self._stop.clear()
        self.llm.reset_counters()
        started = time.time()
        ctx = AgentContext(order=order, history=self.history, model=model)

        # The vision model is slow and single-threaded. Ambient curiosity must never compete with
        # an order the user actually gave.
        self.vigil.pause()
        try:
            yield from self._run_inner(order, model, ctx, started)
        except Exception as e:  # noqa: BLE001 — a crash must still produce a spoken reply
            yield {"type": "response", "text": f"Something went wrong on my side, Boss — {e}"}
        finally:
            self.vigil.resume()

    def _run_inner(
        self, order: str, model: str, ctx: AgentContext, started: float
    ) -> Generator[Dict[str, Any], None, None]:
        intent = self.triage.classify(order, ctx)
        yield {"type": "status", "message": f"Intent: {intent.kind} (via {intent.source})"}

        # --- lanes that never touch the screen ------------------------------------------------
        if intent.kind == "chat":
            reply = self.narrator.small_talk(order, ctx)
            yield from self._finish(order, reply, ctx, started, detail=f'Small talk: "{order}"')
            return

        if intent.kind == "clarify":
            question = self.narrator.clarify(str(intent.slots.get("question") or ""))
            yield from self._finish(order, question, ctx, started, detail="Asked for clarification.")
            return

        if intent.kind == "screen_query":
            question = str(intent.slots.get("question") or order)

            # VIGIL may already understand this exact screen. When it does the reply is instant
            # instead of the 5-19 seconds a cold vision call costs, and no model runs at all.
            cached = self.vigil.current_view()
            if cached is not None:
                yield {"type": "status", "message": f"Already watching — seen {cached.age:.0f}s ago"}
                spoken = self.narrator.answer_question(question, cached.text, ctx)
                yield {"type": "tool_exec", "tool": "vigil_cache",
                       "input": {"question": question},
                       "output": {"status": "success", "answer": cached.text,
                                  "age_seconds": round(cached.age, 1), "source": "ambient"}}
                detail = (f'Question: "{question}"\n\nAnswered from the ambient observer '
                          f'(seen {cached.age:.0f}s ago in "{cached.window_title}").\n\n{cached.text}')
                yield from self._finish(order, spoken, ctx, started, detail=detail)
                return

            yield {"type": "status", "message": "Looking at your screen..."}
            answer = self.retina.ask(question, ctx)
            if not answer:
                yield from self._finish(order, "I couldn't get a clear read on the screen, Boss.",
                                        ctx, started, detail="Vision returned nothing.")
                return
            snap = self.retina.snapshot(ctx=ctx)
            spoken = self.narrator.answer_question(question, answer, ctx)
            yield {"type": "tool_exec", "tool": "ask_vision", "input": {"question": question},
                   "output": {"status": "success", "answer": answer}}
            detail = (f'Question: "{question}"\n\nObserved:\n{answer}\n\n'
                      f'Active window: "{snap.window_title}" ({len(snap.elements)} controls visible)')
            yield from self._finish(order, spoken, ctx, started, detail=detail)
            return

        # --- deterministic single action: no plan, no tool schemas, no verification call -------
        if intent.kind == "deterministic":
            action = str(intent.slots.get("action") or "")
            args = dict(intent.slots.get("args") or {})
            yield {"type": "status", "message": f"Executing: {action}"}
            result = self.pathfinder._invoke(action, args, ctx)
            if result.get("status") != "success" and intent.slots.get("fallback"):
                fb = intent.slots["fallback"]
                yield {"type": "status", "message": f"Falling back to {fb.get('action')}"}
                result = self.pathfinder._invoke(str(fb.get("action")), dict(fb.get("args") or {}), ctx)
            yield {"type": "tool_exec", "tool": action, "input": args, "output": result}
            self.retina.invalidate()

            spoken = self.narrator.action_line(str(result.get("tool") or action), args, result)
            self._record(order, result, success=result.get("status") == "success", ctx=ctx)
            detail = self.narrator.detail_report(
                Plan(goal=order, source="single",
                     steps=[PlanStep(description=order, done=result.get("status") == "success",
                                     evidence=str(result.get("status")))]),
                [result], time.time() - started, self._stats(),
            )
            yield from self._finish(order, spoken, ctx, started, detail=detail)
            return

        # --- planned execution -----------------------------------------------------------------
        yield from self._run_planned(order, model, ctx, started, intent.kind)

    # ------------------------------------------------------------------

    def _run_planned(
        self, order: str, model: str, ctx: AgentContext, started: float, kind: str
    ) -> Generator[Dict[str, Any], None, None]:
        snapshot = self.retina.snapshot(query=order, ctx=ctx)

        if kind == "single_action":
            plan = self.architect.single_step_plan(order)
        else:
            yield {"type": "status", "message": "Working out the steps..."}
            plan = self.architect.plan(
                order, ctx, screen_context=snapshot.describe_for_prompt(query=order, budget=14), model=model
            )
        ctx.plan = plan
        yield {"type": "status", "message": f"Plan: {len(plan.steps)} step(s)"}
        for i, step in enumerate(plan.steps, 1):
            yield {"type": "thought", "text": f"{i}. {step.description}"
                   + (f" (target: {step.target})" if step.target else "")}

        results: List[Dict[str, Any]] = []
        tool_counts: Dict[str, int] = {}
        budget = plan.budget()
        correction = ""

        for iteration in range(budget):
            if self._stop.is_set():
                yield {"type": "response", "text": "Stopped, Boss."}
                return
            if time.time() - started > MAX_WALL_SECONDS:
                yield {"type": "status", "message": "Hit the time limit for this task."}
                break

            step = plan.current()
            if step is None:
                break
            if step.attempts >= MAX_STEP_ATTEMPTS:
                step.evidence = step.evidence or "I tried this several times without it taking effect"
                break
            step.attempts += 1

            yield {"type": "status",
                   "message": f"Step {plan.steps.index(step) + 1}/{len(plan.steps)}: {step.description}"}

            before = self.retina.snapshot(query=step.target or step.description, ctx=ctx)

            # --- perception-only step ------------------------------------------------------
            if step.lane == "perception":
                answer = self.retina.ask(step.description, ctx)
                result = {"status": "success" if answer else "error", "tool": "ask_vision",
                          "args": {"question": step.description}, "answer": answer or "",
                          "message": "" if answer else "vision returned nothing"}
                results.append(result)
                yield {"type": "tool_exec", "tool": "ask_vision",
                       "input": {"question": step.description}, "output": result}
                if answer:
                    step.done, step.evidence = True, "observed on screen"
                continue

            # --- choose the executor lane --------------------------------------------------
            lane = step.lane if step.lane in ("pathfinder", "hands") else "hands"

            # The cap has to be enforced *before* the executor runs. Counting afterwards means the
            # call that breaches it has already produced its real side effect — the very thing the
            # cap exists to prevent.
            predicted = (self.pathfinder.predict_tool(step, before.window_title) if lane == "pathfinder"
                         else self.hands.predict_tool(step))
            if predicted and tool_counts.get(predicted, 0) >= MAX_CALLS_PER_TOOL:
                yield {"type": "status",
                       "message": f"Refusing to call {predicted} more than {MAX_CALLS_PER_TOOL} times this task."}
                step.evidence = f"{predicted} was already tried {MAX_CALLS_PER_TOOL} times without finishing this"
                break

            if lane == "pathfinder":
                result = self.pathfinder.execute(step, ctx, active_window=before.window_title, model=model)
            else:
                result = self.hands.execute(
                    step, before, ctx,
                    vision_locate=lambda d: self.retina.locate(d, ctx),
                    history=self.history, model=model, correction=correction,
                )
            correction = ""

            tool = str(result.get("tool") or "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            if tool_counts[tool] > MAX_CALLS_PER_TOOL:
                # Backstop for the model-chosen path, where the tool isn't knowable in advance.
                yield {"type": "status", "message": f"Blocking further calls to {tool} this task."}
                step.evidence = f"{tool} was already tried {MAX_CALLS_PER_TOOL} times"
                break

            results.append(result)
            yield {"type": "tool_exec", "tool": tool, "input": result.get("args", {}), "output": result}

            # --- verify --------------------------------------------------------------------
            self.retina.invalidate()
            settled = self.retina.wait_for_change(timeout=1.5)
            after = self.retina.snapshot(query=step.target or step.description, ctx=ctx)
            verdict = self.sentinel.verify(step, result, before, after, ctx)

            if verdict.done:
                step.done = True
                step.evidence = verdict.evidence
                yield {"type": "status", "message": f"Verified: {verdict.evidence}"}
            else:
                step.evidence = verdict.evidence
                yield {"type": "status", "message": f"Not yet: {verdict.evidence}"}
                if result.get("correction"):
                    correction = str(result["correction"])
                elif not settled and result.get("status") == "success":
                    correction = ("That action produced no visible change on screen. Try a different "
                                  "approach rather than repeating it.")

            if plan.complete:
                break

        # --- compose the reply -----------------------------------------------------------------
        elapsed = time.time() - started
        response = self.narrator.compose(plan, results, ctx, elapsed=elapsed, stats=self._stats())
        # Only a plan that was genuinely one action can teach SCHOLAR a one-action shortcut. A
        # multi-step run logs several actions against the same order text, and promoting any one of
        # them would make a future repeat of that order do a fraction of the work.
        learnable = len(plan.steps) == 1
        for result in results:
            self._record(order, result, success=plan.complete and result.get("status") == "success",
                         ctx=ctx, learnable=learnable)
        yield {"type": "detail", "text": response.detail, "data": response.data}
        yield from self._finish(order, response.spoken, ctx, started, detail=None, already_detailed=True)

    # ------------------------------------------------------------------

    def _finish(
        self, order: str, spoken: str, ctx: AgentContext, started: float,
        detail: Optional[str] = None, already_detailed: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        spoken = speechify(spoken) or "Done, Boss."
        self.history.append({"user": order, "bot": spoken})
        if len(self.history) > 40:
            self.history = self.history[-40:]
        if detail is not None and not already_detailed:
            yield {"type": "detail", "text": detail, "data": {}}
        stats = self._stats()
        stats["elapsed_seconds"] = round(time.time() - started, 2)
        self.last_run = stats
        self._log(order, "final_response", {}, spoken, "success")
        yield {"type": "response", "text": spoken, "stats": stats}

    def _record(self, order: str, result: Dict[str, Any], success: bool, ctx: AgentContext,
                learnable: bool = True) -> None:
        tool = str(result.get("tool") or "")
        args = dict(result.get("args") or {})
        self._log(order, tool, args, json.dumps(result)[:500], str(result.get("status") or "unknown"))
        if tool and learnable:
            self.scholar.record(order, tool, args, success=success, ctx=ctx)

    def _log(self, order: str, tool: str, args: Dict[str, Any], output: str, status: str) -> None:
        if self.memory is None:
            return
        try:
            self.memory.log_action(order, "", tool, json.dumps(args), output, status)
        except Exception:
            pass

    def _stats(self) -> Dict[str, Any]:
        retina = self.retina.stats()
        return {
            "llm_calls": self.llm.calls,
            "llm_calls_by_agent": dict(self.llm.calls_by_agent),
            "tree_walks": retina["tree_walks"],
            "walks_avoided": retina["walks_avoided"],
            "vision_calls": retina["vision_calls"],
            "grounding": {
                "resolutions": self.anchor.groundings,
                "model_disambiguations": self.anchor.model_disambiguations,
                "vision_escalations": self.anchor.vision_escalations,
            },
            "verification": self.sentinel.stats(),
            "learned": self.scholar.stats(),
        }

    def agent_stats(self) -> Dict[str, Any]:
        """Full per-agent instrumentation, for the dashboard."""
        return {
            "TRIAGE": self.triage.stats(),
            "RETINA": self.retina.stats(),
            "ANCHOR": {
                "resolutions": self.anchor.groundings,
                "model_disambiguations": self.anchor.model_disambiguations,
                "vision_escalations": self.anchor.vision_escalations,
            },
            "ARCHITECT": self.architect.stats(),
            "PATHFINDER": self.pathfinder.stats(),
            "HANDS": self.hands.stats(),
            "SENTINEL": self.sentinel.stats(),
            "NARRATOR": self.narrator.stats(),
            "SCHOLAR": self.scholar.stats(),
            "EARS": self.ears.stats(),
            "VIGIL": self.vigil.stats(),
            "llm_calls_total": self.llm.calls,
        }
