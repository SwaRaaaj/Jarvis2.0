"""NARRATOR — the response composer.

Splits what JARVIS *says* from what JARVIS *shows*, because they have opposite requirements and
the monolith conflated them.

Spoken output goes through pyttsx3. Markdown headers, bullet points and asterisks read aloud as
unintelligible noise — the existing `ask_vision` docstring documents exactly this problem with
gemma3's structured answers, and patches it with a formatting instruction glued onto every prompt.
But the constraint cuts the other way too: keeping every reply to one short sentence, as the old
system prompt demanded, is why JARVIS answered a screen question with "Done, Boss" and nothing else.

NARRATOR produces both registers from the same run trace:

    spoken  - one or two clean sentences, no markup, safe for TTS
    detail  - the full breakdown for the console and the dashboard: what was planned, what was
              actually clicked, what the evidence was, what didn't happen and why

It also reports honestly. `partial()` says which steps completed and which didn't rather than
emitting the old blanket "I ran out of steps before fully confirming this was done", and any step
ANCHOR flagged as off-target is named as unfinished instead of being quietly counted as success.

Most replies are templated and cost nothing. A model call is spent only where phrasing genuinely
carries information — summarising a screen, or answering a question in the user's own terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import FAST_MODEL, Agent, AgentContext, LLMClient, Plan, PlanStep, Verdict

_MARKDOWN = re.compile(r"[*_`#>|~]+")
_BULLETS = re.compile(r"^\s*[-*•]\s*", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s*", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]+"
)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def speechify(text: str, max_sentences: int = 3) -> str:
    """Makes text safe and pleasant to hear.

    Strips the markup a language model reaches for by default, collapses whitespace, and caps
    length — a paragraph read at 175 words per minute is a long time to wait for a desktop
    assistant to stop talking.
    """
    if not text:
        return ""
    out = _CODE_FENCE.sub(" ", text)
    out = _BULLETS.sub("", out)
    out = _NUMBERED.sub("", out)
    out = _MARKDOWN.sub("", out)
    out = _EMOJI.sub("", out)
    out = re.sub(r"https?://\S+", "that link", out)
    out = re.sub(r"\s+", " ", out).strip()
    sentences = re.split(r"(?<=[.!?])\s+", out)
    trimmed = " ".join(s for s in sentences[:max_sentences] if s).strip()
    return trimmed or out[:280]


@dataclass
class Response:
    """What to say, what to show, and the machine-readable trace behind both."""

    spoken: str = ""
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"spoken": self.spoken, "detail": self.detail, "data": self.data}


class NarratorAgent(Agent):
    """Turns a run trace into a spoken reply plus a detailed report."""

    name = "NARRATOR"

    def __init__(self, llm: Optional[LLMClient] = None, address: str = "Boss"):
        super().__init__(llm)
        self.address = address
        self.compositions: int = 0
        self.model_compositions: int = 0

    # ==================================================================
    # Templated replies — zero model calls
    # ==================================================================

    def action_line(self, tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> str:
        """One natural sentence describing a completed action, built from the tool result alone."""
        who = self.address
        matched = result.get("matched_name") or ""
        if tool in ("click_coordinate", "click_element", "locate_and_click"):
            name = matched or args.get("text") or args.get("description")
            return f"Clicked '{name}' for you, {who}." if name else f"Clicked that for you, {who}."
        if tool == "type_text":
            text = str(args.get("text") or "")
            preview = text if len(text) <= 40 else text[:37] + "..."
            return f"Typed \"{preview}\", {who}." if preview else f"Typed that in, {who}."
        if tool == "launch_app":
            return f"Opened {str(args.get('app_name', 'that')).title()}, {who}."
        if tool == "switch_window":
            return f"Switched to {result.get('title', 'that window')}, {who}."
        if tool == "open_url":
            return f"Opened that page, {who}."
        if tool == "open_social_inbox":
            return f"Opened your {str(args.get('platform', '')).title()} inbox, {who}."
        if tool == "search_google":
            return f"Searched Google for {args.get('query', 'that')}, {who}."
        if tool == "search_youtube":
            return f"Searched YouTube for {args.get('query', 'that')}, {who}."
        if tool == "get_time_date":
            return result.get("formatted") or f"Got the time for you, {who}."
        if tool == "scroll":
            return f"Scrolled {args.get('direction', 'down')}, {who}."
        if tool in ("close_tab", "close_all_tabs", "close_window"):
            return f"Closed that, {who}."
        if tool == "key_combo":
            return f"Pressed {args.get('keys', 'that')}, {who}."
        return f"Done, {who}."

    # ==================================================================
    # Full-run composition
    # ==================================================================

    def compose(
        self,
        plan: Plan,
        results: List[Dict[str, Any]],
        ctx: Optional[AgentContext] = None,
        elapsed: float = 0.0,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Builds the reply for a finished run — complete or not."""
        self.compositions += 1
        done = [s for s in plan.steps if s.done]
        missed = [s for s in plan.steps if not s.done]

        if not missed and done:
            spoken = self._success_line(plan, done, results)
        elif done and missed:
            spoken = self._partial_line(done, missed)
        elif missed:
            spoken = self._failure_line(missed, results)
        else:
            spoken = f"Nothing to do there, {self.address}."

        detail = self.detail_report(plan, results, elapsed, stats)
        return Response(spoken=speechify(spoken), detail=detail, data={
            "plan": plan.as_dict(),
            "steps_done": len(done),
            "steps_total": len(plan.steps),
            "elapsed_seconds": round(elapsed, 2),
            "stats": stats or {},
        })

    def _success_line(self, plan: Plan, done: List[PlanStep], results: List[Dict[str, Any]]) -> str:
        # An informational step's answer *is* the reply — never bury it under "Done".
        for result in reversed(results):
            answer = result.get("answer")
            if answer:
                return str(answer)
        if len(done) == 1 and results:
            last = results[-1]
            return self.action_line(str(last.get("tool") or ""), last.get("args") or {}, last)
        what = "; ".join(s.description.rstrip(".") for s in done[:3])
        if len(done) > 3:
            what += f", and {len(done) - 3} more"
        return f"All done, {self.address} — {what}."

    def _partial_line(self, done: List[PlanStep], missed: List[PlanStep]) -> str:
        first_missed = missed[0]
        blocker = first_missed.evidence or "I couldn't confirm it went through"
        return (
            f"Partly done, {self.address}. I finished {len(done)} of {len(done) + len(missed)} steps, "
            f"but I stopped at \"{first_missed.description.rstrip('.')}\" — {blocker}."
        )

    def _failure_line(self, missed: List[PlanStep], results: List[Dict[str, Any]]) -> str:
        reason = ""
        for result in reversed(results):
            if result.get("status") != "success":
                reason = str(result.get("message") or "")
                break
            if result.get("on_target") is False:
                reason = str(result.get("scope_reason") or "")
                break
        step = missed[0].description.rstrip(".")
        if reason:
            return f"I couldn't do that, {self.address} — {reason}. The step was \"{step}\"."
        return f"I couldn't complete \"{step}\", {self.address}. Want me to try a different way?"

    # ==================================================================
    # The detailed register
    # ==================================================================

    def detail_report(
        self,
        plan: Plan,
        results: List[Dict[str, Any]],
        elapsed: float = 0.0,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """The console/dashboard breakdown. Plain text, but structured — this one is read, not heard."""
        lines: List[str] = [f'Order: "{plan.goal}"']
        if plan.steps:
            lines.append("")
            lines.append(f"Plan ({plan.source}, {len(plan.steps)} steps):")
            for i, step in enumerate(plan.steps, 1):
                mark = "[done]" if step.done else "[not done]"
                lines.append(f"  {i}. {mark} {step.description}")
                if step.target:
                    lines.append(f"       target: {step.target}")
                if step.evidence:
                    lines.append(f"       evidence: {step.evidence}")
                elif not step.done and step.success_criteria:
                    lines.append(f"       expected: {step.success_criteria}")

        if results:
            lines.append("")
            lines.append("Actions taken:")
            for result in results:
                tool = result.get("tool", "?")
                status = result.get("status", "?")
                bits = [f"  - {tool} -> {status}"]
                if result.get("matched_name"):
                    bits.append(f"hit '{result['matched_name']}'")
                grounding = result.get("grounding") or {}
                if grounding.get("method"):
                    bits.append(f"grounded via {grounding['method']} ({grounding.get('confidence')})")
                if result.get("on_target") is False:
                    bits.append("OFF-TARGET")
                if status != "success" and result.get("message"):
                    bits.append(str(result["message"])[:120])
                lines.append(" | ".join(bits))

        lines.append("")
        summary = f"Finished in {elapsed:.1f}s"
        if stats:
            calls = stats.get("llm_calls")
            if calls is not None:
                summary += f" using {calls} model call{'s' if calls != 1 else ''}"
            walks = stats.get("tree_walks")
            avoided = stats.get("walks_avoided")
            if walks is not None:
                summary += f", {walks} screen scan{'s' if walks != 1 else ''}"
                if avoided:
                    summary += f" ({avoided} avoided by cache)"
        lines.append(summary + ".")
        return "\n".join(lines)

    # ==================================================================
    # Model-composed replies — used only where phrasing carries information
    # ==================================================================

    def answer_question(
        self, question: str, evidence: str, ctx: Optional[AgentContext] = None
    ) -> str:
        """Turns raw evidence (usually a vision answer) into a natural spoken reply."""
        if not evidence:
            return f"I couldn't get a clear read on that, {self.address}."
        # A short, already-conversational answer needs no second model pass.
        if len(evidence) <= 220 and not _MARKDOWN.search(evidence) and evidence.count("\n") <= 1:
            return speechify(evidence)
        self.model_compositions += 1
        system = (
            f"You are JARVIS, a desktop assistant speaking out loud to your user (address them as "
            f"{self.address}). Rewrite the given observation as 1-2 natural spoken sentences that "
            f"directly answer their question. Plain conversational text only — no markdown, no "
            f"headers, no bullet points, no lists."
        )
        user = f'They asked: "{question}"\n\nWhat was observed:\n{evidence}'
        text, error = self.llm.text_call(system, user, agent=self.name, model=FAST_MODEL,
                                         temperature=0.3, timeout=15, max_tokens=140)
        if error or not text:
            return speechify(evidence)
        return speechify(text)

    def small_talk(self, order: str, ctx: Optional[AgentContext] = None) -> str:
        self.model_compositions += 1
        system = (
            f"You are JARVIS, {self.address}'s desktop AI assistant. Reply to small talk warmly and "
            f"concisely in ONE short sentence, plain text only — no markdown, no lists."
        )
        text, error = self.llm.text_call(system, order, agent=self.name, model=FAST_MODEL,
                                         temperature=0.5, timeout=12, max_tokens=60)
        if error or not text:
            return f"Online and ready, {self.address}."
        return speechify(text, max_sentences=2)

    def clarify(self, question: str) -> str:
        return speechify(question or f"What would you like me to do, {self.address}?")

    def stats(self) -> Dict[str, Any]:
        return {"compositions": self.compositions, "model_compositions": self.model_compositions}
