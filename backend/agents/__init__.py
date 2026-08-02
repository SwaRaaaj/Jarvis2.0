"""JARVIS multi-agent cortex.

The original design ran every command through one monolithic ReAct agent that held all 21 tool
schemas, the intent routing, the execution loop, the completion judgment and the phrasing of the
reply. That agent paid for all of those jobs on every single step: ~2k tokens of tool schemas per
Groq call, a fresh full UI-Automation tree walk per step, and a *second* Groq call after every
terminal action just to ask "are we done yet?".

This package splits that work across specialised agents that each do one job with a small tool
surface and a small prompt. Each is independently testable with no screen and no network.

    ANCHOR      Binds the literal words of an order to a real on-screen target, and refuses to
                let an action count as "following the order" when the thing it touched isn't the
                thing that was named. This is the agent that keeps JARVIS inside screen context.
    RETINA      Perception. Owns screen capture, the UI-Automation tree, change detection,
                caching, query-aware element ranking, and when to spend a vision call.
    TRIAGE      Intent router. One cheap classification instead of falling through regex maps
                into the expensive loop.
    ARCHITECT   Planner. Turns a multi-clause order into an explicit checklist with per-step
                success criteria.
    PATHFINDER  Executor for "get to the right place" actions (launch, switch, navigate).
    HANDS       Executor for "manipulate what's on screen" actions (click, type, scroll).
    SENTINEL    Verifier. Answers "did that actually work?" from screen evidence, usually with
                no model call at all.
    NARRATOR    Composes the spoken reply and the detailed breakdown from the run trace.
    SCHOLAR     Mines the execution log into reusable rules so repeat orders get faster.
    EARS        Voice gate. Decides whether speech was even addressed to JARVIS.
    CORTEX      Orchestrator that wires the above into the existing process_command() contract.
"""

from .base import (
    AgentContext,
    AgentEvent,
    LLMClient,
    ScreenElement,
    ScreenSnapshot,
    GroundedTarget,
    Intent,
    Plan,
    PlanStep,
    Verdict,
)

__all__ = [
    "AgentContext",
    "AgentEvent",
    "LLMClient",
    "ScreenElement",
    "ScreenSnapshot",
    "GroundedTarget",
    "Intent",
    "Plan",
    "PlanStep",
    "Verdict",
]
