"""PATHFINDER — the navigation executor.

Handles the "get to the right place" half of the tool surface: launching apps, switching windows,
opening sites and inboxes, running searches, reading the clock. Seven tools instead of twenty-one.

The token arithmetic is the point. The monolith sent all 21 function schemas — roughly 2,000 tokens
— on every step of every task, including steps that could only possibly be a launch. Groq bills per
token and enforces a per-minute token bucket that this project already hits in production (the 429
retry-after handler exists because it happens). Cutting the schema payload by ~70% on navigation
steps cuts latency, cost and rate-limit pressure at once.

The second benefit is accuracy. With a small, coherent tool set the model cannot pick a screen-
manipulation tool for a navigation job — a whole class of error the monolith papered over with
prompt warnings like "never guess a URL for a specific person's chat".
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from .base import SMART_MODEL, Agent, AgentContext, LLMClient, PlanStep, normalize

try:
    from os_automation import OSAutomation, APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES
except Exception:  # pragma: no cover
    OSAutomation, APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES = None, {}, {}, {}


TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "launch_app",
        "description": "Launches an application that isn't already running (chrome, notepad, code, "
                       "calculator, explorer, cmd, powershell, spotify, edge, word, excel).",
        "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
    }},
    {"type": "function", "function": {
        "name": "switch_window",
        "description": "Brings an ALREADY-OPEN window (title contains this text) to the foreground.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
    }},
    {"type": "function", "function": {
        "name": "open_url",
        "description": "Opens a real, well-known URL. Never invent a URL for a specific person's chat or DM.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "open_social_inbox",
        "description": "Opens the real inbox of a chat platform. Chat platforms do not support deep-linking "
                       "to a person's thread — open the inbox, then locate the person on screen.",
        "parameters": {"type": "object", "properties": {
            "platform": {"type": "string", "enum": ["instagram", "whatsapp", "telegram", "messenger"]}},
            "required": ["platform"]},
    }},
    {"type": "function", "function": {
        "name": "search_google",
        "description": "Searches Google for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "search_youtube",
        "description": "Searches YouTube for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_time_date",
        "description": "Returns the current system time and date.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]

_SYSTEM = (
    "You are the navigation module of a Windows desktop agent. Your only job is getting to the "
    "right place: launching apps, switching to open windows, opening sites, running searches.\n"
    "Call exactly one tool. Never guess a URL for an individual person's chat or DM — open that "
    "platform's inbox instead. If the target is already the active window, prefer switch_window."
)


class PathfinderAgent(Agent):
    """Executes navigation steps, deterministically wherever possible."""

    name = "PATHFINDER"

    def __init__(self, llm: Optional[LLMClient] = None, os_api: Any = None):
        super().__init__(llm)
        self.os = os_api or OSAutomation
        self.executions: int = 0
        self.deterministic: int = 0

    # ------------------------------------------------------------------

    def execute(
        self,
        step: PlanStep,
        ctx: Optional[AgentContext] = None,
        active_window: str = "",
        model: str = SMART_MODEL,
    ) -> Dict[str, Any]:
        self.executions += 1
        text = (step.target or step.description or "").strip()

        direct = self.resolve_deterministic(text, active_window)
        if direct:
            self.deterministic += 1
            tool, args = direct
            self._emit(ctx, "status", f"{tool} (no model call needed)")
            return self._invoke(tool, args, ctx)

        return self._execute_via_model(step, ctx, active_window, model)

    # ------------------------------------------------------------------

    _VERB_RE = re.compile(
        r"^(open|launch|start|run|fire up|boot up|pull up|bring up|go to|switch to|switch|"
        r"focus on|focus|jump to|visit|load|get me)\s+(.+)$",
        re.IGNORECASE,
    )
    _SWITCH_VERBS = {"switch to", "switch", "go to", "focus on", "focus", "jump to", "bring up"}

    def predict_tool(self, step: PlanStep, active_window: str = "") -> Optional[str]:
        """The tool this step will use, when knowable without a model call — lets the orchestrator
        enforce the per-tool cap before the side effect rather than after it."""
        resolved = self.resolve_deterministic((step.target or step.description or "").strip(), active_window)
        return resolved[0] if resolved else None

    def resolve_deterministic(self, text: str, active_window: str = ""):
        """Maps an order onto a navigation tool with no model call, or returns None."""
        raw = (text or "").strip().lower()
        if not raw:
            return None

        if any(p in raw for p in ("what time", "the time", "what day", "the date", "today's date")):
            return ("get_time_date", {})

        match = self._VERB_RE.match(raw)
        verb = match.group(1).lower() if match else ""
        # Kept in raw form because normalisation strips punctuation, and "github.com" without its
        # dot is no longer recognisable as a URL.
        target_raw = (match.group(2) if match else raw).strip(" ?!.,")
        target_raw = re.sub(r"^(the|a|an|my)\s+", "", target_raw).strip()
        target_raw = re.sub(r"\s+(please|now|for me|window|app|application|website|site)$", "", target_raw).strip()
        target = normalize(target_raw)
        if not target:
            return None

        if re.match(r"^(https?://|www\.)\S+$", target_raw) or \
                re.match(r"^[\w-]+(\.[\w-]+)*\.(com|org|net|io|dev|co|in|ai|app|gg)$", target_raw):
            return ("open_url", {"url": target_raw})

        # An explicit "switch to X" means bring the existing window forward. _invoke falls back to
        # launching when the window turns out not to be open, so this stays correct either way.
        if verb in self._SWITCH_VERBS:
            return ("switch_window", {"title": target})

        # Already there — re-navigating to the window we are in is a wasted action that also muddies
        # completion checking, since nothing on screen changes.
        if active_window and target in normalize(active_window):
            return ("switch_window", {"title": target})

        if target in APP_ALIASES:
            return ("launch_app", {"app_name": target})
        if target in SOCIAL_INBOX_URLS:
            return ("open_social_inbox", {"platform": target})
        if target in WEBSITE_ALIASES:
            return ("open_url", {"url": WEBSITE_ALIASES[target]})
        return None

    # ------------------------------------------------------------------

    def _invoke(self, tool: str, args: Dict[str, Any], ctx: Optional[AgentContext]) -> Dict[str, Any]:
        if self.os is None:
            return {"status": "error", "message": "OS automation unavailable"}
        try:
            if tool == "launch_app":
                result = self.os.launch_app(args.get("app_name", ""))
            elif tool == "switch_window":
                result = self.os.switch_to_window(args.get("title", ""))
                if result.get("status") != "success":
                    # The window isn't open. Launching is the correct recovery, and doing it here
                    # saves an entire model round-trip that would only reach the same conclusion.
                    name = args.get("title", "")
                    if normalize(name) in APP_ALIASES:
                        self._emit(ctx, "status", f"'{name}' isn't open — launching it instead")
                        result = self.os.launch_app(name)
            elif tool == "open_url":
                result = self.os.open_url(args.get("url", "https://google.com"))
            elif tool == "open_social_inbox":
                result = self.os.open_social_inbox(args.get("platform", ""))
            elif tool == "search_google":
                result = self.os.search_google(args.get("query", ""))
            elif tool == "search_youtube":
                result = self.os.search_youtube(args.get("query", ""))
            elif tool == "get_time_date":
                result = self.os.get_time_and_date()
            else:
                return {"status": "error", "message": f"'{tool}' is not a navigation tool"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "tool": tool, "message": str(e)}
        result = dict(result or {})
        result["tool"] = tool
        result["args"] = args
        self._emit(ctx, "tool_exec", f"{tool} -> {result.get('status')}", tool=tool, input=args, output=result)
        return result

    def _execute_via_model(
        self, step: PlanStep, ctx: Optional[AgentContext], active_window: str, model: str
    ) -> Dict[str, Any]:
        user = f'Step to perform: "{step.description}"'
        if step.target:
            user += f'\nTarget named in the order: "{step.target}"'
        if active_window:
            user += f'\nActive window right now: "{active_window}"'
        message, error = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            agent=self.name, model=model, tools=TOOLS, tool_choice="required",
            temperature=0.0, timeout=25,
        )
        if error or not message:
            return {"status": "error", "message": error or "no response from the model"}
        calls = message.get("tool_calls") or []
        if not calls:
            return {"status": "error", "message": "the navigation module chose no tool"}
        call = calls[0]
        tool = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        return self._invoke(tool, args, ctx)

    def stats(self) -> Dict[str, Any]:
        return {
            "executions": self.executions,
            "deterministic": self.deterministic,
            "model_calls_saved": self.deterministic,
        }
