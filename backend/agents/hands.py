"""HANDS — the on-screen interaction executor.

Clicks, types, scrolls, keystrokes: everything that manipulates what is already in front of JARVIS.
It never decides *what* to click — that is ANCHOR's job — which is precisely the separation the
monolith lacked.

In the old design one 70B call per step had to simultaneously choose a tool from 21 options, guess
which element name to pass it, and remember not to over-reach. Tool choice and target choice are
different skills, and folding them together is what produced the failure the code base is full of
comments about: told to open a specific person's chat, it picked a plausible tool with a plausible
argument and clicked a generic inbox tab.

Here the split is structural:
    ANCHOR decides WHICH element      (string scoring; usually zero model calls)
    HANDS  decides WHAT ACTION        (verb detection; usually zero model calls)
and only a step that defeats both spends a model call, over 9 tools rather than 21.

Every action that names a target runs through ANCHOR's scope guard afterwards, so "the click
succeeded" can never be mistaken for "the right thing was clicked".
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    SMART_MODEL,
    Agent,
    AgentContext,
    GroundedTarget,
    LLMClient,
    PlanStep,
    ScreenSnapshot,
    normalize,
)
from .anchor import AnchorAgent, Referent
from .fovea import FoveaAgent

try:
    from os_automation import OSAutomation
except Exception:  # pragma: no cover
    OSAutomation = None


TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "click_element",
        "description": "Click a visible control by its exact on-screen name.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The control's visible label"},
            "double": {"type": "boolean", "description": "optional, default false"}},
            "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "click_coordinate",
        "description": "Click an exact pixel coordinate. Only when a named control cannot be used.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "integer"}, "y": {"type": "integer"},
            "double": {"type": "boolean"}, "button": {"type": "string", "enum": ["left", "right"]}},
            "required": ["x", "y"]},
    }},
    {"type": "function", "function": {
        "name": "type_text",
        "description": "Type into whatever input currently has focus. Click the input first if it isn't focused.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
            "press_enter": {"type": "boolean", "description": "optional, default true"}},
            "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "key_combo",
        "description": "Press a keyboard shortcut, e.g. 'ctrl+w', 'enter', 'escape', 'alt+tab', 'ctrl+l'.",
        "parameters": {"type": "object", "properties": {"keys": {"type": "string"}}, "required": ["keys"]},
    }},
    {"type": "function", "function": {
        "name": "scroll",
        "description": "Scroll the mouse wheel up or down.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["up", "down"]},
            "amount": {"type": "integer", "description": "optional, default 5"}},
            "required": ["direction"]},
    }},
    {"type": "function", "function": {
        "name": "close_tab",
        "description": "Close the active browser tab (Ctrl+W).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "close_all_tabs",
        "description": "Close every open browser tab.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "close_window",
        "description": "Close the current foreground window (Alt+F4).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "move_mouse",
        "description": "Move the cursor without clicking, e.g. to reveal hover-only UI.",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                       "required": ["x", "y"]},
    }},
]

_SYSTEM = (
    "You are the hands of a Windows desktop agent. The correct on-screen target has already been "
    "located for you when one exists; your only job is choosing the right physical action.\n"
    "Call exactly one tool. Do not invent targets, recipients or messages that are not in the step. "
    "If a previous attempt failed, choose a genuinely different approach rather than repeating it."
)

# Verb -> action routing. Deterministic because these mappings are not judgment calls, and paying a
# 70B round-trip to learn that "scroll down" means scroll(down) is pure waste.
_CLICK_VERBS = ("click", "press", "tap", "hit", "push", "select", "choose", "open", "pick", "play")
_TYPE_VERBS = ("type", "write", "enter", "input", "send", "say", "message", "reply", "text", "dm")
_SCROLL_RE = re.compile(r"\bscroll\s+(up|down)\b", re.IGNORECASE)
_KEY_RE = re.compile(r"\b((?:ctrl|alt|shift|win)(?:\s*\+\s*\w+)+|enter|escape|esc|tab|backspace|delete|space)\b",
                     re.IGNORECASE)
_CLOSE_TAB_RE = re.compile(r"\bclose\b.*\b(tab|this tab|current tab)\b", re.IGNORECASE)
_CLOSE_ALL_RE = re.compile(r"\bclose\b.*\ball\b.*\btabs?\b", re.IGNORECASE)
_CLOSE_WIN_RE = re.compile(r"\bclose\b.*\b(window|app|application|program)\b", re.IGNORECASE)


class HandsAgent(Agent):
    """Executes on-screen actions against ANCHOR-resolved targets."""

    name = "HANDS"

    def __init__(self, llm: Optional[LLMClient] = None, anchor: Optional[AnchorAgent] = None,
                 os_api: Any = None, fovea: Optional[FoveaAgent] = None):
        super().__init__(llm)
        self.anchor = anchor or AnchorAgent(llm)
        self.fovea = fovea or FoveaAgent(llm=llm)
        self.os = os_api or OSAutomation
        self.executions: int = 0
        self.deterministic: int = 0
        self.scope_blocks: int = 0

    # ------------------------------------------------------------------

    def execute(
        self,
        step: PlanStep,
        snapshot: ScreenSnapshot,
        ctx: Optional[AgentContext] = None,
        vision_locate: Any = None,
        history: Optional[List[Dict[str, str]]] = None,
        model: str = SMART_MODEL,
        correction: str = "",
        ambient_hint: str = "",
    ) -> Dict[str, Any]:
        """Performs one step. Returns the tool result, annotated with grounding and scope info."""
        self.executions += 1
        order_text = step.description or step.target or ""
        referent = self.anchor.extract_referent(step.target or order_text)

        action = self._detect_action(order_text)

        # --- actions that need no target at all --------------------------------------------
        if action in ("close_tab", "close_all_tabs", "close_window", "scroll", "key_combo"):
            self.deterministic += 1
            args = self._args_for(action, order_text)
            self._emit(ctx, "status", f"{action} (no model call needed)")
            return self._invoke(action, args, ctx, referent)

        # --- typing: the payload is the content, the target is where to put it -------------
        if action == "type_text":
            payload = referent.payload or self._extract_payload(order_text)
            if payload:
                self.deterministic += 1
                self._emit(ctx, "status", "typing (no model call needed)")
                return self._invoke("type_text", {"text": payload, "press_enter": True}, ctx, referent)

        # --- clicking: ANCHOR resolves the target, then act --------------------------------
        # An anaphoric step ("do it again", "the other one") names nothing itself, but ANCHOR can
        # recover the referent from history — so it belongs on the grounded path, not the model one.
        if (action == "click_element" or referent.text or referent.ordinal is not None
                or referent.is_anaphoric):
            target = self.anchor.ground_order(
                step.target or order_text, snapshot, ctx,
                vision_locate=vision_locate, history=history, ambient_hint=ambient_hint,
            )
            if not target.resolved and vision_locate is not None and (referent.text or ambient_hint):
                # Last resort before giving up: describe it to the vision model. The accessibility
                # tree simply cannot see canvas-rendered content, video thumbnails or icon-only
                # controls, and refusing to act there is what made JARVIS feel unable to click
                # "any part" of a page.
                described = referent.text or self.anchor._condense_hint(ambient_hint)
                self._emit(ctx, "status", f"not in the accessibility tree — looking for {described!r} visually")
                coords = None
                try:
                    coords = vision_locate(described)
                except Exception:
                    coords = None
                if coords:
                    fine = self.fovea.refine(coords, described, ctx)
                    result = self._invoke("click_coordinate", {"x": fine.x, "y": fine.y},
                                          ctx, referent, matched_name=fine.snapped_to or described)
                    result["grounding"] = {"method": f"vision+{fine.method}",
                                           "confidence": fine.confidence, "referent": described}
                    result["precision"] = fine.as_dict()
                    return result

            if target.resolved:
                self.deterministic += 1
                double = bool(re.search(r"\bdouble[- ]?click\b", order_text, re.IGNORECASE)) \
                    or self._looks_like_file(target.element.name)

                x, y = target.element.x, target.element.y
                precision = None
                if target.method == "vision":
                    # A grid-derived point can sit up to ~80px from the target on a 1080p screen,
                    # which misses a 30px button entirely. FOVEA snaps it onto the real control.
                    fine = self.fovea.refine((x, y), referent.text or target.referent, ctx)
                    x, y, precision = fine.x, fine.y, fine.as_dict()

                result = self._invoke(
                    "click_coordinate", {"x": x, "y": y, "double": double},
                    ctx, referent, matched_name=target.element.name,
                )
                result["grounding"] = target.as_dict()
                if precision:
                    result["precision"] = precision
                return self._apply_scope_guard(result, referent, order_text, ctx)
            hint = self._browser_accessibility_hint()
            self._emit(ctx, "status",
                       f"couldn't place {referent.describe()} on screen — {target.reason}{hint}")

        # --- defeated both: one model call over 9 tools ------------------------------------
        return self._execute_via_model(step, snapshot, ctx, model, referent, correction)

    # ------------------------------------------------------------------

    def predict_tool(self, step: PlanStep) -> Optional[str]:
        """The tool this step will use, when that is knowable without a model call.

        The orchestrator needs this to enforce the per-tool cap *before* the side effect happens.
        Counting after execution means the call that breaches the cap has already run.
        """
        action = self._detect_action(step.description or step.target or "")
        if action in ("close_tab", "close_all_tabs", "close_window", "scroll", "key_combo", "type_text"):
            return action
        if action == "click_element":
            return "click_coordinate"
        return None

    @staticmethod
    def _detect_action(text: str) -> str:
        low = (text or "").lower().strip()
        if _CLOSE_ALL_RE.search(low):
            return "close_all_tabs"
        if _CLOSE_TAB_RE.search(low):
            return "close_tab"
        if _CLOSE_WIN_RE.search(low):
            return "close_window"
        if _SCROLL_RE.search(low):
            return "scroll"
        first = low.split()[0] if low.split() else ""
        if first in _TYPE_VERBS or any(re.match(rf"^{v}\b", low) for v in _TYPE_VERBS):
            return "type_text"
        if _KEY_RE.search(low) and re.match(r"^(press|hit|push)\b", low):
            return "key_combo"
        if first in _CLICK_VERBS or any(re.match(rf"^{v}\b", low) for v in _CLICK_VERBS):
            return "click_element"
        return "unknown"

    @staticmethod
    def _args_for(action: str, text: str) -> Dict[str, Any]:
        if action == "scroll":
            m = _SCROLL_RE.search(text)
            direction = m.group(1).lower() if m else "down"
            amount = 5
            n = re.search(r"\b(\d{1,2})\s*(?:times|clicks|notches)\b", text, re.IGNORECASE)
            if n:
                amount = max(1, min(int(n.group(1)), 20))
            return {"direction": direction, "amount": amount}
        if action == "key_combo":
            m = _KEY_RE.search(text)
            keys = m.group(1).lower().replace(" ", "") if m else "enter"
            return {"keys": "escape" if keys == "esc" else keys}
        return {}

    @staticmethod
    def _extract_payload(text: str) -> str:
        m = re.search(r"[\"'‘“]([^\"'’”]{1,300})[\"'’”]", text)
        if m:
            return m.group(1)
        m = re.search(r"\b(?:saying|that says|to say|type|write|enter)\s+(.+)$", text, re.IGNORECASE)
        if m:
            return m.group(1).strip(" ,.!?")
        return ""

    def _browser_accessibility_hint(self) -> str:
        """Explains *why* a click failed when the cause is a browser hiding its page.

        A generic "I couldn't find it" is useless here, because the fix is concrete and one step
        away: a Chromium browser started without renderer accessibility exposes its tabs and
        toolbar but none of the page, and relaunching it through JARVIS turns every page element
        into an exactly-addressable control.
        """
        try:
            from screen_vision import ScreenVision  # local import keeps the agent layer portable
        except Exception:
            return ""
        vision = getattr(self, "_vision_probe", None)
        if vision is None:
            try:
                vision = ScreenVision()
                self._vision_probe = vision
            except Exception:
                return ""
        try:
            if vision.page_content_accessible() is False:
                return (" — this browser is running without accessibility, so I can see its tabs "
                        "but not the page. Ask me to open Chrome and I'll start it so I can read "
                        "the page properly.")
        except Exception:
            pass
        return ""

    @staticmethod
    def _looks_like_file(name: str) -> bool:
        return bool(re.search(r"\.(py|js|jsx|ts|tsx|html|css|txt|json|md|png|jpg|pdf|docx?|xlsx?)$",
                              name or "", re.IGNORECASE))

    # ------------------------------------------------------------------

    def _invoke(
        self, tool: str, args: Dict[str, Any], ctx: Optional[AgentContext],
        referent: Optional[Referent] = None, matched_name: str = "",
    ) -> Dict[str, Any]:
        if self.os is None:
            return {"status": "error", "message": "OS automation unavailable"}
        try:
            if tool == "click_coordinate":
                result = self.os.click(int(args.get("x", 0)), int(args.get("y", 0)),
                                       clicks=2 if args.get("double") else 1,
                                       button=args.get("button", "left"))
            elif tool == "click_element":
                # Reached only from the model path; ANCHOR already handles the named-target case.
                result = {"status": "error", "message": "click_element must be grounded by ANCHOR first"}
            elif tool == "type_text":
                result = self.os.type_text(args.get("text", ""), press_enter=bool(args.get("press_enter", True)))
            elif tool == "key_combo":
                result = self.os.key_combination(args.get("keys", "enter"))
            elif tool == "scroll":
                result = self.os.scroll(args.get("direction", "down"), int(args.get("amount", 5)))
            elif tool == "move_mouse":
                result = self.os.move_mouse(int(args.get("x", 0)), int(args.get("y", 0)))
            elif tool == "close_tab":
                result = self.os.close_active_tab()
            elif tool == "close_all_tabs":
                result = self.os.close_all_browser_tabs(max_tabs=15)
            elif tool == "close_window":
                result = self.os.close_active_window()
            else:
                return {"status": "error", "message": f"'{tool}' is not an interaction tool"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "tool": tool, "message": str(e)}
        result = dict(result or {})
        result["tool"] = tool
        result["args"] = args
        if matched_name:
            result["matched_name"] = matched_name
        self._emit(ctx, "tool_exec", f"{tool} -> {result.get('status')}", tool=tool, input=args, output=result)
        return result

    def _apply_scope_guard(
        self, result: Dict[str, Any], referent: Referent, order: str, ctx: Optional[AgentContext]
    ) -> Dict[str, Any]:
        """Attaches ANCHOR's on-target verdict so the orchestrator can never treat a successful but
        misdirected action as progress."""
        on_target, reason = self.anchor.verify_on_target(referent, result)
        result["on_target"] = on_target
        result["scope_reason"] = reason
        if not on_target:
            self.scope_blocks += 1
            result["correction"] = self.anchor.scope_report(order, referent, result)
            self._emit(ctx, "verify", f"off-target: {reason}")
        return result

    def _execute_via_model(
        self, step: PlanStep, snapshot: ScreenSnapshot, ctx: Optional[AgentContext],
        model: str, referent: Referent, correction: str,
    ) -> Dict[str, Any]:
        query = step.target or step.description
        user = (
            f'Step to perform: "{step.description}"\n'
            f'{f"Target named in the order: {step.target}" if step.target else ""}\n\n'
            f"{snapshot.describe_for_prompt(query=query, budget=24)}"
        )
        if correction:
            user += f"\n\n{correction}"
        message, error = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            agent=self.name, model=model, tools=TOOLS, tool_choice="required",
            temperature=0.0, timeout=25,
        )
        if error or not message:
            return {"status": "error", "message": error or "no response from the model"}
        calls = message.get("tool_calls") or []
        if not calls:
            return {"status": "error", "message": "the interaction module chose no tool"}
        call = calls[0]
        tool = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except Exception:
            args = {}

        # A model-chosen click_element still gets grounded rather than trusted — the whole point of
        # ANCHOR is that a label the model produced is a hypothesis, not a screen coordinate.
        if tool == "click_element":
            text = str(args.get("text") or "")
            target = self.anchor.ground(self.anchor.extract_referent(text), snapshot, ctx, allow_model=False)
            if not target.resolved:
                return {"status": "error", "tool": tool, "args": args,
                        "message": f"no on-screen control matches '{text}'"}
            result = self._invoke("click_coordinate",
                                  {"x": target.element.x, "y": target.element.y, "double": bool(args.get("double"))},
                                  ctx, referent, matched_name=target.element.name)
            result["grounding"] = target.as_dict()
            return self._apply_scope_guard(result, referent, step.description, ctx)

        result = self._invoke(tool, args, ctx, referent)
        return self._apply_scope_guard(result, referent, step.description, ctx)

    def stats(self) -> Dict[str, Any]:
        return {
            "executions": self.executions,
            "deterministic": self.deterministic,
            "model_calls_saved": self.deterministic,
            "scope_blocks": self.scope_blocks,
        }
