import os
import requests
import json
import re
import time
import threading
from typing import Dict, Any, List, Generator, Optional, Tuple

from pc_telemetry import PCTelemetry
from os_automation import OSAutomation, APP_ALIASES, SOCIAL_INBOX_URLS, WEBSITE_ALIASES
from screen_vision import ScreenVision
from memory_vault import MemoryVault

# Reasoning + tool-calling now runs on Groq's cloud API instead of local Ollama — verified live
# (2026-07-28) that llama-3.3-70b-versatile's native tool-calling correctly refuses to invent a tool
# outside the declared set (unlike the local 3B model, which needed JSON-schema-constrained decoding
# to achieve the same guarantee), responds in under a second, and gives well-reasoned completion
# judgments. Groq currently has NO vision-capable model on this account (checked the live model list —
# text models only), so screenshot understanding (ask_vision / locate_via_vision) stays on the local
# gemma3:4b via Ollama, unchanged from before.


def _load_local_env_file() -> Dict[str, str]:
    """Fallback for GROQ_API_KEY (and any other secret) when the OS environment variable hasn't
    reached this process yet. A newly-set persistent Windows env var only propagates to processes
    started *after* it was set, inherited from a parent (explorer.exe, a shell) that has itself
    refreshed its cached environment — an already-running terminal/IDE, or a fresh shell spawned
    from an already-running parent process tree, can keep serving a stale snapshot indefinitely.
    Reading a local .env file sidesteps that timing problem entirely."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    values: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return values
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        pass
    return values


_LOCAL_ENV = _load_local_env_file()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or _LOCAL_ENV.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "gemma3:4b"  # local, via screen_vision.py — Groq has no vision model on this account

MAX_REACT_STEPS = 8  # absolute ceiling; most tasks get a tighter task-scoped budget, see _estimate_step_budget
MAX_WALL_SECONDS = 90  # hard wall-clock budget for a whole task, regardless of step count
MAX_CALLS_PER_TOOL = 3  # hard cap per tool name per task — stops repeated real side effects outright
MAX_HISTORY_TURNS = 4  # sliding window of turns kept in the prompt — bounds token growth against Groq's per-minute token limit

# Tools that themselves constitute "doing the thing" the user asked for, as opposed to
# information-gathering tools (ask_vision, wait) that never complete a task on their own — used by
# the single-action auto-complete short-circuit below.
TERMINAL_ACTION_TOOLS = {
    "click_element", "click_coordinate", "locate_and_click", "type_text", "launch_app",
    "open_url", "switch_window", "open_social_inbox", "key_combo", "key_combination",
    "scroll", "close_tab", "close_all_tabs", "close_window", "find_and_click_chat"
}

# Purely informational/positioning helpers that essentially never complete a task on their own —
# skipped by the completion checker to avoid a wasted extra model call on every single scroll/wait.
NON_TERMINAL_TOOLS = {"wait", "scroll", "move_mouse"}

# Tools whose successful result can itself BE the final answer for a purely informational request
# ("what can you see", "is there a red light on"). Found live: for a single-clause info request, the
# model correctly called ask_vision and got a good answer, then — since ask_vision wasn't treated as
# "the task could be done now" — kept going and invented an entirely unrelated action (it actually
# clicked a real Minimize button that was never asked for). Treating a successful ask_vision as
# completion-eligible, same as a physical action, closes that gap at the code level instead of
# hoping the model remembers to stop.
INFO_ANSWER_TOOLS = {"ask_vision"}

# Formal OpenAI-style function-calling tool definitions for Groq's native tool-calling — this
# replaces the old text-embedded tool list + JSON-schema-constrained decoding that Ollama needed.
# Native tool_calls are a stronger guarantee: verified live that even when deliberately prompted to
# invent a fake tool ("banana_launcher"), the model picked a real declared tool instead every time.
GROQ_TOOLS = [
    {"type": "function", "function": {
        "name": "click_element",
        "description": "Click a UI element by visible text and/or control type. For ordinals ('3rd "
                        "profile') omit text, pass control_type + ordinal. Falls back to vision if no "
                        "match found.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Visible label substring (optional)"},
                "control_type": {"type": "string", "description": "e.g. Button, ListItem, TabItem, Edit, Hyperlink (optional)"},
                "ordinal": {"type": "integer", "description": "1-indexed position among matches, e.g. 3 for '3rd profile' (optional, default 1)"},
                "double": {"type": "boolean", "description": "Double-click instead of single-click (optional, default false)"}
            },
            "required": []
        }
    }},
    {"type": "function", "function": {
        "name": "click_coordinate",
        "description": "Clicks an exact known pixel coordinate.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"}, "y": {"type": "integer"},
                "double": {"type": "boolean", "description": "optional, default false"},
                "button": {"type": "string", "enum": ["left", "right"], "description": "optional, default left"}
            },
            "required": ["x", "y"]
        }
    }},
    {"type": "function", "function": {
        "name": "locate_and_click",
        "description": "For icon-only/image targets with no accessible name — vision locates and clicks it.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Natural language description of the target"},
                "double": {"type": "boolean", "description": "optional, default false"}
            },
            "required": ["description"]
        }
    }},
    {"type": "function", "function": {
        "name": "ask_vision",
        "description": "Ask a visual question the accessibility tree can't answer (e.g. 'which chat row "
                        "shows 4+ unread?'). Answer is plain text — put it straight into speak_final.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"]
        }
    }},
    {"type": "function", "function": {
        "name": "find_and_click_chat",
        "description": "Finds and clicks a specific chat/conversation ROW by contact name and/or "
                        "minimum unread count — this is the tool for 'open the chat of/with <name>', "
                        "not click_element (which tends to match a generic tab/menu item like an inbox "
                        "tab instead of the actual named conversation).",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_hint": {"type": "string", "description": "name or partial name (optional)"},
                "min_unread": {"type": "integer", "description": "optional, default 0"}
            },
            "required": []
        }
    }},
    {"type": "function", "function": {
        "name": "type_text",
        "description": "Types into whatever input is currently focused. Click the input field first if "
                        "it isn't already focused.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "press_enter": {"type": "boolean", "description": "optional, default true"}
            },
            "required": ["text"]
        }
    }},
    {"type": "function", "function": {
        "name": "key_combo",
        "description": "Presses a keyboard shortcut, e.g. 'ctrl+w', 'enter', 'escape', 'alt+tab', 'ctrl+l'.",
        "parameters": {"type": "object", "properties": {"keys": {"type": "string"}}, "required": ["keys"]}
    }},
    {"type": "function", "function": {
        "name": "scroll",
        "description": "Scrolls the mouse wheel up or down.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {"type": "integer", "description": "optional, default 5"}
            },
            "required": ["direction"]
        }
    }},
    {"type": "function", "function": {
        "name": "move_mouse",
        "description": "Moves the mouse cursor to a coordinate without clicking (e.g. to trigger hover UI).",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}
    }},
    {"type": "function", "function": {
        "name": "launch_app",
        "description": "Launches an application that isn't already running (chrome, notepad, code, "
                        "calculator, explorer, cmd, powershell, spotify, edge, etc.).",
        "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}
    }},
    {"type": "function", "function": {
        "name": "switch_window",
        "description": "Brings an ALREADY-OPEN window (title contains this text) to the foreground. Always "
                        "works for real open windows.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    }},
    {"type": "function", "function": {
        "name": "open_url",
        "description": "Opens a real, well-known URL (search engines, youtube, a site's home/inbox page, "
                        "or m.me/<user>). Never invent a URL for a specific person's chat/DM.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    }},
    {"type": "function", "function": {
        "name": "open_social_inbox",
        "description": "Opens the real inbox of a chat platform. Never guess a per-contact URL — open the "
                        "inbox then find the person via click_element/find_and_click_chat.",
        "parameters": {
            "type": "object",
            "properties": {"platform": {"type": "string", "enum": ["instagram", "whatsapp", "telegram", "messenger"]}},
            "required": ["platform"]
        }
    }},
    {"type": "function", "function": {
        "name": "search_youtube",
        "description": "Searches YouTube for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "search_google",
        "description": "Searches Google for a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Pauses briefly (max 5s) after navigation/clicks that need a moment to load.",
        "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}
    }},
    {"type": "function", "function": {
        "name": "get_time_date",
        "description": "Returns the current system time and date.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "close_tab",
        "description": "Closes the active browser tab (Ctrl+W).",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "close_all_tabs",
        "description": "Closes all open browser tabs.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "close_window",
        "description": "Closes the current foreground window (Alt+F4).",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "speak_final",
        "description": "Call this when — and only when — the task is fully complete, to speak the final "
                        "response to the user. This is the ONLY way to end the task.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "What to say out loud, short and natural"}},
            "required": ["message"]
        }
    }},
]

SYSTEM_PROMPT = """You are JARVIS, an autonomous Windows desktop control agent. You accomplish tasks by
directly perceiving what's on screen and acting on it (clicking, typing, launching apps) — exactly like a
human using the mouse and keyboard. You are NOT a chatbot; you are a hands-on operator.

SCOPE DISCIPLINE (most important rule): only do what the task literally asked for. The instant every
explicit clause is done, call speak_final and STOP. Never invent a new sub-task, recipient, or message.

Each turn you see the active window and visible on-screen UI elements with pixel coordinates. Call
exactly one tool per turn.

NEVER guess a URL for a specific person's chat/DM (instagram.com/direct/t/bob does NOT work). Call
open_social_inbox instead, then find the person on screen via click_element/find_and_click_chat.
(Exception: m.me/<username> is a real Messenger link, fine via open_url.)

For "the Nth item" (e.g. "play the 2nd video"), prefer click_element with control_type + ordinal over
vision-based tools — the accessibility tree gives an exact position.

If a tool result errors or nothing matched, try a genuinely different approach — never repeat the same
failed call. Never fabricate success. Keep speak_final short and natural (1 sentence).
"""


class OllamaEngine:
    """Multi-step ReAct agent: perceives the live screen (UI Automation tree + local gemma3:4b vision
    fallback), decides one action at a time via Groq's cloud API (native tool-calling), executes it,
    re-observes, and repeats until the task is complete or a step budget is exhausted."""

    def __init__(self, memory_vault: MemoryVault, screen_vision: ScreenVision):
        self.memory = memory_vault
        self.vision = screen_vision
        self.preferred_model = GROQ_MODEL
        self.history: List[Dict[str, str]] = []
        self._stop_event = threading.Event()

        # The multi-agent cortex replaces this class's single ReAct loop. The legacy loop below is
        # kept intact and reachable via JARVIS_LEGACY_ENGINE=1 — it is the fallback if the agent
        # layer can't be constructed (a missing dependency, say), so a bad import degrades to the
        # old behaviour instead of taking the assistant offline.
        self.cortex = None
        self.use_cortex = os.environ.get("JARVIS_LEGACY_ENGINE", "").strip() not in ("1", "true", "yes")
        if self.use_cortex:
            try:
                from agents.cortex import Cortex
                from agents.base import LLMClient

                self.cortex = Cortex(
                    memory=memory_vault,
                    vision=screen_vision,
                    telemetry=PCTelemetry,
                    os_api=OSAutomation,
                    llm=LLMClient(api_key=GROQ_API_KEY),
                    address=(memory_vault.get_user_info("user_name") or "Boss").split("/")[-1].strip() or "Boss",
                )
                self.history = self.cortex.history
                threading.Thread(target=self._mine_history_async, daemon=True).start()
            except Exception as e:
                print(f"[Cortex init failed, using legacy engine]: {e}")
                self.cortex = None
                self.use_cortex = False

    def _mine_history_async(self):
        """Backfills SCHOLAR's rules from the existing execution log, off the startup path so a
        large log can't delay the assistant coming online."""
        try:
            report = self.cortex.mine_history()
            if report.get("promoted"):
                print(f"[SCHOLAR] learned {report['promoted']} shortcuts from {report['scanned']} log entries")
        except Exception as e:
            print(f"[SCHOLAR mining skipped]: {e}")

    def cancel(self):
        """Requests the current in-flight task to stop before its next step (safety kill-switch)."""
        self._stop_event.set()
        if self.cortex is not None:
            self.cortex.cancel()

    def agent_stats(self) -> Dict[str, Any]:
        """Per-agent instrumentation for the dashboard; empty when running the legacy engine."""
        return self.cortex.agent_stats() if self.cortex is not None else {}

    def _call_groq_chat(self, payload: Dict[str, Any], timeout: int = 30) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """POSTs to Groq's OpenAI-compatible /chat/completions with one short automatic retry, returning
        (message, friendly_error) — exactly one of the two is not None. `message` is the raw assistant
        message dict (so callers can read either .content or .tool_calls as needed)."""
        if not GROQ_API_KEY:
            return None, "I don't have a Groq API key configured, Boss — set the GROQ_API_KEY environment variable."
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        last_err = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = requests.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"], None
                if resp.status_code == 429:
                    # Groq's per-minute token bucket is real and can be hit mid-task (verified live:
                    # it returns a "retry-after" header with the actual wait needed) — worth a bounded
                    # automatic wait-and-retry rather than failing the whole task outright.
                    if attempt < max_attempts - 1:
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
                    err_body = resp.json().get("error", {}).get("message", resp.text[:200])
                except Exception:
                    err_body = resp.text[:200]
                last_err = f"Groq API error (HTTP {resp.status_code}): {err_body}"
            except requests.exceptions.ConnectionError:
                last_err = "I can't reach Groq's API right now, Boss — check your internet connection."
            except requests.exceptions.Timeout:
                last_err = "Groq's API is taking too long to respond, Boss."
            except Exception as e:
                last_err = f"Execution error: {e}"
            if attempt < max_attempts - 1:
                time.sleep(1.0)
        return None, last_err

    # Recognized small-talk that has no on-screen action to take — answered directly by a single
    # plain-text chat call with no tools, instead of being forced through the tool-calling ReAct loop
    # (which was confusing "hello Jarvis" for a click target with the old local model).
    CONVERSATIONAL_PATTERNS = [
        r'^(hi|hello|hey)\b', r'\bhow are you\b', r'\bwho are you\b', r'\bwhat can you do\b',
        r'\bwhat are you\b', r'^(thanks|thank you)\b', r'^good\s?(morning|afternoon|evening|night)\b',
        r'^(bye|goodbye|see you)\b', r'\bare you (there|online|awake|listening)\b'
    ]

    def _try_conversational(self, user_cmd: str) -> Optional[str]:
        cmd = user_cmd.lower().strip()
        if not any(re.search(p, cmd) for p in self.CONVERSATIONAL_PATTERNS):
            return None
        payload = {
            "model": self.preferred_model,
            "messages": [
                {"role": "system", "content": "You are JARVIS, the user's desktop AI assistant. Reply to small talk warmly and concisely in ONE short sentence, plain text only — no JSON, no tool calls."},
                {"role": "user", "content": user_cmd}
            ],
            "temperature": 0.4,
            "max_tokens": 60
        }
        message, error = self._call_groq_chat(payload, timeout=15)
        if message and message.get("content"):
            return message["content"].strip()
        return error or "Hello, Boss! JARVIS online and ready."

    # "Can you see my screen" / "what's on screen" has exactly one correct action — ask_vision —
    # but observed live that leaving this to the tool-calling loop is unreliable: asked "can you see
    # my screen", the model skipped ask_vision entirely and clicked a real "Maximize" button instead,
    # then the single-action auto-complete confidently reported "Done, Boss — clicked Maximize" to a
    # question that had nothing to do with clicking anything. Routing this deterministically straight
    # to ask_vision removes the tool-selection risk for this common phrasing entirely.
    SCREEN_DESCRIPTION_PATTERNS = [
        r'\bcan you see (my |the )?screen\b', r'\bwhat can you see\b',
        r"\bwhat'?s (currently )?on (my |the )?(current )?screen\b",
        r'\bwhat is (currently )?on (my |the )?(current )?screen\b',
        r'\bdescribe (my |the )?screen\b', r'\bwhat do you see\b'
    ]

    def _try_screen_description(self, user_cmd: str) -> Optional[str]:
        cmd = user_cmd.lower().strip()
        cmd = re.sub(r'\bjarvis\b', '', cmd)
        cmd = re.sub(r'\s+', ' ', cmd).strip(' ,.?')
        if not any(re.search(p, cmd) for p in self.SCREEN_DESCRIPTION_PATTERNS):
            return None
        answer = self.vision.ask_vision("Briefly, what's the main thing visible on screen right now?")
        return answer or "I'm having trouble reading the screen right now, Boss."

    def list_local_models(self) -> List[str]:
        """Returns the models actually in use: Groq's live model list (reasoning) plus the local vision
        model — kept local since Groq has no vision-capable model on this account."""
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            resp = requests.get(f"{GROQ_BASE_URL}/models", headers=headers, timeout=5)
            if resp.status_code == 200:
                groq_models = [m.get("id") for m in resp.json().get("data", []) if m.get("active")]
                if groq_models:
                    return groq_models + [f"{VISION_MODEL} (vision, local)"]
        except Exception:
            pass
        return [GROQ_MODEL, f"{VISION_MODEL} (vision, local)"]

    # ------------------------------------------------------------------
    # Ultra-fast path: mouse-cursor pointing ("see this"/"click this") and single-noun direct
    # clicks resolve in one UI Automation scan with no LLM round-trip at all (<100ms typical).
    # Compound / multi-step commands are intentionally NOT handled here — they fall through to
    # the full ReAct loop below, which reasons about them one step at a time.
    # ------------------------------------------------------------------

    def subsecond_element_locator(self, user_cmd: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Sub-second (<100ms) screen element and mouse-cursor-pointing shortcut for simple, single-step
        commands only. Returns matched=False for anything that looks like a multi-step instruction so
        it can be handled by the full ReAct loop instead."""
        cmd = user_cmd.lower().strip()

        # Strip the wake word — it's just how the user addresses the assistant, never semantic
        # content to search for. Critically, this project's own folder is named "jarvis", so any
        # window title in this very VS Code / repo contains "jarvis" as a literal substring — without
        # this strip, saying "...Jarvis?" in almost any sentence would fuzzy-match that window title
        # and trigger a blind click on it (this was an observed real bug, not theoretical).
        cmd = re.sub(r'\bjarvis\b', '', cmd)
        cmd = re.sub(r'\s+', ' ', cmd).strip(' ,.')

        # Compound/multi-step commands must go through the full ReAct loop, not this shortcut.
        compound_markers = [" then ", " and then ", " after that", ";", " next "]
        if any(m in cmd for m in compound_markers) or cmd.count(",") >= 2:
            return False, None, ""

        # 0. Deterministic zero-LLM queries. These have one obviously-correct answer, so answering
        # them via the ReAct loop would waste a round-trip for nothing — worse, a small model can talk
        # itself into ignoring the obvious tool entirely (observed in testing: it tried to read the
        # system clock off a screenshot instead of just calling get_time_date).
        time_phrases = ["what time is it", "what's the time", "current time", "what day is it", "what's the date", "today's date", "what is the date", "what is the time"]
        if any(p in cmd for p in time_phrases):
            res = OSAutomation.get_time_and_date()
            return True, res, res.get("formatted", "I couldn't get the time, Boss.")

        # 0b. Deterministic app-launch / social-inbox / website intent. A known single-app or
        # well-known-site request has exactly one correct action, so it never needs an LLM round-trip.
        launch_match = re.match(r'^(?:open|launch|start|run)\s+(.+)$', cmd)
        if launch_match:
            target = launch_match.group(1).strip(' ?.!')
            target = re.sub(r'^(the|a|an)\s+', '', target).strip()
            target = re.sub(r'\s+(please|now|for me|for me please)$', '', target).strip()
            if target in APP_ALIASES:
                res = OSAutomation.launch_app(target)
                if res.get("status") == "success":
                    return True, res, f"Opening {target.title()} for you, Boss!"
                return True, res, f"I tried to open {target.title()} but ran into an issue: {res.get('message', 'unknown error')}"
            if target in SOCIAL_INBOX_URLS:
                res = OSAutomation.open_social_inbox(target)
                return True, res, f"Opening your {target.title()} inbox, Boss!"
            if target in WEBSITE_ALIASES:
                res = OSAutomation.open_url(WEBSITE_ALIASES[target])
                return True, res, f"Opening {target.title()} for you, Boss!"

        # 0c. Deterministic window-switch intent ("switch to chrome", "go to notepad", "focus vs code").
        switch_match = re.match(r'^(?:switch to|switch|go to|focus(?: on)?|bring up)\s+(.+)$', cmd)
        if switch_match:
            target = switch_match.group(1).strip(' ?.!')
            target = re.sub(r'^(the|a|an)\s+', '', target).strip()
            target = re.sub(r'\s+(window|please|now|for me)$', '', target).strip()
            if target:
                res = OSAutomation.switch_to_window(target)
                if res.get("status") == "success":
                    return True, res, f"Switched to {res.get('title', target)}, Boss."
                # Fall through to the ReAct loop rather than failing outright — the target might be
                # something not yet open (needs launch_app) rather than a real window-switch failure.

        # 1. Mouse Cursor Pointing ("see this", "look here", "click this", "what is under cursor")
        pointing_phrases = ["see this", "look here", "click this", "under cursor", "under my cursor", "where cursor is", "what am i pointing"]
        if any(p in cmd for p in pointing_phrases):
            el = self.vision.get_element_at_cursor()
            x, y = el["x"], el["y"]
            item_name = el.get("name", "Target Item")
            res = OSAutomation.click(x, y)
            msg = f"I see '{item_name}' right under your cursor at ({x}, {y}), and I've clicked it, Boss!"
            return True, res, msg

        # 2. Direct single-target click/open on something already visible on screen. Requires an
        # explicit leading action verb and excludes interrogative/conversational phrasing entirely —
        # otherwise any short sentence that happens to share a word with something on screen (like the
        # wake word above) could trigger a blind click on the wrong thing.
        if re.match(r'^(why|what|who|how|when|where|is|are|can|could|would|should|do|does|did)\b', cmd):
            return False, None, ""

        action_match = re.match(r'^(?:open|click(?: on)?|target|select|show|go to)\s+(.+)$', cmd)
        if not action_match:
            return False, None, ""

        if len(cmd.split()) > 8:
            return False, None, ""

        clean_target = re.sub(
            r'^(the|a|an)\s*(chat of|dm of|message of|chat with|chat|message|dm)?\s*',
            '', action_match.group(1), flags=re.IGNORECASE
        ).strip(' ?.')

        if not clean_target or len(clean_target) < 2:
            return False, None, ""

        search_terms = [clean_target] + [w for w in clean_target.split() if len(w) > 2]

        ui_elements = self.vision.find_visible_ui_elements(max_depth=6)

        best_match = None
        highest_score = 0.0
        for el in ui_elements:
            name_lower = el["name"].lower()
            if not name_lower or len(name_lower) < 2:
                continue
            score = 0.0
            for term in search_terms:
                if term in name_lower:
                    score += 2.0
                elif any(w in name_lower for w in term.split() if len(w) > 2):
                    score += 1.0
            if score > highest_score and score >= 1.0:
                highest_score = score
                best_match = el

        if best_match:
            x, y = best_match["x"], best_match["y"]
            is_file = best_match["name"].endswith(('.py', '.js', '.ts', '.html', '.css', '.txt', '.json', '.md'))
            res = OSAutomation.click(x, y, clicks=2 if is_file else 1)
            msg = f"Clicked '{best_match['name']}' at ({x}, {y}) on screen for you, Boss!"
            return True, res, msg

        return False, None, ""

    # ------------------------------------------------------------------
    # Generic tool dispatch used by the ReAct loop
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "click_element":
                return self._tool_click_element(args)
            elif tool_name == "click_coordinate":
                clicks = 2 if args.get("double") else 1
                return OSAutomation.click(int(args.get("x", 0)), int(args.get("y", 0)), clicks=clicks, button=args.get("button", "left"))
            elif tool_name == "locate_and_click":
                return self._tool_locate_and_click(args)
            elif tool_name == "ask_vision":
                answer = self.vision.ask_vision(args.get("question", "Describe what's currently on screen."))
                if answer is None:
                    return {"status": "error", "message": "Vision model did not respond."}
                return {"status": "success", "answer": answer}
            elif tool_name == "find_and_click_chat":
                return self._tool_find_and_click_chat(args)
            elif tool_name == "type_text":
                return OSAutomation.type_text(args.get("text", ""), press_enter=bool(args.get("press_enter", True)))
            elif tool_name == "key_combo" or tool_name == "key_combination":
                return OSAutomation.key_combination(args.get("keys", "enter"))
            elif tool_name == "scroll":
                return OSAutomation.scroll(args.get("direction", "down"), int(args.get("amount", 5)))
            elif tool_name == "move_mouse":
                return OSAutomation.move_mouse(int(args.get("x", 0)), int(args.get("y", 0)))
            elif tool_name == "launch_app":
                return OSAutomation.launch_app(args.get("app_name", ""))
            elif tool_name == "switch_window":
                return OSAutomation.switch_to_window(args.get("title", ""))
            elif tool_name == "open_url":
                return OSAutomation.open_url(args.get("url", "https://google.com"))
            elif tool_name == "open_social_inbox":
                return OSAutomation.open_social_inbox(args.get("platform", ""))
            elif tool_name == "search_youtube":
                return OSAutomation.search_youtube(args.get("query", ""))
            elif tool_name == "search_google":
                return OSAutomation.search_google(args.get("query", ""))
            elif tool_name == "wait":
                seconds = max(0.0, min(float(args.get("seconds", 1.0)), 5.0))
                time.sleep(seconds)
                return {"status": "success", "action": "wait", "seconds": seconds}
            elif tool_name == "get_time_date":
                return OSAutomation.get_time_and_date()
            elif tool_name == "close_tab":
                return OSAutomation.close_active_tab()
            elif tool_name == "close_all_tabs":
                return OSAutomation.close_all_browser_tabs(max_tabs=15)
            elif tool_name == "close_window":
                return OSAutomation.close_active_window()
            else:
                return {
                    "status": "error",
                    "message": f"Unknown tool '{tool_name}'. Valid tools: click_element, click_coordinate, "
                               f"locate_and_click, ask_vision, find_and_click_chat, type_text, key_combo, scroll, "
                               f"move_mouse, launch_app, switch_window, open_url, open_social_inbox, search_youtube, "
                               f"search_google, wait, get_time_date, close_tab, close_all_tabs, close_window, speak_final."
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _tool_click_element(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text = (args.get("text") or "").strip()
        control_type = (args.get("control_type") or "").strip()
        ordinal = args.get("ordinal")
        double = bool(args.get("double", False))

        elements = self.vision.find_visible_ui_elements(max_depth=8)
        pool = elements
        if control_type:
            pool = [e for e in pool if control_type.lower() in e["type"].lower()]
        if text:
            pool = [e for e in pool if text.lower() in e["name"].lower()]

        if not pool:
            if text:
                coords = self.vision.locate_via_vision(text)
                if coords:
                    res = OSAutomation.click(coords[0], coords[1], clicks=2 if double else 1)
                    res["method"] = "vision_fallback"
                    return res
            return {"status": "error", "message": f"No element found matching text='{text}' control_type='{control_type}'."}

        pool = sorted(pool, key=lambda e: (e["top"], e["left"]))
        idx = max(0, min((int(ordinal) - 1) if ordinal else 0, len(pool) - 1))
        el = pool[idx]
        res = OSAutomation.click(el["x"], el["y"], clicks=2 if double else 1)
        res["matched_name"] = el["name"]
        res["matched_type"] = el["type"]
        return res

    def _tool_locate_and_click(self, args: Dict[str, Any]) -> Dict[str, Any]:
        description = (args.get("description") or "").strip()
        double = bool(args.get("double", False))
        if not description:
            return {"status": "error", "message": "description is required"}
        coords = self.vision.locate_via_vision(description)
        if not coords:
            return {"status": "error", "message": f"Vision model could not locate '{description}' on screen."}
        res = OSAutomation.click(coords[0], coords[1], clicks=2 if double else 1)
        res["method"] = "vision"
        res["description"] = description
        return res

    def _tool_find_and_click_chat(self, args: Dict[str, Any]) -> Dict[str, Any]:
        contact_hint = (args.get("contact_hint") or "").strip()
        min_unread = int(args.get("min_unread", 0) or 0)

        elements = self.vision.find_visible_ui_elements(max_depth=10)
        rows = [e for e in elements if ("List" in e["type"] or "Button" in e["type"]) and len(e["name"]) > 1]
        if contact_hint:
            hinted = [r for r in rows if contact_hint.lower() in r["name"].lower()]
            if hinted:
                rows = hinted

        if min_unread > 0:
            question = (
                f"Look at the conversation/chat list visible on screen. Which single conversation row "
                f"shows {min_unread} or more unread messages (bold text and/or a numbered badge)? "
                f"Reply with just the contact/conversation name exactly as shown, or NONE if none qualify."
            )
            answer = self.vision.ask_vision(question)
            if not answer or "none" in answer.lower():
                return {"status": "error", "message": "No conversation with that many unread messages was found.", "vision_answer": answer}
            for r in rows:
                if r["name"].lower() in answer.lower() or answer.lower() in r["name"].lower():
                    res = OSAutomation.click(r["x"], r["y"])
                    res["matched_name"] = r["name"]
                    res["vision_answer"] = answer
                    return res
            return {"status": "error", "message": f"Vision suggested '{answer}' but it didn't match a clickable row.", "vision_answer": answer}

        if rows:
            target = rows[0]
            res = OSAutomation.click(target["x"], target["y"])
            res["matched_name"] = target["name"]
            return res
        return {"status": "error", "message": "No chat/conversation rows detected on screen."}

    # ------------------------------------------------------------------
    # Screen observation + step budgeting + completion checking
    # ------------------------------------------------------------------

    def _build_observation(self, last_result: Optional[Dict[str, Any]] = None) -> str:
        active = PCTelemetry.get_active_window()
        elements = self.vision.find_visible_ui_elements(max_depth=8)

        priority_types = ("Button", "ListItem", "TabItem", "MenuItem", "Edit", "Hyperlink", "CheckBox", "ComboBox", "RadioButton")

        def rank(e):
            return 0 if any(t in e["type"] for t in priority_types) else 1

        elements = sorted(elements, key=lambda e: (rank(e), e["top"], e["left"]))[:25]

        lines = [f"Active window: \"{active.get('title')}\" (process: {active.get('app')})", "", "Visible UI elements (type, name, center x,y):"]
        for e in elements:
            short_type = e["type"].replace("Control", "")
            name = e["name"][:60]
            lines.append(f"- [{short_type}] \"{name}\" at ({e['x']},{e['y']})")
        if not elements:
            lines.append("(no named UI elements detected on screen — try ask_vision or locate_and_click instead)")

        if last_result is not None:
            lines.append("")
            lines.append(f"Result of your last action ({last_result['tool']}): {json.dumps(last_result['result'])[:400]}")

        return "\n".join(lines)

    def _estimate_step_budget(self, user_command: str) -> int:
        """Derives a task-scoped step budget from how many sub-actions the command actually implies, so
        a short 2-clause request can't drift for the full MAX_REACT_STEPS. Kept as defense in depth
        alongside the "stop the instant the task is done" system prompt rule, even with a much more
        capable cloud model — a hard budget costs nothing when the model behaves and is cheap insurance
        when it doesn't."""
        clauses = self._count_clauses(user_command)
        return max(4, min(MAX_REACT_STEPS, clauses * 2 + 2))

    @staticmethod
    def _craft_completion_message(tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Builds a natural confirmation for the single-action auto-complete path, without another
        LLM round-trip — the tool result already has everything needed to describe what happened."""
        if tool_name == "ask_vision":
            return result.get("answer") or "I couldn't get a clear read on the screen, Boss."
        if tool_name in ("click_element", "click_coordinate", "locate_and_click"):
            name = result.get("matched_name") or args.get("text") or args.get("description")
            return f"Done, Boss — clicked '{name}'." if name else "Done, Boss — clicked that for you."
        if tool_name == "type_text":
            return "Typed that in for you, Boss."
        if tool_name == "launch_app":
            return f"Opened {args.get('app_name', 'that')} for you, Boss."
        if tool_name == "open_url":
            return "Opened that page for you, Boss."
        if tool_name == "switch_window":
            return f"Switched to {result.get('title', 'that window')}, Boss."
        if tool_name == "open_social_inbox":
            return f"Opened your {args.get('platform', '')} inbox, Boss."
        if tool_name == "find_and_click_chat":
            name = result.get("matched_name")
            return f"Opened the chat with {name}, Boss." if name else "Opened that chat for you, Boss."
        if tool_name in ("key_combo", "key_combination"):
            return "Done, Boss."
        if tool_name == "scroll":
            return "Scrolled for you, Boss."
        if tool_name in ("close_tab", "close_all_tabs", "close_window"):
            return "Closed that for you, Boss."
        return "Done, Boss."

    @staticmethod
    def _count_clauses(user_command: str) -> int:
        cmd = user_command.lower()
        parts = re.split(r',| then | and then | after that|;| next ', cmd)
        return max(1, len([p for p in parts if p.strip()]))

    _AUTO_COMPLETE_LEADING_VERBS = {
        "Open", "Click", "Go", "Show", "Select", "Type", "Search", "Close", "Switch",
        "Launch", "Find", "Message", "Send", "Text", "Chat"
    }

    @classmethod
    def _extract_named_target(cls, user_command: str) -> Optional[str]:
        """Pulls a likely proper-noun target out of the command (e.g. 'Arundhati' from 'open the chat
        of Arundhati'). Used to stop the single-action auto-complete from declaring victory on a click
        that succeeded but clearly wasn't the thing the task named — observed live: asked to 'open the
        chat of Arundhati', the model clicked a generic 'Instagram Messages' tab (a reasonable *first*
        step) and auto-complete immediately called that "done" without ever looking for Arundhati."""
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', user_command)
        candidates = [w for w in words if w not in cls._AUTO_COMPLETE_LEADING_VERBS]
        return candidates[0] if candidates else None

    def _check_task_complete(self, user_command: str, actions_taken: List[str], model: str) -> Tuple[bool, str]:
        """Narrow yes/no check: 'given these actions, is the task done?' — asked as its own separate,
        easy classification question instead of folded into the harder open-ended 'what should I do
        next' reasoning, which is exactly where a small local model kept inventing unrelated extra
        steps after the real task was already finished. Verified live with Groq: correctly says "not
        complete" on a genuinely partial 2-step task and "complete" once it's actually done, including
        catching a subtle case (search text typed but never confirmed navigated to google.com)."""
        summary = "\n".join(actions_taken) if actions_taken else "(none yet)"
        active = PCTelemetry.get_active_window()
        screen_state = f"Active window right now: \"{active.get('title')}\""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": (
                    "You judge whether a desktop automation task is fully complete, based on the actions "
                    "already taken and the current screen state. Respond with ONLY a JSON object of the "
                    "shape {\"task_complete\": true|false, \"reason\": \"...\"} and nothing else."
                )},
                {"role": "user", "content": f"Task: {user_command}\n\nActions taken so far:\n{summary}\n\n{screen_state}\n\nIs the task now fully complete?"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        message, error = self._call_groq_chat(payload, timeout=20)
        if error or not message:
            return False, ""
        try:
            data = json.loads(message.get("content") or "{}")
            return bool(data.get("task_complete", False)), str(data.get("reason", ""))
        except Exception:
            return False, ""

    # ------------------------------------------------------------------
    # Main entry point: multi-step ReAct loop
    # ------------------------------------------------------------------

    def process_command(self, user_command: str, model_override: str = None) -> Generator[Dict[str, Any], None, None]:
        """Processes a user command end-to-end.

        Delegates to the multi-agent cortex (TRIAGE -> ARCHITECT -> ANCHOR -> HANDS/PATHFINDER ->
        SENTINEL -> NARRATOR), falling back to the original single-loop engine below when the agent
        layer is unavailable or JARVIS_LEGACY_ENGINE=1 is set.

        The yielded event shapes are unchanged — {"type": "status"|"thought"|"tool_exec"|"response"}
        — so every existing consumer keeps working. One new optional event, {"type": "detail"},
        carries the long-form breakdown; consumers that don't know it simply ignore it.
        """
        if self.cortex is not None:
            yield from self.cortex.run(user_command, model=model_override or self.preferred_model)
            return
        yield from self._process_command_legacy(user_command, model_override)

    def _process_command_legacy(self, user_command: str, model_override: str = None) -> Generator[Dict[str, Any], None, None]:
        """The original monolithic ReAct loop, retained as a fallback path."""
        self._stop_event.clear()
        start_time = time.time()

        convo_reply = self._try_conversational(user_command)
        if convo_reply is not None:
            yield {"type": "status", "message": "Replying..."}
            self.history.append({"user": user_command, "bot": convo_reply})
            self.memory.log_action(user_command, "conversational", "chat_reply", "{}", convo_reply, "success")
            yield {"type": "response", "text": convo_reply}
            return

        screen_desc = self._try_screen_description(user_command)
        if screen_desc is not None:
            yield {"type": "status", "message": "Looking at your screen..."}
            yield {"type": "tool_exec", "tool": "ask_vision", "input": {"question": "screen description"}, "output": {"status": "success", "answer": screen_desc}}
            self.history.append({"user": user_command, "bot": screen_desc})
            self.memory.log_action(user_command, "screen description shortcut", "ask_vision", "{}", screen_desc, "success")
            yield {"type": "response", "text": screen_desc}
            return

        matched, output, speech_response = self.subsecond_element_locator(user_command)
        if matched:
            elapsed = round((time.time() - start_time) * 1000, 1)
            yield {"type": "status", "message": f"Targeted item on screen ({elapsed}ms)..."}
            yield {"type": "tool_exec", "tool": "subsecond_locator", "input": {"command": user_command}, "output": output}
            self.history.append({"user": user_command, "bot": speech_response})
            self.memory.log_action(user_command, "subsecond shortcut", "subsecond_locator", "{}", json.dumps(output)[:500], output.get("status", "unknown"))
            yield {"type": "response", "text": speech_response, "tool_outputs": [output]}
            return

        model = model_override or self.preferred_model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {user_command}\n\n{self._build_observation()}"}
        ]

        last_failed_call = None  # (tool_name, args_json) of the previous failed attempt, to catch repeats
        tool_call_counts: Dict[str, int] = {}  # per-tool-name call count this task, regardless of args/outcome
        last_success = None  # most recent successful tool_exec, so a budget-exhausted task can still
                              # report real progress instead of a blanket "couldn't finish" apology
        clause_count = self._count_clauses(user_command)
        is_single_action_task = clause_count == 1
        actions_taken: List[str] = []  # human-readable log fed to the completion checker below
        terminal_action_count = 0  # successful TERMINAL_ACTION_TOOLS calls this task

        step_budget = self._estimate_step_budget(user_command)
        for step in range(step_budget):
            if self._stop_event.is_set():
                yield {"type": "response", "text": "Stopped, Boss."}
                return
            if time.time() - start_time > MAX_WALL_SECONDS:
                yield {"type": "response", "text": "This is taking longer than it should, Boss — I'll stop here rather than keep you waiting."}
                return

            if len(messages) > (2 * MAX_HISTORY_TURNS + 1):
                messages = [messages[0], messages[1]] + messages[-(2 * MAX_HISTORY_TURNS):]

            yield {"type": "status", "message": f"Step {step + 1}/{step_budget}: thinking via {model}..."}

            assistant_msg, error = self._call_groq_chat({
                "model": model,
                "messages": messages,
                "tools": GROQ_TOOLS,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "temperature": 0.1
            })
            if error:
                yield {"type": "response", "text": error}
                return

            if assistant_msg.get("content"):
                yield {"type": "thought", "text": assistant_msg["content"]}

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                messages.append({"role": "assistant", "content": assistant_msg.get("content") or ""})
                messages.append({"role": "user", "content": "You must call exactly one tool from the available list."})
                continue

            messages.append({
                "role": "assistant",
                "content": assistant_msg.get("content"),
                "tool_calls": tool_calls
            })

            call = tool_calls[0]
            tool_name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            call_id = call["id"]

            if tool_name in ("speak_final", "final", "respond", "done", "finish"):
                message = args.get("message") or args.get("text") or "Task completed, Boss."
                self.history.append({"user": user_command, "bot": message})
                self.memory.log_action(user_command, "", "speak_final", json.dumps(args), message, "success")
                yield {"type": "response", "text": message}
                return

            call_signature = (tool_name, json.dumps(args, sort_keys=True))
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

            if tool_call_counts[tool_name] > MAX_CALLS_PER_TOOL:
                # Hard cap, independent of args or success/failure: this is what actually stops a
                # confused model from repeatedly executing real side effects — a text nudge alone
                # wasn't enough with the old local model, and it's cheap insurance here too.
                result = {
                    "status": "error",
                    "message": f"'{tool_name}' has already been called {MAX_CALLS_PER_TOOL} times this task "
                               f"and is now blocked from repeating further. Choose a genuinely different tool, "
                               f"or call speak_final now with your best answer/explanation."
                }
                yield {"type": "status", "message": f"Blocking repeated call to {tool_name} (called {tool_call_counts[tool_name]}x)..."}
            else:
                yield {"type": "status", "message": f"Executing: {tool_name}..."}
                result = self._execute_tool(tool_name, args)

            self.memory.log_action(user_command, "", tool_name, json.dumps(args), json.dumps(result)[:500], result.get("status", "unknown"))
            yield {"type": "tool_exec", "tool": tool_name, "input": args, "output": result}
            if result.get("status") == "success":
                last_success = {"tool": tool_name, "result": result}

            info_answer_ready = tool_name in INFO_ANSWER_TOOLS and bool(result.get("answer"))
            named_target = self._extract_named_target(user_command)
            target_mismatch = False
            if named_target and tool_name in ("click_element", "click_coordinate", "locate_and_click", "find_and_click_chat"):
                matched = (result.get("matched_name") or "").lower()
                if named_target.lower() not in matched:
                    target_mismatch = True

            if (is_single_action_task and result.get("status") == "success"
                    and (tool_name in TERMINAL_ACTION_TOOLS or info_answer_ready) and not target_mismatch):
                # For a genuinely single-action command ("click the X button", "open Y", "what can you
                # see") one successful primary action or a real vision answer IS the whole task — don't
                # spend another round-trip waiting for the model to notice and call speak_final (this
                # is exactly where it previously invented an unrelated action, like clicking Minimize,
                # after already answering an informational question). The target_mismatch check guards
                # a related failure: the task named a specific target (e.g. "the chat of Arundhati")
                # but the click that succeeded didn't actually match that name (it clicked a generic
                # "Instagram Messages" tab instead) — a successful click isn't proof it was the RIGHT
                # click, so don't let a name-mismatched one auto-complete the task.
                auto_message = self._craft_completion_message(tool_name, args, result)
                self.history.append({"user": user_command, "bot": auto_message})
                self.memory.log_action(user_command, "", "auto_complete", "{}", auto_message, "success")
                yield {"type": "response", "text": auto_message}
                return

            action_summary = f"{tool_name}({json.dumps(args, sort_keys=True)}) -> {result.get('status')}"
            extra_detail = result.get("matched_name") or result.get("answer") or result.get("title")
            if extra_detail:
                action_summary += f" ({extra_detail})"
            actions_taken.append(action_summary)
            if result.get("status") == "success" and (tool_name in TERMINAL_ACTION_TOOLS or tool_name in INFO_ANSWER_TOOLS):
                terminal_action_count += 1

            # Only ask the completion-judgment question once there's already objective evidence enough
            # real actions have happened — gating on a plain count rather than trusting the judgment
            # call to always fire at the right time. Skipped on a target_mismatch: we already have
            # direct evidence the click didn't match the task's named target, so there's no point
            # spending a call asking a question we already know the answer to.
            if (result.get("status") == "success" and tool_name not in NON_TERMINAL_TOOLS
                    and terminal_action_count >= clause_count and not target_mismatch):
                yield {"type": "status", "message": "Checking whether the task is complete..."}
                is_done, done_reason = self._check_task_complete(user_command, actions_taken, model)
                if is_done:
                    final_message = done_reason or "Done, Boss!"
                    self.history.append({"user": user_command, "bot": final_message})
                    self.memory.log_action(user_command, "", "completion_check", "{}", final_message, "success")
                    yield {"type": "response", "text": final_message}
                    return

            observation = self._build_observation(last_result={"tool": tool_name, "args": args, "result": result})
            tool_content = f"Result: {json.dumps(result)}\n\n{observation}"
            if target_mismatch:
                tool_content += (
                    f"\n\nNote: the task mentions '{named_target}' but what you just clicked/matched "
                    f"('{result.get('matched_name', '')}') doesn't contain that name — this is very "
                    f"likely NOT the specific thing the task asked for yet. If '{named_target}' is a "
                    f"person's name, try find_and_click_chat with contact_hint='{named_target}', or "
                    f"click_element with text='{named_target}', now that you may be on the right page."
                )
            if result.get("status") == "error" and call_signature == last_failed_call:
                # The model repeated the exact same failing call — force a course-correction instead
                # of burning the whole step budget on it.
                tool_content += (
                    "\n\nYou just repeated that exact failing call again. Do not repeat it a third time. "
                    "Either pick a genuinely different tool from the list, or if you already have enough "
                    "information from a previous ask_vision/tool result, call speak_final now with your "
                    "best answer."
                )
            last_failed_call = call_signature if result.get("status") == "error" else None
            messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_content})

        if last_success:
            progress = self._craft_completion_message(last_success["tool"], {}, last_success["result"])
            final_msg = f"I ran out of steps before fully confirming this was done, Boss. Progress so far: {progress} Let me know if you want me to keep going."
        else:
            final_msg = "I tried several steps but couldn't confidently finish this one, Boss — want me to keep going or try a different approach?"
        self.history.append({"user": user_command, "bot": final_msg})
        yield {"type": "response", "text": final_msg}
