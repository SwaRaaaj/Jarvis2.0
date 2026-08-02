"""Test doubles for the agent layer.

Every agent is built to take its screen, OS and model dependencies by injection, so the whole
cortex runs here with no Windows API, no microphone, no Groq account and no Ollama daemon.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import LLMClient, ScreenElement, ScreenSnapshot  # noqa: E402


# ----------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------


class FakeLLM(LLMClient):
    """An LLMClient whose transport is a Python function.

    Subclassing rather than duck-typing keeps the real counting, retry and JSON-parsing code paths
    under test — only the network hop is replaced.
    """

    def __init__(self, handler: Optional[Callable[[Dict[str, Any]], Any]] = None):
        super().__init__(api_key="test-key")
        self.handler = handler
        self.payloads: List[Dict[str, Any]] = []
        self.fail_with: Optional[str] = None

    def _post(self, payload, timeout):
        self.payloads.append(payload)
        if self.fail_with:
            return None, self.fail_with
        if self.handler is None:
            return {"content": "{}"}, None
        result = self.handler(payload)
        if result is None:
            return None, "no canned response"
        if isinstance(result, str):
            return {"content": result}, None
        return result, None

    # -- helpers for assertions ----------------------------------------------------

    def tool_schema_count(self) -> int:
        """Largest number of tool schemas sent in any single request — the payload-size metric the
        21-tool monolith paid on every step."""
        return max((len(p.get("tools") or []) for p in self.payloads), default=0)

    def models_used(self) -> List[str]:
        return [p.get("model") for p in self.payloads]


def tool_call(name: str, **args) -> Dict[str, Any]:
    """Builds an OpenAI-style tool_calls message body."""
    import json

    return {
        "content": None,
        "tool_calls": [{
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def route(**by_agent_marker: Any) -> Callable[[Dict[str, Any]], Any]:
    """Dispatches canned responses by matching a marker string in the system prompt.

    Agents are identified by distinctive phrases in their own system prompts, so a single handler
    can serve a whole end-to-end run without the test needing to know call ordering.
    """
    markers = {
        "triage": "route desktop-assistant orders",
        "architect": "break a desktop-automation order",
        "sentinel": "judge whether one step",
        "anchor": "match a user's spoken target",
        "hands": "hands of a Windows desktop agent",
        "pathfinder": "navigation module",
        "narrator": "JARVIS",
    }

    def handler(payload: Dict[str, Any]):
        system = ""
        for message in payload.get("messages", []):
            if message.get("role") == "system":
                system = message.get("content") or ""
                break
        for key, marker in markers.items():
            if marker in system and key in by_agent_marker:
                value = by_agent_marker[key]
                return value(payload) if callable(value) else value
        return by_agent_marker.get("default", "{}")

    return handler


# ----------------------------------------------------------------------------------
# Screen
# ----------------------------------------------------------------------------------


class FakeVision:
    """Stands in for ScreenVision: a scriptable element tree and a colour-coded frame."""

    def __init__(self, elements: Optional[List[Dict[str, Any]]] = None, colour: int = 10):
        self.elements = elements or []
        self.colour = colour
        self.walks = 0
        self.vision_questions: List[str] = []
        self.vision_answer = "There is a chat window open with several conversations."
        self.locate_result = None

    def capture_screen_pil(self):
        return Image.new("RGB", (64, 64), (self.colour, self.colour, self.colour))

    def find_visible_ui_elements(self, max_depth: int = 8, max_elements: int = 400):
        self.walks += 1
        return list(self.elements)

    def ask_vision(self, question: str, **kwargs):
        self.vision_questions.append(question)
        return self.vision_answer

    def locate_via_vision(self, description: str, **kwargs):
        return self.locate_result

    # -- scripting helpers ---------------------------------------------------------

    def change_screen(self, elements=None, colour=None, title=None):
        """Simulates the screen changing, so RETINA's digest differs and SENTINEL sees an effect."""
        if elements is not None:
            self.elements = elements
        self.colour = colour if colour is not None else (self.colour + 40) % 250


class FakeTelemetry:
    def __init__(self, title: str = "Desktop", app: str = "explorer.exe"):
        self.title = title
        self.app = app

    def get_active_window(self):
        return {"title": self.title, "app": self.app}


class FakeOS:
    """Records every OS action instead of performing one."""

    def __init__(self, fail: Optional[set] = None, on_action: Optional[Callable] = None):
        self.calls: List[Dict[str, Any]] = []
        self.fail = fail or set()
        self.window_titles: List[str] = []
        # Lets a test model the world responding to an action — a click that opens a conversation,
        # a launch that changes the foreground window. Without it the fake screen never changes and
        # SENTINEL correctly concludes nothing happened.
        self.on_action = on_action

    def _record(self, action: str, **kw):
        self.calls.append({"action": action, **kw})
        if action in self.fail:
            return {"status": "error", "action": action, "message": f"{action} failed"}
        if self.on_action:
            self.on_action(action, kw)
        return {"status": "success", "action": action, **kw}

    def click(self, x, y, button="left", clicks=1):
        return self._record("click", x=x, y=y, button=button, clicks=clicks)

    def type_text(self, text, press_enter=True):
        return self._record("type_text", text=text, press_enter=press_enter)

    def key_combination(self, keys):
        return self._record("key_combination", keys=keys)

    def scroll(self, direction="down", amount=5):
        return self._record("scroll", direction=direction, amount=amount)

    def move_mouse(self, x, y):
        return self._record("move_mouse", x=x, y=y)

    def close_active_tab(self):
        return self._record("close_active_tab")

    def close_all_browser_tabs(self, max_tabs=15):
        return self._record("close_all_browser_tabs")

    def close_active_window(self):
        return self._record("close_active_window")

    def launch_app(self, app_name):
        return self._record("launch_app", app=app_name)

    def switch_to_window(self, title):
        if title in self.window_titles:
            return self._record("switch_to_window", title=title)
        self.calls.append({"action": "switch_to_window", "title": title})
        return {"status": "error", "message": f"No open window found matching '{title}'."}

    def open_url(self, url):
        return self._record("open_url", url=url)

    def open_social_inbox(self, platform):
        return self._record("open_social_inbox", platform=platform)

    def search_google(self, query):
        return self._record("search_google", query=query)

    def search_youtube(self, query):
        return self._record("search_youtube", query=query)

    def get_time_and_date(self):
        return {"status": "success", "time": "3:04 PM", "date": "Monday, August 03, 2026",
                "formatted": "The current time is 3:04 PM on Monday, August 03, 2026."}

    def tools_used(self):
        return [c["action"] for c in self.calls]


# ----------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------


def el(name, type_="ButtonControl", x=100, y=100, w=80, h=24):
    return {"name": name, "type": type_, "x": x, "y": y,
            "left": x - w // 2, "top": y - h // 2, "width": w, "height": h}


def snapshot(elements, title="Test Window", app="test.exe"):
    return ScreenSnapshot(
        window_title=title,
        window_app=app,
        elements=[ScreenElement.from_dict(e) for e in elements],
        frame_hash="deadbeef",
    )


@pytest.fixture
def instagram_elements():
    """A realistic Instagram DM list — the scenario the monolith's comments describe failing."""
    return [
        el("Instagram", "TextControl", 200, 40),
        el("Messages", "TabItemControl", 300, 80),
        el("Instagram Messages", "TabItemControl", 420, 80),
        el("Search", "EditControl", 300, 130),
        el("Alice Johnson", "ListItemControl", 300, 220),
        el("Alicia Jones", "ListItemControl", 300, 280),
        el("Bob Miller", "ListItemControl", 300, 340),
        el("Carol Diaz", "ListItemControl", 300, 400),
        el("Message...", "EditControl", 700, 800),
        el("Send", "ButtonControl", 900, 800),
    ]


@pytest.fixture
def memory(tmp_path):
    from memory_vault import MemoryVault

    return MemoryVault(db_path=str(tmp_path / "test_memory.db"))
