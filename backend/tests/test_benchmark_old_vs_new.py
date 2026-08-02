"""Head-to-head grounding benchmark: the old monolith vs ANCHOR.

WHAT THIS MEASURES
    Target-grounding accuracy on a fixed set of (order, screen) cases: given an order and the
    on-screen controls, does the system click the thing the order actually named?

    It also measures a second, separate property: when the wrong thing IS clicked, does the system
    *notice* — or does it report success anyway? That is the difference between a task that fails
    visibly and one that fails silently, and it is the failure mode this project's own source
    comments describe repeatedly.

WHAT THIS DOES NOT MEASURE
    Live end-to-end task success against real applications. That needs a real desktop, real network
    latency and real UI timing, and no synthetic benchmark can stand in for it. These numbers are
    about the grounding subsystem only.

FAIRNESS
    The comparison is deliberately rigged in the OLD code's favour:
      * OLD is driven by an ORACLE model that always picks the best option it can see, so every
        failure counted against it is caused by the code, never by a weak model.
      * NEW runs with allow_model=False — pure string scoring, no model calls at all, no vision.
    If anything, this understates the gap.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.anchor import AnchorAgent, CONFIDENT
from agents.base import ScreenElement, rank_elements
from conftest import FakeLLM, el, snapshot


# ======================================================================
# Benchmark cases
# ======================================================================
# Each: (label, order, elements, expected_name or None meaning "must refuse to click")

INBOX = [
    el("Instagram", "TextControl", 200, 40),
    el("Messages", "TabItemControl", 300, 80),
    el("Instagram Messages", "TabItemControl", 420, 80),
    el("Search", "EditControl", 300, 130),
    el("Alice Johnson", "ListItemControl", 300, 220),
    el("Alicia Jones", "ListItemControl", 300, 280),
    el("Bob Miller", "ListItemControl", 300, 340),
    el("Message...", "EditControl", 700, 800),
    el("Send", "ButtonControl", 900, 800),
]

LONG_LIST = [el(f"Conversation {i}", "ListItemControl", 300, 100 + i * 30) for i in range(30)] + [
    el("Alice Johnson", "ListItemControl", 300, 1000),
]

VSCODE = [
    el("jarvis - Visual Studio Code", "WindowControl", 960, 20, w=1900, h=40),
    el("Explorer", "ButtonControl", 30, 100),
    el("Search", "ButtonControl", 30, 140),
    el("main.py", "ListItemControl", 150, 200),
    el("ollama_engine.py", "ListItemControl", 150, 230),
    el("Terminal", "TabItemControl", 400, 900),
]

CASES = [
    # --- the documented failure: a named person vs a generic tab -------------------
    ("named person over generic tab", "open the chat of Alice", INBOX, "Alice Johnson"),
    ("same, lowercase (voice input)", "open the chat of alice", INBOX, "Alice Johnson"),
    ("possessive form", "open Alice's chat", INBOX, "Alice Johnson"),

    # --- simple cases the old code should also get right ---------------------------
    ("exact button name", "click Send", INBOX, "Send"),
    ("button with type noun", "click the Send button", INBOX, "Send"),
    ("exact file name", "open main.py", VSCODE, "main.py"),

    # --- truncation: target beyond the 25-element observation window ---------------
    ("target below the fold", "open the chat of Alice", LONG_LIST, "Alice Johnson"),

    # --- wake-word contamination (this repo's folder is named "jarvis") ------------
    ("wake word must not match window", "jarvis open main.py", VSCODE, "main.py"),

    # --- ordinal reference ---------------------------------------------------------
    ("ordinal reference", "open the 2nd conversation", LONG_LIST, "Conversation 1"),

    # --- content vs recipient ------------------------------------------------------
    ("quoted content is not the target", "send 'Alicia Jones' to Bob", INBOX, "Bob Miller"),

    # --- must refuse rather than blind-click --------------------------------------
    ("absent target must refuse", "click the Publish button", VSCODE, None),
    ("absent person must refuse", "open the chat of Zara", INBOX, None),

    # --- prefix / partial names ----------------------------------------------------
    ("partial first name", "open the chat of Bob", INBOX, "Bob Miller"),
    ("target named mid-sentence", "can you open main.py for me", VSCODE, "main.py"),

    # --- interrogative phrasing must not become a click ---------------------------
    ("question is not a command", "is Send visible on screen", INBOX, None),

    # --- type-noun disambiguation --------------------------------------------------
    ("tab vs button of same name", "click the Search button", VSCODE, "Search"),

    # --- below-the-fold variants ---------------------------------------------------
    ("below fold, exact name", "click Alice Johnson", LONG_LIST, "Alice Johnson"),
    ("below fold, partial name", "open Alice", LONG_LIST, "Alice Johnson"),

    # --- polite / indirect phrasing ------------------------------------------------
    ("polite phrasing", "could you please open ollama_engine.py", VSCODE, "ollama_engine.py"),
    ("filler words", "just click Send please", INBOX, "Send"),
]


# ======================================================================
# OLD pipeline, reconstructed faithfully and driven by an oracle model
# ======================================================================


class _NoClickOS:
    """Captures clicks instead of performing them."""

    last = None

    @classmethod
    def click(cls, x, y, button="left", clicks=1):
        cls.last = (x, y)
        return {"status": "success", "action": "click", "x": x, "y": y}


class _StubVision:
    def __init__(self, elements):
        self.elements = elements

    def find_visible_ui_elements(self, max_depth=8, max_elements=400):
        return list(self.elements)

    def get_element_at_cursor(self):
        return {"name": "Target Item", "type": "Control", "x": 0, "y": 0}

    def locate_via_vision(self, text):
        return None  # the old vision fallback; unavailable offline, same as NEW gets here

    def ask_vision(self, q, **kw):
        return None


def _old_observation_window(elements):
    """Replicates `_build_observation`: rank interactive first, then reading order, hard-cut at 25.

    This is what the old model could actually see. An element outside this window was invisible to
    it, so no amount of prompting could have made it click the right thing.
    """
    priority = ("Button", "ListItem", "TabItem", "MenuItem", "Edit", "Hyperlink", "CheckBox",
                "ComboBox", "RadioButton")

    def rank(e):
        return 0 if any(t in e["type"] for t in priority) else 1

    return sorted(elements, key=lambda e: (rank(e), e["top"], e["left"]))[:25]


# The old engine's own target-cleanup regex, lifted verbatim from subsecond_element_locator. This
# is the text the old system derived from an order, with no help from anything newer.
_OLD_CLEANUP = re.compile(
    r"^(the|a|an)\s*(chat of|dm of|message of|chat with|chat|message|dm)?\s*", re.IGNORECASE
)
_OLD_VERB = re.compile(r"^(?:open|click(?: on)?|target|select|show|go to)\s+(.+)$", re.IGNORECASE)


def _pick_realistic(order, expected, visible):
    """What the old model actually passed to click_element(text=...).

    Uses the old engine's own text handling — strip the action verb, strip the relational prefix —
    and nothing else. This is the configuration that reflects how the system behaved in practice.
    """
    body = re.sub(r"\bjarvis\b", " ", order, flags=re.IGNORECASE).strip()
    match = _OLD_VERB.match(body)
    if match:
        body = match.group(1)
    text = _OLD_CLEANUP.sub("", body).strip(" ?.!,")
    return text


def _pick_oracle(order, expected, visible):
    """An upper bound: a model that always names the correct element when it can see it.

    Any failure surviving this configuration is caused by the old *code* — truncation hiding the
    element, or the matching logic mishandling it — never by a weak model.
    """
    names = [e["name"] for e in visible]
    if expected and expected in names:
        return expected
    return _pick_realistic(order, expected, visible)


def _old_resolve(order, elements, expected, pick=_pick_realistic):
    """Runs the old grounding pipeline end to end. Returns (clicked_name, flagged_wrong)."""
    os.environ["JARVIS_LEGACY_ENGINE"] = "1"
    import tempfile

    import ollama_engine
    from memory_vault import MemoryVault

    real_os = ollama_engine.OSAutomation
    ollama_engine.OSAutomation = _NoClickOS
    # A file, not ":memory:" — MemoryVault opens a fresh connection per operation, and an in-memory
    # database is scoped to a single connection, so the schema would vanish between calls.
    db = os.path.join(tempfile.mkdtemp(), "bench.db")
    try:
        engine = ollama_engine.OllamaEngine(
            memory_vault=MemoryVault(db_path=db), screen_vision=_StubVision(elements)
        )

        # Path A: the sub-second fast path (no LLM).
        _NoClickOS.last = None
        matched, output, _ = engine.subsecond_element_locator(order)
        clicked = None
        if matched and _NoClickOS.last:
            for e in elements:
                if (e["x"], e["y"]) == _NoClickOS.last:
                    clicked = e["name"]
                    break

        # Path B: the ReAct loop's click_element.
        if clicked is None:
            visible = _old_observation_window(elements)
            args = {}
            ordinal = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", order) or \
                re.search(r"\b(first|second|third|fourth|fifth)\b", order, re.IGNORECASE)
            if ordinal:
                # The old click_element did support control_type + ordinal, so a fair run gives it
                # that path rather than forcing a name match it was never meant to do.
                words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
                raw = ordinal.group(1).lower()
                args = {"control_type": "ListItem", "ordinal": words.get(raw, 0) or int(raw)}
            else:
                text = pick(order, expected, visible)
                if text:
                    args = {"text": text}
            if args:
                result = engine._tool_click_element(args)
                clicked = result.get("matched_name")

        # The old scope guard: _extract_named_target + a plain substring test.
        flagged = False
        if clicked:
            named = engine._extract_named_target(order)
            if named and named.lower() not in clicked.lower():
                flagged = True
        return clicked, flagged
    finally:
        ollama_engine.OSAutomation = real_os
        os.environ.pop("JARVIS_LEGACY_ENGINE", None)


def _new_resolve(order, elements, expected):
    """ANCHOR with no model and no vision — the hardest configuration for the new code."""
    anchor = AnchorAgent(FakeLLM())
    snap = snapshot(elements)
    target = anchor.ground_order(order, snap, allow_model=False)
    if not target.resolved or target.confidence < 0.42:
        return None, False
    referent = anchor.extract_referent(order)
    on_target, _ = anchor.verify_on_target(
        referent, {"status": "success", "matched_name": target.element.name}
    )
    return target.element.name, (not on_target)


# ======================================================================
# The benchmark
# ======================================================================


def _score(resolver):
    correct = caught = silent = 0
    rows = []
    for label, order, elements, expected in CASES:
        clicked, flagged = resolver(order, elements, expected)
        if expected is None:
            ok = clicked is None or flagged
            verdict = "correct (refused)" if ok else f"WRONG (clicked {clicked!r})"
        else:
            ok = clicked == expected
            verdict = "correct" if ok else f"WRONG (clicked {clicked!r})"
        if ok:
            correct += 1
        elif flagged:
            caught += 1
            verdict += " [flagged]"
        else:
            silent += 1
            verdict += " [reported as success]"
        rows.append((label, verdict))
    return correct, caught, silent, rows


def test_benchmark_and_report(capsys):
    old_real = _score(lambda o, e, x: _old_resolve(o, e, x, pick=_pick_realistic))
    old_best = _score(lambda o, e, x: _old_resolve(o, e, x, pick=_pick_oracle))
    new = _score(_new_resolve)
    total = len(CASES)

    with capsys.disabled():
        print(f"\n{'=' * 92}\nGROUNDING BENCHMARK - {total} cases")
        print("OLD-realistic : old code, old text handling (how it actually behaved)")
        print("OLD-bestcase  : old code, but a model that always names the right element if visible")
        print("NEW           : ANCHOR with allow_model=False - no model calls, no vision")
        print("=" * 92)
        print(f"{'case':<34} {'OLD-realistic':<19} {'OLD-bestcase':<19} {'NEW':<19}")
        print("-" * 92)
        for (label, r_v), (_, b_v), (_, n_v) in zip(old_real[3], old_best[3], new[3]):
            print(f"{label:<34} {r_v[:18]:<19} {b_v[:18]:<19} {n_v[:18]:<19}")
        print("-" * 92)

        def pct(s):
            return f"{s[0]}/{total} ({s[0] / total:.0%})"

        print(f"{'CORRECT TARGET':<34} {pct(old_real):<19} {pct(old_best):<19} {pct(new):<19}")
        print(f"{'wrong, silently reported OK':<34} {old_real[2]:<19} {old_best[2]:<19} {new[2]:<19}")
        print("=" * 92)

    assert new[0] > old_real[0], "the new pipeline must ground more cases correctly"
    assert new[0] >= old_best[0], "it must also match or beat the old code's best case"
    assert new[2] <= old_real[2], "and must not fail silently more often"


@pytest.mark.parametrize("label,order,elements,expected", CASES, ids=[c[0] for c in CASES])
def test_new_pipeline_grounds_each_case(label, order, elements, expected):
    """Per-case assertions, so a regression names the exact scenario it broke."""
    clicked, flagged = _new_resolve(order, elements, expected)
    if expected is None:
        assert clicked is None or flagged, f"{label}: must refuse rather than blind-click"
    else:
        assert clicked == expected, f"{label}: expected {expected!r}, got {clicked!r}"
