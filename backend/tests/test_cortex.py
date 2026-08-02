"""CORTEX — end-to-end runs through the whole agent pipeline.

Nothing here touches Windows, a microphone, Groq or Ollama. The efficiency assertions are the
point of the refactor, so they are written as hard bounds rather than observations.
"""

import pytest

from agents.cortex import Cortex
from conftest import FakeLLM, FakeOS, FakeTelemetry, FakeVision, el, route, tool_call


INBOX = [
    el("Instagram", "TextControl", 200, 40),
    el("Instagram Messages", "TabItemControl", 420, 80),
    el("Alice Johnson", "ListItemControl", 300, 220),
    el("Bob Miller", "ListItemControl", 300, 340),
]

OPEN_CHAT = [
    el("Alice Johnson", "TextControl", 500, 40),
    el("Message...", "EditControl", 700, 800),
    el("Send", "ButtonControl", 900, 800),
]


def build(handler=None, elements=None, title="Desktop", memory=None):
    """Assembles a cortex whose screen reacts to actions."""
    vision = FakeVision(elements if elements is not None else [])
    telemetry = FakeTelemetry(title)
    os_api = FakeOS()

    def react(action, kw):
        # Model the world responding: navigation changes the window, a click opens the chat.
        if action == "open_social_inbox":
            telemetry.title = "Instagram Direct"
            vision.change_screen(elements=INBOX)
        elif action == "launch_app":
            telemetry.title = f"{kw.get('app', 'App')} Window"
            vision.change_screen(elements=[el("New App", "ButtonControl")])
        elif action == "click":
            telemetry.title = "Instagram Direct - Alice Johnson"
            vision.change_screen(elements=OPEN_CHAT)
        elif action == "type_text":
            vision.change_screen(elements=OPEN_CHAT + [el(kw.get("text", "sent"), "TextControl", 600, 700)])
        else:
            vision.change_screen()

    os_api.on_action = react
    llm = FakeLLM(handler)
    return Cortex(memory=memory, vision=vision, telemetry=telemetry, os_api=os_api, llm=llm), llm, os_api


def collect(cortex, order):
    events = list(cortex.run(order))
    return events, [e for e in events if e["type"] == "response"]


# ======================================================================
# Event contract — the UIs must keep working unchanged
# ======================================================================


def test_event_types_stay_within_the_existing_contract():
    """main.py, jarvis_desktop.py and the React dashboard already handle these. "detail" is new and
    additive — consumers that don't know it ignore it."""
    cortex, _, _ = build()
    events, _ = collect(cortex, "open chrome")
    known = {"status", "thought", "tool_exec", "response", "detail"}
    assert {e["type"] for e in events} <= known


def test_every_run_ends_in_exactly_one_response():
    cortex, _, _ = build()
    for order in ["hi", "open chrome", "what time is it"]:
        _, responses = collect(cortex, order)
        assert len(responses) == 1
        assert responses[0]["text"]


def test_tool_exec_events_keep_their_shape():
    cortex, _, _ = build()
    events, _ = collect(cortex, "open chrome")
    execs = [e for e in events if e["type"] == "tool_exec"]
    assert execs
    for event in execs:
        assert "tool" in event and "input" in event and "output" in event


# ======================================================================
# Lane routing and its cost
# ======================================================================


def test_deterministic_order_costs_zero_model_calls():
    """The monolith ran "open chrome" through the full loop: one 70B call with 21 tool schemas,
    then a second 70B call to ask whether it was done."""
    cortex, llm, os_api = build()
    _, responses = collect(cortex, "open chrome")

    assert llm.calls == 0
    assert "launch_app" in os_api.tools_used()
    assert "Chrome" in responses[0]["text"]


def test_time_query_costs_zero_model_calls():
    cortex, llm, _ = build()
    _, responses = collect(cortex, "what time is it")
    assert llm.calls == 0
    assert "3:04 PM" in responses[0]["text"]


def test_small_talk_costs_one_cheap_call():
    cortex, llm, os_api = build(lambda p: "All good here, Boss.")
    _, responses = collect(cortex, "hey how are you")

    assert llm.calls == 1
    assert llm.payloads[0]["model"] == "llama-3.1-8b-instant"
    assert os_api.calls == [], "small talk must never touch the screen"
    assert "All good" in responses[0]["text"]


def test_screen_question_answers_instead_of_clicking():
    """Asked "can you see my screen", the monolith clicked a real Maximize button and reported
    "Done, Boss — clicked Maximize"."""
    cortex, llm, os_api = build(elements=[el("Maximize", "ButtonControl", 1800, 20)])
    cortex.retina.vision.vision_answer = "You're looking at a code editor with three Python files."
    _, responses = collect(cortex, "what can you see")

    assert os_api.calls == [], "a question must not produce an action"
    assert "code editor" in responses[0]["text"]


def test_screen_question_reply_is_the_answer_not_done():
    cortex, _, _ = build()
    cortex.retina.vision.vision_answer = "A browser is open on a news site."
    _, responses = collect(cortex, "what's on my screen")
    assert "news site" in responses[0]["text"]
    assert responses[0]["text"] != "Done, Boss."


# ======================================================================
# The headline case: a multi-step order, grounded and verified
# ======================================================================


PLAN_JSON = """{"steps": [
  {"description": "Open the Instagram inbox", "target": "instagram", "lane": "pathfinder",
   "success_criteria": "the Instagram direct inbox is on screen"},
  {"description": "Open the conversation with Alice", "target": "Alice", "lane": "hands",
   "success_criteria": "the conversation with Alice is open"}
]}"""


def test_multi_step_order_completes_and_stays_on_target():
    cortex, llm, os_api = build(route(
        triage='{"kind": "multi_step", "confidence": 0.95, "reason": "two actions"}',
        architect=PLAN_JSON,
    ))
    events, responses = collect(cortex, "open instagram and open the chat of alice")

    assert "open_social_inbox" in os_api.tools_used()
    click = [c for c in os_api.calls if c["action"] == "click"]
    assert click, "the second step must actually click something"
    assert click[0]["x"] == 300 and click[0]["y"] == 220, "it must click Alice's row, not the tab"

    assert cortex.last_run["llm_calls"] <= 3, "triage + plan, and nothing per-step"
    assert "All done" in responses[0]["text"] or "done" in responses[0]["text"].lower()


def test_multi_step_run_costs_far_fewer_calls_than_the_monolith():
    """Monolith arithmetic for this order: 2 ReAct steps x 1 call (21 schemas, 70B), plus 1
    completion-check call per terminal action, plus a final speak_final call = ~5-6 70B calls.
    Here: 1 cheap classification + 1 planning call, and every step resolves deterministically."""
    cortex, llm, _ = build(route(
        triage='{"kind": "multi_step", "confidence": 0.95, "reason": "two actions"}',
        architect=PLAN_JSON,
    ))
    collect(cortex, "open instagram and open the chat of alice")

    assert llm.calls <= 3
    assert llm.tool_schema_count() == 0, "no step needed a tool-calling round-trip at all"
    assert llm.calls_by_agent.get("SENTINEL", 0) == 0, "screen evidence answered every check"


def test_off_target_click_does_not_complete_the_step():
    """Told to open a specific person's chat with only a generic tab on screen, the run must not
    report success."""
    cortex, llm, os_api = build(
        route(triage='{"kind": "single_action", "confidence": 0.9, "reason": "one click"}',
              hands=tool_call("click_element", text="Instagram Messages")),
        elements=[el("Instagram Messages", "TabItemControl", 420, 80)],
    )
    _, responses = collect(cortex, "open the chat of Alice")

    text = responses[0]["text"].lower()
    assert "all done" not in text
    assert any(w in text for w in ("couldn't", "could not", "partly", "stopped"))


def test_detail_event_carries_the_full_breakdown():
    cortex, _, _ = build(route(
        triage='{"kind": "multi_step", "confidence": 0.95, "reason": "two"}',
        architect=PLAN_JSON,
    ))
    events, _ = collect(cortex, "open instagram and open the chat of alice")

    details = [e for e in events if e["type"] == "detail"]
    assert len(details) == 1
    text = details[0]["text"]
    assert "Plan (" in text
    assert "Alice" in text
    assert "model call" in text
    assert details[0]["data"]["steps_total"] == 2


def test_spoken_reply_is_never_markdown():
    cortex, _, _ = build(route(
        triage='{"kind": "multi_step", "confidence": 0.9, "reason": "two"}',
        architect=PLAN_JSON,
    ))
    _, responses = collect(cortex, "open instagram and open the chat of alice")
    spoken = responses[0]["text"]
    assert not any(ch in spoken for ch in "*#`_|")
    assert "\n" not in spoken


# ======================================================================
# Safety invariants carried over from the monolith
# ======================================================================


def test_cancellation_stops_before_the_next_step():
    cortex, _, os_api = build(route(
        triage='{"kind": "multi_step", "confidence": 0.9, "reason": "many"}',
        architect='{"steps": [' + ",".join(
            f'{{"description": "step {i}", "target": "thing{i}", "lane": "hands"}}' for i in range(6)
        ) + "]}",
    ))
    events = []
    for event in cortex.run("do six things"):
        events.append(event)
        if len(events) > 4:
            cortex.cancel()

    assert any("Stopped" in e.get("text", "") for e in events if e["type"] == "response")


def test_repeated_tool_use_is_capped():
    """A hard cap is what actually stops a confused loop from repeating a real side effect."""
    cortex, _, os_api = build(route(
        triage='{"kind": "multi_step", "confidence": 0.9, "reason": "many"}',
        architect='{"steps": [' + ",".join(
            f'{{"description": "scroll down", "lane": "hands"}}' for _ in range(8)
        ) + "]}",
    ))
    collect(cortex, "scroll a lot")
    assert os_api.tools_used().count("scroll") <= 3


def test_a_crash_still_produces_a_spoken_reply():
    cortex, _, _ = build()

    def explode(*a, **kw):
        raise RuntimeError("boom")

    cortex.triage.classify = explode
    _, responses = collect(cortex, "open chrome")
    assert len(responses) == 1
    assert "went wrong" in responses[0]["text"]


def test_model_outage_degrades_instead_of_failing():
    cortex, llm, os_api = build()
    llm.fail_with = "Groq is unreachable"
    _, responses = collect(cortex, "open chrome then close the tab")

    assert len(responses) == 1
    assert os_api.calls, "rule-based planning and deterministic execution still work offline"


# ======================================================================
# Cross-agent behaviour
# ======================================================================


def test_history_is_read_not_just_written():
    """The monolith appended to self.history in seven places and read it in zero, so follow-ups
    had no context at all."""
    cortex, _, os_api = build(
        route(triage='{"kind": "single_action", "confidence": 0.9, "reason": "click"}'),
        elements=INBOX,
    )
    events, _ = collect(cortex, "open the chat of Alice")
    assert cortex.history
    first = [e for e in events if e["type"] == "tool_exec"][0]
    assert first["output"]["matched_name"] == "Alice Johnson"

    # The screen has legitimately moved on by now, so the follow-up must re-resolve the same
    # *referent* against the current screen rather than replay a stale coordinate.
    os_api.calls.clear()
    events, _ = collect(cortex, "do it again")
    execs = [e for e in events if e["type"] == "tool_exec"]
    assert execs, "an anaphoric follow-up must still act"
    assert execs[0]["output"].get("matched_name") == "Alice Johnson"


def test_a_repeated_order_becomes_free(memory):
    """SCHOLAR's whole purpose: the third identical order costs nothing."""
    from agents.scholar import PROMOTION_THRESHOLD

    cortex, llm, _ = build(memory=memory)
    for _ in range(PROMOTION_THRESHOLD + 1):
        collect(cortex, "open chrome")

    cortex.scholar.invalidate_cache()
    assert cortex.scholar.lookup("open chrome") is not None
    assert llm.calls == 0


def test_screen_is_not_re_walked_on_every_perception_request():
    """The monolith rebuilt its observation — a full UI Automation tree walk — on every step. Here
    a run asks RETINA for the screen repeatedly but only walks when it actually changed."""
    cortex, _, _ = build(route(
        triage='{"kind": "multi_step", "confidence": 0.95, "reason": "two"}',
        architect=PLAN_JSON,
    ), elements=INBOX)
    collect(cortex, "open instagram and open the chat of alice")

    stats = cortex.retina.stats()
    assert stats["snapshot_requests"] > stats["tree_walks"]
    assert stats["walks_avoided"] >= 1


def test_agent_stats_expose_every_agent():
    cortex, _, _ = build()
    collect(cortex, "open chrome")
    stats = cortex.agent_stats()
    for agent in ["TRIAGE", "RETINA", "ANCHOR", "ARCHITECT", "PATHFINDER", "HANDS",
                  "SENTINEL", "NARRATOR", "SCHOLAR", "EARS"]:
        assert agent in stats
