"""TRIAGE (intent routing) and ARCHITECT (planning)."""

import pytest

from agents.architect import ArchitectAgent
from agents.triage import TriageAgent
from conftest import FakeLLM


# ======================================================================
# TRIAGE
# ======================================================================


@pytest.fixture
def triage():
    return TriageAgent(FakeLLM())


@pytest.mark.parametrize("order,kind,action", [
    ("open chrome", "deterministic", "launch_app"),
    ("open notepad", "deterministic", "launch_app"),
    ("open youtube", "deterministic", "open_url"),
    ("open instagram", "deterministic", "open_social_inbox"),
    ("what time is it", "deterministic", "get_time_date"),
    ("search google for python tutorials", "deterministic", "search_google"),
    ("play lofi beats on youtube", "deterministic", "search_youtube"),
])
def test_deterministic_orders_cost_nothing(triage, order, kind, action):
    intent = triage.classify(order)
    assert intent.kind == kind
    assert intent.slots["action"] == action
    assert triage.llm.calls == 0


@pytest.mark.parametrize("order", ["hi", "hello there", "yo", "how are you", "thanks", "who are you"])
def test_small_talk_is_recognised_without_a_model_call(triage, order):
    assert triage.classify(order).kind == "chat"
    assert triage.llm.calls == 0


@pytest.mark.parametrize("order", [
    "what can you see", "what's on my screen", "describe my screen", "what do you see",
])
def test_screen_questions_are_routed_to_perception(triage, order):
    intent = triage.classify(order)
    assert intent.kind == "screen_query"
    assert triage.llm.calls == 0


def test_unknown_phrasing_falls_to_one_cheap_call_not_the_expensive_loop():
    """"fire up chrome" misses the monolith's `^(?:open|launch|start|run)` regex entirely and took
    the most expensive path available. Here an unmatched phrasing costs one 8B classification."""
    llm = FakeLLM(lambda p: '{"kind": "single_action", "confidence": 0.9, "reason": "one launch"}')
    triage = TriageAgent(llm)
    intent = triage.classify("would you mind bringing up my email client")

    assert intent.kind == "single_action"
    assert intent.source == "model"
    assert llm.calls == 1
    assert llm.payloads[0]["model"] == "llama-3.1-8b-instant"
    assert not llm.payloads[0].get("tools"), "classification never ships tool schemas"


def test_fire_up_chrome_still_resolves_deterministically(triage):
    """The alias table is consulted after a wider verb list than the monolith used."""
    intent = triage.classify("fire up chrome")
    assert intent.kind == "deterministic"
    assert intent.slots["action"] == "launch_app"
    assert triage.llm.calls == 0


def test_compound_orders_are_not_short_circuited(triage):
    """"open chrome and then search for cats" must not be answered by the launch fast path — that
    would silently drop half the order."""
    llm = FakeLLM(lambda p: '{"kind": "multi_step", "confidence": 0.9, "reason": "two actions"}')
    triage = TriageAgent(llm)
    intent = triage.classify("open chrome and then search for cats")
    assert intent.kind == "multi_step"


def test_switch_carries_a_launch_fallback(triage):
    intent = triage.classify("switch to spotify")
    assert intent.slots["action"] == "switch_window"
    assert intent.slots["fallback"]["action"] == "launch_app"


def test_find_the_send_button_is_not_a_web_search(triage):
    llm = FakeLLM(lambda p: '{"kind": "single_action", "confidence": 0.9, "reason": "screen click"}')
    triage = TriageAgent(llm)
    intent = triage.classify("find the send button")
    assert intent.kind != "deterministic"


def test_learned_rules_beat_everything(triage):
    hits = []

    def lookup(order):
        hits.append(order)
        return {"tool": "launch_app", "args": {"app_name": "spotify"}, "trigger": "put on music"}

    triage = TriageAgent(FakeLLM(), rule_lookup=lookup)
    intent = triage.classify("put on music")

    assert intent.source == "learned"
    assert intent.slots["action"] == "launch_app"
    assert triage.llm.calls == 0
    assert hits == ["put on music"]


def test_model_failure_falls_back_conservatively():
    llm = FakeLLM()
    llm.fail_with = "network down"
    triage = TriageAgent(llm)
    intent = triage.classify("do something complicated with the window")
    assert intent.source == "fallback"
    assert intent.kind in ("single_action", "multi_step")


def test_empty_order_asks_rather_than_acting(triage):
    assert triage.classify("   ").kind == "clarify"


def test_zero_cost_rate_is_reported(triage):
    for order in ["hi", "open chrome", "what time is it", "thanks"]:
        triage.classify(order)
    assert triage.stats()["zero_cost_rate"] == 1.0


# ======================================================================
# ARCHITECT
# ======================================================================


@pytest.fixture
def architect():
    return ArchitectAgent(FakeLLM())


def test_rule_plan_splits_on_real_connectives(architect):
    plan = architect.rule_plan("open instagram then find arundhati and then send her a message")
    assert len(plan.steps) == 3
    assert plan.steps[0].lane == "pathfinder"
    assert plan.source == "rule"


def test_budget_reflects_plan_shape_not_punctuation(architect):
    """The monolith sized its budget as `clauses * 2 + 2` from splitting on commas and the literal
    word "then". An order phrased any other way was under-budgeted."""
    one = architect.single_step_plan("click send")
    assert one.budget() == 4

    five = architect.rule_plan("a; b; c; d; e")
    assert five.budget() == 12


def test_model_plan_is_parsed_into_verifiable_steps():
    llm = FakeLLM(lambda p: """{"steps": [
        {"description": "Open the Instagram inbox", "target": "instagram", "lane": "pathfinder",
         "success_criteria": "the Instagram direct inbox is on screen"},
        {"description": "Open the conversation with Arundhati", "target": "Arundhati", "lane": "hands",
         "success_criteria": "the conversation with Arundhati is open"}
    ]}""")
    plan = ArchitectAgent(llm).plan("message arundhati on instagram")

    assert len(plan.steps) == 2
    assert plan.source == "model"
    assert plan.steps[1].target == "Arundhati"
    assert plan.steps[1].success_criteria
    assert llm.calls == 1


def test_planning_failure_degrades_to_rules_not_to_nothing():
    llm = FakeLLM()
    llm.fail_with = "rate limited"
    plan = ArchitectAgent(llm).plan("open chrome then close the tab")
    assert plan.source == "rule"
    assert len(plan.steps) == 2


def test_malformed_plan_json_degrades_to_rules():
    plan = ArchitectAgent(FakeLLM(lambda p: "I think you should open chrome!")).plan("open chrome")
    assert plan.source == "rule"
    assert len(plan.steps) == 1


def test_plan_completion_tracking(architect):
    plan = architect.rule_plan("open chrome then close the tab")
    assert not plan.complete
    assert plan.current().description == "open chrome"

    plan.steps[0].done = True
    assert plan.current().description == "close the tab"

    plan.steps[1].done = True
    assert plan.complete
    assert plan.current() is None
