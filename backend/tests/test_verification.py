"""SENTINEL (verification) and NARRATOR (response composition)."""

import pytest

from agents.base import Plan, PlanStep
from agents.narrator import NarratorAgent, speechify
from agents.sentinel import SentinelAgent
from conftest import FakeLLM, el, snapshot


# ======================================================================
# SENTINEL — the ladder
# ======================================================================


@pytest.fixture
def sentinel():
    return SentinelAgent(FakeLLM())


def test_rung1_tool_failure_needs_no_model_call(sentinel):
    verdict = sentinel.verify(
        PlanStep(description="click Send"),
        {"status": "error", "tool": "click_coordinate", "message": "out of bounds"},
        snapshot([]), snapshot([]),
    )
    assert verdict.done is False
    assert verdict.method == "tool"
    assert sentinel.llm.calls == 0


def test_rung2_off_target_is_not_done(sentinel):
    """A successful click on the wrong element is not progress."""
    verdict = sentinel.verify(
        PlanStep(description="open Arundhati's chat", target="Arundhati"),
        {"status": "success", "tool": "click_coordinate", "on_target": False,
         "scope_reason": "landed on 'Instagram Messages'"},
        snapshot([]), snapshot([]),
    )
    assert verdict.done is False
    assert "Instagram Messages" in verdict.evidence
    assert sentinel.llm.calls == 0


def test_rung3_success_criteria_newly_visible_means_done(sentinel):
    before = snapshot([el("Inbox", "ListItemControl")], title="Instagram")
    after = snapshot([el("Arundhati Sharma", "ListItemControl"), el("Message...", "EditControl")],
                     title="Instagram")
    verdict = sentinel.verify(
        PlanStep(description="open the chat with Arundhati", target="Arundhati"),
        {"status": "success", "tool": "click_coordinate", "matched_name": "Arundhati Sharma"},
        before, after,
    )
    assert verdict.done is True
    assert verdict.method == "screen"
    assert sentinel.llm.calls == 0


def test_already_present_text_is_not_evidence(sentinel):
    """"Instagram" is in the window title the whole time, so its presence proves nothing about
    whether this step just succeeded. Only a change is evidence."""
    before = snapshot([el("Instagram", "TextControl")], title="Instagram")
    after = snapshot([el("Instagram", "TextControl")], title="Instagram")
    verdict = sentinel.verify(
        PlanStep(description="open instagram", target="Instagram"),
        {"status": "success", "tool": "click_coordinate"},
        before, after, allow_model=False,
    )
    assert verdict.done is False


def test_rung4_no_screen_change_means_the_click_missed(sentinel):
    identical = [el("Send", "ButtonControl")]
    verdict = sentinel.verify(
        PlanStep(description="click Send", target="Send"),
        {"status": "success", "tool": "click_coordinate", "matched_name": "Send"},
        snapshot(identical), snapshot(identical), allow_model=False,
    )
    assert verdict.done is False
    assert "identical" in verdict.evidence


def test_navigation_lands_when_the_window_title_matches(sentinel):
    verdict = sentinel.verify(
        PlanStep(description="open chrome", target="chrome", lane="pathfinder"),
        {"status": "success", "tool": "launch_app", "args": {"app_name": "chrome"}},
        snapshot([], title="Desktop"), snapshot([], title="Google Chrome"),
    )
    assert verdict.done is True
    assert sentinel.llm.calls == 0


def test_self_evident_tools_are_not_second_guessed(sentinel):
    verdict = sentinel.verify(
        PlanStep(description="scroll down"),
        {"status": "success", "tool": "scroll"},
        snapshot([]), snapshot([]),
    )
    assert verdict.done is True
    assert sentinel.llm.calls == 0


def test_grounded_click_with_visible_effect_is_done(sentinel):
    before = snapshot([el("A", "ButtonControl")])
    after = snapshot([el("A", "ButtonControl"), el("B", "ButtonControl")])
    verdict = sentinel.verify(
        PlanStep(description="click A"),
        {"status": "success", "tool": "click_coordinate", "on_target": True,
         "grounding": {"element": {"name": "A"}, "confidence": 0.9, "method": "exact"}},
        before, after,
    )
    assert verdict.done is True
    assert sentinel.llm.calls == 0


def test_only_ambiguous_cases_reach_the_model():
    llm = FakeLLM(lambda p: '{"done": true, "evidence": "the compose box is focused", "confidence": 0.8}')
    sentinel = SentinelAgent(llm)
    before = snapshot([el("A", "ButtonControl")])
    after = snapshot([el("A", "ButtonControl"), el("C", "ButtonControl")])
    verdict = sentinel.verify(
        PlanStep(description="something hard to judge"),
        {"status": "success", "tool": "type_text", "args": {"text": "hi"}},
        before, after,
    )
    assert verdict.done is True
    assert verdict.method == "model"
    assert llm.calls == 1
    assert llm.payloads[0]["model"] == "llama-3.1-8b-instant", "verification uses the cheap model"


def test_verification_failure_fails_closed():
    """Failing open would let an unverified step count as done. Failing closed costs one retry."""
    llm = FakeLLM()
    llm.fail_with = "network down"
    sentinel = SentinelAgent(llm)
    verdict = sentinel.verify(
        PlanStep(description="ambiguous"),
        {"status": "success", "tool": "type_text"},
        snapshot([el("A")]), snapshot([el("A"), el("B")]),
    )
    assert verdict.done is False


def test_free_rate_is_reported(sentinel):
    for _ in range(4):
        sentinel.verify(PlanStep(description="scroll"), {"status": "success", "tool": "scroll"},
                        snapshot([]), snapshot([]))
    assert sentinel.stats()["free_rate"] == 1.0


# ======================================================================
# NARRATOR
# ======================================================================


@pytest.fixture
def narrator():
    return NarratorAgent(FakeLLM(), address="Boss")


@pytest.mark.parametrize("raw,forbidden", [
    ("## Heading\n- bullet one\n- bullet two", "#"),
    ("**bold** and _italic_", "*"),
    ("Here: `code`", "`"),
    ("1. first\n2. second", "1."),
])
def test_speechify_strips_what_tts_cannot_read(raw, forbidden):
    """gemma3 defaults to markdown reports; read aloud, those are noise. The existing ask_vision
    docstring documents exactly this."""
    assert forbidden not in speechify(raw)


def test_speechify_caps_length():
    long = " ".join(f"Sentence number {i}." for i in range(20))
    assert speechify(long).count(".") <= 3


def test_speechify_replaces_urls():
    assert "https://" not in speechify("Go to https://example.com/very/long/path now")


@pytest.mark.parametrize("tool,args,result,expected", [
    ("launch_app", {"app_name": "chrome"}, {}, "Opened Chrome"),
    ("click_coordinate", {}, {"matched_name": "Send"}, "Clicked 'Send'"),
    ("type_text", {"text": "hello"}, {}, 'Typed "hello"'),
    ("open_social_inbox", {"platform": "instagram"}, {}, "Opened your Instagram inbox"),
    ("scroll", {"direction": "down"}, {}, "Scrolled down"),
])
def test_action_lines_are_templated_and_free(narrator, tool, args, result, expected):
    line = narrator.action_line(tool, args, result)
    assert expected in line
    assert narrator.llm.calls == 0


def test_informational_answer_is_the_reply_not_buried_under_done(narrator):
    """The monolith's one-sentence rule meant a screen question got answered with "Done, Boss"."""
    plan = Plan(goal="what's on screen", steps=[PlanStep(description="look", done=True)])
    results = [{"status": "success", "tool": "ask_vision", "answer": "A code editor with three files open."}]
    response = narrator.compose(plan, results)
    assert "code editor" in response.spoken
    assert "Done" not in response.spoken


def test_partial_completion_is_reported_honestly(narrator):
    plan = Plan(goal="open instagram then message arundhati", steps=[
        PlanStep(description="open instagram", done=True),
        PlanStep(description="message arundhati", done=False, evidence="I couldn't find her in the list"),
    ])
    response = narrator.compose(plan, [{"status": "success", "tool": "open_social_inbox"}])

    assert "1 of 2" in response.spoken
    assert "message arundhati" in response.spoken
    assert "couldn't find her" in response.spoken


def test_detail_report_shows_plan_grounding_and_evidence(narrator):
    plan = Plan(goal="open the chat of Arundhati", steps=[
        PlanStep(description="open the chat of Arundhati", target="Arundhati", done=True,
                 evidence="'Arundhati Sharma' is now visible"),
    ])
    results = [{"tool": "click_coordinate", "status": "success", "matched_name": "Arundhati Sharma",
                "grounding": {"method": "exact", "confidence": 0.95}}]
    detail = narrator.detail_report(plan, results, elapsed=1.8,
                                    stats={"llm_calls": 2, "tree_walks": 2, "walks_avoided": 3})

    assert "[done]" in detail
    assert "Arundhati Sharma" in detail
    assert "grounded via exact" in detail
    assert "2 model calls" in detail
    assert "3 avoided by cache" in detail


def test_detail_report_names_off_target_actions(narrator):
    plan = Plan(goal="x", steps=[PlanStep(description="x", done=False)])
    detail = narrator.detail_report(plan, [{"tool": "click_coordinate", "status": "success",
                                            "matched_name": "Wrong Thing", "on_target": False}])
    assert "OFF-TARGET" in detail
    assert "[not done]" in detail


def test_short_evidence_needs_no_second_model_pass(narrator):
    answer = narrator.answer_question("what's on screen", "A browser showing a news site.")
    assert "news site" in answer
    assert narrator.llm.calls == 0


def test_long_markdown_evidence_is_rewritten_for_speech():
    llm = FakeLLM(lambda p: "You've got a code editor open with a few Python files.")
    narrator = NarratorAgent(llm)
    messy = "## Screen Contents\n\n- **Editor**: VS Code\n- **Files**: three\n" + "x" * 250
    answer = narrator.answer_question("what's on screen", messy)

    assert llm.calls == 1
    assert "#" not in answer and "*" not in answer
