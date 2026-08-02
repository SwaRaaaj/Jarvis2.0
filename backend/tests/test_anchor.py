"""ANCHOR — screen-order grounding.

These tests encode the specific failures the monolith's own comments document, so a regression
would have to break a named, explained case rather than a vague assertion.
"""

import pytest

from agents.anchor import AnchorAgent, CONFIDENT
from agents.base import ScreenElement
from conftest import FakeLLM, el, snapshot, tool_call


@pytest.fixture
def anchor():
    return AnchorAgent(FakeLLM())


# ======================================================================
# 1. What did the order name?
# ======================================================================


@pytest.mark.parametrize("order,expected_text,expected_type,expected_ordinal", [
    ("open the chat of Arundhati", "arundhati", "ListItem", None),
    ("click the Send button", "send", "Button", None),
    ("play the 3rd video", "", "ListItem", 3),
    ("open instagram", "instagram", "", None),
    ("switch to chrome", "chrome", "", None),
    ("click the second tab", "", "TabItem", 2),
    ("open Bob's chat", "bob", "ListItem", None),
    ("show me the settings menu", "settings", "MenuItem", None),
    ("open the last email", "email", "", -1),
    ("click on the search bar", "", "Edit", None),
])
def test_extract_referent(anchor, order, expected_text, expected_type, expected_ordinal):
    ref = anchor.extract_referent(order)
    assert ref.text.lower() == expected_text
    assert ref.type_hint == expected_type
    assert ref.ordinal == expected_ordinal


def test_wake_word_is_never_a_target(anchor):
    """This repo's own folder is named "jarvis", so every VS Code window title here contains it.
    Leaving the wake word in the referent made "...Jarvis?" fuzzy-match that window and click it —
    an observed bug, not a hypothetical."""
    ref = anchor.extract_referent("JARVIS open notepad")
    assert "jarvis" not in ref.text.lower()
    assert ref.text.lower() == "notepad"

    snap = snapshot([el("jarvis - Visual Studio Code", "WindowControl", 500, 300)])
    target = anchor.ground(anchor.extract_referent("hey jarvis what's up"), snap, allow_model=False)
    assert not target.resolved


def test_quoted_span_is_content_for_writing_verbs(anchor):
    """"send 'hello there' to Bob" must ground on Bob, not on the message text."""
    ref = anchor.extract_referent("send 'hello there' to Bob")
    assert ref.text == "Bob"
    assert ref.payload == "hello there"


def test_quoted_span_is_the_target_for_clicking_verbs(anchor):
    ref = anchor.extract_referent("click 'Sign in with Google'")
    assert ref.text == "Sign in with Google"
    assert ref.is_quoted
    assert ref.payload == ""


def test_recipient_and_content_split_without_quotes(anchor):
    ref = anchor.extract_referent("message Arundhati saying hey are you free")
    assert ref.text == "Arundhati"
    assert ref.payload == "hey are you free"


@pytest.mark.parametrize("question", [
    "is Send visible on screen",
    "what is on the screen",
    "do you see the Send button",
    "where is main.py",
    "can you see my screen",
    "is that Arundhati?",
])
def test_questions_never_become_clicks(anchor, question, instagram_elements):
    """Found by the old-vs-new benchmark: ANCHOR happily grounded "is Send visible on screen" to
    the Send button and clicked it. A question about the screen must never act on it."""
    ref = anchor.extract_referent(question)
    assert ref.is_question is True
    assert anchor.ground(ref, snapshot(instagram_elements), allow_model=False).resolved is False


@pytest.mark.parametrize("order,expected", [
    ("could you please open main.py", "main.py"),
    ("can you open notepad", "notepad"),
    ("would you click Send", "Send"),
    ("please open instagram", "instagram"),
    ("just click Send please", "Send"),
])
def test_polite_requests_are_still_commands(anchor, order, expected):
    """The old engine's interrogative guard was a flat leading-word regex, so it also rejected
    "could you open main.py" — an ordinary polite order. A modal followed by a real action verb is
    a request, not a question."""
    ref = anchor.extract_referent(order)
    assert ref.is_question is False
    assert ref.text.lower() == expected.lower()


def test_do_it_again_is_imperative_not_interrogative(anchor):
    """"do" opens both "does it work" (question) and "do it again" (order)."""
    ref = anchor.extract_referent("do it again")
    assert ref.is_question is False
    assert ref.is_anaphoric is True


def test_lowercase_names_are_found(anchor):
    """The old `_extract_named_target` regex was `\\b[A-Z][a-z]{2,}\\b`, so it found nothing at all
    in lowercase voice transcription — which is what a microphone actually produces."""
    ref = anchor.extract_referent("open the chat of arundhati")
    assert ref.text == "arundhati"


# ======================================================================
# 2. Where is it on screen?
# ======================================================================


def test_grounds_exact_name_without_a_model_call(anchor, instagram_elements):
    snap = snapshot(instagram_elements)
    target = anchor.ground_order("open the chat of Arundhati", snap)
    assert target.resolved
    assert target.element.name == "Arundhati Sharma"
    assert target.confidence >= CONFIDENT
    assert anchor.llm.calls == 0, "an unambiguous name must never cost a model call"


def test_prefers_the_named_row_over_a_generic_tab(anchor, instagram_elements):
    """The documented monolith failure: told to open a specific person's chat, it clicked the
    generic "Instagram Messages" tab because that was a plausible first step."""
    snap = snapshot(instagram_elements)
    target = anchor.ground_order("open the chat of Arundhati", snap)
    assert target.element.name == "Arundhati Sharma"
    assert "Messages" not in target.element.name


def test_full_element_list_is_searched_not_just_the_first_25(anchor):
    """The old observation builder sorted by screen position and truncated to [:25], so an element
    further down was invisible to the model and grounding failed for reasons prompting can't fix."""
    filler = [el(f"Item {i}", "ListItemControl", 100, 100 + i * 20) for i in range(40)]
    target_row = el("Arundhati Sharma", "ListItemControl", 100, 1000)
    snap = snapshot(filler + [target_row])
    assert len(snap.elements) == 41

    target = anchor.ground_order("open the chat of Arundhati", snap)
    assert target.resolved
    assert target.element.name == "Arundhati Sharma"


def test_ordinal_selection_uses_reading_order(anchor):
    rows = [
        el("Video A", "ListItemControl", 100, 300),
        el("Video B", "ListItemControl", 100, 100),
        el("Video C", "ListItemControl", 100, 200),
    ]
    target = anchor.ground_order("play the 3rd video", snapshot(rows))
    assert target.resolved
    assert target.method == "ordinal"
    assert target.element.name == "Video A"  # topmost first: B(100), C(200), A(300)


def test_ambiguous_candidates_escalate_to_one_cheap_model_call(instagram_elements):
    llm = FakeLLM(lambda payload: '{"choice": 2, "confidence": 0.8}')
    anchor = AnchorAgent(llm)
    snap = snapshot([
        el("Arun Kumar", "ListItemControl", 100, 100),
        el("Arun Kumaran", "ListItemControl", 100, 140),
    ])
    target = anchor.ground_order("open the chat of Arun", snap)
    assert target.resolved
    assert llm.calls == 1, "ambiguity costs exactly one call"
    assert llm.payloads[0]["model"] == "llama-3.1-8b-instant", "and it uses the cheap model"


def test_vision_is_only_used_when_the_tree_has_nothing(anchor):
    calls = []

    def locate(description):
        calls.append(description)
        return (640, 480)

    snap = snapshot([el("Unrelated", "ButtonControl", 10, 10)])
    target = anchor.ground_order("click the record button", snap, vision_locate=locate, allow_model=False)
    assert calls == ["record"]
    assert target.method == "vision"
    assert target.coords == (640, 480)


def test_vision_is_not_used_when_the_tree_already_matches(anchor, instagram_elements):
    calls = []
    snap = snapshot(instagram_elements)
    anchor.ground_order("click Send", snap, vision_locate=lambda d: calls.append(d) or (1, 1))
    assert calls == [], "a confident tree match must not spend a vision call"


@pytest.mark.parametrize("order,expected", [
    ("click Saved Tab Groups", "Saved Tab Groups"),
    ("click saved tab groups", "Saved Tab Groups"),
    ("could you please click Saved Tab Groups", "Saved Tab Groups"),
    ("click Chrome Legacy Window", "Chrome Legacy Window"),
])
def test_type_noun_inside_a_proper_name_is_not_a_control_hint(anchor, order, expected):
    """Found by running against a live Chrome window: "Saved Tab Groups" is a Button whose name
    merely contains "tab". Reading that as a control-type hint stripped it from the target text and
    narrowed the pool to TabItems, hiding the exact element being asked for. This single case
    accounted for every grounding miss in that run."""
    snap = snapshot([
        el("Saved Tab Groups", "ButtonControl", 200, 60),
        el("Chrome Legacy Window", "PaneControl", 900, 500, w=1800, h=900),
        el("New Tab", "TabItemControl", 400, 20),
        el("Pull Request #1", "TabItemControl", 600, 20),
        el("Chrome", "ButtonControl", 50, 20),
    ])
    target = anchor.ground_order(order, snap, allow_model=False)
    assert target.resolved
    assert target.element.name == expected


def test_trailing_type_noun_still_works_as_a_hint(anchor):
    """The fallback must not undo the ordinary case: "the Send button" should still prefer the
    Button named Send over a list item that happens to share the word."""
    snap = snapshot([
        el("Send", "ListItemControl", 100, 400),
        el("Send", "ButtonControl", 900, 800),
    ])
    target = anchor.ground_order("click the Send button", snap, allow_model=False)
    assert target.element.type == "ButtonControl"


def test_oversized_containers_are_penalised(anchor):
    snap = snapshot([
        el("Settings", "PaneControl", 960, 540, w=1900, h=1000),
        el("Settings", "ButtonControl", 300, 200, w=90, h=30),
    ])
    target = anchor.ground_order("click Settings", snap, allow_model=False)
    assert target.element.type == "ButtonControl", "a full-screen pane merely contains the match"


# ======================================================================
# 3. Did we actually hit it? (the scope guard)
# ======================================================================


def test_scope_guard_rejects_a_successful_but_misdirected_click(anchor):
    ref = anchor.extract_referent("open the chat of Arundhati")
    result = {"status": "success", "matched_name": "Instagram Messages"}
    ok, reason = anchor.verify_on_target(ref, result)
    assert ok is False
    assert "Arundhati" in reason


def test_scope_guard_accepts_the_right_target(anchor):
    ref = anchor.extract_referent("open the chat of Arundhati")
    ok, _ = anchor.verify_on_target(ref, {"status": "success", "matched_name": "Arundhati Sharma"})
    assert ok is True


def test_scope_guard_treats_failure_as_off_target(anchor):
    ref = anchor.extract_referent("click Send")
    ok, _ = anchor.verify_on_target(ref, {"status": "error", "message": "nope"})
    assert ok is False


def test_scope_guard_defers_when_no_name_was_reported(anchor):
    """Keystrokes and coordinate clicks report no name; that is 'unknown', not 'mismatch'."""
    ref = anchor.extract_referent("click Send")
    ok, reason = anchor.verify_on_target(ref, {"status": "success"})
    assert ok is True
    assert "deferring" in reason


def test_scope_report_gives_an_actionable_correction(anchor):
    ref = anchor.extract_referent("open the chat of Arundhati")
    report = anchor.scope_report("open the chat of Arundhati", ref,
                                 {"status": "success", "matched_name": "Instagram Messages"})
    assert "SCOPE WARNING" in report
    assert "Arundhati" in report
    assert "Do not report this as done" in report


# ======================================================================
# Follow-ups
# ======================================================================


def test_anaphora_resolves_from_history(anchor, instagram_elements):
    history = [{"user": "open the chat of Arundhati", "bot": "Clicked 'Arundhati Sharma' for you, Boss."}]
    snap = snapshot(instagram_elements)
    target = anchor.ground_order("do it again", snap, history=history, allow_model=False)
    assert target.resolved
    assert target.element.name == "Arundhati Sharma"


def test_no_history_means_no_blind_click(anchor, instagram_elements):
    target = anchor.ground_order("do it again", snapshot(instagram_elements), history=[], allow_model=False)
    assert not target.resolved, "with nothing to refer back to, clicking anything would be a guess"
