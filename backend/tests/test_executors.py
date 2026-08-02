"""PATHFINDER (navigation) and HANDS (on-screen interaction)."""

import pytest

from agents.anchor import AnchorAgent
from agents.base import PlanStep
from agents.hands import HandsAgent, TOOLS as HANDS_TOOLS
from agents.pathfinder import PathfinderAgent, TOOLS as PATH_TOOLS
from conftest import FakeLLM, FakeOS, el, snapshot, tool_call


# ======================================================================
# Tool-surface size — the token-budget claim
# ======================================================================


def test_tool_surface_is_split_not_duplicated():
    """The monolith shipped 21 schemas (~2k tokens) on every step of every task. Neither executor
    may carry the whole surface, or the split bought nothing."""
    assert len(PATH_TOOLS) == 7
    assert len(HANDS_TOOLS) == 9
    assert len(PATH_TOOLS) + len(HANDS_TOOLS) < 21

    path_names = {t["function"]["name"] for t in PATH_TOOLS}
    hands_names = {t["function"]["name"] for t in HANDS_TOOLS}
    assert not (path_names & hands_names), "lanes must not overlap"


# ======================================================================
# PATHFINDER
# ======================================================================


@pytest.fixture
def pathfinder():
    return PathfinderAgent(FakeLLM(), os_api=FakeOS())


@pytest.mark.parametrize("order,action", [
    ("open chrome", "launch_app"),
    ("launch notepad", "launch_app"),
    ("open youtube", "open_url"),
    ("open instagram", "open_social_inbox"),
    ("what time is it", "get_time_date"),
    ("open github.com", "open_url"),
])
def test_known_destinations_need_no_model_call(pathfinder, order, action):
    result = pathfinder.execute(PlanStep(description=order))
    assert result["status"] == "success"
    assert result["tool"] == action
    assert pathfinder.llm.calls == 0


def test_switch_to_a_closed_window_launches_instead(pathfinder):
    """Recovering in code saves a whole model round-trip that could only reach the same conclusion."""
    result = pathfinder.execute(PlanStep(description="switch to notepad"))
    assert result["status"] == "success"
    assert pathfinder.os.tools_used()[-1] == "launch_app"
    assert pathfinder.llm.calls == 0


def test_switch_to_an_open_window_does_not_relaunch():
    os_api = FakeOS()
    os_api.window_titles = ["chrome"]
    pathfinder = PathfinderAgent(FakeLLM(), os_api=os_api)
    result = pathfinder.execute(PlanStep(description="switch to chrome"))
    assert result["status"] == "success"
    assert "launch_app" not in os_api.tools_used()


def test_unknown_destination_uses_one_call_with_seven_schemas():
    llm = FakeLLM(lambda p: tool_call("open_url", url="https://arxiv.org"))
    pathfinder = PathfinderAgent(llm, os_api=FakeOS())
    result = pathfinder.execute(PlanStep(description="open the arxiv preprint server"))

    assert result["status"] == "success"
    assert llm.calls == 1
    assert llm.tool_schema_count() == 7, "navigation must never see the interaction tools"


# ======================================================================
# HANDS
# ======================================================================


@pytest.fixture
def hands():
    llm = FakeLLM()
    return HandsAgent(llm, anchor=AnchorAgent(llm), os_api=FakeOS())


@pytest.mark.parametrize("order,expected", [
    ("close this tab", "close_tab"),
    ("close all tabs", "close_all_tabs"),
    ("close the window", "close_window"),
    ("scroll down", "scroll"),
    ("scroll up", "scroll"),
    ("type hello", "type_text"),
    ("click the Send button", "click_element"),
])
def test_action_detection_is_deterministic(hands, order, expected):
    assert hands._detect_action(order) == expected


def test_targetless_actions_need_no_model_call(hands):
    result = hands.execute(PlanStep(description="scroll down 3 times"), snapshot([]))
    assert result["status"] == "success"
    assert result["args"] == {"direction": "down", "amount": 3}
    assert hands.llm.calls == 0


def test_click_is_grounded_by_anchor_with_no_model_call(hands, instagram_elements):
    step = PlanStep(description="open the chat of Arundhati", target="Arundhati")
    result = hands.execute(step, snapshot(instagram_elements))

    assert result["status"] == "success"
    assert result["matched_name"] == "Arundhati Sharma"
    assert result["args"]["x"] == 300 and result["args"]["y"] == 220
    assert result["on_target"] is True
    assert hands.llm.calls == 0, "grounding + acting must both be free for a clear target"


def test_typing_extracts_the_payload_not_the_recipient(hands, instagram_elements):
    step = PlanStep(description="send 'see you at 6' to Arundhati")
    result = hands.execute(step, snapshot(instagram_elements))
    assert result["tool"] == "type_text"
    assert result["args"]["text"] == "see you at 6"
    assert hands.llm.calls == 0


def test_off_target_click_is_flagged_not_celebrated(hands):
    """A click that succeeds on the wrong element must not read as progress. This is the exact
    failure the monolith's comments describe: it clicked a generic Instagram tab when asked for a
    specific person's chat, and auto-complete reported success."""
    elements = [el("Instagram Messages", "TabItemControl", 420, 80)]
    step = PlanStep(description="open the chat of Arundhati", target="Arundhati")
    result = hands.execute(step, snapshot(elements))

    if result.get("matched_name"):
        assert result["on_target"] is False
        assert "Arundhati" in result["correction"]
        assert hands.scope_blocks == 1


def test_double_click_for_files(hands):
    step = PlanStep(description="open main.py", target="main.py")
    result = hands.execute(step, snapshot([el("main.py", "ListItemControl", 200, 300)]))
    assert result["args"]["double"] is True


def test_explicit_double_click_is_honoured(hands):
    step = PlanStep(description="double click the Refresh button", target="Refresh")
    result = hands.execute(step, snapshot([el("Refresh", "ButtonControl", 200, 300)]))
    assert result["args"]["double"] is True


def test_model_chosen_label_is_still_grounded_never_trusted():
    """A label the model produced is a hypothesis, not a coordinate. It goes through ANCHOR too."""
    llm = FakeLLM(lambda p: tool_call("click_element", text="Submit"))
    hands = HandsAgent(llm, anchor=AnchorAgent(llm), os_api=FakeOS())
    result = hands.execute(PlanStep(description="finalise the form"),
                           snapshot([el("Submit", "ButtonControl", 500, 600)]))

    assert result["status"] == "success"
    assert result["args"]["x"] == 500 and result["args"]["y"] == 600
    assert result["grounding"]["method"] in ("exact", "strong")


def test_model_label_that_matches_nothing_errors_instead_of_clicking():
    llm = FakeLLM(lambda p: tool_call("click_element", text="Nonexistent Control"))
    hands = HandsAgent(llm, anchor=AnchorAgent(llm), os_api=FakeOS())
    result = hands.execute(PlanStep(description="do the thing"), snapshot([el("Other", "ButtonControl")]))

    assert result["status"] == "error"
    assert hands.os.calls == [], "a hallucinated label must never produce a real click"


def test_hands_only_ever_sees_nine_schemas():
    llm = FakeLLM(lambda p: tool_call("key_combo", keys="ctrl+s"))
    hands = HandsAgent(llm, anchor=AnchorAgent(llm), os_api=FakeOS())
    hands.execute(PlanStep(description="save whatever is in front of me"), snapshot([]))
    assert llm.tool_schema_count() == 9
