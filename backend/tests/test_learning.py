"""SCHOLAR (learning from the execution log) and EARS (voice gating).

Also covers the MemoryVault repairs SCHOLAR depends on — including `get_recent_logs`, which
main.py has always called and which never existed.
"""

import json

import pytest

from agents.ears import EarsAgent
from agents.scholar import PROMOTION_THRESHOLD, ScholarAgent


# ======================================================================
# MemoryVault repairs
# ======================================================================


def test_get_recent_logs_exists_and_works(memory):
    """GET /api/memory called memory.get_recent_logs(15) against a method that was never defined,
    so the endpoint raised AttributeError on every request."""
    for i in range(5):
        memory.log_action(f"order {i}", "", "launch_app", "{}", "ok", "success")

    logs = memory.get_recent_logs(3)
    assert len(logs) == 3
    assert logs[0]["user_input"] == "order 4", "newest first"
    assert set(logs[0]) >= {"user_input", "tool_used", "status", "timestamp"}


def test_learned_rules_round_trip(memory):
    memory.upsert_learned_rule("open chrome", '{"tool": "launch_app"}', reward_delta=1.0)
    memory.upsert_learned_rule("open chrome", '{"tool": "launch_app"}', reward_delta=1.0)

    rule = memory.find_learned_rule("open chrome")
    assert rule["score"] == 2.0, "reinforcing an existing rule must not duplicate it"


def test_rules_can_be_demoted_below_serving_threshold(memory):
    memory.upsert_learned_rule("open thing", '{"tool": "launch_app"}', reward_delta=3.0)
    assert len(memory.get_active_rules(min_score=3.0)) == 1

    memory.penalise_learned_rule("open thing", '{"tool": "launch_app"}', penalty=5.0)
    assert memory.get_active_rules(min_score=3.0) == []


# ======================================================================
# SCHOLAR
# ======================================================================


@pytest.fixture
def scholar(memory):
    return ScholarAgent(memory=memory)


def test_rule_key_normalises_phrasing(scholar):
    assert scholar.rule_key("Open Chrome please") == scholar.rule_key("open chrome")
    assert scholar.rule_key("JARVIS open chrome now") == scholar.rule_key("open chrome")


def test_repeated_successes_become_a_free_shortcut(scholar, memory):
    for _ in range(PROMOTION_THRESHOLD):
        memory.log_action("open chrome", "", "launch_app", '{"app_name": "chrome"}', "ok", "success")

    report = scholar.mine()
    assert report["promoted"] == 1

    hit = scholar.lookup("open chrome")
    assert hit["tool"] == "launch_app"
    assert hit["args"] == {"app_name": "chrome"}


def test_below_threshold_is_not_promoted(scholar, memory):
    for _ in range(PROMOTION_THRESHOLD - 1):
        memory.log_action("open chrome", "", "launch_app", '{"app_name": "chrome"}', "ok", "success")
    scholar.mine()
    assert scholar.lookup("open chrome") is None


def test_any_failure_blocks_promotion(scholar, memory):
    """A rule fires without review, so the bar has to be higher than "usually works"."""
    for _ in range(5):
        memory.log_action("open thing", "", "launch_app", '{"app_name": "thing"}', "ok", "success")
    memory.log_action("open thing", "", "launch_app", '{"app_name": "thing"}', "no", "error")

    scholar.mine()
    assert scholar.lookup("open thing") is None


def test_compound_orders_are_never_promoted_as_a_single_action(scholar, memory):
    """Regression, found by mining this project's real log: a multi-step run writes one entry per
    action against the *full* order text, so "open chrome then search google for openai" was
    promoted to launch_app(chrome). Serving that would have launched Chrome and silently dropped
    the search half of the order on every future repeat."""
    for _ in range(5):
        memory.log_action("open chrome then search google for openai", "", "launch_app",
                          '{"app_name": "chrome"}', "ok", "success")
        memory.log_action("open chrome then search google for openai", "", "search_google",
                          '{"query": "openai"}', "ok", "success")

    scholar.mine()
    assert scholar.lookup("open chrome then search google for openai") is None


def test_multi_tool_orders_are_never_promoted_even_without_connectives(scholar, memory):
    """"message alice on instagram" has no "then" but still takes two distinct actions."""
    for _ in range(5):
        memory.log_action("message alice on instagram", "", "open_social_inbox",
                          '{"platform": "instagram"}', "ok", "success")
        memory.log_action("message alice on instagram", "", "click_coordinate",
                          '{"x": 300, "y": 220}', "ok", "success")

    scholar.mine()
    assert scholar.lookup("message alice on instagram") is None


def test_single_action_orders_are_still_promoted(scholar, memory):
    """The guards above must not block the case the agent exists for."""
    for _ in range(PROMOTION_THRESHOLD):
        memory.log_action("open chrome", "", "launch_app", '{"app_name": "chrome"}', "ok", "success")
    scholar.mine()
    assert scholar.lookup("open chrome") is not None


def test_coordinate_clicks_are_never_promoted(scholar, memory):
    """"click at (840, 512)" is a coincidence of one screen layout, not reusable knowledge."""
    for _ in range(10):
        memory.log_action("click send", "", "click_coordinate", '{"x": 840, "y": 512}', "ok", "success")
    scholar.mine()
    assert scholar.lookup("click send") is None


def test_typing_is_never_promoted(scholar, memory):
    for _ in range(10):
        memory.log_action("send a message", "", "type_text", '{"text": "hello"}', "ok", "success")
    scholar.mine()
    assert scholar.lookup("send a message") is None


def test_anaphoric_orders_are_never_promoted(scholar, memory):
    """"do it again" means something different every time it is said."""
    for _ in range(10):
        memory.log_action("do it again", "", "launch_app", '{"app_name": "chrome"}', "ok", "success")
    scholar.mine()
    assert scholar.lookup("do it again") is None


def test_online_reinforcement_promotes_after_enough_successes(scholar):
    for _ in range(PROMOTION_THRESHOLD):
        scholar.record("open spotify", "launch_app", {"app_name": "spotify"}, success=True)
    scholar.invalidate_cache()
    assert scholar.lookup("open spotify") is not None


def test_a_failure_demotes_a_learned_rule(scholar):
    for _ in range(PROMOTION_THRESHOLD + 1):
        scholar.record("open spotify", "launch_app", {"app_name": "spotify"}, success=True)
    scholar.invalidate_cache()
    assert scholar.lookup("open spotify") is not None

    for _ in range(3):
        scholar.record("open spotify", "launch_app", {"app_name": "spotify"}, success=False)
    scholar.invalidate_cache()
    assert scholar.lookup("open spotify") is None, "a UI change must self-correct, not fail forever"


def test_lookup_is_free_of_model_and_network(scholar, memory):
    for _ in range(PROMOTION_THRESHOLD):
        memory.log_action("open chrome", "", "launch_app", '{"app_name": "chrome"}', "ok", "success")
    scholar.mine()

    for _ in range(50):
        scholar.lookup("open chrome")
    assert scholar.stats()["hits"] == 50


def test_mining_survives_corrupt_log_rows(scholar, memory):
    memory.log_action("weird", "", "launch_app", "not json at all", "ok", "success")
    report = scholar.mine()
    assert "error" not in report


def test_rules_written_under_an_older_policy_are_retired(scholar, memory):
    """Rules outlive the code that created them, so closing a policy gap has to clean up the rules
    already on disk — otherwise a fixed bug keeps misbehaving in the field."""
    memory.upsert_learned_rule(
        "open chrome then search google for openai",
        '{"tool": "launch_app", "args": {"app_name": "chrome"}}',
        reward_delta=10.0,
    )
    scholar.invalidate_cache()
    assert scholar.lookup("open chrome then search google for openai") is not None

    assert scholar.purge_invalid_rules() == 1
    assert scholar.lookup("open chrome then search google for openai") is None


def test_purge_leaves_valid_rules_alone(scholar, memory):
    memory.upsert_learned_rule("open chrome", '{"tool": "launch_app", "args": {"app_name": "chrome"}}',
                               reward_delta=10.0)
    assert scholar.purge_invalid_rules() == 0
    assert scholar.lookup("open chrome") is not None


def test_no_memory_backend_is_a_no_op():
    lone = ScholarAgent(memory=None)
    assert lone.lookup("anything") is None
    assert lone.mine() == {"scanned": 0, "promoted": 0}


# ======================================================================
# EARS
# ======================================================================


@pytest.fixture
def ears():
    return EarsAgent()


@pytest.mark.parametrize("utterance", [
    "open chrome", "close this tab", "what time is it", "send a message to bob",
    "scroll down", "search for python tutorials",
])
def test_real_commands_are_dispatched(ears, utterance):
    assert ears.gate(utterance).dispatch is True


@pytest.mark.parametrize("utterance", [
    "um", "yeah", "okay", "hmm", "thanks", "", "   ",
])
def test_filler_is_dropped(ears, utterance):
    assert ears.gate(utterance).dispatch is False


def test_ambient_conversation_does_not_become_a_desktop_action(ears):
    """The listener loop sent every transcribed utterance straight to send_command, so nearby
    conversation caused real clicks, launches and closed tabs."""
    assert ears.gate("she said the meeting is at four").dispatch is False
    assert ears.gate("i was thinking about lunch").dispatch is False


def test_wake_word_alone_is_an_attention_call_not_an_order(ears):
    decision = ears.gate("jarvis")
    assert decision.dispatch is False
    assert decision.had_wake_word is True


def test_wake_word_is_stripped_from_the_dispatched_command(ears):
    decision = ears.gate("hey jarvis open chrome")
    assert decision.dispatch is True
    assert "jarvis" not in decision.cleaned
    assert decision.cleaned == "open chrome"


def test_wake_word_lets_non_imperative_phrasing_through(ears):
    """Addressing JARVIS directly is explicit consent, so the imperative filter is bypassed."""
    assert ears.gate("jarvis i need the volume turned down").dispatch is True


def test_duplicate_transcripts_are_debounced(ears):
    """Speech recognition re-cuts phrase boundaries and returns the same phrase twice; acting on
    both means doing the action twice, which for a click is a real double-action."""
    assert ears.gate("open chrome").dispatch is True
    assert ears.gate("open chrome").dispatch is False


def test_strict_mode_requires_the_wake_word():
    strict = EarsAgent(require_wake_word=True)
    assert strict.gate("open chrome").dispatch is False
    assert strict.gate("jarvis open chrome").dispatch is True


def test_calibration_is_time_based_not_every_iteration(ears):
    """The original loop called adjust_for_ambient_noise every iteration, blocking the mic for
    0.3s each time for a measurement stable over minutes."""
    assert ears.calibration_due() is True
    ears.note_calibration()
    assert ears.calibration_due() is False


def test_drop_rate_is_reported(ears):
    for u in ["um", "yeah", "open chrome", "hmm"]:
        ears.gate(u)
    assert ears.stats()["dispatched"] == 1
    assert ears.stats()["drop_rate"] == 0.75
