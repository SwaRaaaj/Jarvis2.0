"""SCHOLAR — the learning agent.

Every command JARVIS has ever run is already in `execution_logs`: the order, the tool, the
arguments, the result, the status. And `learned_rules` — a table with a trigger, an action and a
reward score — has existed since the first commit, with `add_learned_rule` and `get_learned_rules`
written and ready.

Nothing in the codebase ever called either one. The reinforcement loop was scaffolding with no
wiring, so JARVIS re-derived from scratch, at full cost, every order it had already executed
correctly a hundred times.

SCHOLAR closes that loop. It mines the log for orders that reliably resolve to the same action,
promotes them to rules, and hands TRIAGE a lookup that answers them with **zero model calls and
zero network**. It is the only agent here that makes JARVIS faster the longer it runs, rather than
faster once.

Safety properties, since this writes rules that later execute for real:
  * only orders seen succeeding at least PROMOTION_THRESHOLD times are promoted
  * any failure of the same order blocks promotion outright — a 3-1 record is not a rule
  * tools with side effects that vary by context (typing, coordinate clicks) are never promoted;
    "click at (840, 512)" is not reusable knowledge, it is a coincidence of one screen layout
  * rules are demoted on failure and stop being served at zero, so a UI change self-corrects
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import Agent, AgentContext, LLMClient, normalize

# How many clean successes before an order becomes a rule. Two is too eager (one lucky repeat),
# five is too slow to ever feel like learning.
PROMOTION_THRESHOLD = 3

# Only tools whose (order -> arguments) mapping is stable across screens can be promoted. A launch
# or a search means the same thing tomorrow; a click at a pixel coordinate does not.
PROMOTABLE_TOOLS = {
    "launch_app", "open_url", "open_social_inbox", "search_google", "search_youtube",
    "get_time_date", "switch_window", "close_tab", "close_all_tabs", "close_window",
}

# Orders too generic to key a rule on — they mean something different every time they are said.
_UNSTABLE_ORDERS = re.compile(
    r"^(do it again|again|same|that one|this|it|yes|no|ok|okay|stop|continue|go on)$", re.IGNORECASE
)

# Multi-clause orders must never become a single-tool rule.
#
# This is not hypothetical: mining this project's real 200-entry log promoted
# "open chrome then search google for openai" -> launch_app(chrome), because a multi-step run logs
# every one of its actions against the full order text. Serving that rule would have launched
# Chrome and silently dropped the search half of the order forever after.
_COMPOUND_MARKERS = (
    " then ", " and then ", " after that", ";", " next ", " followed by ", " also ", " and search ",
    " and open ", " and click ", " and type ", " and send ", " and close ",
)


class ScholarAgent(Agent):
    """Mines the execution log into reusable shortcuts."""

    name = "SCHOLAR"

    def __init__(self, memory: Any = None, llm: Optional[LLMClient] = None):
        super().__init__(llm)
        self.memory = memory
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded_at: float = 0.0
        self.last_mined_ts: float = 0.0
        self.promotions: int = 0
        self.demotions: int = 0
        self.hits: int = 0
        self.misses: int = 0

    # ------------------------------------------------------------------
    # Key normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def rule_key(order: str) -> str:
        """Canonical form of an order, so "open Chrome please" and "Open chrome" share a rule.

        Strips the wake word for the same reason every other agent does: this repo's own folder is
        named "jarvis", so leaving it in a stored rule key would make the key match window titles.
        """
        text = re.sub(r"\bjarvis\b", " ", order or "", flags=re.IGNORECASE)
        text = re.sub(r"\b(please|now|for me|thanks|thank you|quickly|real quick)\b", " ", text, flags=re.IGNORECASE)
        return normalize(text)

    def promotable(self, order: str, tool: str, args: Dict[str, Any]) -> bool:
        key = self.rule_key(order)
        if not key or len(key) < 3 or _UNSTABLE_ORDERS.match(key):
            return False
        if any(marker in f" {key} " for marker in _COMPOUND_MARKERS) or key.count(",") >= 2:
            return False
        if tool not in PROMOTABLE_TOOLS:
            return False
        # An order that carries free text (a message, a search phrase not in the args) is not a
        # fixed mapping — the same words next time will mean different content.
        if any(isinstance(v, str) and len(v) > 80 for v in (args or {}).values()):
            return False
        return True

    # ------------------------------------------------------------------
    # Lookup — the hot path, called by TRIAGE on every order
    # ------------------------------------------------------------------

    def lookup(self, order: str) -> Optional[Dict[str, Any]]:
        """Returns a learned (tool, args) for this order, or None. No model call, no network."""
        if self.memory is None:
            return None
        key = self.rule_key(order)
        if not key:
            return None
        self._ensure_cache()
        hit = self._cache.get(key)
        if not hit:
            self.misses += 1
            return None
        self.hits += 1
        try:
            self.memory.touch_learned_rule(key)
        except Exception:
            pass
        return {"tool": hit["tool"], "args": dict(hit.get("args") or {}), "trigger": key,
                "score": hit.get("score", 0.0)}

    def _ensure_cache(self, ttl: float = 30.0) -> None:
        """Rules live in SQLite but are read on every single order, so they are cached in memory
        and refreshed on a timer rather than hitting the database each time."""
        if self._cache and (time.time() - self._cache_loaded_at) < ttl:
            return
        self._cache = {}
        try:
            rules = self.memory.get_active_rules(min_score=float(PROMOTION_THRESHOLD))
        except Exception:
            rules = []
        for rule in rules:
            try:
                action = json.loads(rule.get("action") or "{}")
            except Exception:
                continue
            tool = action.get("tool")
            if not tool or tool not in PROMOTABLE_TOOLS:
                continue
            trigger = rule.get("trigger") or ""
            if trigger and trigger not in self._cache:
                self._cache[trigger] = {"tool": tool, "args": action.get("args") or {},
                                        "score": rule.get("score", 0.0)}
        self._cache_loaded_at = time.time()

    def invalidate_cache(self) -> None:
        self._cache_loaded_at = 0.0

    # ------------------------------------------------------------------
    # Online reinforcement — called after every run
    # ------------------------------------------------------------------

    def record(self, order: str, tool: str, args: Dict[str, Any], success: bool,
               ctx: Optional[AgentContext] = None) -> None:
        """Reinforces or demotes the rule for an order that just ran."""
        if self.memory is None or not self.promotable(order, tool, args):
            return
        key = self.rule_key(order)
        action = json.dumps({"tool": tool, "args": args}, sort_keys=True)
        try:
            if success:
                self.memory.upsert_learned_rule(key, action, reward_delta=1.0)
                existing = self.memory.find_learned_rule(key)
                if existing and float(existing.get("score") or 0) >= PROMOTION_THRESHOLD:
                    if key not in self._cache:
                        self.promotions += 1
                        self._emit(ctx, "status", f"learned a shortcut for \"{key}\"")
                        self.invalidate_cache()
            else:
                self.memory.penalise_learned_rule(key, action, penalty=2.0)
                self.demotions += 1
                self.invalidate_cache()
        except Exception as e:  # noqa: BLE001 — learning must never break a run
            self._emit(ctx, "status", f"couldn't record that outcome: {e}")

    # ------------------------------------------------------------------
    # Offline mining — the backfill pass over history
    # ------------------------------------------------------------------

    def purge_invalid_rules(self, ctx: Optional[AgentContext] = None) -> int:
        """Re-validates every stored rule against the current promotion policy and demotes any that
        no longer qualify.

        Rules outlive the code that created them. When a policy gap is closed — as it was for
        compound orders, after mining this project's real log produced
        "open chrome then search google for openai" -> launch_app — the bad rules are already
        written to disk. Self-healing on each mining pass is what stops a fixed bug from continuing
        to misbehave in the field.
        """
        if self.memory is None:
            return 0
        try:
            rules = self.memory.get_active_rules(min_score=0.0)
        except Exception:
            return 0
        purged = 0
        for rule in rules:
            trigger = rule.get("trigger") or ""
            raw_action = rule.get("action") or "{}"
            try:
                action = json.loads(raw_action)
            except Exception:
                action = {}
            tool = action.get("tool") or ""
            args = action.get("args") or {}
            if trigger and tool and self.promotable(trigger, tool, args):
                continue
            try:
                self.memory.penalise_learned_rule(trigger, raw_action, penalty=999.0)
                purged += 1
            except Exception:
                continue
        if purged:
            self.invalidate_cache()
            self._emit(ctx, "status", f"retired {purged} rule(s) that no longer meet the promotion policy")
        return purged

    def mine(self, ctx: Optional[AgentContext] = None, since: Optional[float] = None) -> Dict[str, Any]:
        """Scans the execution log and promotes every order with a clean, repeated success record.

        Run at startup (in the background) and periodically. On an existing installation this
        immediately converts months of accumulated history into shortcuts.
        """
        if self.memory is None:
            return {"scanned": 0, "promoted": 0}
        purged = self.purge_invalid_rules()
        window_start = self.last_mined_ts if since is None else since
        try:
            logs = self.memory.get_logs_since(window_start)
        except Exception as e:  # noqa: BLE001
            return {"scanned": 0, "promoted": 0, "purged": purged, "error": str(e)}

        successes: Dict[Tuple[str, str], int] = defaultdict(int)
        failures: Dict[str, int] = defaultdict(int)
        actions_per_key: Dict[str, set] = defaultdict(set)

        for entry in logs:
            order = entry.get("user_input") or ""
            tool = entry.get("tool_used") or ""
            key = self.rule_key(order)
            if not key:
                continue
            try:
                args = json.loads(entry.get("tool_input") or "{}")
            except Exception:
                args = {}
            if entry.get("status") != "success":
                failures[key] += 1
                continue
            # Tracked before the promotable() filter, so a run that mixed a promotable tool with a
            # non-promotable one (a launch followed by a click) still counts as multi-action.
            if tool:
                actions_per_key[key].add(tool)
            if not self.promotable(order, tool, args):
                continue
            action = json.dumps({"tool": tool, "args": args}, sort_keys=True)
            successes[(key, action)] += 1

        promoted = 0
        for (key, action), count in successes.items():
            # A single failure of the same order is disqualifying. A rule fires without review, so
            # the bar for creating one has to be higher than "usually works".
            if failures.get(key):
                continue
            # An order that took several different tools is a multi-step task. Promoting one of its
            # tools as "the" rule for that order would drop the rest of the work silently.
            if len(actions_per_key.get(key, ())) > 1:
                continue
            if count < PROMOTION_THRESHOLD:
                continue
            try:
                self.memory.upsert_learned_rule(key, action, reward_delta=float(count))
                promoted += 1
            except Exception:
                continue

        self.promotions += promoted
        self.last_mined_ts = time.time()
        self.invalidate_cache()
        if promoted:
            self._emit(ctx, "status", f"learned {promoted} shortcut{'s' if promoted != 1 else ''} from history")
        return {"scanned": len(logs), "promoted": promoted, "purged": purged, "candidates": len(successes)}

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "rules_cached": len(self._cache),
            "lookups": total,
            "hits": self.hits,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "promotions": self.promotions,
            "demotions": self.demotions,
        }
