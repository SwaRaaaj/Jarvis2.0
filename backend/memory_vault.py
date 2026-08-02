import sqlite3
import os
import json
import time
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_memory.db")

class MemoryVault:
    """Stores user information, reinforcement learning rules, custom macros, and activity logs."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # User profile table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Reinforcement Learning & Learned Rules Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_rule TEXT NOT NULL,
                    learned_action TEXT NOT NULL,
                    reward_score REAL DEFAULT 1.0,
                    created_at REAL NOT NULL
                )
            ''')

            # Execution logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    thought_chain TEXT,
                    tool_used TEXT,
                    tool_input TEXT,
                    tool_output TEXT,
                    status TEXT,
                    timestamp REAL NOT NULL
                )
            ''')

            # Reading the log is the whole point of SCHOLAR's mining pass, and it scans by recency
            # and by status. Without these the scan degrades to a full table walk as the log grows.
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_ts ON execution_logs(timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_status ON execution_logs(status)')

            # Additive migration: existing databases predate these columns, and the original schema
            # had no way to tell a rule used a hundred times from one used once.
            for column, ddl in (
                ("hit_count", "ALTER TABLE learned_rules ADD COLUMN hit_count INTEGER DEFAULT 0"),
                ("last_used_at", "ALTER TABLE learned_rules ADD COLUMN last_used_at REAL DEFAULT 0"),
            ):
                try:
                    cursor.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column already present

            conn.commit()

        self.seed_defaults()

    def seed_defaults(self):
        defaults = {
            "user_name": "SWARAJ LAYEK / Boss",
            "assistant_name": "JARVIS AI Brain",
            "preferred_voice_speed": "175",
            "emergency_stop_key": "ESC"
        }
        for k, v in defaults.items():
            if not self.get_user_info(k):
                self.set_user_info(k, v)

    def get_user_info(self, key: str) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM user_profile WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_user_info(self) -> Dict[str, str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM user_profile")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def set_user_info(self, key: str, value: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_profile (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            ''', (key, value, time.time()))
            conn.commit()

    def add_learned_rule(self, trigger_rule: str, learned_action: str, reward_score: float = 1.0):
        """Adds a new reinforcement learning rule based on user training."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learned_rules (trigger_rule, learned_action, reward_score, created_at)
                VALUES (?, ?, ?, ?)
            ''', (trigger_rule, learned_action, reward_score, time.time()))
            conn.commit()

    def get_learned_rules(self) -> List[Dict[str, Any]]:
        """Fetches all reinforcement learning rules sorted by highest reward score."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT trigger_rule, learned_action, reward_score FROM learned_rules
                ORDER BY reward_score DESC, id DESC
            ''')
            rows = cursor.fetchall()
            return [{"trigger": r[0], "action": r[1], "score": r[2]} for r in rows]

    def log_action(self, user_input: str, thought_chain: str, tool_used: str, tool_input: str, tool_output: str, status: str = "success"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO execution_logs (user_input, thought_chain, tool_used, tool_input, tool_output, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_input, thought_chain, tool_used, tool_input, tool_output, status, time.time()))
            conn.commit()

    # ------------------------------------------------------------------
    # Log reading. `get_recent_logs` is called by GET /api/memory in main.py but never existed,
    # so that endpoint raised AttributeError on every request.
    # ------------------------------------------------------------------

    def get_recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Most recent execution log entries, newest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, user_input, thought_chain, tool_used, tool_input, tool_output, status, timestamp
                FROM execution_logs ORDER BY timestamp DESC LIMIT ?
            ''', (int(limit),))
            return [
                {
                    "id": r[0], "user_input": r[1], "thought_chain": r[2], "tool_used": r[3],
                    "tool_input": r[4], "tool_output": r[5], "status": r[6], "timestamp": r[7],
                }
                for r in cursor.fetchall()
            ]

    def get_logs_since(self, since_ts: float = 0.0, limit: int = 2000) -> List[Dict[str, Any]]:
        """Log entries newer than a timestamp — SCHOLAR's incremental mining window."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_input, tool_used, tool_input, status, timestamp
                FROM execution_logs WHERE timestamp > ? ORDER BY timestamp ASC LIMIT ?
            ''', (float(since_ts), int(limit)))
            return [
                {"user_input": r[0], "tool_used": r[1], "tool_input": r[2], "status": r[3], "timestamp": r[4]}
                for r in cursor.fetchall()
            ]

    # ------------------------------------------------------------------
    # Learned rules. The table and both accessors existed from the start but nothing in the
    # codebase ever called them — the reinforcement loop was scaffolding with no wiring.
    # ------------------------------------------------------------------

    def find_learned_rule(self, trigger_rule: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, trigger_rule, learned_action, reward_score, hit_count
                FROM learned_rules WHERE trigger_rule = ? ORDER BY reward_score DESC LIMIT 1
            ''', (trigger_rule,))
            row = cursor.fetchone()
            if not row:
                return None
            return {"id": row[0], "trigger": row[1], "action": row[2], "score": row[3], "hits": row[4] or 0}

    def upsert_learned_rule(self, trigger_rule: str, learned_action: str, reward_delta: float = 1.0) -> None:
        """Adds a rule, or reinforces it if the same trigger/action pair is already known."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, reward_score FROM learned_rules WHERE trigger_rule = ? AND learned_action = ?',
                           (trigger_rule, learned_action))
            row = cursor.fetchone()
            if row:
                cursor.execute('UPDATE learned_rules SET reward_score = ? WHERE id = ?',
                               (float(row[1] or 0.0) + reward_delta, row[0]))
            else:
                cursor.execute('''
                    INSERT INTO learned_rules (trigger_rule, learned_action, reward_score, created_at, hit_count, last_used_at)
                    VALUES (?, ?, ?, ?, 0, 0)
                ''', (trigger_rule, learned_action, reward_delta, time.time()))
            conn.commit()

    def penalise_learned_rule(self, trigger_rule: str, learned_action: str, penalty: float = 1.0) -> None:
        """Demotes a rule that stopped working. Rules that fall to zero or below stop being served,
        so a UI change that invalidates a shortcut self-corrects instead of failing forever."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE learned_rules SET reward_score = reward_score - ?
                WHERE trigger_rule = ? AND learned_action = ?
            ''', (penalty, trigger_rule, learned_action))
            conn.commit()

    def touch_learned_rule(self, trigger_rule: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE learned_rules SET hit_count = COALESCE(hit_count, 0) + 1, last_used_at = ?
                WHERE trigger_rule = ?
            ''', (time.time(), trigger_rule))
            conn.commit()

    def get_active_rules(self, min_score: float = 1.0) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT trigger_rule, learned_action, reward_score, COALESCE(hit_count, 0)
                FROM learned_rules WHERE reward_score >= ? ORDER BY reward_score DESC, id DESC
            ''', (float(min_score),))
            return [{"trigger": r[0], "action": r[1], "score": r[2], "hits": r[3]} for r in cursor.fetchall()]
