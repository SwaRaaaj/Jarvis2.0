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
