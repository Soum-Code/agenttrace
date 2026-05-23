"""
AgentTrace — SQLite Persistence Layer
======================================
Manages saving, retrieving, listing, and deleting analyzed agent trajectories.

Author: P. Somnath Reddy (Research Lead)
"""

import os
import json
import sqlite3
from typing import Any, Dict, List, Optional
from config import CONFIG

DB_PATH = os.path.join(CONFIG.paths.data_dir, "trajectories.db")

def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection to trajectories.db."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the database and creates the necessary tables."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    num_steps INTEGER NOT NULL,
                    num_hallucinated INTEGER NOT NULL,
                    overall_confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    steps_raw_json TEXT NOT NULL
                )
            """)
    finally:
        conn.close()

# Auto-initialize on import
init_db()

def save_trajectory(
    trajectory_id: str,
    task: str,
    num_steps: int,
    num_hallucinated: int,
    overall_confidence: float,
    created_at: str,
    response: dict,
    steps_raw: list
) -> None:
    """Saves or updates a trajectory in the SQLite database."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories (
                    trajectory_id, task, num_steps, num_hallucinated,
                    overall_confidence, created_at, response_json, steps_raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory_id,
                    task,
                    num_steps,
                    num_hallucinated,
                    overall_confidence,
                    created_at,
                    json.dumps(response, ensure_ascii=False),
                    json.dumps(steps_raw, ensure_ascii=False)
                )
            )
    finally:
        conn.close()

def get_trajectory(trajectory_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a saved trajectory (response and raw steps) by ID."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response_json, steps_raw_json FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "response": json.loads(row["response_json"]),
            "steps_raw": json.loads(row["steps_raw_json"])
        }
    finally:
        conn.close()

def list_trajectories() -> List[Dict[str, Any]]:
    """Returns a list of all stored trajectories (responses)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT response_json FROM trajectories ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [json.loads(row["response_json"]) for row in rows]
    finally:
        conn.close()

def delete_trajectory(trajectory_id: str) -> bool:
    """Deletes a trajectory by ID. Returns True if a row was deleted, False otherwise."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,)
            )
            return cursor.rowcount > 0
    finally:
        conn.close()
