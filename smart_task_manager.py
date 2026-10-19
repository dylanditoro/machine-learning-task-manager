"""Smart Task Manager.

A small Flask app that stores tasks in SQLite, scores their priority from the
text of the task, estimates how long they will take, and adjusts those
estimates using the durations of tasks you have already completed.

The scoring is rule-based (keywords and deadline phrases). The only learning
is the calibration step in ``TaskIntelligence.calibration_factor``.
"""

import os
import re
import sqlite3
from datetime import datetime, timedelta

from flask import Flask, jsonify, request

SORT_FIELDS = {"priority_score", "deadline", "created_at", "predicted_duration"}
STATUSES = {"all", "pending", "in_progress", "completed"}
UPDATABLE_FIELDS = ("title", "description", "priority_score", "deadline", "status", "category")

URGENCY_WORDS = {"urgent", "asap", "emergency", "critical", "immediate", "deadline"}
IMPORTANCE_WORDS = {"important", "crucial", "vital", "key", "essential", "priority"}
DURATION_WORDS = {
    "quick": 0.5, "fast": 0.5, "brief": 0.5,
    "short": 1, "medium": 3, "long": 6,
    "detailed": 4, "complex": 5, "thorough": 4,
}
COMPLEX_WORDS = {"analyze", "research", "develop", "create", "design", "implement"}
SIMPLE_WORDS = {"check", "review", "update", "send", "call", "email"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

CATEGORY_KEYWORDS = {
    "work": {"meeting", "report", "presentation", "project", "deadline", "client"},
    "personal": {"home", "family", "health", "exercise", "shopping", "personal"},
    "learning": {"study", "course", "tutorial", "learn", "research", "book"},
    "maintenance": {"fix", "repair", "clean", "organize", "update", "maintenance"},
}

DEFAULT_DAYS_UNTIL_DEADLINE = 7
MIN_SAMPLES_FOR_CALIBRATION = 3
CALIBRATION_RANGE = (0.5, 2.0)


def tokenize(text):
    """Lower-case words, so 'shortcut' does not count as 'short'."""
    return re.findall(r"[a-z']+", text.lower())


class TaskIntelligence:
    """Rule-based task scoring plus a history-based duration calibration."""

    def parse_deadline_days(self, text, now=None):
        """Return days until a deadline mentioned in the text, or None."""
        now = now or datetime.now()
        t = text.lower()

        match = re.search(r"within (\d+) hours?", t)
        if match:
            return int(match.group(1)) / 24
        match = re.search(r"in (\d+) days?", t)
        if match:
            return float(match.group(1))
        if re.search(r"\b(today|tonight)\b", t):
            return 0.5
        if re.search(r"\btomorrow\b", t):
            return 1.0
        match = re.search(r"\b(?:by|due|on)\s+(" + "|".join(WEEKDAYS) + r")\b", t)
        if match:
            target = WEEKDAYS.index(match.group(1))
            ahead = (target - now.weekday()) % 7
            return float(ahead or 7)
        return None

    def extract_features(self, text, now=None):
        words = tokenize(text)
        word_set = set(words)

        estimated_duration = 2.0
        for word, hours in DURATION_WORDS.items():
            if word in word_set:
                estimated_duration = hours
                break

        days = self.parse_deadline_days(text, now)
        return {
            "urgency_score": len(word_set & URGENCY_WORDS),
            "importance_score": len(word_set & IMPORTANCE_WORDS),
            "estimated_duration": estimated_duration,
            "days_until_deadline": DEFAULT_DAYS_UNTIL_DEADLINE if days is None else days,
            "text_length": len(words),
        }

    def priority_score(self, title, description="", now=None):
        """Score from 0 to 10. Urgent words, importance words and near deadlines raise it."""
        features = self.extract_features(f"{title} {description}", now)
        raw = (
            features["urgency_score"] * 3
            + features["importance_score"] * 2
            + 10 / max(features["days_until_deadline"], 0.1)
            + min(features["text_length"] / 10, 2)
        )
        return round(min(max(raw, 0), 10), 2)

    def base_duration(self, title, description=""):
        """Hours estimated from keywords alone, before calibration."""
        text = f"{title} {description}"
        hours = self.extract_features(text)["estimated_duration"]
        word_set = set(tokenize(text))
        if word_set & COMPLEX_WORDS:
            hours *= 1.5
        elif word_set & SIMPLE_WORDS:
            hours *= 0.7
        return round(hours, 2)

    def calibration_factor(self, history):
        """Learn a correction from completed tasks.

        ``history`` is a list of (base_duration, actual_duration) pairs. With
        enough samples, the factor is the average of actual / base, kept
        within CALIBRATION_RANGE so a few odd tasks cannot swing estimates.
        """
        ratios = [actual / base for base, actual in history if base and base > 0 and actual and actual > 0]
        if len(ratios) < MIN_SAMPLES_FOR_CALIBRATION:
            return 1.0
        factor = sum(ratios) / len(ratios)
        low, high = CALIBRATION_RANGE
        return round(min(max(factor, low), high), 3)

    def suggest_category(self, text):
        words = set(tokenize(text))
        for category, keywords in CATEGORY_KEYWORDS.items():
            if words & keywords:
                return category
        return "general"


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    priority_score REAL,
    predicted_duration REAL,
    base_duration REAL,
    deadline TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT,
    completed_at TEXT,
    actual_duration REAL,
    tags TEXT,
    category TEXT
)
"""


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        if "base_duration" not in columns:  # databases created by older versions
            conn.execute("ALTER TABLE tasks ADD COLUMN base_duration REAL")
        conn.commit()
    finally:
        conn.close()


def create_app(db_path=None):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["DB_PATH"] = db_path or os.environ.get("TASKS_DB", "tasks.db")
    ai = TaskIntelligence()
    init_db(app.config["DB_PATH"])

    def connect():
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        return conn

    def error(message, status=400):
        return jsonify({"error": message}), status

    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/tasks")
    def list_tasks():
        status = request.args.get("status", "all")
        sort_by = request.args.get("sort", "priority_score")
        if status not in STATUSES:
            return error(f"status must be one of {sorted(STATUSES)}")
        if sort_by not in SORT_FIELDS:
            return error(f"sort must be one of {sorted(SORT_FIELDS)}")

        query = "SELECT * FROM tasks"
        params = ()
        if status != "all":
            query += " WHERE status = ?"
            params = (status,)
        query += f" ORDER BY {sort_by} DESC"  # sort_by is checked against SORT_FIELDS above

        conn = connect()
        try:
            rows = [dict(row) for row in conn.execute(query, params)]
        finally:
            conn.close()
        return jsonify(rows)

    @app.post("/api/tasks")
    def create_task():
        data = request.get_json(silent=True) or {}
        title = str(data.get("title", "")).strip()
        if not title:
            return error("title is required")
        description = str(data.get("description", ""))
        category = data.get("category") or "general"

        deadline = data.get("deadline")
        if deadline:
            try:
                datetime.fromisoformat(deadline)
            except (TypeError, ValueError):
                return error("deadline must be an ISO date such as 2026-11-02T17:00")

        conn = connect()
        try:
            history = [
                (row["base_duration"], row["actual_duration"])
                for row in conn.execute(
                    "SELECT base_duration, actual_duration FROM tasks "
                    "WHERE status = 'completed' AND category = ?",
                    (category,),
                )
            ]
            base = ai.base_duration(title, description)
            predicted = round(base * ai.calibration_factor(history), 2)
            priority = ai.priority_score(title, description)

            if not deadline:
                days_ahead = max(1, int(predicted / 8))  # assume 8 working hours a day
                deadline = (datetime.now() + timedelta(days=days_ahead)).isoformat()

            cursor = conn.execute(
                "INSERT INTO tasks (title, description, priority_score, predicted_duration, "
                "base_duration, deadline, created_at, tags, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (title, description, priority, predicted, base, deadline,
                 datetime.now().isoformat(), data.get("tags", ""), category),
            )
            conn.commit()
            task_id = cursor.lastrowid
        finally:
            conn.close()

        return jsonify({
            "id": task_id,
            "title": title,
            "priority_score": priority,
            "predicted_duration": predicted,
            "deadline": deadline,
            "message": "Task created",
        }), 201

    @app.put("/api/tasks/<int:task_id>")
    def update_task(task_id):
        data = request.get_json(silent=True) or {}
        conn = connect()
        try:
            if not conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone():
                return error("task not found", 404)

            if data.get("status") == "completed":
                try:
                    actual = float(data.get("actual_duration") or 0)
                except (TypeError, ValueError):
                    return error("actual_duration must be a number")
                if actual < 0:
                    return error("actual_duration cannot be negative")
                conn.execute(
                    "UPDATE tasks SET status = 'completed', completed_at = ?, actual_duration = ? WHERE id = ?",
                    (datetime.now().isoformat(), actual, task_id),
                )
            else:
                fields = [f for f in UPDATABLE_FIELDS if f in data]
                if fields:
                    assignments = ", ".join(f"{f} = ?" for f in fields)  # names come from UPDATABLE_FIELDS
                    conn.execute(
                        f"UPDATE tasks SET {assignments} WHERE id = ?",
                        [data[f] for f in fields] + [task_id],
                    )
            conn.commit()
        finally:
            conn.close()
        return jsonify({"message": "Task updated"})

    @app.delete("/api/tasks/<int:task_id>")
    def delete_task(task_id):
        conn = connect()
        try:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()
        if cursor.rowcount == 0:
            return error("task not found", 404)
        return jsonify({"message": "Task deleted"})

    @app.get("/api/analytics")
    def analytics():
        conn = connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            completed = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'").fetchone()[0]
            avg_time = conn.execute(
                "SELECT AVG(actual_duration) FROM tasks WHERE status = 'completed' AND actual_duration > 0"
            ).fetchone()[0] or 0
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            weekly = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'completed' AND completed_at > ?", (week_ago,)
            ).fetchone()[0]
            error_hours = conn.execute(
                "SELECT AVG(ABS(predicted_duration - actual_duration)) FROM tasks "
                "WHERE status = 'completed' AND actual_duration > 0"
            ).fetchone()[0] or 0
            distribution = dict(conn.execute(
                "SELECT CAST(priority_score AS INTEGER) AS bucket, COUNT(*) FROM tasks GROUP BY bucket"
            ).fetchall())
        finally:
            conn.close()

        return jsonify({
            "total_tasks": total,
            "completed_tasks": completed,
            "completion_rate": round(completed / max(total, 1) * 100, 1),
            "avg_completion_time": round(avg_time, 2),
            "weekly_completed": weekly,
            "productivity_score": round(min(weekly / 7 * 10, 10), 2),
            "priority_distribution": distribution,
            "prediction_accuracy": round(error_hours, 2),
        })

    @app.post("/api/smart-suggestions")
    def smart_suggestions():
        text = str((request.get_json(silent=True) or {}).get("text", ""))
        features = ai.extract_features(text)

        recommendations = []
        if features["urgency_score"] > 0:
            recommendations.append("Urgent wording found. Consider doing this first.")
        if features["estimated_duration"] > 4:
            recommendations.append("This looks large. Consider splitting it into smaller tasks.")
        if features["days_until_deadline"] < 2:
            recommendations.append("Tight deadline. Block out focused time.")

        return jsonify({
            "category": ai.suggest_category(text),
            "tags": re.findall(r"\b[a-zA-Z]{4,}\b", text)[:5],
            "recommendations": recommendations,
        })

    return app


if __name__ == "__main__":
    # Debug mode lets anyone who can reach the server run code, so it is off
    # by default and the server only listens on this machine.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    print("Smart Task Manager running at http://127.0.0.1:5000")
    create_app().run(host="127.0.0.1", port=5000, debug=debug)
