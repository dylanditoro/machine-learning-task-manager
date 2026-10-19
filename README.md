# Smart Task Manager

A small Flask web app that stores tasks in SQLite, scores how urgent each one is, and estimates how long it will take. Estimates adjust over time using the durations of tasks you have already finished.

## What it does

- Create, list, update, complete and delete tasks through a web page or a JSON API.
- Scores priority from 0 to 10 using the wording of the task: urgent and important keywords, deadline phrases such as "by Friday", "tomorrow" or "within 6 hours", and how long the text is.
- Estimates duration in hours from keywords ("quick", "complex", "design", "update") and then scales the estimate by how long your completed tasks in the same category took compared with their estimates.
- Shows completion rate, tasks finished this week and the average estimate error.
- Suggests a category, tags and a few recommendations for a piece of text.

## How the scoring works

The scoring is rule-based, not a trained model. Keyword sets and fixed weights set the priority. The part that learns is small and deliberate: once at least three tasks in a category are completed with a recorded actual duration, new estimates in that category are multiplied by the average of actual divided by estimated hours, limited to between 0.5 and 2.0 so a few odd tasks cannot swing it.

A natural next step is replacing the keyword rules with a model trained on completed tasks, for example scikit-learn regression on TF-IDF features. The estimate and the actual duration are already stored for that purpose.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 smart_task_manager.py
```

Open http://127.0.0.1:5000. Tasks are saved in `tasks.db` in the current folder. Set `TASKS_DB` to use another file. The server only listens on your own machine and debug mode is off unless you set `FLASK_DEBUG=1`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/tasks?status=all&sort=priority_score` | List tasks. Status is `all`, `pending`, `in_progress` or `completed`. Sort is `priority_score`, `deadline`, `created_at` or `predicted_duration`. |
| POST | `/api/tasks` | Create a task. Needs `title`. Optional `description`, `category`, `deadline` (ISO date), `tags`. |
| PUT | `/api/tasks/<id>` | Update fields, or send `{"status": "completed", "actual_duration": 2.5}` to complete a task. |
| DELETE | `/api/tasks/<id>` | Delete a task. |
| GET | `/api/analytics` | Totals, completion rate, weekly count and estimate error. |
| POST | `/api/smart-suggestions` | Send `{"text": "..."}` for a category, tags and recommendations. |

Invalid input returns a JSON error with a 400 or 404 status.

## Tests

```bash
python3 -m pytest
```

The tests cover deadline parsing, scoring, the calibration step, every API route, rejection of bad sort values, and the page being served.

## Project layout

```
smart_task_manager.py   scoring logic and Flask app
static/index.html       web interface
tests/                  pytest suite
requirements.txt
```

## Limits

- Single user, no login. It is meant to run on your own machine.
- Keyword rules only understand English phrases of the forms listed above.
- Categories are fixed to general, work, personal, learning and maintenance in the form.
