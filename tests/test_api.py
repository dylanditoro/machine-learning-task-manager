import pytest

from smart_task_manager import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    app.config["TESTING"] = True
    return app.test_client()


def make_task(client, title="Write report", **extra):
    response = client.post("/api/tasks", json={"title": title, **extra})
    assert response.status_code == 201
    return response.get_json()


def test_create_and_list(client):
    task = make_task(client, "Urgent fix", description="asap")
    tasks = client.get("/api/tasks").get_json()
    assert [t["id"] for t in tasks] == [task["id"]]
    assert tasks[0]["priority_score"] == task["priority_score"]


def test_title_is_required(client):
    assert client.post("/api/tasks", json={"title": "  "}).status_code == 400
    assert client.post("/api/tasks", data="not json").status_code == 400


def test_bad_deadline_is_rejected(client):
    response = client.post("/api/tasks", json={"title": "x", "deadline": "next week"})
    assert response.status_code == 400


def test_sort_parameter_cannot_inject_sql(client):
    make_task(client)
    response = client.get("/api/tasks?sort=id; DROP TABLE tasks")
    assert response.status_code == 400
    assert len(client.get("/api/tasks").get_json()) == 1


def test_sorting_by_priority(client):
    low = make_task(client, "Tidy shelf")
    high = make_task(client, "Urgent critical fix", description="due tomorrow")
    tasks = client.get("/api/tasks?sort=priority_score").get_json()
    assert [t["id"] for t in tasks] == [high["id"], low["id"]]


def test_status_filter(client):
    done = make_task(client, "Done")
    make_task(client, "Open")
    client.put(f"/api/tasks/{done['id']}", json={"status": "completed", "actual_duration": 1})
    completed = client.get("/api/tasks?status=completed").get_json()
    assert [t["id"] for t in completed] == [done["id"]]
    assert client.get("/api/tasks?status=bogus").status_code == 400


def test_complete_update_and_delete(client):
    task = make_task(client)
    assert client.put(f"/api/tasks/{task['id']}", json={"title": "Renamed"}).status_code == 200
    assert client.get("/api/tasks").get_json()[0]["title"] == "Renamed"

    client.put(f"/api/tasks/{task['id']}", json={"status": "completed", "actual_duration": 2.5})
    assert client.get("/api/tasks").get_json()[0]["actual_duration"] == 2.5

    assert client.delete(f"/api/tasks/{task['id']}").status_code == 200
    assert client.get("/api/tasks").get_json() == []


def test_missing_task_returns_404(client):
    assert client.put("/api/tasks/999", json={"title": "x"}).status_code == 404
    assert client.delete("/api/tasks/999").status_code == 404


def test_negative_actual_duration_is_rejected(client):
    task = make_task(client)
    response = client.put(f"/api/tasks/{task['id']}", json={"status": "completed", "actual_duration": -1})
    assert response.status_code == 400


def test_estimates_adjust_after_three_completed_tasks(client):
    first = make_task(client, "Review notes", category="work")
    baseline = first["predicted_duration"]
    # Every task in this category takes twice the estimate.
    for _ in range(3):
        task = make_task(client, "Review notes", category="work")
        client.put(f"/api/tasks/{task['id']}", json={"status": "completed", "actual_duration": baseline * 2})

    adjusted = make_task(client, "Review notes", category="work")["predicted_duration"]
    assert adjusted == pytest.approx(baseline * 2, abs=0.05)

    # A different category has no history and keeps the raw estimate.
    other = make_task(client, "Review notes", category="personal")["predicted_duration"]
    assert other == baseline


def test_analytics(client):
    task = make_task(client)
    make_task(client, "Another")
    client.put(f"/api/tasks/{task['id']}", json={"status": "completed", "actual_duration": 1})
    stats = client.get("/api/analytics").get_json()
    assert stats["total_tasks"] == 2
    assert stats["completed_tasks"] == 1
    assert stats["completion_rate"] == 50.0
    assert stats["weekly_completed"] == 1


def test_smart_suggestions(client):
    body = client.post("/api/smart-suggestions", json={"text": "Urgent client report due tomorrow"}).get_json()
    assert body["category"] == "work"
    assert any("Urgent" in r for r in body["recommendations"])


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Smart Task Manager" in response.data
