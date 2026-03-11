import pytest
from fastapi.testclient import TestClient

from poolvr.api import app


@pytest.fixture
def client():
    return TestClient(app)


# --- POST /simulations ---

def test_create_simulation_defaults(client):
    resp = client.post("/api/v1/simulations", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert "sim_id" in data
    assert len(data["ball_positions"]) == 16
    assert len(data["pocket_positions"]) == 6
    assert data["t"] == 0.0


def test_create_simulation_custom_physics(client):
    resp = client.post("/api/v1/simulations", json={
        "physics": {"ball_collision_model": "fsimulated"}
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["physics"]["ball_collision_model"] == "fsimulated"


def test_create_simulation_invalid_model(client):
    resp = client.post("/api/v1/simulations", json={
        "physics": {"ball_collision_model": "nonexistent"}
    })
    assert resp.status_code == 400


# --- GET /simulations/{sim_id} ---

def test_get_simulation(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.get(f"/api/v1/simulations/{sim_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sim_id"] == sim_id
    assert data["num_events"] > 0  # initial rest events


def test_get_simulation_not_found(client):
    resp = client.get("/api/v1/simulations/nonexistent")
    assert resp.status_code == 404


# --- DELETE /simulations/{sim_id} ---

def test_delete_simulation(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.delete(f"/api/v1/simulations/{sim_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    # confirm gone
    resp = client.get(f"/api/v1/simulations/{sim_id}")
    assert resp.status_code == 404


# --- POST /simulations/{sim_id}/strikes ---

def test_strike_ball(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    ball_pos = resp.json()["ball_positions"][0]
    R = resp.json()["physics"]["ball_radius"]
    contact = [ball_pos[0], ball_pos[1] + R * 0.5, ball_pos[2] - R * 0.8]
    resp = client.post(f"/api/v1/simulations/{sim_id}/strikes", json={
        "ball_index": 0,
        "contact_point": contact,
        "cue_velocity": [0.0, 0.0, -1.5],
        "cue_mass": 0.54,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) > 0
    assert data["balls_at_rest_time"] is not None or data["balls_at_rest_time"] is None
    assert "event_summary" in data
    # check event structure
    first_event = data["events"][0]
    assert "type" in first_event
    assert "t" in first_event
    assert "index" in first_event


def test_strike_ball_not_found_sim(client):
    resp = client.post("/api/v1/simulations/nonexistent/strikes", json={
        "ball_index": 0,
        "contact_point": [0.0, 0.77, 0.66],
        "cue_velocity": [0.0, 0.0, -1.0],
        "cue_mass": 0.54,
    })
    assert resp.status_code == 404


# --- GET /simulations/{sim_id}/state ---

def test_get_state(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.get(f"/api/v1/simulations/{sim_id}/state", params={"t": 0.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["t"] == 0.0
    assert "0" in data["balls"]
    assert "position" in data["balls"]["0"]
    assert "velocity" in data["balls"]["0"]


def test_get_state_specific_balls(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.get(f"/api/v1/simulations/{sim_id}/state",
                      params={"t": 0.0, "balls": "0,1"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["balls"].keys()) == {"0", "1"}


def test_get_state_with_energy(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.get(f"/api/v1/simulations/{sim_id}/state",
                      params={"t": 0.0, "include": "positions,energy"})
    assert resp.status_code == 200
    data = resp.json()
    assert "energy" in data
    assert data["energy"] == 0.0  # all balls at rest initially


# --- GET /simulations/{sim_id}/events ---

def test_list_events(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.get(f"/api/v1/simulations/{sim_id}/events")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 16  # at least one rest event per ball
    assert len(data["events"]) <= data["total"]


def test_list_events_filter_type(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    resp = client.get(f"/api/v1/simulations/{sim_id}/events",
                      params={"type": "BallRestEvent"})
    assert resp.status_code == 200
    data = resp.json()
    for e in data["events"]:
        assert e["type"] == "BallRestEvent"


# --- POST /simulations/{sim_id}/reset ---

def test_reset(client):
    resp = client.post("/api/v1/simulations", json={})
    sim_id = resp.json()["sim_id"]
    original_positions = resp.json()["ball_positions"]
    # strike to change state
    R = resp.json()["physics"]["ball_radius"]
    ball_pos = original_positions[0]
    contact = [ball_pos[0], ball_pos[1] + R * 0.5, ball_pos[2] - R * 0.8]
    client.post(f"/api/v1/simulations/{sim_id}/strikes", json={
        "ball_index": 0,
        "contact_point": contact,
        "cue_velocity": [0.0, 0.0, -1.5],
        "cue_mass": 0.54,
    })
    # reset
    resp = client.post(f"/api/v1/simulations/{sim_id}/reset", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["t"] == 0.0


# --- POST /oneshot/strike ---

def test_oneshot_strike(client):
    resp = client.post("/api/v1/oneshot/strike", json={
        "strike": {
            "ball_index": 0,
            "contact_point": [0.0, 0.7692, 0.66125],
            "cue_velocity": [-0.01, 0.0, -1.6],
            "cue_mass": 0.54,
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) > 0
    assert "initial_ball_positions" in data
    assert "final_ball_positions" in data
    assert data["balls_at_rest_time"] is not None
    assert "event_summary" in data


def test_oneshot_with_fsimulated(client):
    resp = client.post("/api/v1/oneshot/strike", json={
        "physics": {"ball_collision_model": "fsimulated"},
        "strike": {
            "ball_index": 0,
            "contact_point": [0.0, 0.7692, 0.66125],
            "cue_velocity": [-0.01, 0.0, -1.6],
            "cue_mass": 0.54,
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) > 10  # break shot should have many events


# --- Multi-shot session test ---

def test_multi_shot_session(client):
    # 1. create
    resp = client.post("/api/v1/simulations", json={})
    assert resp.status_code == 201
    sim_id = resp.json()["sim_id"]
    ball_pos = resp.json()["ball_positions"][0]
    R = resp.json()["physics"]["ball_radius"]
    # 2. first strike (simple shot)
    contact = [ball_pos[0], ball_pos[1] + R * 0.5, ball_pos[2] - R * 0.8]
    resp = client.post(f"/api/v1/simulations/{sim_id}/strikes", json={
        "ball_index": 0,
        "contact_point": contact,
        "cue_velocity": [0.0, 0.0, -1.0],
        "cue_mass": 0.54,
    })
    assert resp.status_code == 200
    rest_time = resp.json()["balls_at_rest_time"]
    # 3. query mid-shot state
    if rest_time and rest_time > 0.1:
        resp = client.get(f"/api/v1/simulations/{sim_id}/state",
                          params={"t": 0.05})
        assert resp.status_code == 200
    # 4. delete
    resp = client.delete(f"/api/v1/simulations/{sim_id}")
    assert resp.status_code == 200
