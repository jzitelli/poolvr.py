"""
REST API for the poolvr.py event-based pool physics engine.

Run with:
    uvicorn poolvr.api:app
"""
import os
import uuid
import threading
from collections import Counter
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .table import PoolTable
from .physics import PoolPhysics, BALL_COLLISION_MODELS
from .physics.events import (
    BallRestEvent, BallSpinningEvent,
    BallCollisionEvent, SegmentCollisionEvent, CornerCollisionEvent,
)

app = FastAPI(title="poolvr.py API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _get_session(sim_id: str) -> dict:
    with _sessions_lock:
        session = _sessions.get(sim_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "SIMULATION_NOT_FOUND",
                "message": f"Simulation {sim_id} not found",
            }
        })
    return session


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TableConfig(BaseModel):
    L: float = 2.54
    W: Optional[float] = None
    H: float = 0.74295
    ball_radius: float = 0.028575
    num_balls: int = 16


class PhysicsConfig(BaseModel):
    ball_mass: float = 0.1406
    ball_radius: float = 0.02625
    mu_r: float = 0.016
    mu_sp: float = 0.044
    mu_s: float = 0.21
    mu_b: float = 0.05
    e: float = 0.89
    g: float = 9.81
    ball_collision_model: str = "simple"
    use_quartic_solver: bool = True


class CreateSimulationRequest(BaseModel):
    table: Optional[TableConfig] = None
    physics: Optional[PhysicsConfig] = None
    balls_on_table: Optional[list[int]] = None
    ball_positions: Optional[list[list[float]]] = None


class StrikeRequest(BaseModel):
    t: Optional[float] = None
    ball_index: int
    ball_position: Optional[list[float]] = None
    contact_point: list[float]
    cue_velocity: list[float]
    cue_mass: float


class ResetRequest(BaseModel):
    ball_positions: Optional[list[list[float]]] = None
    balls_on_table: Optional[list[int]] = None


class OneshotRequest(BaseModel):
    table: Optional[TableConfig] = None
    physics: Optional[PhysicsConfig] = None
    balls_on_table: Optional[list[int]] = None
    ball_positions: Optional[list[list[float]]] = None
    strike: StrikeRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_physics(req_table: Optional[TableConfig],
                   req_physics: Optional[PhysicsConfig],
                   balls_on_table: Optional[list[int]],
                   ball_positions: Optional[list[list[float]]]):
    table_cfg = req_table or TableConfig()
    phys_cfg = req_physics or PhysicsConfig()
    if phys_cfg.ball_collision_model not in BALL_COLLISION_MODELS:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_PARAMETER",
                "message": f"Unknown collision model: {phys_cfg.ball_collision_model}",
                "details": {"field": "physics.ball_collision_model",
                            "value": phys_cfg.ball_collision_model},
            }
        })
    table_kwargs = table_cfg.model_dump()
    table = PoolTable(**table_kwargs)
    bp = np.array(ball_positions, dtype=np.float64) if ball_positions else None
    physics = PoolPhysics(
        num_balls=table_cfg.num_balls,
        ball_mass=phys_cfg.ball_mass,
        ball_radius=phys_cfg.ball_radius,
        mu_r=phys_cfg.mu_r,
        mu_sp=phys_cfg.mu_sp,
        mu_s=phys_cfg.mu_s,
        mu_b=phys_cfg.mu_b,
        e=phys_cfg.e,
        g=phys_cfg.g,
        ball_collision_model=phys_cfg.ball_collision_model,
        use_quartic_solver=phys_cfg.use_quartic_solver,
        table=table,
        balls_on_table=balls_on_table,
        ball_positions=bp,
    )
    return physics, table, table_cfg, phys_cfg


def _serialize_events(events, start_index=0):
    result = []
    for idx, e in enumerate(events):
        d = e.to_dict()
        d['index'] = start_index + idx
        result.append(d)
    return result


def _event_summary(events):
    return dict(Counter(e.__class__.__name__ for e in events))


def _do_strike(physics: PoolPhysics, req: StrikeRequest):
    i = req.ball_index
    if not physics._on_table[i]:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "BALL_NOT_ON_TABLE",
                "message": f"Ball {i} is not on the table",
            }
        })
    t = req.t
    if t is None:
        t = physics.balls_at_rest_time
        if t is None:
            # balls still in motion — use time of last event
            t = physics.events[-1].t if physics.events else 0.0
    r_i = np.array(req.ball_position, dtype=np.float64) if req.ball_position else \
        physics.eval_positions(t, balls=[i])[0]
    r_c = np.array(req.contact_point, dtype=np.float64)
    V = np.array(req.cue_velocity, dtype=np.float64)
    M = req.cue_mass
    num_before = len(physics.events)
    physics.strike_ball(t, i, r_i, r_c, V, M)
    new_events = physics.events[num_before:]
    return new_events, t


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(os.path.join(_frontend_dir, "index.html")) as f:
        return f.read()


@app.post("/api/v1/simulations", status_code=201)
def create_simulation(req: CreateSimulationRequest = None):
    if req is None:
        req = CreateSimulationRequest()
    physics, table, table_cfg, phys_cfg = _build_physics(
        req.table, req.physics, req.balls_on_table, req.ball_positions)
    sim_id = str(uuid.uuid4())
    session = {
        "physics": physics,
        "table": table,
        "table_config": table_cfg.model_dump(),
        "physics_config": phys_cfg.model_dump(),
        "lock": threading.Lock(),
    }
    with _sessions_lock:
        _sessions[sim_id] = session
    positions = physics.eval_positions(0.0).tolist()
    return {
        "sim_id": sim_id,
        "table": session["table_config"],
        "physics": session["physics_config"],
        "ball_positions": positions,
        "pocket_positions": table.pocket_positions.tolist(),
        "t": 0.0,
    }


@app.get("/api/v1/simulations/{sim_id}")
def get_simulation(sim_id: str):
    session = _get_session(sim_id)
    physics = session["physics"]
    return {
        "sim_id": sim_id,
        "table": session["table_config"],
        "physics": session["physics_config"],
        "balls_on_table": physics.balls_on_table.tolist(),
        "balls_at_rest_time": physics.balls_at_rest_time,
        "num_events": len(physics.events),
    }


@app.get("/api/v1/simulations/{sim_id}/table_geometry")
def get_table_geometry(sim_id: str):
    session = _get_session(sim_id)
    table = session["table"]
    return {
        "corners": table._corners.tolist(),
        "pocket_positions": table.pocket_positions.tolist(),
        "L": table.L,
        "W": table.W,
        "H": table.H,
        "w": table.w,
        "h": table.h,
        "ball_radius": table.ball_radius,
        "width_rail": table.width_rail,
        "R_c": table.R_c,
        "R_s": table.R_s,
    }


@app.delete("/api/v1/simulations/{sim_id}")
def delete_simulation(sim_id: str):
    with _sessions_lock:
        if sim_id not in _sessions:
            raise HTTPException(status_code=404, detail={
                "error": {
                    "code": "SIMULATION_NOT_FOUND",
                    "message": f"Simulation {sim_id} not found",
                }
            })
        del _sessions[sim_id]
    return {"status": "deleted", "sim_id": sim_id}


@app.post("/api/v1/simulations/{sim_id}/strikes")
def strike_ball(sim_id: str, req: StrikeRequest):
    session = _get_session(sim_id)
    physics = session["physics"]
    with session["lock"]:
        event_offset = len(physics.events)
        new_events, t_strike = _do_strike(physics, req)
    return {
        "events": _serialize_events(new_events, start_index=event_offset),
        "balls_at_rest_time": physics.balls_at_rest_time,
        "event_summary": _event_summary(new_events),
    }


@app.get("/api/v1/simulations/{sim_id}/events")
def list_events(sim_id: str,
                t_min: Optional[float] = None,
                t_max: Optional[float] = None,
                ball_index: Optional[int] = None,
                type: Optional[str] = None,
                offset: int = 0,
                limit: int = 1000):
    session = _get_session(sim_id)
    physics = session["physics"]
    events = physics.events
    # filter
    filtered = []
    for e in events:
        if t_min is not None and e.t < t_min:
            continue
        if t_max is not None and e.t > t_max:
            continue
        if type is not None and e.__class__.__name__ != type:
            continue
        if ball_index is not None:
            if isinstance(e, BallCollisionEvent):
                if e.i != ball_index and e.j != ball_index:
                    continue
            elif hasattr(e, 'i'):
                if e.i != ball_index:
                    continue
            else:
                continue
        filtered.append(e)
    total = len(filtered)
    page = filtered[offset:offset + limit]
    return {
        "events": _serialize_events(page, start_index=offset),
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/v1/simulations/{sim_id}/state")
def get_state(sim_id: str,
              t: float,
              balls: Optional[str] = None,
              include: Optional[str] = None):
    session = _get_session(sim_id)
    physics = session["physics"]
    if balls is not None:
        ball_indices = [int(b) for b in balls.split(",")]
    else:
        ball_indices = list(physics.balls_on_table)
    includes = set((include or "positions,velocities,angular_velocities").split(","))
    result: dict = {"t": t, "balls": {}}
    if "positions" in includes:
        positions = physics.eval_positions(t, balls=ball_indices)
    if "velocities" in includes:
        velocities = physics.eval_velocities(t, balls=ball_indices)
    if "angular_velocities" in includes:
        angular_velocities = physics.eval_angular_velocities(t, balls=ball_indices)
    if "energy" in includes:
        energy = float(physics.eval_energy(t, balls=ball_indices))
        result["energy"] = energy
    if "active_events" in includes:
        active = physics.find_active_events(t, balls=ball_indices)
        result["active_events"] = _serialize_events(active)
    for ii, i in enumerate(ball_indices):
        ball_data: dict = {}
        if "positions" in includes:
            ball_data["position"] = positions[ii].tolist()
        if "velocities" in includes:
            ball_data["velocity"] = velocities[ii].tolist()
        if "angular_velocities" in includes:
            ball_data["angular_velocity"] = angular_velocities[ii].tolist()
        result["balls"][str(i)] = ball_data
    return result


@app.post("/api/v1/simulations/{sim_id}/reset")
def reset_simulation(sim_id: str, req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    session = _get_session(sim_id)
    physics = session["physics"]
    bp = np.array(req.ball_positions, dtype=np.float64) if req.ball_positions else None
    with session["lock"]:
        physics.reset(ball_positions=bp, balls_on_table=req.balls_on_table)
    positions = physics.eval_positions(0.0).tolist()
    return {
        "ball_positions": positions,
        "balls_on_table": physics.balls_on_table.tolist(),
        "t": 0.0,
    }


@app.post("/api/v1/oneshot/strike")
def oneshot_strike(req: OneshotRequest):
    physics, table, table_cfg, phys_cfg = _build_physics(
        req.table, req.physics, req.balls_on_table, req.ball_positions)
    initial_positions = physics.eval_positions(0.0).tolist()
    new_events, t_strike = _do_strike(physics, req.strike)
    rest_time = physics.balls_at_rest_time
    final_positions = physics.eval_positions(rest_time).tolist() if rest_time is not None else None
    return {
        "events": _serialize_events(new_events),
        "balls_at_rest_time": rest_time,
        "event_summary": _event_summary(new_events),
        "initial_ball_positions": initial_positions,
        "final_ball_positions": final_positions,
    }


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")
