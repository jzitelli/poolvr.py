# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Three.js web frontend for the poolvr.py pool physics engine. Communicates with a Python FastAPI backend via REST API. No build tooling — vanilla ES modules served as static files.

## Development

### Running the Frontend

Start the backend API server (from repo root):
```bash
uvicorn poolvr.api:app --reload --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000` in a browser. The backend serves `index.html` and `js/` as static files.

No build step, bundler, or package manager is used. Edit JS files directly and refresh the browser.

### Dependencies

Three.js v0.170.0 loaded via ES module importmap from unpkg CDN (configured in `index.html`). No npm/node dependencies.

## Architecture

### Module Dependency Graph

```
main.js  (entry point)
├── api.js         — REST client for /api/v1/* endpoints
├── animation.js   — AnimationEngine: interpolates ball motion from physics events
├── controls.js    — AimingController: input state machine for aiming and striking
├── balls.js       — Ball mesh creation (colors, striped textures, shadows)
└── table.js       — Table geometry (surface, cushions, rails, pockets, legs)
```

### Backend API (`api.js`)

All endpoints are under `/api/v1`. Key operations:
- `POST /simulations` — create new simulation session
- `GET /simulations/{id}/table_geometry` — fetch table dimensions for 3D mesh construction
- `POST /simulations/{id}/strikes` — execute cue strike, returns physics events
- `GET /simulations/{id}/events` — query physics events with filters

### Animation System (`animation.js`)

`AnimationEngine` stores per-ball event sequences received from the backend. Each event contains polynomial coefficients for position (`a[0] + a[1]*τ + a[2]*τ²`) and angular velocity (`b[0] + b[1]*τ`). The engine uses binary search to find the active event at any time `t`, then evaluates the polynomials.

### Controls State Machine (`controls.js`)

`AimingController` manages user interaction through four states:

`IDLE` → click cue ball → `AIMING` → hold spacebar → `CHARGING` → release spacebar → `ANIMATING` → animation complete → `IDLE`

- **AIMING**: mouse left/right rotates aim direction; ESC cancels
- **CHARGING**: mouse up/down adjusts strike power (0.2–4.0); release spacebar fires
- Strike calls `api.strike()` with `{ball_index, cue_velocity, contact_point, cue_mass}`

### Ball Rotation (`main.js`)

Ball quaternions are integrated each frame from angular velocity using small-timestep quaternion updates:
```
q.w -= 0.5 * dt * dot(ω, q.xyz)
q.xyz += 0.5 * dt * (q.w * ω + cross(ω, q.xyz))
```

### Coordinate System

Y-axis is up. Table surface sits at height `H`. Ball positions are `[x, H + radius, z]` (Y is constant on the table surface). Physics events from the backend use `[x, z]` 2D coordinates which the frontend maps to 3D.
