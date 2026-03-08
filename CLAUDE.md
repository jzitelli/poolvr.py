# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

poolvr.py is a VR pool/billiards simulator written in Python. It features an event-based physics engine based on the paper "An Event-Based Pool Physics Simulator" by Leckie & Greenspan.

## Build and Development Commands

### Installation
```bash
python setup.py install
```

### Running the Application
```bash
# VR mode (requires pyopenvr)
poolvr

# Non-VR mode
poolvr --novr

# See all options
poolvr -h
```

### Running Tests
```bash
# Run all tests from the test directory
cd test && pytest

# Run a single test
cd test && pytest test_physics.py::test_strike_ball

# Run with a specific collision model
cd test && pytest --collision-model=fsimulated

# Generate and save plots
cd test && pytest --save-plots

# Run with OpenGL rendering
cd test && pytest --render
```

### Building Fortran Extensions
The physics engine uses Fortran for performance-critical collision calculations. Shared libraries (`_collisions.so`, `_poly_solvers.so`) are built from `.f90` files in `poolvr/physics/`.

## Architecture

### Core Components

**Physics Engine** (`poolvr/physics/`):
- `__init__.py` - `PoolPhysics` class: main event-based physics simulator that manages ball states and determines collision sequences
- `events.py` - Event hierarchy: `PhysicsEvent` base class with specialized events (`BallSlidingEvent`, `BallRollingEvent`, `BallSpinningEvent`, `BallRestEvent`, `BallCollisionEvent`, `SegmentCollisionEvent`, `CornerCollisionEvent`)
- `collisions.py` - Ball-ball collision model (Python wrapper + Fortran implementation in `collisions.f90`)
- `poly_solvers.py` - Quartic/polynomial solvers for collision time calculation (Fortran in `poly_solvers.f90`)

**Ball Collision Models** (configured via `--collision-model`):
- `simple` - SimpleBallCollisionEvent
- `simulated` - SimulatedBallCollisionEvent (Python implementation)
- `fsimulated` - FSimulatedBallCollisionEvent (Fortran implementation)

**Game Layer** (`poolvr/`):
- `game.py` - `PoolGame`: high-level game state, ball positions/velocities/quaternions
- `table.py` - `PoolTable`: table geometry, pocket positions, cushion boundaries
- `app.py` / `glfw_app.py` - Main application loop with GLFW window management
- `__main__.py` - CLI entry point with argument parsing

**Rendering** (`poolvr/`):
- `gl_rendering.py` - OpenGL mesh/material/renderer abstractions
- `gl_primitives.py` - Basic geometric primitives
- `gl_techniques.py` - Shader techniques (Lambert, EGA, etc.)
- `pyopenvr_renderer.py` - OpenVR integration for VR headsets
- `shaders/` - GLSL vertex/fragment shaders

### Physics Event Flow

1. `CueStrikeEvent` initiates ball motion
2. Ball transitions through motion states: `BallSlidingEvent` -> `BallRollingEvent` -> `BallRestEvent` (or `BallSpinningEvent`)
3. `PoolPhysics._determine_next_event()` finds earliest collision or state transition
4. Collision events (`BallCollisionEvent`, `SegmentCollisionEvent`, `CornerCollisionEvent`) spawn child motion events

### Test Fixtures

Tests use pytest fixtures defined in `test/conftest.py`:
- `pool_physics` - Parametrized across collision models (simple, simulated, fsimulated)
- `pool_table` - Standard pool table geometry
- `plot_energy`, `plot_motion_timelapse` - Enable with `--save-plots` or `--show-plots`
- `gl_rendering` - Enable with `--render` or `--vr`
