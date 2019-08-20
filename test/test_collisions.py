import logging
_logger = logging.getLogger(__name__)
import numpy as np


from poolvr.physics.collisions import collide_balls


_k = np.array([0.0, 0.0, 1.0])
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi


def test_collide_balls():
    e = 0.9
    mu_s = 0.21
    mu_b = 0.06
    M = 0.1406
    R = 1.125 * 0.0254
    r_i = np.zeros(3)
    rd = np.array([0.0, 1.0, 0.0])
    r_j = r_i + 2 * R * rd
    r_ij = r_j - r_i
    r_ij_mag = np.sqrt(np.dot(r_ij, r_ij))
    y_loc = r_ij / r_ij_mag
    z_loc = _k
    x_loc = np.cross(y_loc, z_loc)
    r_c = r_i + R * y_loc
    v_j = np.zeros(3, dtype=np.float64)
    omega_j = np.zeros(3, dtype=np.float64)
    for cue_ball_velocity, topspin, cut_angle in [
        (1.539, 58.63, 33.83),
        (1.032, 39.31, 26.36),
        (1.364, 51.96, 40.52),
        (1.731, 65.94, 46.5),
        (0.942, 35.89, 18.05)
    ]:
        c, s = np.cos(cut_angle*DEG2RAD), np.sin(cut_angle*DEG2RAD)
        v_i = np.array([cue_ball_velocity*s, cue_ball_velocity*c, 0.0])
        omega_i = np.array([-topspin*DEG2RAD, 0.0, 0.0])
        v_i1, omega_i1, v_j1, omega_j1 = collide_balls(r_c,
                                                       r_i, v_i, omega_i,
                                                       r_j, v_j, omega_j,
                                                       e, mu_s, mu_b,
                                                       M, R,
                                                       9.81,
                                                       4000)
        # convert back to local frame:
        v_i  = np.array([np.dot( v_i, x_loc), np.dot( v_i, y_loc),  v_i[2]])
        v_il = np.array([np.dot(v_i1, x_loc), np.dot(v_i1, y_loc), v_i1[2]])
        v_j  = np.array([np.dot( v_j, x_loc), np.dot( v_j, y_loc),  v_j[2]])
        v_jl = np.array([np.dot(v_j1, x_loc), np.dot(v_j1, y_loc), v_j1[2]])
        omega_i1 = np.array([np.dot(omega_i1, x_loc), np.dot(omega_i1, y_loc), omega_i1[2]])
        omega_j1 = np.array([np.dot(omega_j1, x_loc), np.dot(omega_j1, y_loc), omega_j1[2]])
        from poolvr.physics.events import PhysicsEvent, BallSlidingEvent
        PhysicsEvent.ball_radius = R
        PhysicsEvent.ball_diameter = 2*R
        PhysicsEvent.ball_mass = M
        PhysicsEvent.ball_I = 2.0/5 * PhysicsEvent.ball_mass * PhysicsEvent.ball_radius**2
        PhysicsEvent.mu_s = mu_s
        PhysicsEvent.mu_b = mu_b
        e_i, e_j = BallSlidingEvent(0.0, 0, r_i, v_i1, omega_i1), BallSlidingEvent(0.0, 1, r_j, v_j1, omega_j1)
        v_is = e_i.next_motion_event.eval_velocity(0.0)
        v_js = e_j.next_motion_event.eval_velocity(0.0)
        _logger.info('''
        v_is = %s
        v_js = %s
        |v_is| = %s
        |v_js| = %s
        arctan2(*v_is[::-2]) = %s
        arctan2(*v_js[::-2]) = %s
        ''',
                     v_is, v_js,
                     np.sqrt(np.dot(v_is, v_is)), np.sqrt(np.dot(v_js, v_js)),
                     np.arctan2(*v_is[:2])*RAD2DEG, np.arctan2(*v_js[:2])*RAD2DEG)
        _logger.info('''
        v_i  = %s
        v_i1 = %s
        v_j  = %s
        v_j1 = %s
        |v_i1| = %s
        |v_j1| = %s
        ''',
                     v_i, v_i1, v_j, v_j1,
                     np.sqrt(np.dot(v_i1, v_i1)), np.sqrt(np.dot(v_j1, v_j1)))




def test_half_ball_shot():
    e = 0.9
    mu_s = 0.2
    mu_b = 0.06
    M = 0.17
    R = 1.125 * 0.0254
    r_i = np.array([0.0, 0.0, 0.0])
    rd = np.array([0.0, 1.0, 0.0])
    r_j = r_i + 2 * R * rd
    r_ij = r_j - r_i
    r_ij_mag = np.sqrt(np.dot(r_ij, r_ij))
    y_loc = r_ij / r_ij_mag
    z_loc = _k
    x_loc = np.cross(y_loc, z_loc)
    r_c = r_i + R * y_loc
    v_i = 2 * (0.5*rd + 0.5*np.sqrt(3)*x_loc)
    v_j = np.zeros(3, dtype=np.float64)
    omega_i = np.zeros(3, dtype=np.float64)
    omega_i[2] = 20.0
    omega_j = np.zeros(3, dtype=np.float64)
    v_i1, omega_i1, v_j1, omega_j1 = collide_balls(r_c,
                                                   r_i, v_i, omega_i,
                                                   r_j, v_j, omega_j,
                                                   e, mu_s, mu_b,
                                                   M, R,
                                                   9.81,
                                                   4000)
    v_il = np.array([np.dot(v_i, x_loc), np.dot(v_i, y_loc), 0])
    v_jl = np.array([np.dot(v_j, x_loc), np.dot(v_j, y_loc), 0])
    v_i1l = np.array([np.dot(v_i1, x_loc), np.dot(v_i1, y_loc), 0])
    v_j1l = np.array([np.dot(v_j1, x_loc), np.dot(v_j1, y_loc), 0])
    v_ijy = np.dot(v_j, y_loc) - np.dot(v_i, y_loc)
    v_ijy1 = np.dot(v_j1, y_loc) - np.dot(v_i1, y_loc)
    _logger.info('''
v_i  = %s
v_i1 = %s
v_j  = %s
v_j1 = %s

(local frame):
v_i  = %s
v_i1 = %s
v_j  = %s
v_j1 = %s

v_ijy  = %s
v_ijy1 = %s

     |v_ijy1 / v_ijy|  = %s
sqrt(|v_ijy1 / v_ijy|) = %s
''',
                  v_i, v_i1, v_j, v_j1,
                  v_il, v_i1l, v_jl, v_j1l,
                  v_ijy, v_ijy1,
                  v_ijy1 / v_ijy,
                  np.sqrt(-v_ijy1 / v_ijy))


def test_A():
    e = 0.89
    mu_s = 0.21
    mu_b = 0.05
    M = 0.1406
    # R = 1.125 * 0.0254
    R = 0.02625
    top_spin = 58.63
    cut_angle = 33.83 * DEG2RAD
    r_i = np.zeros(3)
    v_i = np.array([0.0, 1.539, 0.0])
    omega_i = np.array([-top_spin, 0.0, 0.0])
    rd = y_loc = np.array([-np.sin(cut_angle), np.cos(cut_angle), 0.0])
    x_loc = np.cross(y_loc, _k)
    r_c = r_i + R*rd
    r_j = r_i + 2*R*rd
    v_j = np.zeros(3)
    omega_j = np.zeros(3)
    v_i1, omega_i1, v_j1, omega_j1 = collide_balls(r_c,
                                                   r_i, v_i, omega_i,
                                                   r_j, v_j, omega_j,
                                                   e, mu_s, mu_b,
                                                   M, R,
                                                   9.81,
                                                   10000)
    # u_i1 = v_i1 + np.cross(_k, omega_i1)
    # u_j1 = v_j1 + np.cross(_k, omega_j1)
    from poolvr.physics.events import BallSlidingEvent
    _r_i = r_i.copy(); _r_i[1], _r_i[2] = _r_i[2], _r_i[1]
    _v_i = v_i1.copy(); _v_i[1], _v_i[2] = _v_i[2], _v_i[1]
    _omega_i = omega_i1.copy(); _omega_i[1], _omega_i[2] = _omega_i[2], _omega_i[1]
    e_i = BallSlidingEvent(0, 0, _r_i, _v_i, _omega_i)
    v_is = e_i.eval_velocity(e_i.T)

    _r_j = r_j.copy(); _r_j[1], _r_j[2] = _r_j[2], _r_j[1]
    _v_j = v_j1.copy(); _v_j[1], _v_j[2] = _v_j[2], _v_j[1]
    _omega_j = omega_j1.copy(); _omega_j[1], _omega_j[2] = _omega_j[2], _omega_j[1]
    e_j = BallSlidingEvent(0, 1, r_j, v_j1, omega_j1)
    v_js = e_j.eval_velocity(e_j.T)

    v_ijy = np.dot(v_j, y_loc) - np.dot(v_i, y_loc)
    v_ijy1 = np.dot(v_j1, y_loc) - np.dot(v_i1, y_loc)
    v_il = np.array([np.dot(v_i, x_loc), np.dot(v_i, y_loc), 0])
    v_jl = np.array([np.dot(v_j, x_loc), np.dot(v_j, y_loc), 0])
    v_i1l = np.array([np.dot(v_i1, x_loc), np.dot(v_i1, y_loc), 0])
    v_j1l = np.array([np.dot(v_j1, x_loc), np.dot(v_j1, y_loc), 0])
    _logger.info('''
v_i  = %s
v_i1 = %s
v_j  = %s
v_j1 = %s

(local frame):
v_i  = %s
v_i1 = %s
v_j  = %s
v_j1 = %s

|v_i|  = %s
|v_i1| = %s
|v_j|  = %s
|v_j1| = %s

|v_is| = %s
|v_js| = %s

v_ijy  = %s
v_ijy1 = %s

     |v_ijy1 / v_ijy|  = %s
sqrt(|v_ijy1 / v_ijy|) = %s
''',
                 v_i, v_i1, v_j, v_j1,
                 v_il, v_i1l, v_jl, v_j1l,
                 np.linalg.norm(v_i), np.linalg.norm(v_i1),
                 np.linalg.norm(v_j), np.linalg.norm(v_j1),
                 np.linalg.norm(v_is), np.linalg.norm(v_js),
                 v_ijy, v_ijy1,
                 v_ijy1 / v_ijy,
                 np.sqrt(-v_ijy1 / v_ijy))
