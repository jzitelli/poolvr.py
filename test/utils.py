import logging
import os
from os import makedirs
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from poolvr.table import PoolTable
from poolvr.physics.events import (CueStrikeEvent,
                                   BallEvent,
                                   BallSlidingEvent, BallRollingEvent, BallSpinningEvent,
                                   BallMotionEvent, BallRestEvent,
                                   RailCollisionEvent, CornerCollisionEvent, BallCollisionEvent,
                                   MarlowBallCollisionEvent, SimpleBallCollisionEvent,
                                   SimulatedBallCollisionEvent, FSimulatedBallCollisionEvent)


_logger = logging.getLogger(__name__)


EVENT_COLORS = {CueStrikeEvent: 'blue',
                BallSlidingEvent: 'orange',
                BallRollingEvent: 'green',
                BallSpinningEvent: 'red',
                BallRestEvent: 'red',
                BallCollisionEvent: 'blue',
                MarlowBallCollisionEvent: 'blue',
                SimpleBallCollisionEvent: 'blue',
                SimulatedBallCollisionEvent: 'blue',
                FSimulatedBallCollisionEvent: 'blue',
                RailCollisionEvent: 'green',
                CornerCollisionEvent: 'orange'}
BALL_COLORS = {0: 'white',
               1: 'yellow',
               2: 'blue',
               3: 'red',
               4: 'purple',
               5: 'orange',
               6: 'green',
               7: 'magenta',
               8: 'black'}
BALL_COLORS.update({i: BALL_COLORS[i-8] for i in range(9, 16)})


def catches_tcl_errors(func):
    from tkinter import TclError
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TclError as err:
            _logger.warn(err)
    return wrapper


def sorrted(roots):
    from poolvr.physics.poly_solvers import sort_complex_conjugate_pairs
    roots.sort()
    sort_complex_conjugate_pairs(roots)
    return roots


def printit(roots):
    return ',  '.join('%5.10f + %5.10fj' % (r.real, r.imag) for r in roots)


@catches_tcl_errors
def plot_ball_motion(i, physics,
                     table=None,
                     title=None,
                     t_0=None, t_1=None, nt=1000,
                     coords=(0,),
                     event_markers=True,
                     collision_depth=0,
                     collision_markers=True,
                     hold=False,
                     filename=None, show=False,
                     dpi=400,
                     figure=None):
    if table is None:
        table = PoolTable()
    if type(coords) == int:
        coords = (coords,)
    if title is None:
        title = 'ball %d position vs time'
    events = physics.ball_events[i]
    if events:
        if t_0 is None:
            t_0 = events[0].t
        else:
            events = [e for e in events if t_0 <= e.t + e.T]
        if t_1 is None:
            t_1 = events[-1].t
            if events[-1].T < float('inf'):
                t_1 += events[-1].T
            # t_1 = min(20.0, events[-1].t + events[-1].T)
    events = [e for e in events if e.t <= t_1]
    if figure is None:
        figure = plt.figure()
    if not hold:
        plt.title(title)
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$%s$ (meters)' % ' / '.join('xyz'[coord] for coord in coords))
        #plt.ylim(-0.5*table.length, 0.5*table.length)
    linewidth = 5 - 2*collision_depth
    if event_markers:
        for e in events:
            plt.axvline(e.t, color=EVENT_COLORS[type(e)], linewidth=linewidth)
    for i_e, e in enumerate(events):
        if isinstance(e.parent_event, BallCollisionEvent):
            parent = e.parent_event
            if collision_depth > 0:
                e_i, e_j = parent.child_events
                other_ball_event = e_j if parent.i == e.i else e_i
                plot_ball_motion(other_ball_event.i, physics, table=table,
                                 t_0=other_ball_event.t,# t_1=t_1,
                                 coords=coords,
                                 collision_depth=collision_depth-1,
                                 hold=True, event_markers=False, collision_markers=False,
                                 figure=figure)
    if events:
        ts = np.linspace(max(t_0, events[0].t),
                         min(t_1, events[-1].t + events[-1].T), nt)
        for coord in coords:
            plt.plot(ts, [physics.eval_positions(t)[i,coord] for t in ts],
                     ['-', '-.', '--'][coord], color=BALL_COLORS[i],
                     label='ball %d (%s)' % (i, 'xyz'[coord]),
                     linewidth=linewidth)
    if not hold:
        # if collision_markers:
        #     for i_e, e in enumerate(events):
        #         if isinstance(e.parent_event, BallCollisionEvent):
        #             parent = e.parent_event
        #             for child in parent.child_events:
        #                 r = child.eval_position(0)
        #                 plt.gcf().gca().add_patch(plt.Circle((child.t, r[0]), physics.ball_radius, color=BALL_COLORS[child.i]))
        #                 plt.gcf().gca().add_patch(plt.Circle((child.t, r[2]), physics.ball_radius, color=BALL_COLORS[child.i]))
        plt.legend()
        if filename:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            try:
                plt.savefig(filename, dpi=300)
                _logger.info('...saved figure to %s', filename)
            except Exception as err:
                _logger.warning('error saving figure:\n%s', err)
        if show:
            plt.show()
        plt.close()


@catches_tcl_errors
def plot_motion_timelapse(physics, table=None,
                          title=None,
                          nt=200,
                          t_0=None, t_1=None,
                          filename=None,
                          show=False,
                          figure=None):
    from itertools import chain
    if table is None:
        table = PoolTable()
    if title is None:
        title = "ball position timelapse"
    events = sorted(chain.from_iterable(physics.ball_events.values()))
    if not events:
        return
    if t_0 is None:
        t_0 = events[0].t
    else:
        events = [e for e in events if t_0 < e.t + e.T]
    if t_1 is None:
        t_1 = events[-1].t
        if events[-1].T < float('inf'):
            t_1 += events[-1].T
    events = [e for e in events if e.t <= t_1]
    if figure is None:
        figure = plt.figure()
    plt.title(title, fontsize='xx-small')
    plt.xticks([]); plt.yticks([])
    ball_colors = dict(BALL_COLORS)
    ball_colors[0] = 'white'
    ts = np.linspace(t_0, t_1, nt)
    positions_ts = np.array([physics.eval_positions(t) for t in ts])
    bot = np.array(list(physics.balls_on_table), dtype=np.int32)
    xlim = -0.5*table.W, 0.5*table.W
    ylim = -0.5*table.L, 0.5*table.L
    if positions_ts.size:
        positions_ts[:,bot,2] *= -1
        xlim = min(xlim[0], *positions_ts[:,bot,0].ravel()), max(xlim[1], *positions_ts[:,bot,0].ravel())
        ylim = min(ylim[0], *positions_ts[:,bot,2].ravel()), max(ylim[1], *positions_ts[:,bot,2].ravel())
    plt.gca().set_xlim(*xlim)
    plt.gca().set_ylim(*ylim)
    plt.gca().set_aspect('equal')
    plt.gca().add_patch(plt.Rectangle((xlim[0], ylim[0]),
                                      xlim[1]-xlim[0], ylim[1]-ylim[0],
                                      color='#141414'))
    plt.gca().add_patch(plt.Rectangle((-0.5*table.W, -0.5*table.L),
                                      table.W, table.L,
                                      color='#013216'))
    for i_t, t in enumerate(ts):
        positions = positions_ts[i_t]
        for i in physics.balls_on_table:
            plt.gca().add_patch(plt.Circle(positions[i,::2], physics.ball_radius,
                                           color=ball_colors[i],
                                           alpha=13/nt,
                                           linewidth=0.001,
                                           antialiased=True))
    for i in physics.balls_on_table:
        start_event, end_event = physics.ball_events[i][0], physics.ball_events[i][-1]
        if nt == 0:
            if t_1 == 0:
                r = start_event.eval_position(0)
            else:
                r = end_event.eval_position(t_1)
            r[2] *= -1
            plt.gca().add_patch(plt.Circle(r[::2], physics.ball_radius,
                                           color=ball_colors[i],
                                           fill=True,
                                           linewidth=0.001,
                                           antialiased=True))
            plt.gca().add_patch(plt.Circle(r[::2], physics.ball_radius,
                                           color='black',
                                           fill=False,
                                           linestyle='solid',
                                           linewidth=0.18,
                                           antialiased=True))
        else:
            r_0, r_1 = start_event.eval_position(t_0), end_event.eval_position(t_1)
            r_0[2] *= -1; r_1[2] *= -1
            plt.gca().add_patch(plt.Circle(r_0[::2], physics.ball_radius,
                                           color=ball_colors[i],
                                           fill=True,
                                           linewidth=0.001,
                                           antialiased=True))
            plt.gca().add_patch(plt.Circle(r_1[::2], physics.ball_radius,
                                           color=ball_colors[i],
                                           fill=True,
                                           linewidth=0.001,
                                           antialiased=True))
            plt.gca().add_patch(plt.Circle(r_0[::2], physics.ball_radius,
                                           color='black',
                                           fill=False,
                                           linestyle='dashed',
                                           linewidth=0.18,
                                           antialiased=True))
    for i in physics.balls_on_table:
        last_event = physics.ball_events[i][-1]
        r_1 = last_event.eval_position(t_1)
        r_1[2] *= -1
        plt.gca().add_patch(plt.Circle(r_1[::2], physics.ball_radius,
                                       color='black',
                                       fill=False,
                                       linestyle='solid',
                                       linewidth=0.18,
                                       antialiased=True))
    if filename:
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=1200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


@catches_tcl_errors
def plot_energy(physics, title=None, nt=1000,
                t_0=None, t_1=None, filename=None,
                show=False, figure=None):
    events = physics.events
    if t_0 is None:
        t_0 = events[0].t
    if t_1 is None:
        t_1 = events[-1].t
    if title is None:
        title = 'kinetic energy vs time'
    if figure is None:
        figure = plt.figure()
    plt.title(title)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('energy (Joules)')
    labeled = set()
    for e in events:
        typee = e.__class__.__name__
        if type(e) in EVENT_COLORS:
            plt.axvline(e.t, color=EVENT_COLORS[type(e)],
                        label=typee if typee not in labeled else None)
            labeled.add(typee)
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    plt.plot(ts, [physics.eval_energy(t) for t in ts], color='black')
    plt.legend()
    if filename:
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=300)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


@catches_tcl_errors
def plot_collision_velocities(deltaPs, v_is, v_js,
                              title='velocities along axis of impulse',
                              show=True, filename=None):
    plt.figure()
    plt.plot(deltaPs, np.array(v_is)[:,0], label='ball i')
    plt.plot(deltaPs, np.array(v_js)[:,0], label='ball j')
    plt.xlabel(r'$P_I$: cumulative impulse along axis of impulse')
    plt.ylabel(r'$v_y$: velocity along axis of impulse')
    plt.title(title)
    plt.legend()
    if filename:
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


@catches_tcl_errors
def plot_collision_angular_velocities(deltaPs, omega_is, omega_js,
                                      title='angular velocities within horizontal plane',
                                      show=True, filename=None):
    plt.figure()
    plt.plot(deltaPs, np.array(omega_is)[:,0], label='ball i (axis of impulse)')
    plt.plot(deltaPs, np.array(omega_js)[:,0], label='ball j (axis of impulse)')
    plt.plot(deltaPs, np.array(omega_is)[:,2], '--', label='ball i (perpendicular axis)')
    plt.plot(deltaPs, np.array(omega_js)[:,2], '--', label='ball j (perpendicular axis)')
    plt.xlabel('$P_I$: cumulative impulse')
    plt.ylabel(r'$\omega_{xy}$: angular velocity within horizontal plane')
    plt.title(title)
    plt.legend()
    if filename:
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


@catches_tcl_errors
def plot_collision_velocity_maps(v_i1s, v_j1s,
                                 V_min=0.0001, V_max=40.0,
                                 theta_min=0.01, theta_max=89.99,
                                 show=True, filename=None):
    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    # axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True, subplot_kw=dict(polar=True))
    vmax = max(v_i1s.max(), v_j1s.max())
    vmin = 0.001
    for ax, v_1, label in zip(axes.ravel(),
                              [v_i1s[::-1,:,0], v_i1s[::-1,:,2], v_j1s[::-1,:,0], v_j1s[::-1,:,2]],
                              [r'$v_{%s}$ (m/s)' % comp for comp in ['iy', 'ix', 'jy', 'jx']]):
        im = ax.imshow(v_1, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        #ax.pcolormesh(theta, V, v_1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(label, fontsize='small')
        # cbar.ax.set_yticks([vmin, np.log10(vmax)])
        # cbar.ax.set_yticklabels(['%s m/s' % vmin, '%s m/s' % vmax], fontsize='x-small')
        ax.set_xlabel(r'$\theta$', fontsize='x-small')
        ax.set_xticks((-0.5, v_1.shape[0] - 0.5))
        ax.set_xticklabels(['%s deg.' % theta_min, '%s deg.' % theta_max], fontsize='x-small')
        ax.set_ylabel(r'$V_0$', fontsize='x-small')
        ax.set_yticks((-0.5, v_1.shape[1] - 0.5)[::-1])
        ax.set_yticklabels(['%s m/s' % V_min, '%s m/s' % V_max], fontsize='x-small')
    if show:
        plt.show()
    if filename:
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    plt.close()


@catches_tcl_errors
def plot_collision_angular_velocity_maps(omega_i1s, omega_j1s,
                                         V_min=0.001, V_max=40.0,
                                         theta_min=0.01, theta_max=89.99,
                                         show=True, filename=None):
    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    vmax = max(omega_i1s.max(), omega_j1s.max())
    for ax, omega_1, label in zip(axes.ravel(),
                                  [omega_i1s[::-1,:,0], omega_i1s[::-1,:,2], omega_j1s[::-1,:,0], omega_j1s[::-1,:,2]],
                                  [r'$\omega_{%s}$ (radians/s)' % comp for comp in ['iy', 'ix', 'jy', 'jx']]):
        im = ax.imshow(omega_1, norm=colors.LogNorm(vmin=0.001, vmax=vmax))
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(label, fontsize='small')
        ax.set_xlabel(r'$\theta$', fontsize='x-small')
        ax.set_xticks((-0.5, omega_1.shape[0] - 0.5))
        ax.set_xticklabels(['%s deg.' % theta_min, '%s deg.' % theta_max], fontsize='x-small')
        ax.set_ylabel(r'$V_0$', fontsize='x-small')
        ax.set_yticks((-0.5, omega_1.shape[1] - 0.5)[::-1])
        ax.set_yticklabels(['%s m/s' % V_min, '%s m/s' % V_max], fontsize='x-small')
    if show:
        plt.show()
    if filename:
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname, exist_ok=True)
        try:
            plt.savefig(filename, dpi=200)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    plt.close()


@catches_tcl_errors
def plot_distance(physics, i, j,
                  t0=None, t1=None,
                  show=True, filename=None):
    from numpy import linalg
    ball_events = physics.ball_events
    i_events = ball_events.get(i, [])
    j_events = ball_events.get(j, [])
    if not (i_events or j_events):
        return
    if t0 is None:
        t0 = 0.0
    collisions = [e for e in physics.events if isinstance(e, BallCollisionEvent)
                  and set((i,j)) & set((e.i,e.j)) and e.t >= t0]
    i_collisions = [e for e in collisions if i in (e.i, e.j)]
    j_collisions = [e for e in collisions if j in (e.i, e.j)]
    i_events = [e for e in sorted(i_events + i_collisions) if e.t >= t0]
    j_events = [e for e in sorted(j_events + j_collisions) if e.t >= t0]
    if t1 is None:
        t1 = max(i_events[-1].t if i_events else t0,
                 j_events[-1].t if j_events else t0)
    ts = np.linspace(t0, t1, 2000)
    events = sorted(i_events + j_events)
    ts = np.concatenate([[a.t] + list(ts[(ts > a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    fig = plt.figure()
    plt.title('distance b/t balls  %d  and  %d' % (i, j))
    plt.xlabel('t (seconds)')
    plt.ylabel('distance (meters)')
    for e in sorted(events, key=lambda e: (e.t, 0 if isinstance(e, BallEvent) else 1)):
        if e.t < t0:
            continue
        typee = e.__class__.__name__
        if isinstance(e, BallEvent):
            plt.axvline(e.t, color=EVENT_COLORS[type(e)],
                        label=typee + ' i=%d' % e.i,
                        linestyle=':')
    plt.hlines(2*physics.ball_radius, ts[0], ts[-1], linestyles='--', label='ball diameter')
    positions = np.array([physics.eval_positions(t)
                          for t in ts])
    deltas = positions[:,j] - positions[:,i]
    plt.plot(ts, linalg.norm(deltas, axis=-1))
    ticks, labels = plt.xticks()
    ticks = np.array(list(ticks) + [c.t for c in collisions])
    labels += ['(%d,%d)  %5.4f' % (c.i, c.j, c.t) for c in collisions]
    argsort = ticks.argsort()
    fig.axes[0].set_xticks(ticks[argsort])
    fig.axes[0].set_xticklabels([labels[ii] for ii in argsort], rotation=70, fontsize='x-small')
    plt.minorticks_on()
    fig.axes[0].set_xticks(0.01*np.arange(np.ceil(ts[-1])), minor=True)
    # plt.xticks([c.t for c in collisions], ['%5.4f (%d,%d)' % (c.t, c.i, c.j) for c in collisions],
    #            rotation=70, fontsize='x-small')
    plt.legend()
    plt.show()
    plt.close()


def gen_filename(name, ext, directory="."):
    from pathlib import Path
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True)
    import re
    matches = (re.match(r"({name}\.{{0,1}})(?P<number>\d*).{ext}".format(name=name, ext=ext),
                        f) for f in os.listdir(directory))
    number = max((int(m.group('number')) for m in matches
                  if m and m.group('number')), default=None)
    if number is None:
        if not matches:
            filename = '{name}.{ext}'.format(name=name, ext=ext)
        else:
            filename = '{name}.0.{ext}'.format(name=name, ext=ext)
    else:
        filename = '{name}.{num}.{ext}'.format(name=name, ext=ext, num=number+1)
    return os.path.abspath(os.path.join(directory, filename))


def git_head_hash():
    from subprocess import run
    ret = run('git reflog --format=%h -n 1'.split(), capture_output=True)
    return ret.stdout.decode().strip()


def check_ball_distances(pool_physics, t=None, filename=None, nt=2000):
    from numpy import sqrt, dot, linspace
    import pickle
    physics = pool_physics
    if t is None:
        ts = linspace(physics.events[0].t, physics.events[-1].t, nt)
    else:
        ts = [t]
    for t in ts:
        positions = physics.eval_positions(t)
        for i, r_i in enumerate(positions):
            for j, r_j in enumerate(positions[i+1:]):
                r_ij = r_j - r_i
                d = sqrt(dot(r_ij, r_ij))
                if d < 2*physics.ball_radius:
                    e_i, e_j = (e for e in physics.find_active_events(t)
                                if e.i == i or e.i == i + j + 1)
                    if isinstance(e_i, BallMotionEvent):
                        if isinstance(e_j, BallMotionEvent) and e_j.i < e_i.i:
                            e_i, e_j = e_j, e_i
                    else:
                        e_i, e_j = e_j, e_i
                    physics.e_i = e_i
                    physics.e_j = e_j
                    physics.i = e_i.i
                    physics.j = e_j.i
                    class BallsPenetratedInsanity(Exception):
                        def __init__(self, physics, *args, **kwargs):
                            fname = '%s.%s.%s.dump' % (self.__class__.__name__.split('.')[-1],
                                                       physics.ball_collision_model,
                                                       git_head_hash())
                            if filename is not None:
                                fname = '%s.%s' % (filename, fname)
                            with open(fname, 'wb') as f:
                                pickle.dump(physics, f)
                            _logger.info('dumped serialized physics to "%s"', fname)
                            super().__init__(*args, **kwargs)
                    raise BallsPenetratedInsanity(physics, '''t = {t}
d = {d} < {ball_diameter}

e_i: {e_i}

e_j: {e_j}
'''.format(t=t, d=d, ball_diameter=2*physics.ball_radius, e_i=e_i, e_j=e_j))
