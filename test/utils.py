import logging
import numpy as np
import matplotlib.pyplot as plt
from poolvr.physics.events import CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent, BallCollisionEvent, SimpleBallCollisionEvent

_logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

EVENT_COLORS = {CueStrikeEvent: 'green',
                BallSlidingEvent: 'yellow',
                BallRollingEvent: 'orange',
                BallRestEvent: 'red',
                BallCollisionEvent: 'blue',
                SimpleBallCollisionEvent: 'blue'}
BALL_COLORS = {0: 'gray',
               1: 'yellow',
               2: 'blue',
               3: 'red',
               4: 'purple',
               5: 'orange',
               6: 'green',
               7: 'magenta',
               8: 'black'}
BALL_COLORS.update({i: BALL_COLORS[i-8] for i in range(9, 16)})


def plot_ball_motion(i, game, title=None, nt=1000,
                     t_0=None, t_1=None, coords=(0,),
                     collision_depth=0, event_markers=True,
                     hold=False, filename=None):
    if type(coords) == int:
        coords = (coords,)
    if title is None:
        title = 'ball %d position vs time'
    physics = game.physics
    events = physics.ball_events[i]
    if t_0 is None:
        t_0 = events[0].t
    else:
        events = [e for e in events if t_0 <= e.t + e.T]
    if t_1 is None:
        t_1 = events[-1].t
        if events[-1].T < float('inf'):
            t_1 += events[-1].T
    else:
        events = [e for e in events if e.t <= t_1]
    if not hold:
        plt.figure()
        plt.title(title)
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$%s$ (meters)' % ' / '.join('xyz'[coord] for coord in coords))
    linewidth = 5 - 2*collision_depth
    if event_markers:
        for e in events:
            plt.axvline(e.t, color=EVENT_COLORS[type(e)], linewidth=linewidth)
    for i_e, e in enumerate(events):
        if e.i != i:
            continue
        if isinstance(e.parent_event, (BallCollisionEvent, SimpleBallCollisionEvent)):
            parent = e.parent_event
            if event_markers:
                plt.axvline(e.t, color=BALL_COLORS[parent.i], ymax=0.5, linewidth=linewidth)
                plt.axvline(e.t, color=BALL_COLORS[parent.j], ymin=0.5, linewidth=linewidth)
                #plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwarg
            if collision_depth > 0:
                e_i, e_j = parent.child_events
                other_ball_event = e_j if parent.i == e.i else e_i
                plot_ball_motion(other_ball_event.i, game,
                                 t_0=other_ball_event.t, t_1=t_1,
                                 coords=coords,
                                 collision_depth=collision_depth-1,
                                 hold=True, event_markers=False)
    if events:
        ts = np.linspace(max(t_0, events[0].t),
                         min(t_1, events[-1].t + events[-1].T), nt)
        for coord in coords:
            plt.plot(ts, [physics.eval_positions(t)[i,coord] for t in ts],
                     ['-', '-.', '--'][coord], color=BALL_COLORS[i],
                     label='ball %d (%s)' % (i, 'xyz'[coord]),
                     linewidth=linewidth)
    if not hold:
        plt.legend()
        if filename:
            try:
                plt.savefig(filename, dpi=400)
                _logger.info('...saved figure to %s', filename)
            except Exception as err:
                _logger.warning('error saving figure:\n%s', err)
        plt.show()


def plot_energy(game, title=None, nt=1000,
                t_0=None, t_1=None, filename=None):
    physics = game.physics
    events = physics.events
    if t_0 is None:
        t_0 = events[0].t
    if t_1 is None:
        t_1 = events[-1].t
    if title is None:
        title = 'kinetic energy vs time'
    plt.figure()
    plt.title(title)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('energy (Joules)')
    for e in events:
        plt.axvline(e.t, color=EVENT_COLORS[type(e)])
        #if e.T < float('inf'):
        #    plt.axvline(e.t + e.T, color='red')
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    plt.plot(ts, [physics._calc_energy(t) for t in ts], color='green')
    if filename:
        try:
            plt.savefig(filename, dpi=400)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    plt.show()
