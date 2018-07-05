import logging
import numpy as np
import matplotlib.pyplot as plt
from poolvr.physics.events import CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent, BallCollisionEvent

_logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

event_colors = {CueStrikeEvent: 'green',
                BallSlidingEvent: 'yellow',
                BallRollingEvent: 'orange',
                BallRestEvent: 'red'}
ball_colors = {0: 'gray',
               1: 'yellow',
               2: 'blue',
               3: 'red',
               4: 'purple',
               5: 'orange',
               6: 'green',
               7: 'magenta',
               8: 'black'}
ball_colors.update({i: ball_colors[i-8] for i in range(9, 16)})


def _savefig(fp):
    """
    Save matplotlib figure to a file
    """
    try:
        plt.savefig(fp, dpi=400)
        _logger.info("...saved figure to %s", fp)
    except Exception as err:
        _logger.warning(err)
        _logger.warning("could not save the figure to %s. i'll just show it to you:", fp)
        plt.show()


def plot_ball_motion(i, game, title=None, nt=1000,
                     t_0=None, t_1=None, coords=(0,),
                     collision_depth=0, hold=False,
                     filename=None):
    if type(coords) == int:
        coords = (coords,)
    if title is None:
        title = 'ball %d position vs time'
    physics = game.physics
    #events = physics.events
    events = physics.ball_events[i]
    if t_0 is None:
        t_0 = events[0].t
    if t_1 is None:
        t_1 = events[-1].t
        if events[-1].T < float('inf'):
            t_1 += events[-1].T
    if not hold:
        plt.figure()
        plt.title(title)
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$%s$ (meters)' % ' / '.join('xyz'[coord] for coord in coords))
    for i_e, e in enumerate(events):
        if e.i != i:
            continue
        if isinstance(e.parent_event, BallCollisionEvent):
            parent = e.parent_event
            plt.axvline(e.t, color=ball_colors[parent.i], ymax=0.5)
            plt.axvline(e.t, color=ball_colors[parent.j], ymin=0.5)
            if collision_depth > 0:
                e_i, e_j = parent.child_events
                other_ball_event = e_j if parent.i == e.i else e_i
                plot_ball_motion(other_ball_event.i, game,
                                 t_0=other_ball_event.t, t_1=t_1,
                                 coords=coords,
                                 collision_depth=collision_depth-1,
                                 hold=True)
        else:
            plt.axvline(e.t, color=event_colors[type(e)])
    linewidth = 5 - 2*collision_depth
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    for coord in coords:
        plt.plot(ts, [physics.eval_positions(t)[i,coord] for t in ts],
                 ['-', '-.', '--'][coord], color=ball_colors[i],
                 label='ball %d (%s)' % (i, 'xyz'[coord]),
                 linewidth=linewidth)
    if not hold:
        plt.legend()
    if filename:
        _savefig(filename)


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
        plt.axvline(e.t, color=event_colors[type(e)])
        #if e.T < float('inf'):
        #    plt.axvline(e.t + e.T, color='red')
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    plt.plot(ts, [physics._calc_energy(t) for t in ts], color='green')
    if filename:
        _savefig(filename)
