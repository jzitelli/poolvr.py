import logging
import numpy as np
import matplotlib.pyplot as plt
from poolvr.physics import PoolPhysics


logging.getLogger('matplotlib').setLevel(logging.WARNING)
_logger = logging.getLogger(__name__)


event_colors = {PoolPhysics.StrikeBallEvent: 'green',
                PoolPhysics.SlideToRollEvent: 'yellow',
                PoolPhysics.RollToRestEvent: 'red',
                PoolPhysics.SlideToRestEvent: 'cyan',
                PoolPhysics.BallCollisionEvent: 'orange',
                PoolPhysics.SimpleBallCollisionEvent: 'orange'}
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


def savefig(fp):
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


def plot_ball_motion(i, game, title=None, nt=400,
                     t_0=None, t_1=None, coords=(0,)):
    if type(coords) == int:
        coords = (coords,)
    if title is None:
        title = 'ball %d position vs time'
    physics = game.physics
    events = physics.events[:-1]
    if t_0 is None:
        t_0 = events[0].t
    if t_1 is None:
        t_1 = events[-1].t
    plt.figure()
    plt.title(title)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('$%s$ (meters)' % ' / '.join('xyz'[coord] for coord in coords))
    for i_e, e in enumerate(events):
        if e.i != i:
            continue
        _logger.debug('event %d: %s', i_e, e)
        plt.axvline(e.t, color=event_colors[type(e)])
        j = getattr(e, 'j', None)
        if j:
            ts = np.linspace(t_0, t_1, nt)
            ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                                 for a, b in zip(events[i_e:-1], events[i_e+1:])])
            for coord in coords:
                plt.plot(ts, [physics.eval_positions(t)[j,coord] for t in ts], color=ball_colors[j], label='ball %d' % j)
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    for coord in coords:
        plt.plot(ts, [physics.eval_positions(t)[0,coord] for t in ts], ['-', '-.', '--'][coord],
                 color=ball_colors[i], label='ball %d (%s)' % (i, 'xyz'[coord]))
    plt.legend()


def plot_energy(game, title=None, nt=400,
                t_0=None, t_1=None):
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
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    plt.plot(ts, [physics._calc_energy(t) for t in ts], color='green')
