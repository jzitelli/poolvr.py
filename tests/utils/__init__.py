import logging
import numpy as np
import matplotlib.pyplot as plt


_logger = logging.getLogger(__name__)


def savefig(fp):
    """
    Save matplotlib figure to a file
    """
    try:
        plt.savefig(fp)
        _logger.info("...saved figure to %s", fp)
    except Exception as err:
        _logger.warning(err)
        _logger.warning("could not save the figure to %s. i'll just show it to you:", fp)
        plt.show()


def plot_ball_motion(i, game, title=None, nt=50,
                     t_0=None, t_1=None):
    fig = plt.figure()
    if title is None:
        title = 'ball %d position vs time'
    plt.title(title)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('$x, y, z$ (meters)')
    physics = game.physics
    events = physics.events
    table = game.table
    plt.axhline(table.height)
    plt.axhline(-0.5 * table.length)
    for e in events:
        plt.axvline(e.t)
        if e.T < float('inf'):
            plt.axvline(e.t + e.T)
    if t_0 is None:
        t_0 = events[0].t
    if t_1 is None:
        t_1 = events[-1].t
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    for i, ls, xyz in zip(range(3), ['-o', '-s', '-d'], 'xyz'):
        plt.plot(ts, [physics.eval_positions(t)[0,i] for t in ts], ls, label='$%s$' % xyz)
    plt.legend()


def plot_energy(game, title=None, nt=50,
                t_0=None, t_1=None):
    physics = game.physics
    table = game.table
    events = physics.events
    if t_0 is None:
        t_0 = events[0].t
    if t_1 is None:
        t_1 = events[-1].t
    if title is None:
        title = 'kinetic energy vs time'
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('energy (Joules)')
    plt.axhline(table.height)
    plt.axhline(-0.5 * table.length)
    for e in events:
        plt.axvline(e.t)
        if e.T < float('inf'):
            plt.axvline(e.t + e.T)
    ts = np.linspace(t_0, t_1, nt)
    ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                         for a, b in zip(events[:-1], events[1:])])
    plt.plot(ts, [physics._calc_energy(t) for t in ts], '-xy')
