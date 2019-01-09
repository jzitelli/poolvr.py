import logging
import numpy as np
import matplotlib.pyplot as plt


from poolvr.table import PoolTable
from poolvr.physics.events import (CueStrikeEvent, BallSlidingEvent, BallRollingEvent, BallRestEvent,
                                   BallCollisionEvent, MarlowBallCollisionEvent, SimpleBallCollisionEvent,
                                   RailCollisionEvent)


_logger = logging.getLogger(__name__)


EVENT_COLORS = {CueStrikeEvent: 'green',
                BallSlidingEvent: 'yellow',
                BallRollingEvent: 'orange',
                BallRestEvent: 'red',
                BallCollisionEvent: 'blue',
                MarlowBallCollisionEvent: 'blue',
                SimpleBallCollisionEvent: 'blue',
                RailCollisionEvent: 'green'}
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
            try:
                plt.savefig(filename, dpi=400)
                _logger.info('...saved figure to %s', filename)
            except Exception as err:
                _logger.warning('error saving figure:\n%s', err)
        if show:
            plt.show()
        plt.close()


def plot_motion_timelapse(physics, table=None,
                          title=None,
                          nt=50,
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
        events = [e for e in events if t_0 <= e.t + e.T]
    if t_1 is None:
        t_1 = events[-1].t
        if events[-1].T < float('inf'):
            t_1 += events[-1].T
    events = [e for e in events if e.t <= t_1]
    if figure is None:
        figure = plt.figure()
    plt.title(title)
    ts = np.linspace(t_0, t_1, nt)
    #ax = plt.subplot(111, facecolor='green')
    plt.gca().set_xlim(-0.5*table.width, 0.5*table.width)
    plt.gca().set_ylim(-0.5*table.length, 0.5*table.length)
    #[-0.5*table.width, 0.5*table.width,
    # -0.5*table.length, 0.5*table.length])
    plt.gca().set_aspect('equal')
    plt.gca().add_patch(plt.Rectangle((-0.5*table.width, -0.5*table.length),
                                      table.width, table.length,
                                      color='#013216'))
    ball_colors = dict(BALL_COLORS)
    ball_colors[0] = 'white'
    for t in ts:
        positions = physics.eval_positions(t)
        for i in physics.balls_on_table:
            plt.gca().add_patch(plt.Circle(positions[i,::2], physics.ball_radius,
                                           color=ball_colors[i], alpha=14/nt))
    #plt.xlim(-0.5*table.width, 0.5*table.width)
    #plt.ylim(-0.5*table.length, 0.5*table.length)
    # plt.xticks(np.linspace(-0.5*table.length, 0.5*table.length, 8))
    # plt.yticks(np.linspace(-0.5*table.width, 0.5*table.width, 8))
    if filename:
        try:
            plt.savefig(filename, dpi=400)
            _logger.info('...saved figure to %s', filename)
        except Exception as err:
            _logger.warning('error saving figure:\n%s', err)
    if show:
        plt.show()
    plt.close()


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
    for e in events:
        plt.axvline(e.t, color=EVENT_COLORS[type(e)])
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
    if show:
        plt.show()
    plt.close()
