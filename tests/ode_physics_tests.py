from sys import stdout
import os.path
import logging
from unittest import TestCase, skip
import traceback
import numpy as np
import matplotlib.pyplot as plt


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.ode_physics import ODEPoolPhysics


class ODEPhysicsTests(TestCase):
    def setUp(self):
        self.game = PoolGame()
        self.table = self.game.table
        self.physics = ODEPoolPhysics(initial_positions=self.game.initial_positions(),
                                      table=self.table)
        self.game.physics = self.physics
        self.cue = PoolCue()
        self.playback_rate = 1


    def test_reset(self):
        self.physics.reset(self.game.initial_positions())
        self.assertEqual(0, len(self.physics.events))
        self.assertLessEqual(np.linalg.norm(self.game.initial_positions() -
                                            self.physics.eval_positions(0.0)),
                             0.001 * self.physics.ball_radius)
        self.assertTrue((self.physics.eval_velocities(0.0) == 0).all())


    def test_strike_ball(self):
        self.game.reset()
        self.physics.on_table[1:] = False
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        self.cue.velocity[2] = -0.8
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        events = self.physics.events
        fig = plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        for e in events:
            plt.axvline(e.t)
            if e.T < float('inf'):
                plt.axvline(e.t + e.T)
        plt.axhline(self.table.height)
        plt.axhline(-0.5 * self.table.length)
        ts = np.linspace(events[0].t, events[-1].t, 50) #int((events[-1].t - events[0].t) * 23 + 1))
        ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                             for a, b in zip(events[:-1], events[1:])])
        for i, ls, xyz in zip(range(3), ['-o', '-s', '-d'], 'xyz'):
            plt.plot(ts, [self.physics.eval_positions(t)[0,i] for t in ts], ls, label='$%s$' % xyz)
        plt.legend()
        plt.show()


    def test_ball_collision(self):
        self.game.reset()
        self.physics.on_table[8:] = False
        self.cue.velocity[2] = -1.6
        self.cue.velocity[0] = -0.02
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        fig = plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        events = self.physics.events
        for e in events:
            plt.axvline(e.t)
            if e.T < float('inf'):
                plt.axvline(e.t + e.T)
        plt.axhline(self.table.height)
        plt.axhline(-0.5 * self.table.length)
        ts = np.linspace(events[0].t, events[-1].t, 50)
        ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                             for a, b in zip(events[:-1], events[1:])])
        for i, ls, xyz in zip(range(3), ['-o', '-s', '-d'], 'xyz'):
            plt.plot(ts, [self.physics.eval_positions(t)[0,i] for t in ts], ls, label='$%s$' % xyz)
        plt.legend()
        plt.show()
        # energy plot:
        plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('energy (Joules)')
        for e in events:
            plt.axvline(e.t)
            if e.T < float('inf'):
                plt.axvline(e.t + e.T)
        plt.axhline(self.table.height)
        plt.axhline(-0.5 * self.table.length)
        plt.plot(ts, [self.physics._calc_energy(t) for t in ts], '-xy')
        plt.show()
