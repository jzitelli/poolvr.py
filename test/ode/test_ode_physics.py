import logging
_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue


def test_strike_ball(ode_pool_physics, ode_gl_rendering):
    physics = ode_pool_physics
    physics.reset(balls_on_table=[0,1])
    r_c = physics.ball_positions[0].copy()
    r_c[2] += physics.ball_radius
    cue = PoolCue()
    cue.velocity[2] = -0.6
    physics.strike_ball(0.0, 0, physics.ball_positions[0], r_c, cue.velocity, cue.mass)


# class ODEPhysicsTests(TestCase):
#     show = True

#     def setUp(self):
#         self.game = PoolGame()
#         self.table = self.game.table
#         self.physics = ODEPoolPhysics(initial_positions=self.game.initial_positions(),
#                                       table=self.table)
#         self.game.physics = self.physics
#         self.cue = PoolCue()
#         self.playback_rate = 1


#     def test_reset(self):
#         self.physics.reset(self.game.initial_positions())
#         self.assertEqual(0, len(self.physics.events))
#         self.assertLessEqual(np.linalg.norm(self.game.initial_positions() -
#                                             self.physics.eval_positions(0.0)),
#                              0.001 * self.physics.ball_radius)
#         self.assertTrue((self.physics.eval_velocities(0.0) == 0).all())




#     def test_ball_collision(self):
#         self.game.reset()
#         self.physics.on_table[8:] = False
#         self.cue.velocity[2] = -1.6
#         self.cue.velocity[0] = -0.02
#         Q = np.array((0.0, 0.0, self.physics.ball_radius))
#         i = 0
#         n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
#         _logger.debug('strike on %d resulted in %d events', i, n_events)
#         test_name = 'ODE_' + traceback.extract_stack(None, 1)[0][2]
#         plot_ball_motion(i, self.game, title=test_name)
#         # savefig(os.path.join(PLOTS_DIR, test_name + '.png'))
#         # plot_energy(self.game, title=test_name + ' - energy')
#         # savefig(os.path.join(PLOTS_DIR, test_name + '_energy.png'))
#         if self.show:
#             show(self.game, title=test_name,
#                  screenshots_dir=SCREENSHOTS_DIR)
