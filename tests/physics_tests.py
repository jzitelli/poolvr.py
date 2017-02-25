from sys import stdout
import os.path
import logging
from unittest import TestCase, skip
import traceback
import numpy as np
import matplotlib.pyplot as plt
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics
from poolvr.gl_rendering import OpenGLRenderer, Texture
from poolvr.app import setup_glfw, BG_COLOR, TEXTURES_DIR
from poolvr.billboards import BillboardParticles
from poolvr.keyboard_controls import init_keyboard


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')


class PhysicsTests(TestCase):
    show = True

    def setUp(self):
        self.table = PoolTable()
        self.game = PoolGame()
        self.cue = PoolCue()
        self.physics = PoolPhysics(initial_positions=self.game.ball_positions)


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
        events = self.physics.strike_ball(0.0, 0, Q, self.cue.velocity, self.cue.mass)
        _logger.info('\n'.join(['  %s' % e for e in events]))
        self.assertEqual(3, len(events))
        self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        self.assertIsInstance(events[2], PoolPhysics.RollToRestEvent)
        fig = plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        ts = np.linspace(events[0].t, events[-1].t, 50) #int((events[-1].t - events[0].t) * 23 + 1))
        ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                             for a, b in zip(events[:-1], events[1:])])
        for e in events:
            plt.axvline(e.t)
            if e.T < float('inf'):
                plt.axvline(e.t + e.T)
        plt.axhline(self.table.height)
        plt.axhline(-0.5 * self.table.length)
        plt.plot(ts, [self.physics.eval_positions(t)[0,0] for t in ts], '-o', label='$x$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,1] for t in ts], '-s', label='$y$')
        plt.plot(ts, [self.physics.eval_positions(t)[0,2] for t in ts], '-d', label='$z$')
        plt.legend()
        self._savefig()
        if self.show: self._view()


    def test_ball_collision(self):
        self.game.reset()
        self.physics.on_table[2:] = False
        self.cue.velocity[2] = -1.3
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        events = self.physics.strike_ball(0.0, 0, Q, self.cue.velocity, self.cue.mass)
        _logger.info('\n'.join(['  %s' % e for e in events]))
        # self.assertEqual(3, len(events))
        # self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        # self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        # self.assertIsInstance(events[2], PoolPhysics.BallCollisionEvent)
        fig = plt.figure()
        plt.xlabel('$t$ (seconds)')
        plt.ylabel('$x, y, z$ (meters)')
        ts = np.linspace(events[0].t, events[-1].t, 50) #int((events[-1].t - events[0].t) * 23 + 1))
        ts = np.concatenate([[a.t] + list(ts[(ts >= a.t) & (ts < b.t)]) + [b.t]
                             for a, b in zip(events[:-1], events[1:])])
        for i, ls, xyz in zip(range(3), ['-o', '-s', '-d'], 'xyz'):
            plt.plot(ts, [self.physics.eval_positions(t)[0,i] for t in ts], ls, label='$%s$' % xyz)
        # plt.plot(ts, [self.physics.eval_positions(t)[0,1] for t in ts], '-s', label='$y$')
        # plt.plot(ts, [self.physics.eval_positions(t)[0,2] for t in ts], '-d', label='$z$')
        for e in events:
            plt.axvline(e.t)
            if e.T < float('inf'):
                plt.axvline(e.t + e.T)
        plt.legend()
        self._savefig()
        if self.show: self._view()


    def _savefig(self):
        title = traceback.extract_stack(None, 2)[0][2]
        pth = os.path.join(PLOTS_DIR, '%s.png' % title)
        plt.title(title)
        try:
            plt.savefig(pth)
            _logger.info("...saved figure to %s", pth)
        except:
            _logger.warning("could not save the plot to %s. i'll just show it to you:", pth)
            plt.show()


    def _view(self, window_size=(800,600)):
        title = traceback.extract_stack(None, 2)[0][2]
        window = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=True,
                            title=title)
        renderer = OpenGLRenderer(window_size=window_size, znear=0.1, zfar=1000)
        camera_world_matrix = renderer.camera_matrix
        camera_position = camera_world_matrix[3,:3]
        game = self.game
        camera_position[1] = game.table.height + 0.6
        camera_position[2] = game.table.length - 0.1
        gl.glViewport(0, 0, window_size[0], window_size[1])
        gl.glClearColor(*BG_COLOR)
        gl.glEnable(gl.GL_DEPTH_TEST)
        physics = game.physics
        cue = PoolCue()
        cue.position[1] = game.table.height + 0.1
        ball_radius = game.table.ball_radius
        ball_billboards = BillboardParticles(Texture(os.path.join(TEXTURES_DIR, 'ball.png')), num_particles=game.num_balls,
                                             scale=2*ball_radius,
                                             color=np.array([[(c&0xff0000) / 0xff0000, (c&0x00ff00) / 0x00ff00, (c&0x0000ff) / 0x0000ff]
                                                             for c in game.ball_colors], dtype=np.float32),
                                             translate=game.ball_positions)
        ball_positions = ball_billboards.primitive.attributes['translate']
        ball_quaternions = np.zeros((game.num_balls, 4), dtype=np.float32)
        ball_quaternions[:,3] = 1.0
        meshes = [game.table.mesh, ball_billboards, cue]
        for mesh in meshes:
            mesh.init_gl(force=True)
        def on_resize(window, width, height):
            gl.glViewport(0, 0, width, height)
            renderer.window_size = (width, height)
            renderer.update_projection_matrix()
        glfw.SetWindowSizeCallback(window, on_resize)
        process_keyboard_input = init_keyboard(window)
        _logger.info('entering render loop...')
        stdout.flush()

        nframes = 0
        max_frame_time = 0.0
        lt = glfw.GetTime()
        t0 = self.physics.events[0].t
        t1 = self.physics.events[-1].t + min(2.0, self.physics.events[-1].T)
        pt = t0
        while not glfw.WindowShouldClose(window) and pt <= t1:
            t = glfw.GetTime()
            dt = t - lt
            lt = t
            pt += dt
            glfw.PollEvents()
            process_keyboard_input(dt, camera_world_matrix, cue=cue)
            renderer.process_input()
            self.physics.eval_positions(pt, out=ball_billboards.primitive.attributes['translate'])
            ball_billboards.update_gl()
            with renderer.render(meshes=meshes):
                pass
            max_frame_time = max(max_frame_time, dt)
            if nframes == 0:
                st = glfw.GetTime()
            nframes += 1
            glfw.SwapBuffers(window)

        _logger.info('...exited render loop: average FPS: %f, maximum frame time: %f',
                     (nframes - 1) / (t - st), max_frame_time)

        renderer.shutdown()
        _logger.info('...shut down renderer')
        glfw.DestroyWindow(window)
        glfw.Terminate()
