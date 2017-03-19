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
import PIL.Image


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable
from poolvr.game import PoolGame
from poolvr.physics import PoolPhysics
from poolvr.gl_rendering import OpenGLRenderer, Texture, Mesh, Material
from poolvr.app import setup_glfw, BG_COLOR, TEXTURES_DIR
from poolvr.billboards import BillboardParticles
from poolvr.keyboard_controls import init_keyboard
from poolvr.primitives import SpherePrimitive
from poolvr.techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


class PhysicsTests(TestCase):
    show = False

    @classmethod
    def set_show(cls, val):
        cls.show = val


    def setUp(self):
        self.game = PoolGame()
        self.physics = self.game.physics
        self.table = self.game.table
        self.cue = PoolCue()
        self.playback_rate = 1
        self.physics.reset(self.game.initial_positions())


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
        self.assertEqual(3, len(events))
        self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        self.assertIsInstance(events[2], PoolPhysics.RollToRestEvent)
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
        self._savefig()
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
        self._savefig(plot_name='energy')
        if self.show:
            self._view()

    @skip
    def test_ball_collision(self):
        self.game.reset()
        # self.physics.on_table[8:] = False
        self.cue.velocity[2] = -1.6
        self.cue.velocity[0] = -0.02
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        # self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        # self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        # self.assertIsInstance(events[2], PoolPhysics.BallCollisionEvent)
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
        self._savefig()
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
        self._savefig(plot_name='energy')
        if self.show:
            self._view()


    def test_simple_ball_collision(self):
        self.physics = PoolPhysics(num_balls=self.game.num_balls,
                                   ball_radius=self.game.ball_radius,
                                   initial_positions=self.game.ball_positions,
                                   use_simple_ball_collision=True)
        self.game.physics = self.physics
        self.game.reset()
        # self.physics.on_table[8:] = False
        self.cue.velocity[2] = -1.6
        self.cue.velocity[0] = -0.02
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        # self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        # self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        # self.assertIsInstance(events[2], PoolPhysics.BallCollisionEvent)
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
        self._savefig()
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
        self._savefig(plot_name='energy')
        if self.show:
            self._view()

    def test_ball_collision_2(self):
        self.game.reset()
        self.physics.on_table[2:8:2] = False
        self.physics.on_table[8:] = False
        self.cue.velocity[2] = -1.6
        self.cue.velocity[0] = -0.02
        Q = np.array((0.0, 0.0, self.physics.ball_radius))
        i = 0
        n_events = self.physics.strike_ball(0.0, i, Q, self.cue.velocity, self.cue.mass)
        _logger.debug('strike on %d resulted in %d events', i, n_events)
        # self.assertIsInstance(events[0], PoolPhysics.StrikeBallEvent)
        # self.assertIsInstance(events[1], PoolPhysics.SlideToRollEvent)
        # self.assertIsInstance(events[2], PoolPhysics.BallCollisionEvent)
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
        self._savefig()
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
        plt.legend()
        self._savefig(plot_name='energy')
        if self.show:
            self._view()


    def _savefig(self, plot_name=''):
        test_name = traceback.extract_stack(None, 2)[0][2]
        title = ' - '.join([test_name, plot_name]) if plot_name else test_name
        pth = os.path.join(PLOTS_DIR, '%s.png' % ('-'.join([test_name, plot_name]) if plot_name else test_name)).replace(' ', '_')
        plt.title(title)
        try:
            plt.savefig(pth)
            _logger.info("...saved figure to %s", pth)
        except:
            _logger.warning("could not save the plot to %s. i'll just show it to you:", pth)
            plt.show()


    def _view(self, window_size=(800,600)):
        title = traceback.extract_stack(None, 2)[0][2]
        window, renderer = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=True,
                                      title=title)
        camera_world_matrix = renderer.camera_matrix
        # camera_world_matrix[:,[1,2]] = camera_world_matrix[:,[2,1]]
        camera_position = camera_world_matrix[3,:3]
        game = self.game
        camera_position[1] = game.table.height + 0.19
        camera_position[2] = 0.183 * game.table.length
        gl.glViewport(0, 0, window_size[0], window_size[1])
        #gl.glClearColor(*BG_COLOR)
        gl.glClearColor(0.24, 0.18, 0.08, 0.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        physics = game.physics
        cue = PoolCue()
        cue.position[1] = game.table.height + 0.1
        ball_radius = game.ball_radius
        ball_billboards = BillboardParticles(Texture(os.path.join(TEXTURES_DIR, 'ball.png')), num_particles=game.num_balls,
                                             scale=2*ball_radius,
                                             color=np.array([[(c&0xff0000) / 0xff0000, (c&0x00ff00) / 0x00ff00, (c&0x0000ff) / 0x0000ff]
                                                             for c in game.ball_colors], dtype=np.float32),
                                             translate=game.ball_positions)
        ball_positions = ball_billboards.primitive.attributes['translate']
        sphere_meshes = [Mesh({Material(LAMBERT_TECHNIQUE,
                                        values={'u_color': [(c&0xff0000) / 0xff0000,
                                                            (c&0x00ff00) / 0x00ff00,
                                                            (c&0x0000ff) / 0x0000ff,
                                                            0.0]})
                               : [SpherePrimitive(radius=ball_radius)]})
                         for c in game.ball_colors]
        for mesh in sphere_meshes:
            list(mesh.primitives.values())[0][0].attributes['a_position'] = list(mesh.primitives.values())[0][0].attributes['vertices']
        sphere_positions = [mesh.world_matrix[3,:3] for mesh in sphere_meshes]
        # meshes = [game.table.mesh, ball_billboards, cue] + sphere_meshes
        meshes = [game.table.mesh] + sphere_meshes
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
            pt += dt * self.playback_rate
            glfw.PollEvents()
            process_keyboard_input(dt, camera_world_matrix, cue=cue)
            renderer.process_input()
            with renderer.render(meshes=sphere_meshes):
                self.physics.eval_positions(pt, out=ball_positions)
                for i, pos in enumerate(ball_positions):
                    if not self.physics.on_table[i]:
                        sphere_positions[i][:] = renderer.camera_position
                    else:
                        sphere_positions[i][:] = pos
                ball_positions[~self.physics.on_table] = renderer.camera_position # hacky way to only show balls that are on table
                # ball_billboards.update_gl()
            max_frame_time = max(max_frame_time, dt)
            if nframes == 0:
                st = glfw.GetTime()
            nframes += 1
            glfw.SwapBuffers(window)

        _logger.info('...exited render loop: average FPS: %f, maximum frame time: %f',
                     (nframes - 1) / (t - st), max_frame_time)

        mWidth, mHeight = glfw.GetWindowSize(window);
        n = 3 * mWidth * mHeight;
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(0,0,mWidth,mHeight, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        pil_image = PIL.Image.frombytes('RGB', (mWidth, mHeight), pixels)
        pil_image = pil_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        filename = title.replace(' ', '_') + '-screenshot.png'
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        pil_image.save(filepath)
        _logger.info('..saved screen capture to "%s"', filepath)

        try:
            renderer.shutdown()
            _logger.info('...shut down renderer')
        except Exception as err:
            _logger.error(err)
        try:
            glfw.PollEvents()
        except Exception as err:
            _logger.error(err)
        try:
            glfw.DestroyWindow(window)
        except Exception as err:
            _logger.error(err)
        try:
            glfw.PollEvents()
        except Exception as err:
            _logger.error(err)
        try:
            glfw.Terminate()
        except Exception as err:
            _logger.error(err)
