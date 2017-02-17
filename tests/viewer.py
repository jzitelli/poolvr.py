import logging
import numpy as np
import cyglfw3 as glfw


_logger = logging.getLogger(__name__)


from poolvr.gl_rendering import OpenGLRenderer
from poolvr.keyboard_controls import init_keyboard
from poolvr.mouse_controls import init_mouse
from poolvr.game import PoolGame


class OpenGLViewer(object):
    def __init__(self):
        self.window = OpenGLViewer._setup_glfw(double_buffered=True)
        self.renderer = OpenGLRenderer()
        self.game = PoolGame()
    @staticmethod
    def _setup_glfw(width=800, height=600, title="poolvr.py", double_buffered=False):
        if not glfw.Init():
            raise Exception('failed to initialize glfw')
        if not double_buffered:
            glfw.WindowHint(glfw.DOUBLEBUFFER, False)
            glfw.SwapInterval(0)
        window = glfw.CreateWindow(width, height, title)
        if not window:
            glfw.Terminate()
            raise Exception('failed to create glfw window')
        glfw.MakeContextCurrent(window)
        _logger.info('GL_VERSION: %s', gl.glGetString(gl.GL_VERSION))
        return window
    def _process_input(self, dt):
        pass
    def _show(self, t0=0.0, t1=None):
        nframes = 0
        max_frame_time = 0.0
        lt = glfw.GetTime()
        ball_positions = np.empty((self.game.num_balls, 3), dtype=np.float32)
        while not glfw.WindowShouldClose(self.window):
            t = glfw.GetTime()
            dt = t - lt
            lt = t
            self._process_input(dt)
            with self.renderer.render(meshes=meshes) as frame_data:
                self.game.physics.eval_positions(t, out=ball_positions)
                # for i, position in cue.aabb_check(ball_positions, ball_radius):
                #     poc = cue.contact(position, ball_radius)
                #     if poc is not None:
                #         cue.world_matrix[:3,:3].dot(poc, out=poc)
                #         poc += cue.position
                #         x, y, z = poc
                #         print(np.linalg.norm(poc))
                #         print('%d: %.4f   %.4f   %.4f' % (i, x, y, z))
                #         if i == 0:
                #             pass
                #         else:
                #             print('scratch (touched %d)' % i)
            max_frame_time = max(max_frame_time, dt)
            if nframes == 0:
                st = glfw.GetTime()
            nframes += 1
            glfw.SwapBuffers(window)
        _logger.info('...exited render loop: average FPS: %f, maximum frame time: %f',
                     (nframes - 1) / (t - st), max_frame_time)
