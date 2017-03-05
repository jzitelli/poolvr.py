from unittest import TestCase, skip
from sys import stdout
import os.path
import logging
import traceback
import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw


_logger = logging.getLogger(__name__)


from poolvr.cue import PoolCue
from poolvr.table import PoolTable, EGA_TECHNIQUE, LAMBERT_TECHNIQUE
from poolvr.gl_rendering import OpenGLRenderer, Texture, Material
from poolvr.app import setup_glfw, BG_COLOR, TEXTURES_DIR
from poolvr.billboards import BillboardParticles
from poolvr.keyboard_controls import init_keyboard
import poolvr.primitives


SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


class OpenGLTests(TestCase):
    show = True


    def test_cone_mesh(self):
        material = Material(EGA_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
        mesh = poolvr.primitives.ConeMesh(material, radius=0.15, height=0.3)
        for prim in mesh.primitives[material]:
            prim.attributes['a_position'] = prim.attributes['vertices']
        mesh.world_matrix[3,2] = -3
        self._view(meshes=[mesh])


    def _view(self, meshes=None, window_size=(800,600)):
        if meshes is None:
            meshes = []
        title = traceback.extract_stack(None, 2)[0][2]
        window, renderer = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=True,
                                      title=title)
        camera_world_matrix = renderer.camera_matrix
        camera_position = camera_world_matrix[3,:3]
        gl.glViewport(0, 0, window_size[0], window_size[1])
        gl.glClearColor(*BG_COLOR)
        gl.glEnable(gl.GL_DEPTH_TEST)
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
        while not glfw.WindowShouldClose(window):
            t = glfw.GetTime()
            dt = t - lt
            lt = t
            glfw.PollEvents()
            renderer.process_input()
            process_keyboard_input(dt, camera_world_matrix)
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
