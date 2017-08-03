from unittest import TestCase, skip
from sys import stdout
import os.path
import logging
import traceback
import json
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw


_logger = logging.getLogger(__name__)
_here = os.path.dirname(__file__)


from poolvr.app import setup_glfw, BG_COLOR
from poolvr.keyboard_controls import init_keyboard
import poolvr.gltfutils as gltfu


SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


class GLTFTests(TestCase):
    show = True
    GLTF_PATH = os.path.join(_here, os.path.pardir, os.path.pardir,
                             'glTF', 'sampleModels', 'Duck', 'glTF',
                             'Duck.gltf')
    @skip
    def test_duck(self):
        with open(self.GLTF_PATH) as f:
            json_str = f.read()
        _logger.debug('read "%s"', self.GLTF_PATH)
        json_dict = json.loads(json_str)
        gltf_dict = gltfu.GLTFDict(json_dict)
        gltf_scene = gltfu.GLTFScene.load_from(gltf_dict)
        self._view(gltf_scene)


    def _view(self, gltf_scene, window_size=(800,600)):
        meshes = [scene]
        title = traceback.extract_stack(None, 2)[0][2]
        window, renderer = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=True,
                                      title=title)
        camera_world_matrix = renderer.camera_matrix
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
