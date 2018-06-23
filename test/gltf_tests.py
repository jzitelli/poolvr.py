from unittest import TestCase, skip
from sys import stdout
import os
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
import poolvr.gltf_utils as gltfu


SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


class GLTFTests(TestCase):
    GLTF_PATH = os.path.join(_here, os.path.pardir, os.path.pardir,
                             'glTF-Sample-Models', '1.0', 'Duck', 'glTF', 'Duck.gltf')
    URI_PATH = os.path.dirname(GLTF_PATH)
    _gltf = None
    show = True

    def setUp(self):
        title = traceback.extract_stack(None, 2)[0][2]
        window_size = (800, 600)
        self.window, self.renderer = setup_glfw(width=window_size[0], height=window_size[1],
                                                double_buffered=True, title=title)
        if GLTFTests._gltf is None:
            with open(GLTFTests.GLTF_PATH) as f:
                GLTFTests._gltf = json.loads(f.read())
        self.gltf = GLTFTests._gltf

    def tearDown(self):
        self.renderer.shutdown()
        glfw.DestroyWindow(self.window)
        glfw.Terminate()


    def test_program(self):
        program = gltfu.GLTFProgram(self.gltf, "program_0", self.URI_PATH)
        program.init_gl()
        #program.use()


    def test_technique(self):
        technique = gltfu.GLTFTechnique(self.gltf, 'technique0', self.URI_PATH)
        technique.init_gl()
        #technique.use()


    def test_material(self):
        material = gltfu.GLTFMaterial(self.gltf, 'blinn3-fx', self.URI_PATH)
        material.init_gl()


    def test_buffer_view(self):
        buffer_view = gltfu.GLTFBufferView(self.gltf, 'bufferView_29',self.URI_PATH)
        buffer_view.init_gl()


    def test_primitive(self):
        primitive = gltfu.GLTFPrimitive(self.gltf, self.gltf['meshes']['LOD3spShape-lib']['primitives'][0],
                                        self.URI_PATH)
        primitive.init_gl()

    @skip
    def test_duck(self):
        gltf_scene = gltfu.GLTFScene.load_from(self.gltf)
        self._view(gltf_scene)


    def _view(self, gltf_scene, window_size=(800,600)):
        meshes = [gltf_scene]
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
        _logger.info('''

                  *****************************

                     entering render loop...

                       meshes: %s

                  *****************************

''', meshes)
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
        _logger.info('''

                  *****************************

                     ...exited render loop.

                       average FPS: %f
                       maximum frame time: %f
                       number of frames renderer: %d

                  *****************************

''', (nframes - 1) / (t - st), max_frame_time, nframes - 1)
        renderer.shutdown()
        _logger.info('...shut down renderer.')
        glfw.DestroyWindow(window)
        glfw.Terminate()
