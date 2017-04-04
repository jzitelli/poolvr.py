from sys import stdout
import os.path
import logging
import time
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw
import numpy as np
import PIL.Image


_logger = logging.getLogger(__name__)


from poolvr.app import TEXTURES_DIR
from poolvr.keyboard_controls import init_keyboard
from poolvr.mouse_controls import init_mouse
from poolvr.gl_rendering import OpenGLRenderer, Texture, Mesh, Material, set_matrix_from_quaternion
from poolvr.techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE
from poolvr.primitives import SpherePrimitive
from poolvr.billboards import BillboardParticles
from poolvr.cue import PoolCue


def show(game,
         title='poolvr.py   ***   GLViewer',
         window_size=(800,600),
         gl_clear_color=(0.24, 0.18, 0.08, 0.0),
         before_frame_cb=None, after_frame_cb=None,
         double_buffered=True,
         playback_rate=1.0,
         screenshots_dir='',
         use_billboards=False):
    if not glfw.Init():
        raise Exception('failed to initialize glfw')
    if not double_buffered:
        glfw.WindowHint(glfw.DOUBLEBUFFER, False)
        glfw.SwapInterval(0)
    window = glfw.CreateWindow(window_size[0], window_size[1], title)
    if not window:
        raise Exception('failed to create glfw window')
    glfw.MakeContextCurrent(window)
    _logger.info('GL_VERSION: %s', gl.glGetString(gl.GL_VERSION))
    renderer = OpenGLRenderer(window_size=window_size, znear=0.1, zfar=1000)
    def on_resize(window, width, height):
        gl.glViewport(0, 0, width, height)
        renderer.window_size = (width, height)
        renderer.update_projection_matrix()
    glfw.SetWindowSizeCallback(window, on_resize)

    table = game.table
    physics = game.physics
    ball_radius = game.ball_radius
    table.setup_balls(game.ball_radius, game.ball_colors[:9], game.ball_positions,
                      striped_balls=set(range(9, game.num_balls)),
                      use_billboards=use_billboards)
    camera_world_matrix = renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = table.height + 0.19
    camera_position[2] = 0.183 * table.length
    cue = PoolCue()
    cue.position[1] = table.height + 0.1
    def on_resize(window, width, height):
        gl.glViewport(0, 0, width, height)
        renderer.window_size = (width, height)
        renderer.update_projection_matrix()
    glfw.SetWindowSizeCallback(window, on_resize)
    process_keyboard_input = init_keyboard(window)

    meshes = [table.mesh] + table.ball_meshes #+ [cue]
    for mesh in meshes:
        mesh.init_gl(force=True)

    ball_positions = game.table.ball_positions
    ball_quaternions = game.table.ball_quaternions
    sphere_positions = [mesh.world_matrix[3,:3] for mesh in game.table.ball_meshes]
    sphere_rotations = [mesh.world_matrix[:3,:3].T for mesh in game.table.ball_meshes]

    gl.glViewport(0, 0, window_size[0], window_size[1])
    gl.glClearColor(*gl_clear_color)
    gl.glEnable(gl.GL_DEPTH_TEST)

    _logger.info('entering render loop...')
    stdout.flush()

    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    t0 = physics.events[0].t
    t1 = physics.events[-1].t + min(2.0, physics.events[-1].T)
    pt = t0
    while not glfw.WindowShouldClose(window) and pt <= t1:
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix, cue=cue)
        renderer.process_input()
        with renderer.render(meshes=meshes) as frame_data:
            ball_positions[~physics.on_table] = renderer.camera_position # hacky way to only show balls that are on table
            # ball_billboards.update_gl()

        physics.step(dt)
        physics.eval_positions(pt, out=ball_positions)
        physics.eval_quaternions(pt, out=ball_quaternions)
        for i, pos in enumerate(ball_positions):
            sphere_positions[i][:] = pos
        for i, quat in enumerate(ball_quaternions):
            set_matrix_from_quaternion(quat, sphere_rotations[i])

        max_frame_time = max(max_frame_time, dt)
        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        pt += dt * playback_rate
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
    filepath = os.path.join(screenshots_dir, filename)
    pil_image.save(filepath)
    _logger.info('..saved screen capture to "%s"', filepath)

    try:
        renderer.shutdown()
        _logger.info('...shut down renderer')
    except Exception as err:
        _logger.error(err)

    glfw.DestroyWindow(window)
    glfw.Terminate()
