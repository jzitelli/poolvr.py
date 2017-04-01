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
from poolvr.gl_rendering import OpenGLRenderer, Texture, Mesh, Material
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
         screenshots_dir=''):
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
    gl.glClearColor(*gl_clear_color)
    gl.glViewport(0, 0, window_size[0], window_size[1])
    gl.glEnable(gl.GL_DEPTH_TEST)
    table = game.table
    physics = game.physics
    ball_radius = game.ball_radius
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
        with renderer.render(meshes=sphere_meshes) as frame_data:
            physics.eval_positions(pt, out=ball_positions)
            ball_positions[~physics.on_table] = renderer.camera_position # hacky way to only show balls that are on table
            for i, pos in enumerate(ball_positions):
                if not physics.on_table[i]:
                    sphere_positions[i][:] = renderer.camera_position
                else:
                    sphere_positions[i][:] = pos
            # ball_billboards.update_gl()
        max_frame_time = max(max_frame_time, dt)
        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        pt += dt * playback_rate
        physics.step(dt)
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
