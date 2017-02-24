import sys
import os.path
import logging
import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw


_logger = logging.getLogger('poolvr')


from .exceptions import TODO
from .gl_rendering import OpenGLRenderer, Texture
try:
    from .pyopenvr_renderer import OpenVRRenderer
except ImportError as err:
    OpenVRRenderer = None
from .billboards import BillboardParticles
from .cue import PoolCue
from .game import PoolGame
from .keyboard_controls import init_keyboard
from .mouse_controls import init_mouse


BG_COLOR = (0.0, 0.0, 0.0, 0.0)


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(__file__),
                            os.path.pardir,
                            'textures')


def setup_glfw(width=800, height=600, double_buffered=False, title="poolvr.py 0.0.1"):
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


def main(window_size=(800,600), novr=False):
    _logger.info('HELLO')
    window = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=novr)
    fallback_renderer = OpenGLRenderer(window_size=window_size, znear=0.1, zfar=1000)
    camera_world_matrix = fallback_renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    game = PoolGame()
    if not novr and OpenVRRenderer is not None:
        try:
            renderer = OpenVRRenderer(window_size=window_size)
        except Exception as err:
            renderer = fallback_renderer
            _logger.error('could not initialize OpenVRRenderer: %s', err)
    else:
        renderer = fallback_renderer
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
        mesh.init_gl()
    def on_resize(window, width, height):
        gl.glViewport(0, 0, width, height)
        renderer.window_size = (width, height)
        renderer.update_projection_matrix()
    glfw.SetWindowSizeCallback(window, on_resize)
    process_keyboard_input = init_keyboard(window)
    process_mouse_input = init_mouse(window)
    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix, cue)
        process_mouse_input(dt, cue)

    _logger.info('entering render loop...')
    sys.stdout.flush()

    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        renderer.process_input()
        with renderer.render(meshes=meshes) as frame_data:

            ##### VR mode: #####

            if frame_data:
                poses, velocities, angular_velocities = frame_data
                if len(poses) > 1:
                    pose = poses[-1]
                    cue.world_matrix[:3,:3] = poses[-1][:,:3].dot(cue.rotation).T
                    cue.world_matrix[3,:3] = poses[-1][:,3]
                    cue.velocity[:] = velocities[-1]
                    cue.angular_velocity = angular_velocities[-1]
                    for i, position in cue.aabb_check(ball_positions, ball_radius):
                        poc = cue.contact(position, ball_radius)
                        if poc is not None:
                            poc -= ball_positions[i]
                            x, y, z = poc
                            renderer.vr_system.triggerHapticPulse(renderer._controller_indices[-1],
                                                                  0, 1300)
                            physics.strike_ball(t, i,
                                                cue.world_matrix[1,:3],
                                                poc - ball_positions[i],
                                                cue.velocity, cue.mass)
                            break
                physics.eval_positions(t, out=ball_positions)
                ball_billboards.update_gl()

            ##### desktop mode: #####

            elif isinstance(renderer, OpenGLRenderer):
                for i, position in cue.aabb_check(ball_positions, ball_radius):
                    poc = cue.contact(position, ball_radius)
                    if poc is not None:
                        cue.world_matrix[:3,:3].dot(poc, out=poc)
                        poc += cue.position
                        x, y, z = poc
                        print(np.linalg.norm(poc))
                        print('%d: %.4f   %.4f   %.4f' % (i, x, y, z))
                        if i == 0:
                            pass
                        else:
                            print('scratch (touched %d)' % i)

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
    _logger.info('GOODBYE')
