import sys
import os.path
import logging
from collections import defaultdict
import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw


_logger = logging.getLogger(__name__)


from .gl_rendering import OpenGLRenderer, Texture
try:
    from .pyopenvr_renderer import OpenVRRenderer
except ImportError as err:
    OpenVRRenderer = None
from .billboards import BillboardParticles
from .cue import Cue
from .game import PoolGame
from .mouse_controls import init_mouse


BG_COLOR = (0.0, 0.0, 0.0, 0.0)
TURN_SPEED = 1.2
MOVE_SPEED = 0.3
CUE_MOVE_SPEED = 0.3


def setup_glfw(width=800, height=600, double_buffered=False):
    if not glfw.Init():
        raise Exception('failed to initialize glfw')
    if not double_buffered:
        glfw.WindowHint(glfw.DOUBLEBUFFER, False)
        glfw.SwapInterval(0)
    window = glfw.CreateWindow(width, height, "gltfview")
    if not window:
        glfw.Terminate()
        raise Exception('failed to create glfw window')
    glfw.MakeContextCurrent(window)
    _logger.info('GL_VERSION: %s' % gl.glGetString(gl.GL_VERSION))
    return window


def main(window_size=(800,600), novr=False):
    window = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=novr)
    renderer = OpenGLRenderer(window_size=window_size, znear=0.1, zfar=1000)
    camera_world_matrix = renderer.camera_matrix
    if not novr and OpenVRRenderer is not None:
        try:
            renderer = OpenVRRenderer(window_size=window_size)
        except Exception as err:
            _logger.error('could not initialize OpenVRRenderer: %s' % err)
    camera_position = camera_world_matrix[3,:3]
    cue = Cue()
    game = PoolGame()
    ball_radius = game.table.ball_radius
    ball_billboards = BillboardParticles(Texture('textures/ball.png'), num_particles=game.num_balls,
                                         scale=2*ball_radius,
                                         color=np.array([[(c & 0xff0000) / 0xff0000, (c & 0x00ff00) / 0x00ff00, (c & 0x0000ff) / 0x0000ff] for c in game.ball_colors], dtype=np.float32),
                                         translate=game.ball_positions)
    ball_positions = ball_billboards.primitive.attributes['translate']
    meshes = [game.table.mesh, ball_billboards, cue]
    for mesh in meshes:
        mesh.init_gl()
    gl.glViewport(0, 0, window_size[0], window_size[1])
    def on_resize(window, width, height):
        gl.glViewport(0, 0, width, height)
        renderer.window_size = (width, height)
        renderer.update_projection_matrix()
    glfw.SetWindowSizeCallback(window, on_resize)
    key_state = defaultdict(bool)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.SetWindowShouldClose(window, gl.GL_TRUE)
        elif action == glfw.PRESS:
            key_state[key] = True
        elif action == glfw.RELEASE:
            key_state[key] = False
    glfw.SetKeyCallback(window, on_keydown)
    process_mouse_input = init_mouse(window)
    theta = 0.0
    def process_input(dt):
        glfw.PollEvents()
        nonlocal theta
        theta += TURN_SPEED * dt * (key_state[glfw.KEY_LEFT] - key_state[glfw.KEY_RIGHT])
        sin, cos = np.sin(theta), np.cos(theta)
        camera_world_matrix[0,0] = cos
        camera_world_matrix[0,2] = -sin
        camera_world_matrix[2,0] = sin
        camera_world_matrix[2,2] = cos
        fb = MOVE_SPEED * dt * (-key_state[glfw.KEY_W] + key_state[glfw.KEY_S])
        lr = MOVE_SPEED * dt * (key_state[glfw.KEY_D] - key_state[glfw.KEY_A])
        ud = MOVE_SPEED * dt * (key_state[glfw.KEY_Q] - key_state[glfw.KEY_Z])
        camera_position[:] += fb * camera_world_matrix[2,:3] + lr * camera_world_matrix[0,:3] + ud * camera_world_matrix[1,:3]
        fb = CUE_MOVE_SPEED * (-key_state[glfw.KEY_I] + key_state[glfw.KEY_K])
        lr = CUE_MOVE_SPEED * (key_state[glfw.KEY_L] - key_state[glfw.KEY_J])
        ud = CUE_MOVE_SPEED * (key_state[glfw.KEY_U] - key_state[glfw.KEY_M])
        cue.world_matrix[:3,:3] = cue.rotation.T
        cue.velocity[:] = fb * cue.world_matrix[2,:3] + lr * cue.world_matrix[0,:3] + ud * cue.world_matrix[1,:3]
        cue.position += cue.velocity * dt
        process_mouse_input(dt, cue)
    gl.glClearColor(*BG_COLOR)
    gl.glEnable(gl.GL_DEPTH_TEST)
    physics = game.physics

    _logger.info('* starting render loop...')
    sys.stdout.flush()

    nframes = 0
    lt = glfw.GetTime()
    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        renderer.process_input()
        with renderer.render(meshes=meshes) as frame_data:

            if frame_data:
                # VR mode:
                poses, velocities, angular_velocities = frame_data
                if len(poses) > 1:
                    pose = poses[-1]
                    cue.world_matrix[:3,:3] = poses[-1][:,:3].dot(cue.rotation).T
                    cue.world_matrix[3,:3] = poses[-1][:,3]
                    cue.velocity[:] = velocities[-1]
                    cue.angular_velocity = angular_velocities[-1]
                    for i, position in cue.aabb_check(ball_positions, ball_radius):
                        contact = cue.contact(position, ball_radius)
                        if contact is not None:
                            renderer.vr_system.triggerHapticPulse(renderer._controller_indices[-1], 0, 1500)
                            cue.world_matrix[:3,:3].dot(contact, out=contact)
                            contact += cue.position
                            #x, y, z = contact
                            #print('%d: %.4f   %.4f   %.4f' % (i, x, y, z))
                            physics.strike_ball(i, cue.mass, contact - ball_positions[i], cue.velocity, cue.angular_velocity)

            elif isinstance(renderer, OpenGLRenderer):
                # desktop mode:
                for i, position in cue.aabb_check(ball_positions, ball_radius):
                    contact = cue.contact(position, ball_radius)
                    if contact is not None:
                        cue.world_matrix[:3,:3].dot(contact, out=contact)
                        contact += cue.position
                        # x, y, z = contact
                        # print('%d: %.4f   %.4f   %.4f' % (i, x, y, z))
                        if i == 0:
                            pass
                        else:
                            print('scratch (touched %d)' % i)

        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        glfw.SwapBuffers(window)
    _logger.info('* FPS (avg): %f' % ((nframes - 1) / (t - st)))
    renderer.shutdown()
    glfw.DestroyWindow(window)
    glfw.Terminate()
