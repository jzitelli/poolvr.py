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
from .gl_rendering import OpenGLRenderer, Texture, Mesh, Material
try:
    from .pyopenvr_renderer import openvr, OpenVRRenderer
except ImportError as err:
    OpenVRRenderer = None
from .techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE
from .primitives import SpherePrimitive
from .billboards import BillboardParticles
from .cue import PoolCue
from .game import PoolGame
from .physics import PoolPhysics
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
    renderer = OpenGLRenderer(window_size=(width, height), znear=0.1, zfar=1000)
    def on_resize(window, width, height):
        gl.glViewport(0, 0, width, height)
        renderer.window_size = (width, height)
        renderer.update_projection_matrix()
    glfw.SetWindowSizeCallback(window, on_resize)
    return window, renderer



def main(window_size=(800,600), novr=False):
    _logger.info('HELLO')
    game = PoolGame()
    physics = game.physics
    cue = PoolCue()
    cue.position[1] = game.table.height + 0.1
    ball_radius = physics.ball_radius
    game.reset()

    window, fallback_renderer = setup_glfw(width=window_size[0], height=window_size[1], double_buffered=novr)
    if not novr and OpenVRRenderer is not None:
        try:
            renderer = OpenVRRenderer(window_size=window_size)
            button_press_callbacks = {openvr.k_EButton_Grip: game.reset,
                                      openvr.k_EButton_ApplicationMenu: game.advance_time}
        except Exception as err:
            renderer = fallback_renderer
            _logger.error('could not initialize OpenVRRenderer: %s', err)
    else:
        renderer = fallback_renderer
        renderer.camera_position[1] = game.table.height + 0.6
        renderer.camera_position[2] = game.table.length - 0.1
    process_keyboard_input = init_keyboard(window)
    process_mouse_input = init_mouse(window)
    camera_world_matrix = fallback_renderer.camera_matrix
    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix, cue)
        process_mouse_input(dt, cue)

    ball_billboards = BillboardParticles(Texture(os.path.join(TEXTURES_DIR, 'ball.png')), num_particles=game.num_balls,
                                         scale=2*physics.ball_radius,
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
    # meshes = [game.table.mesh, ball_billboards, cue]
    meshes = [game.table.mesh] + sphere_meshes + [cue]

    gl.glViewport(0, 0, window_size[0], window_size[1])
    gl.glClearColor(*BG_COLOR)
    gl.glEnable(gl.GL_DEPTH_TEST)
    for mesh in meshes:
        mesh.init_gl()

    _logger.info('entering render loop...')
    sys.stdout.flush()

    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    while not glfw.WindowShouldClose(window):
        with renderer.render(meshes=meshes) as frame_data:

            t = glfw.GetTime()
            dt = t - lt
            lt = t
            process_input(dt)

            ##### VR mode: #####

            if frame_data:
                renderer.process_input(button_press_callbacks=button_press_callbacks)
                poses, velocities, angular_velocities = frame_data
                hmd_pose = poses[0]
                if len(poses) > 1:
                    pose = poses[-1]
                    cue.world_matrix[:3,:3] = poses[-1][:,:3].dot(cue.rotation).T
                    cue.world_matrix[3,:3] = poses[-1][:,3]
                    cue.velocity[:] = velocities[-1]
                    cue.angular_velocity = angular_velocities[-1]
                    if game.t >= game.ntt:
                        for i, position in cue.aabb_check(ball_positions, ball_radius):
                            poc = cue.contact(position, ball_radius)
                            if poc is not None:
                                poc[:] = [0.0, 0.0, ball_radius]
                                renderer.vr_system.triggerHapticPulse(renderer._controller_indices[-1],
                                                                      0, 1300)
                                physics.strike_ball(game.t, i, poc, cue.velocity, cue.mass)
                                game.ntt = physics.next_turn_time()
                                break
                physics.eval_positions(game.t, out=ball_positions)
                for i, pos in enumerate(ball_positions):
                    sphere_positions[i][:] = pos
                np.array(sphere_positions)[~physics.on_table] = hmd_pose[:,3] # hacky way to only show balls that are on table
                # physics.eval_positions(game.t, out=ball_positions)
                # ball_positions[~physics.on_table] = hmd_pose[:,3] # hacky way to only show balls that are on table
                # ball_billboards.update_gl()

            ##### desktop mode: #####

            elif isinstance(renderer, OpenGLRenderer):
                for i, position in cue.aabb_check(ball_positions, ball_radius):
                    poc = cue.contact(position, ball_radius)
                    if poc is not None:
                        pass
                physics.eval_positions(game.t, out=ball_positions)
                ball_positions[~physics.on_table] = renderer.camera_position # hacky way to only show balls that are on table
                ball_billboards.update_gl()

        game.t += dt
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
