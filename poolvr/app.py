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


from .gl_rendering import OpenGLRenderer#, set_matrix_from_quaternion, set_quaternion_from_matrix
try:
    from .pyopenvr_renderer import openvr, OpenVRRenderer
except ImportError as err:
    _logger.warning('could not import pyopenvr_renderer:\n%s', err)
    _logger.warning('\n\n\n**** VR FEATURES ARE NOT AVAILABLE! ****\n\n\n')
    OpenVRRenderer = None
# from .gl_text import TexturedText
from .cue import PoolCue
from .game import PoolGame
from .keyboard_controls import init_keyboard, set_on_keydown_callback
from .mouse_controls import init_mouse
from .sound import init_sound
from .room import floor_mesh, skybox_mesh

try:
    from .ode_physics import ODEPoolPhysics
except ImportError as err:
    _logger.warning('could not import ode_physics:\n%s', err)
    ODEPoolPhysics = None


BG_COLOR = (0.0, 0.0, 0.0, 0.0)


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.pardir,
                            'textures')


def setup_glfw(window_size=(800,600), double_buffered=False,
               title="poolvr.py 0.0.1", multisample=0):
    if not glfw.Init():
        raise Exception('failed to initialize glfw')
    if not double_buffered:
        glfw.WindowHint(glfw.DOUBLEBUFFER, False)
        glfw.SwapInterval(0)
    if multisample:
        glfw.WindowHint(glfw.SAMPLES, multisample)
    width, height = window_size
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
    renderer.init_gl()
    return window, renderer


def capture_window(window,
                   filename='screenshot.png'):
    import OpenGL.GL as gl
    import PIL
    if not filename.endswith('.png'):
        filename += '.png'
    _logger.info('saving screen capture...')
    mWidth, mHeight = glfw.GetWindowSize(window)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    pixels = gl.glReadPixels(0, 0, mWidth, mHeight, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    pil_image = PIL.Image.frombytes('RGB', (mWidth, mHeight), pixels)
    pil_image = pil_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    pil_image.save(filename)
    _logger.info('...saved screen capture to "%s"', filename)


def main(window_size=(800,600),
         novr=False,
         use_simple_ball_collisions=False,
         use_ode=False,
         multisample=0,
         use_bb_particles=False):
    """
    The main routine.

    Performs initializations, setups, kicks off the render loop.
    """
    _logger.info('HELLO')
    window, fallback_renderer = setup_glfw(window_size=window_size,
                                           double_buffered=novr, multisample=multisample)
    if not novr and OpenVRRenderer is not None:
        try:
            renderer = OpenVRRenderer(window_size=window_size, multisample=multisample)
        except Exception as err:
            renderer = fallback_renderer
            _logger.error('could not initialize OpenVRRenderer: %s', err)
    else:
        renderer = fallback_renderer
    game = PoolGame(use_simple_ball_collisions=use_simple_ball_collisions)
    # if use_ode and ODEPoolPhysics is not None:
    #     game.physics = ODEPoolPhysics(num_balls=game.num_balls,
    #                                   ball_radius=game.ball_radius,
    #                                   initial_positions=game.ball_positions,
    #                                   table=game.table)
    physics = game.physics
    game.reset()
    ball_meshes = game.table.ball_meshes
    if use_bb_particles:
        ball_shadow_meshes = []
    else:
        ball_shadow_meshes = [mesh.shadow_mesh for mesh in ball_meshes]
    cue = PoolCue()
    # cue_body, cue_geom = game.physics.add_cue(cue)
    cue.position[1] = game.table.height + 0.1
    cue.position[2] += game.table.length * 0.3
    # textured_text = TexturedText()
    meshes = [skybox_mesh, floor_mesh, game.table.mesh] + ball_meshes + ball_shadow_meshes + [cue.shadow_mesh, cue]
    for mesh in meshes:
        mesh.init_gl()
    if use_bb_particles:
        billboard_particles = ball_meshes[0]
        ball_mesh_positions = billboard_particles.primitive.attributes['translate']
        #ball_mesh_rotations = np.array(game.num_balls * [np.eye(3)])
    else:
        ball_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_meshes]
        #ball_mesh_rotations = [mesh.world_matrix[:3,:3].T for mesh in ball_meshes]
        ball_shadow_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_shadow_meshes]

    camera_world_matrix = fallback_renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = game.table.height + 0.6
    camera_position[2] = game.table.length - 0.1

    process_mouse_input = init_mouse(window)
    process_keyboard_input = init_keyboard(window)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_R and action == glfw.PRESS:
            game.reset()
            for k in last_contact_t.keys():
                last_contact_t[k] = float('-inf')
    set_on_keydown_callback(window, on_keydown)

    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix)
        process_mouse_input(dt, cue)

    if isinstance(renderer, OpenVRRenderer):
        button_press_callbacks = {openvr.k_EButton_Grip: game.reset,
                                  }#openvr.k_EButton_ApplicationMenu: game.advance_time}
        def on_cue_ball_collision(renderer=renderer, game=game, physics=physics, impact_speed=None):
            if impact_speed > 0.0015:
                renderer.vr_system.triggerHapticPulse(renderer._controller_indices[0], 0,
                                                      int(max(0.8, impact_speed**2 / 4.0 + impact_speed**3 / 16.0) * 2750))
        #physics.set_cue_ball_collision_callback(on_cue_ball_collision)
        def on_cue_surface_collision(renderer=renderer, game=game, physics=physics, impact_speed=None):
            if impact_speed > 0.003:
                renderer.vr_system.triggerHapticPulse(renderer._controller_indices[0], 0,
                                                      int(max(0.75, 0.2*impact_speed**2 + 0.07*impact_speed**3) * 2500))
        #physics.set_cue_surface_collision_callback(on_cue_surface_collision)

    init_sound()

    _logger.info('entering render loop...')
    sys.stdout.flush()

    last_contact_t = {i: float('-inf') for i in range(physics.num_balls)}
    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()

    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        with renderer.render(meshes=meshes) as frame_data:
            if isinstance(renderer, OpenVRRenderer) and frame_data:
                renderer.process_input(button_press_callbacks=button_press_callbacks)
                hmd_pose = frame_data['hmd_pose']
                camera_position[:] = hmd_pose[:, 3]
                for i, pose in enumerate(frame_data['controller_poses'][:1]):
                    velocity = frame_data['controller_velocities'][i]
                    angular_velocity = frame_data['controller_angular_velocities'][i]
                    cue.world_matrix[:3, :3] = pose[:, :3].dot(cue.rotation).T
                    cue.world_matrix[3, :3] = pose[:, 3]
                    cue.velocity[:] = velocity #cue.rotation.T.dot(velocity)
                    cue.angular_velocity = angular_velocity
                    #set_quaternion_from_matrix(pose[:, :3], cue.quaternion)
            elif isinstance(renderer, OpenGLRenderer):
                # set_quaternion_from_matrix(cue.rotation.dot(cue.world_matrix[:3, :3].T),
                #                            cue.quaternion)
                pass
            if use_bb_particles:
                billboard_particles.update_gl()
            else:
                for i, pos in enumerate(game.ball_positions):
                    ball_mesh_positions[i][:] = pos
                    ball_shadow_mesh_positions[i][0::2] = pos[0::2]
                    ball_shadow_mesh_positions[i][1] = game.table.height + 0.03
                # for i, quat in enumerate(game.ball_quaternions):
                #     set_matrix_from_quaternion(quat, ball_mesh_rotations[i])
            cue.shadow_mesh.update()
            # sdf_text.set_text("%9.3f" % dt)
            # sdf_text.update_gl()

        # cue_body.setPosition(cue.world_position)
        # w = cue.quaternion[3]; cue.quaternion[1:] = cue.quaternion[:3]; cue.quaternion[0] = w
        # cue_body.setQuaternion(cue.quaternion)
        # cue_geom.setQuaternion(cue.quaternion)
        # cue_body.setLinearVel(cue.velocity)
        # cue_body.setAngularVel(cue.angular_velocity)

        for i, position in cue.aabb_check(game.ball_positions, physics.ball_radius):
            if game.t - last_contact_t[i] < 0.5:
                continue
            r_c = cue.contact(position, physics.ball_radius)
            if r_c is not None:
                last_contact_t[i] = game.t
                if isinstance(renderer, OpenVRRenderer):
                    renderer.vr_system.triggerHapticPulse(renderer._controller_indices[-1],
                                                          0, int(np.linalg.norm(cue.velocity)**2 / 1.7 * 2700))
                _logger.debug('cue.velocity = %s', cue.velocity)
                physics.strike_ball(game.t, i, game.ball_positions[i], r_c, cue.velocity, cue.mass)
                # game.ntt = physics.next_turn_time
                # _logger.debug('next_turn_time = %s', game.ntt)
                break

        game.step(dt)

        max_frame_time = max(max_frame_time, dt)
        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        glfw.SwapBuffers(window)

    if nframes > 1:
        _logger.info('...exited render loop: average FPS: %f, maximum frame time: %f, average frame time: %f',
                     (nframes - 1) / (t - st), max_frame_time, (t - st) / (nframes - 1))

    renderer.shutdown()
    _logger.info('...shut down renderer')
    glfw.DestroyWindow(window)
    glfw.Terminate()
    _logger.info('GOODBYE')
