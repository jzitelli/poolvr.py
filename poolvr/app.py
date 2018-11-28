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


from .gl_rendering import OpenGLRenderer, set_quaternion_from_matrix, set_matrix_from_quaternion
try:
    from .pyopenvr_renderer import openvr, OpenVRRenderer
except ImportError as err:
    _logger.warning('could not import pyopenvr_renderer:\n%s', err)
    _logger.warning('\n\n\n**** VR FEATURES ARE NOT AVAILABLE! ****\n\n\n')
    OpenVRRenderer = None
from .physics import PoolPhysics
from .table import PoolTable
from .cue import PoolCue
from .game import PoolGame
from .keyboard_controls import init_keyboard, set_on_keydown_callback
from .mouse_controls import init_mouse
from .sound import init_sound
from .room import floor_mesh
# from .gl_text import TexturedText


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
    on_resize(window, window_size[0], window_size[1])
    return window, renderer


def capture_window(window,
                   filename='screenshot.png'):
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
         ball_collision_model='simple',
         use_ode=False,
         multisample=0,
         use_bb_particles=False,
         cube_map=None):
    """
    The main routine.

    Performs initializations, setups, kicks off the render loop.
    """
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

    init_sound()

    table = PoolTable()
    if use_ode:
        try:
            from .ode_physics import ODEPoolPhysics
            physics = ODEPoolPhysics(num_balls=16, table=table)
        except ImportError as err:
            physics = PoolPhysics(num_balls=16, table=table,
                                  ball_collision_model=ball_collision_model)
            _logger.warning('could not import ode_physics:\n%s', err)
            ODEPoolPhysics = None
    else:
        physics = PoolPhysics(num_balls=16, table=table,
                              ball_collision_model=ball_collision_model,
                              enable_sanity_check=novr)
    game = PoolGame(table=table,
                    physics=physics)
    cue = PoolCue()
    cue.position[1] = game.table.height + 0.1
    cue.position[2] += game.table.length * 0.3
    game.physics.add_cue(cue)
    game.reset()

    ball_meshes = game.table.ball_meshes
    if use_bb_particles:
        ball_shadow_meshes = []
    else:
        ball_shadow_meshes = [mesh.shadow_mesh for mesh in ball_meshes]
    # textured_text = TexturedText()
    if use_bb_particles:
        billboard_particles = ball_meshes[0]
        ball_mesh_positions = billboard_particles.primitive.attributes['translate']
        ball_mesh_rotations = np.array(game.num_balls * [np.eye(3)])
    else:
        ball_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_meshes]
        ball_mesh_rotations = [mesh.world_matrix[:3,:3].T for mesh in ball_meshes]
        ball_shadow_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_shadow_meshes]
    meshes = [floor_mesh, game.table.mesh] + ball_meshes + ball_shadow_meshes + [cue.shadow_mesh, cue]
    if cube_map:
        from .room import skybox_mesh
        meshes.insert(0, skybox_mesh)
    for mesh in meshes:
        mesh.init_gl()
    camera_world_matrix = fallback_renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = game.table.height + 0.6
    camera_position[2] = game.table.length - 0.1
    last_contact_t = float('-inf')
    def reset():
        nonlocal last_contact_t
        game.reset()
        last_contact_t = float('-inf')
        cue.position[0] = 0
        cue.position[1] = game.table.height + 0.1
        cue.position[2] = game.table.length * 0.3
    process_mouse_input = init_mouse(window)
    process_keyboard_input = init_keyboard(window)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_R and action == glfw.PRESS:
            reset()
    set_on_keydown_callback(window, on_keydown)
    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix)
        process_mouse_input(dt, cue)

    if isinstance(renderer, OpenVRRenderer):
        cue_offset = np.zeros(3, dtype=np.float64)
        offset_adjustment_mode = 0
        def toggle_touchpad_fb_ud():
            nonlocal offset_adjustment_mode
            offset_adjustment_mode = 1 - offset_adjustment_mode
        def cue_position_fb_ud(rAxis):
            if offset_adjustment_mode == 0:
                cue_offset[2] -= 0.008 * rAxis.y
            elif offset_adjustment_mode == 1:
                cue_offset[1] += 0.008 * rAxis.y
        axis_callbacks = {
            openvr.k_EButton_Axis0: cue_position_fb_ud,
            #openvr.k_EButton_Axis1: lock_to_cue
        }
        button_press_callbacks = {
            openvr.k_EButton_Grip           : toggle_touchpad_fb_ud,
            openvr.k_EButton_ApplicationMenu: reset,
            #openvr.k_EButton_ApplicationMenu: toggle_vr_menu,
            #openvr.k_EButton_Grip: reset,
            #openvr.k_EButton_ApplicationMenu: game.advance_time,
        }
        if use_ode and ODEPoolPhysics is not None:
            def on_cue_ball_collision(renderer=renderer, game=game, physics=physics, impact_speed=None):
                if impact_speed > 0.0015:
                    renderer.vr_system.triggerHapticPulse(renderer._controller_indices[0], 0,
                                                          int(max(0.8, impact_speed**2 / 4.0 + impact_speed**3 / 16.0) * 2750))
            physics.set_cue_ball_collision_callback(on_cue_ball_collision)
            # def on_cue_surface_collision(renderer=renderer, game=game, physics=physics, impact_speed=None):
            #     if impact_speed > 0.003:
            #         renderer.vr_system.triggerHapticPulse(renderer._controller_indices[0], 0,
            #                                               int(max(0.75, 0.2*impact_speed**2 + 0.07*impact_speed**3) * 2500))
            # physics.set_cue_surface_collision_callback(on_cue_surface_collision)

    _logger.info('entering render loop...')
    sys.stdout.flush()

    last_contact_t = float('-inf')
    contact_last_frame = False
    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    controller_positions = np.zeros((2, 3), dtype=np.float64)
    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        with renderer.render(meshes=meshes) as frame_data:
            if isinstance(renderer, OpenVRRenderer) and frame_data:
                renderer.process_input(button_press_callbacks=button_press_callbacks,
                                       axis_callbacks=axis_callbacks)
                hmd_pose = frame_data['hmd_pose']
                camera_position[:] = hmd_pose[:, 3]
                for i, pose in enumerate(frame_data['controller_poses'][:1]):
                    velocity = frame_data['controller_velocities'][i]
                    angular_velocity = frame_data['controller_angular_velocities'][i]
                    cue.world_matrix[:3, :3] = pose[:, :3].dot(cue.rotation).T
                    position = controller_positions[i]
                    position[:] = pose[:, 3] + cue_offset[2] * pose[:,2]
                    position[1] += cue_offset[1]
                    cue.world_matrix[3, :3] = position
                    cue.velocity[:] = velocity
                    cue.angular_velocity = angular_velocity
                    set_quaternion_from_matrix(pose[:, :3], cue.quaternion)
                # r_0, r_1 = controller_positions
                # r_01 = r_1 - r_0
                # rot = np.eye(3, dtype=np.float64);
                # cue.world_matrix[1,:3] = cue.world_matrix[:3,:3].T.dot(
            elif isinstance(renderer, OpenGLRenderer):
                set_quaternion_from_matrix(cue.rotation.dot(cue.world_matrix[:3, :3].T),
                                           cue.quaternion)
            if use_bb_particles:
                billboard_particles.update_gl()
            else:
                for i, pos in enumerate(game.ball_positions):
                    ball_mesh_positions[i][:] = pos
                    ball_shadow_mesh_positions[i][0::2] = pos[0::2]
                for i, quat in enumerate(game.ball_quaternions):
                    set_matrix_from_quaternion(quat, ball_mesh_rotations[i])
            cue.shadow_mesh.update()
            # sdf_text.set_text("%9.3f" % dt)
            # sdf_text.update_gl()

        if not contact_last_frame:
            if game.t - last_contact_t >= 2:
                for i, position in cue.aabb_check(game.ball_positions[:1], physics.ball_radius):
                    r_c = cue.contact(position, physics.ball_radius)
                    if r_c is not None:
                        physics.strike_ball(game.t, i, game.ball_positions[i], r_c, cue.velocity, cue.mass)
                        last_contact_t = game.t
                        contact_last_frame = True
                        if isinstance(renderer, OpenVRRenderer):
                            renderer.vr_system.triggerHapticPulse(renderer._controller_indices[-1],
                                                                  0, int(np.linalg.norm(cue.velocity)**2 / 1.7 * 2700))
                        break
        else:
            contact_last_frame = False
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
