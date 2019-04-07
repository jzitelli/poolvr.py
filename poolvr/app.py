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
from .gl_techniques import EGA_TECHNIQUE, LAMBERT_TECHNIQUE
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


_window_renderer = None


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
    global _window_renderer
    _window_renderer = renderer
    def on_resize(window, width, height):
        global _window_renderer
        gl.glViewport(0, 0, width, height)
        _window_renderer.window_size = (width, height)
        _window_renderer.update_projection_matrix()
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
         cube_map=None,
         speed=1.0,
         glyphs=False,
         realtime=False,
         balls_on_table=None,
         technique=LAMBERT_TECHNIQUE,
         use_quartic_solver=False):
    """
    The main routine.

    Performs initializations, setups, kicks off the render loop.
    """
    window, fallback_renderer = setup_glfw(window_size=window_size,
                                           double_buffered=novr, multisample=multisample)
    if not novr and OpenVRRenderer is not None:
        try:
            renderer = OpenVRRenderer(window_size=window_size, multisample=multisample)
            global _window_renderer
            _window_renderer = renderer
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
            physics = ODEPoolPhysics(num_balls=16, table=table,
                                     balls_on_table=balls_on_table)
        except ImportError as err:
            _logger.error('could not import ode_physics:\n%s', err)
            ODEPoolPhysics = None
            physics = PoolPhysics(num_balls=16, table=table,
                                  ball_collision_model=ball_collision_model,
                                  enable_sanity_check=novr,
                                  enable_occlusion=False,
                                  realtime=realtime,
                                  balls_on_table=balls_on_table,
                                  use_quartic_solver=use_quartic_solver)
    else:
        physics = PoolPhysics(num_balls=16, table=table,
                              ball_collision_model=ball_collision_model,
                              enable_sanity_check=novr,
                              enable_occlusion=False,
                              realtime=realtime,
                              balls_on_table=balls_on_table,
                              use_quartic_solver=use_quartic_solver)
    game = PoolGame(table=table,
                    physics=physics)
    cue = PoolCue()
    cue.position[1] = game.table.H + 0.1
    cue.position[2] += game.table.L * 0.3
    game.physics.add_cue(cue)
    game.reset(balls_on_table=balls_on_table)
    table_mesh = game.table.export_mesh(surface_technique=technique, cushion_technique=technique)
    ball_meshes = game.table.export_ball_meshes(technique=technique,
                                                use_bb_particles=use_bb_particles)
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
    meshes = [floor_mesh, table_mesh] + ball_meshes + ball_shadow_meshes + [cue.shadow_mesh, cue]
    if cube_map:
        from .room import skybox_mesh
        meshes.insert(0, skybox_mesh)
    for mesh in meshes:
        mesh.init_gl()
    for i, mesh in enumerate(ball_meshes):
        if i not in balls_on_table:
            mesh.visible = False
            ball_shadow_meshes[i].visible = False
    camera_world_matrix = fallback_renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = game.table.H + 0.6
    camera_position[2] = game.table.L - 0.1
    last_contact_t = float('-inf')
    def reset():
        nonlocal last_contact_t
        nonlocal balls_on_table
        game.reset(balls_on_table=balls_on_table)
        last_contact_t = float('-inf')
        cue.position[0] = 0
        cue.position[1] = game.table.H + 0.1
        cue.position[2] = game.table.L * 0.3
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
        from .vr_input import calc_cue_transformation, calc_cue_contact_velocity, axis_callbacks, button_press_callbacks
        button_press_callbacks[openvr.k_EButton_ApplicationMenu] = reset
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
    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        if glyphs:
            glyph_meshes = physics.glyph_meshes(game.t)
        else:
            glyph_meshes = []
        with renderer.render(meshes=meshes+glyph_meshes) as frame_data:
            if isinstance(renderer, OpenVRRenderer) and frame_data:
                renderer.process_input(button_press_callbacks=button_press_callbacks,
                                       axis_callbacks=axis_callbacks)
                hmd_pose = frame_data['hmd_pose']
                camera_position[:] = hmd_pose[:, 3]
                controller_poses = frame_data['controller_poses']
                if len(controller_poses) > 0:
                    if len(controller_poses) == 2:
                        pose_0, pose_1 = controller_poses
                    elif len(controller_poses) == 1:
                        controller_indices = frame_data['controller_indices']
                        if controller_indices[0] == 0:
                            pose_0 = controller_poses[0]
                            pose_1 = np.zeros((3,4), dtype=np.float64)
                            pose_1[0,0] = pose_1[1,1] = pose_1[2,2] = 1
                        else:
                            pose_0 = np.zeros((3,4), dtype=np.float64)
                            pose_0[0,0] = pose_0[1,1] = pose_0[2,2] = 1
                            pose_1 = controller_poses[0]
                    calc_cue_transformation(pose_0, pose_1, out=cue.world_matrix)
                    cue.velocity = frame_data['controller_velocities'][0]
                    cue.angular_velocity = frame_data['controller_angular_velocities'][0]
                    if use_ode and isinstance(physics, ODEPoolPhysics):
                        set_quaternion_from_matrix(pose_0[:, :3], cue.quaternion)
            elif isinstance(renderer, OpenGLRenderer):
                if use_ode and isinstance(physics, ODEPoolPhysics):
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

        game.step(speed*dt)

        if not contact_last_frame:
            if game.t - last_contact_t >= 2:
                for i, position in cue.aabb_check(game.ball_positions[:1], physics.ball_radius):
                    r_c = cue.contact(position, physics.ball_radius)
                    if r_c is not None:
                        if isinstance(renderer, OpenVRRenderer) and frame_data and len(frame_data['controller_poses']) == 2:
                            pose_0, pose_1 = frame_data['controller_poses']
                            r_0, r_1 = pose_0[:,3], pose_1[:,3]
                            v_0, v_1 = frame_data['controller_velocities']
                            v_c = calc_cue_contact_velocity(r_c, r_0, r_1, v_0, v_1)
                        else:
                            v_c = cue.velocity
                        physics.strike_ball(game.t, i, game.ball_positions[i], r_c, v_c, cue.mass)
                        last_contact_t = game.t
                        contact_last_frame = True
                        if isinstance(renderer, OpenVRRenderer):
                            renderer.vr_system.triggerHapticPulse(renderer._controller_indices[0],
                                                                  0, int(np.linalg.norm(cue.velocity)**2 / 1.7 * 2700))
                        break
        else:
            contact_last_frame = False

        max_frame_time = max(max_frame_time, dt)
        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        glfw.SwapBuffers(window)

    if nframes > 1:
        _logger.info('...exited render loop: average FPS: %f, maximum frame time: %f, average frame time: %f',
                     (nframes - 1) / (t - st), max_frame_time, (t - st) / (nframes - 1))

    from .physics.events import PhysicsEvent
    _logger.debug(PhysicsEvent.events_str(physics.events))

    renderer.shutdown()
    _logger.info('...shut down renderer')
    glfw.DestroyWindow(window)
    glfw.Terminate()
    _logger.info('GOODBYE')
