import logging
_logger = logging.getLogger(__name__)
from sys import stdout
import numpy as np
import pytest


from poolvr.ode_physics import ODEPoolPhysics


@pytest.fixture
def ode_pool_physics(pool_table):
    return ODEPoolPhysics(table=pool_table,
                          initial_positions=np.array(pool_table.calc_racked_positions(),
                                                     dtype=np.float64))


@pytest.fixture
def ode_gl_rendering(ode_pool_physics, pool_table, request):
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = True
    import cyglfw3 as glfw
    yield
    from poolvr.glfw_app import setup_glfw
    from poolvr.keyboard_controls import init_keyboard, set_on_keydown_callback
    from poolvr.game import PoolGame
    logging.getLogger('poolvr.gl_rendering').setLevel(logging.WARNING)
    physics = ode_pool_physics
    table = pool_table
    window_size = [960, 680]
    title = request.function.__name__
    window, renderer = setup_glfw(width=window_size[0], height=window_size[1],
                                  double_buffered=True, multisample=4,
                                  title=title)

    camera_world_matrix = renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = table.height + 0.6
    camera_position[2] = table.length - 0.1
    game = PoolGame(physics=physics, table=table)
    ball_meshes = table.ball_meshes
    ball_shadow_meshes = [mesh.shadow_mesh for mesh in ball_meshes]
    for i, (ball_mesh, shadow_mesh) in enumerate(zip(ball_meshes, ball_shadow_meshes)):
        if i not in physics.balls_on_table:
            ball_mesh.visible = False
            shadow_mesh.visible = False
    meshes = [table.mesh] + ball_meshes + ball_shadow_meshes
    for mesh in meshes:
        mesh.init_gl(force=True)
    ball_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_meshes]
    ball_shadow_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_shadow_meshes]
    process_keyboard_input = init_keyboard(window)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_R and action == glfw.PRESS:
            game.reset()
    set_on_keydown_callback(window, on_keydown)
    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix)
    _logger.info('entering render loop...')
    stdout.flush()
    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    t_end = 5.0
    while not glfw.WindowShouldClose(window) and game.t < t_end:
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        with renderer.render(meshes=meshes):# as frame_data:
            for i, pos in enumerate(game.ball_positions):
                ball_mesh_positions[i][:] = pos
                ball_shadow_mesh_positions[i][0::2] = pos[0::2]
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
    glfw.DestroyWindow(window)
    glfw.Terminate()
