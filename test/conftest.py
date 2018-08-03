import logging
_logger = logging.getLogger(__name__)
from sys import stdout
import os.path
import numpy as np
import pytest


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')
logging.getLogger('matplotlib').setLevel(logging.WARNING)


@pytest.fixture
def pool_table():
    from poolvr.table import PoolTable
    return PoolTable()


@pytest.fixture(params=['simple', 'marlow'])
def pool_physics(request, pool_table):
    from poolvr.physics import PoolPhysics
    return PoolPhysics(initial_positions=np.array(pool_table.calc_racked_positions(), dtype=np.float64),
                       ball_collision_model=request.param)


@pytest.fixture
def plot_motion(pool_physics, request):
    from .utils import plot_ball_motion as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(0, pool_physics,
         title=test_name + ' (position)',
         coords=(0,2),
         filename=os.path.join(PLOTS_DIR, test_name + '.png'),
         show=True)


@pytest.fixture
def plot_motion_x_position(pool_physics, request):
    from .utils import plot_ball_motion as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(0, pool_physics,
         title=test_name + " ($x$ position)",
         coords=(0,),
         filename=os.path.join(PLOTS_DIR, test_name + '_x.png'),
         show=True)


@pytest.fixture
def plot_motion_z_position(pool_physics, request):
    from .utils import plot_ball_motion as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(0, pool_physics,
         title=test_name + " ($z$ position)",
         coords=(2,),
         collision_depth=1,
         filename=os.path.join(PLOTS_DIR, test_name + '_z.png'),
         show=True)


@pytest.fixture
def plot_energy(pool_physics, request):
    from .utils import plot_energy as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(pool_physics, title=test_name + ' (energy)',
         filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'),
         show=True)


@pytest.fixture
def plot_motion_timelapse(pool_physics, pool_table, request):
    from .utils import plot_motion_timelapse as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(pool_physics, table=pool_table,
         title=test_name + ' (timelapse)',
         filename=os.path.join(PLOTS_DIR, test_name + '_timelapse.png'),
         show=True)


@pytest.fixture
def gl_rendering(pool_physics, pool_table):
    yield
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = True
    import cyglfw3 as glfw
    from poolvr.glfw_app import setup_glfw
    from poolvr.keyboard_controls import init_keyboard, set_on_keydown
    from poolvr.game import PoolGame
    logging.getLogger('poolvr.gl_rendering').setLevel(logging.WARNING)
    physics = pool_physics
    table = pool_table
    window_size = [960, 680]
    window, renderer = setup_glfw(width=window_size[0], height=window_size[1],
                                  double_buffered=True, multisample=4)
    camera_world_matrix = renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = table.height + 0.6
    camera_position[2] = table.length - 0.1
    game = PoolGame(physics=physics, table=table)
    ball_positions = game.ball_positions
    ball_meshes = table.setup_balls(physics.ball_radius, game.ball_colors[:9], ball_positions,
                                    striped_balls=set(range(9, physics.num_balls)),
                                    use_bb_particles=False)
    ball_shadow_meshes = [mesh.shadow_mesh for mesh in ball_meshes]
    meshes = [table.mesh] + ball_meshes + ball_shadow_meshes
    for mesh in meshes:
        mesh.init_gl(force=True)
    ball_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_meshes]
    ball_shadow_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_shadow_meshes]
    process_keyboard_input = init_keyboard(window)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_R and action == glfw.PRESS:
            physics.reset()
    set_on_keydown(window, on_keydown)
    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix)
    _logger.info('entering render loop...')
    stdout.flush()
    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        with renderer.render(meshes=meshes):# as frame_data:
            physics.eval_positions(physics.t, out=ball_positions)
            ball_positions[~pool_physics._on_table] = camera_position # hacky way to only show balls that are on table
            for i, pos in enumerate(ball_positions):
                ball_mesh_positions[i][:] = pos
                ball_shadow_mesh_positions[i][0::2] = pos[0::2]
        physics.t += dt
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
