import logging
_logger = logging.getLogger(__name__)
from sys import stdout
import os.path
import pytest


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption("--render", help="display OpenGL-rendered view of test results",
                     action="store_true", default=False)
    parser.addoption("--speed", help="playback speed of rendered view of test results",
                     default='1.0')
    parser.addoption("--screenshot", help="save screenshot of OpenGL-rendered test results",
                     action="store_true", default=False)
    parser.addoption("--save-plots", help="save plots created by tests",
                     action="store_true", default=False)
    parser.addoption("--show-plots", help="show plots created by tests",
                     action="store_true", default=False)
    parser.addoption('--glyphs', help='render velocity and angular velocity glyphs',
                     action='store_true', default=False)
    parser.addoption('--msaa', help='multisample anti-aliasing level (defaults to 4)',
                     default='4')
    parser.addoption('--resolution', help='OpenGL viewport resolution, e.g. 960x680',
                     default='960x680')
    parser.addoption('--keep-render-window', help='do not close OpenGL render window when all events have finished',
                     action='store_true')


@pytest.fixture
def pool_table():
    from poolvr.table import PoolTable
    return PoolTable()


@pytest.fixture
def pool_physics(pool_table, request):
    from poolvr.physics import PoolPhysics
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model='simple',
                       enable_sanity_check=False,
                       enable_occlusion=False)


@pytest.fixture
def pool_physics_realtime(pool_table, request):
    from poolvr.physics import PoolPhysics
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model='simple',
                       enable_sanity_check=False,
                       enable_occlusion=False,
                       realtime=True)


@pytest.fixture
def plot_motion(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_ball_motion as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(0, pool_physics,
         title=test_name + ' (position)',
         coords=(0,2),
         filename=os.path.join(PLOTS_DIR, test_name + '.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_motion_x_position(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_ball_motion as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(0, pool_physics,
         title=test_name + " ($x$ position)",
         coords=(0,),
         filename=os.path.join(PLOTS_DIR, test_name + '-x.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_motion_z_position(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_ball_motion as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(0, pool_physics,
         title=test_name + " ($z$ position)",
         coords=(2,),
         collision_depth=1,
         filename=os.path.join(PLOTS_DIR, test_name + '-z.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_energy(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_energy as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    _logger.debug('plotting energy for %s...', request.function.__name__)
    plot(pool_physics, title=test_name + ' (energy)',
         filename=os.path.join(PLOTS_DIR, test_name + '-energy.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_motion_timelapse(pool_physics, pool_table, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_motion_timelapse as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(pool_physics, table=pool_table,
         title=test_name + ' (timelapse)',
         filename=os.path.join(PLOTS_DIR, test_name + '-timelapse.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_initial_positions(pool_physics, pool_table, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_motion_timelapse as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    plot(pool_physics, table=pool_table,
         nt=0,
         t_0=0.0, t_1=0.0,
         title=test_name + ' (initial positions)',
         filename=os.path.join(PLOTS_DIR, test_name + '-initial-positions.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_final_positions(pool_physics, pool_table, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_motion_timelapse as plot
    yield
    test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    events = pool_physics.events
    if events:
        t1 = events[-1].t
        if events[-1].T < float('inf'):
            t1 += events[-1].T
    else:
        t1 = 0.0
    plot(pool_physics, table=pool_table,
         nt=0,
         t_0=t1, t_1=t1,
         title=test_name + ' (final positions)',
         filename=os.path.join(PLOTS_DIR, test_name + '-final-positions.png') if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_occlusion(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    import matplotlib.pyplot as plt
    yield
    plt.imshow(pool_physics._occ_ij)
    if show_plots:
        plt.show()
    if save_plots:
        try:
            test_name = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
            filename = os.path.join(os.path.dirname(__file__), 'plots', test_name + '-occlusion.png')
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(filename)
            _logger.info('saved plot to "%s"', filename)
        except Exception as err:
            _logger.error(err)
    plt.close()


@pytest.fixture
def gl_rendering(pool_physics, pool_table, request):
    should_render = request.config.getoption('--render')
    should_screenshot = request.config.getoption('--screenshot')
    if not (should_render or should_screenshot):
        yield
        return
    xres, yres = [int(n) for n in request.config.getoption('--resolution').split('x')]
    msaa = request.config.getoption('--msaa')
    glyphs = request.config.getoption('--glyphs')
    speed = float(request.config.getoption('--speed'))
    keep_render_window = request.config.getoption('--keep-render-window')
    yield

    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = True
    import cyglfw3 as glfw
    from poolvr.glfw_app import setup_glfw, capture_window
    from poolvr.keyboard_controls import init_keyboard
    from poolvr.game import PoolGame
    from poolvr.gl_rendering import set_matrix_from_quaternion
    logging.getLogger('poolvr.gl_rendering').setLevel(logging.WARNING)
    physics = pool_physics
    table = pool_table
    game = PoolGame(physics=pool_physics, table=table)
    window_size = [xres, yres]
    title = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    window, renderer = setup_glfw(width=window_size[0],
                                  height=window_size[1],
                                  double_buffered=True,
                                  multisample=int(msaa),
                                  title=title)
    camera_world_matrix = renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = table.H + 0.6
    camera_position[2] = table.L - 0.1
    ball_meshes = table.export_ball_meshes()
    ball_shadow_meshes = [mesh.shadow_mesh for mesh in ball_meshes]
    for ball_mesh, shadow_mesh, on_table in zip(ball_meshes, ball_shadow_meshes, physics._on_table):
        if not on_table:
            ball_mesh.visible = False
            shadow_mesh.visible = False
    meshes = [table.export_mesh()] + ball_meshes + ball_shadow_meshes
    for mesh in meshes:
        mesh.init_gl(force=True)
    ball_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_meshes]
    ball_mesh_rotations = [mesh.world_matrix[:3,:3].T for mesh in ball_meshes]
    ball_shadow_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_shadow_meshes]
    process_keyboard_input = init_keyboard(window)
    def process_input(dt):
        glfw.PollEvents()
        process_keyboard_input(dt, camera_world_matrix)
    _logger.info('entering render loop...')
    stdout.flush()
    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    if should_render:
        t_end = physics.events[-1].t if physics.events else 5.0
    else:
        t_end = game.t
    while not glfw.WindowShouldClose(window) and (keep_render_window or game.t < t_end):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        if glyphs:
            glyph_meshes = physics.glyph_meshes(game.t)
        else:
            glyph_meshes = []
        with renderer.render(meshes=meshes+glyph_meshes):
            for i, pos in enumerate(game.ball_positions):
                ball_mesh_positions[i][:] = pos
                set_matrix_from_quaternion(game.ball_quaternions[i], out=ball_mesh_rotations[i])
                ball_shadow_mesh_positions[i][0::2] = pos[0::2]
        game.step(speed*dt)
        max_frame_time = max(max_frame_time, dt)
        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        glfw.SwapBuffers(window)

    if nframes > 1:
        _logger.info('''...exited render loop:
        average FPS: %f
        minimum FPS: %f
        average frame time: %f
        maximum frame time: %f
        ''',
                     (nframes - 1) / (t - st),
                     max_frame_time,
                     1 / max_frame_time,
                     (t - st) / (nframes - 1))

    if should_screenshot:
        with renderer.render(meshes=meshes):
            physics.eval_positions(t_end, out=game.ball_positions)
            for i, pos in enumerate(game.ball_positions):
                ball_mesh_positions[i][:] = pos
                set_matrix_from_quaternion(game.ball_quaternions[i], out=ball_mesh_rotations[i])
                ball_shadow_mesh_positions[i][0::2] = pos[0::2]
            glfw.SwapBuffers(window)
        capture_window(window,
                       filename=os.path.join(os.path.dirname(__file__), 'screenshots',
                                             title.replace(' ', '_') + '.png'))

    renderer.shutdown()
    glfw.DestroyWindow(window)
    glfw.Terminate()
