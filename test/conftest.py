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
                     default='1680x1050')
    parser.addoption('--keep-render-window', help='do not close OpenGL render window when all events have finished',
                     action='store_true')
    parser.addoption('--vr', help='display in VR',
                     action='store_true')
    parser.addoption('--sanity-check', help='enable physics sanity check',
                     action='store_true')
    parser.addoption('--render-method', metavar='<method name>',
                     help='OpenGL rendering method/style to use, one of: "ega", "lambert", "billboards", "raycast"',
                     default='lambert')


@pytest.fixture
def pool_table():
    from poolvr.table import PoolTable
    return PoolTable()


@pytest.fixture
def pool_physics(pool_table, request):
    from poolvr.physics import PoolPhysics
    enable_sanity_check = request.config.getoption('--sanity-check')
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model='simple',
                       enable_sanity_check=enable_sanity_check,
                       enable_occlusion=False)


@pytest.fixture
def pool_physics_realtime(pool_table, request):
    from poolvr.physics import PoolPhysics
    enable_sanity_check = request.config.getoption('--sanity-check')
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model='simple',
                       enable_sanity_check=enable_sanity_check,
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
def meshes():
    return []


@pytest.fixture
def gl_rendering(pool_physics, pool_table, request, meshes):
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
    render_method = request.config.getoption('--render-method')
    yield

    import numpy as np
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = True
    import cyglfw3 as glfw
    from poolvr.glfw_app import setup_glfw, capture_window
    from poolvr.app import KB_MOVE_SPEED, KB_TURN_SPEED
    from poolvr.keyboard_controls import (init_keyboard, key_state,
                                          KEY_LEFT, KEY_RIGHT,
                                          KEY_W, KEY_S, KEY_A, KEY_D, KEY_Q, KEY_Z)
    from poolvr.game import PoolGame
    from poolvr.gl_rendering import set_matrix_from_quaternion
    logging.getLogger('poolvr.gl_rendering').setLevel(logging.WARNING)
    if render_method == 'ega':
        from poolvr.gl_techniques import EGA_TECHNIQUE as technique
    elif render_method == 'lambert':
        from poolvr.gl_techniques import LAMBERT_TECHNIQUE as technique
    physics = pool_physics
    table = pool_table
    game = PoolGame(physics=pool_physics, table=table)
    title = '_'.join([request.function.__name__, pool_physics.ball_collision_model])
    window, renderer = setup_glfw(window_size=[xres, yres], double_buffered=True,
                                  multisample=int(msaa), title=title)
    camera_world_matrix = renderer.camera_matrix
    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = table.H + 0.6
    camera_position[2] = table.L - 0.1
    table_mesh = table.export_mesh(surface_technique=technique, cushion_technique=technique, rail_technique=technique)
    ball_meshes = table.export_ball_meshes(technique=technique)
    ball_shadow_meshes = [mesh.shadow_mesh for mesh in ball_meshes]
    for ball_mesh, shadow_mesh, on_table in zip(ball_meshes, ball_shadow_meshes, physics._on_table):
        if not on_table:
            ball_mesh.visible = False
            shadow_mesh.visible = False
    if not meshes:
        meshes = [table_mesh] + ball_meshes + ball_shadow_meshes
    else:
        glyphs = False
    for mesh in meshes:
        mesh.init_gl(force=True)
    ball_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_meshes]
    ball_mesh_rotations = [mesh.world_matrix[:3,:3].T for mesh in ball_meshes]
    ball_shadow_mesh_positions = [mesh.world_matrix[3,:3] for mesh in ball_shadow_meshes]
    init_keyboard(window)
    theta = 0.0
    def process_keyboard_input(dt, camera_world_matrix):
        nonlocal theta
        theta += KB_TURN_SPEED * dt * (key_state[KEY_LEFT] - key_state[KEY_RIGHT])
        sin, cos = np.sin(theta), np.cos(theta)
        camera_world_matrix[0,0] = cos
        camera_world_matrix[0,2] = -sin
        camera_world_matrix[2,0] = sin
        camera_world_matrix[2,2] = cos
        dist = dt * KB_MOVE_SPEED
        camera_world_matrix[3,:3] += \
            dist*(key_state[KEY_S]-key_state[KEY_W]) * camera_world_matrix[2,:3] \
          + dist*(key_state[KEY_D]-key_state[KEY_A]) * camera_world_matrix[0,:3] \
          + dist*(key_state[KEY_Q]-key_state[KEY_Z]) * camera_world_matrix[1,:3]
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
        game.step(speed*dt)
        with renderer.render(meshes=meshes+glyph_meshes, dt=dt):
            for i, pos in enumerate(game.ball_positions):
                ball_mesh_positions[i][:] = pos
                set_matrix_from_quaternion(game.ball_quaternions[i], out=ball_mesh_rotations[i])
                ball_shadow_mesh_positions[i][0::2] = pos[0::2]
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
                     1 / max_frame_time,
                     (t - st) / (nframes - 1),
                     max_frame_time)
    if should_screenshot:
        if physics.events:
            t_end = physics.events[-1].t
        with renderer.render(meshes=meshes):
            physics.eval_positions(t_end, out=game.ball_positions)
            for i, pos in enumerate(game.ball_positions):
                ball_mesh_positions[i][:] = pos
                set_matrix_from_quaternion(game.ball_quaternions[i], out=ball_mesh_rotations[i])
                ball_shadow_mesh_positions[i][0::2] = pos[0::2]
        glfw.SwapBuffers(window)
        capture_window(window, filename=os.path.join(os.path.dirname(__file__), 'screenshots',
                                                     title.replace(' ', '_') + '.png'))
    renderer.shutdown()
    glfw.DestroyWindow(window)
    glfw.Terminate()


@pytest.fixture
def render_meshes(request):
    should_render = request.config.getoption('--render')
    should_screenshot = request.config.getoption('--screenshot')
    if not (should_render or should_screenshot):
        yield None
        return
    xres, yres = [int(n) for n in request.config.getoption('--resolution').split('x')]
    msaa = int(request.config.getoption('--msaa'))
    vr = request.config.getoption('--vr')
    import numpy as np
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = True
    import cyglfw3 as glfw
    from poolvr.glfw_app import setup_glfw, capture_window
    from poolvr.app import KB_MOVE_SPEED, KB_TURN_SPEED
    from poolvr.keyboard_controls import (init_keyboard, key_state,
                                          KEY_LEFT, KEY_RIGHT,
                                          KEY_W, KEY_S, KEY_A, KEY_D, KEY_Q, KEY_Z)
    window_size = [xres, yres]
    title = request.function.__name__
    window, renderer = setup_glfw(window_size=window_size,
                                  double_buffered=not vr,
                                  multisample=msaa,
                                  title=title)
    camera_world_matrix = renderer.camera_matrix
    if vr:
        from poolvr.pyopenvr_renderer import OpenVRRenderer
        renderer = OpenVRRenderer(multisample=msaa, window_size=window_size)
    class Yielded(list):
        def __init__(self, renderer, window):
            self.renderer = renderer; self.window = window
    meshes = Yielded(renderer, window)
    yield meshes

    camera_position = camera_world_matrix[3,:3]
    camera_position[1] = 0.5
    camera_position[2] = 0.75
    for mesh in meshes:
        mesh.init_gl(force=True)
    init_keyboard(window)
    theta = 0.0
    def process_keyboard_input(dt, camera_world_matrix):
        nonlocal theta
        theta += KB_TURN_SPEED * dt * (key_state[KEY_LEFT] - key_state[KEY_RIGHT])
        sin, cos = np.sin(theta), np.cos(theta)
        camera_world_matrix[0,0] = cos
        camera_world_matrix[0,2] = -sin
        camera_world_matrix[2,0] = sin
        camera_world_matrix[2,2] = cos
        dist = dt * KB_MOVE_SPEED
        camera_world_matrix[3,:3] += \
            dist*(key_state[KEY_S]-key_state[KEY_W]) * camera_world_matrix[2,:3] \
          + dist*(key_state[KEY_D]-key_state[KEY_A]) * camera_world_matrix[0,:3] \
          + dist*(key_state[KEY_Q]-key_state[KEY_Z]) * camera_world_matrix[1,:3]
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
        with renderer.render(meshes=meshes, dt=dt):
            pass
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
                     1 / max_frame_time,
                     (t - st) / (nframes - 1),
                     max_frame_time)
    if should_screenshot:
        with renderer.render(meshes=meshes, dt=0.0):
            pass
        glfw.SwapBuffers(window)
        capture_window(window,
                       filename=os.path.join(os.path.dirname(__file__), 'screenshots',
                                             title.replace(' ', '_') + '.png'))
    renderer.shutdown()
    glfw.DestroyWindow(window)
    glfw.Terminate()
