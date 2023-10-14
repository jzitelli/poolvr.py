import logging
_logger = logging.getLogger(__name__)
from sys import stdout
import os.path
import pytest


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption("--save-plots", help="save plots created by tests",
                     action="store_true", default=False)
    parser.addoption("--show-plots", help="show plots created by tests",
                     action="store_true", default=False)
    parser.addoption('--sanity-check', help='enable physics sanity check',
                     action='store_true')
    parser.addoption('--collision-model', metavar='<name of collision model>',
                     help="set the ball-to-ball collision model",
                     default=None)
    parser.addoption('--quartic-solver', metavar='<name of quartic solver function>',
                     help="set the function used to solve quartic polynomials",
                     default=None)
    parser.addoption('--no-distance-check',
                     help="disable checking that every pair of balls is separated by at least one ball diameter",
                     action="store_true")


def pytest_generate_tests(metafunc):
    if "pool_physics" in metafunc.fixturenames or "pool_physics_realtime" in metafunc.fixturenames:
        if metafunc.config.getoption("--collision-model"):
            metafunc.parametrize('ball_collision_model',
                                 [metafunc.config.getoption("--collision-model")])
        else:
            metafunc.parametrize('ball_collision_model',
                                 ['simple', 'simulated', 'fsimulated'])
    if "poly_solver" in metafunc.fixturenames:
        if metafunc.config.getoption("--quartic-solver"):
            metafunc.parametrize('func', [metafunc.config.getoption("--quartic-solver")])
        else:
            metafunc.parametrize('func', ['quartic_solve', 'c_quartic_solve', 'f_quartic_solve'])


@pytest.mark.parametrize("func", ['quartic_solve', 'c_quartic_solve', 'f_quartic_solve'])
@pytest.fixture
def poly_solver(request, func):
    import poolvr.physics.poly_solvers as poly_solvers
    return getattr(poly_solvers, func)


@pytest.fixture
def pool_table():
    from poolvr.table import PoolTable
    from poolvr.physics.events import PhysicsEvent
    return PoolTable(ball_radius=PhysicsEvent.ball_radius)


@pytest.mark.parametrize("ball_collision_model", ['simple', 'simulated', 'fsimulated'])
@pytest.fixture
def pool_physics(pool_table, request, ball_collision_model):
    from poolvr.physics import PoolPhysics
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model=ball_collision_model)


@pytest.fixture
def pool_physics_realtime(pool_table, request, ball_collision_model):
    from poolvr.physics import PoolPhysics
    return PoolPhysics(initial_positions=pool_table.calc_racked_positions(),
                       ball_collision_model=ball_collision_model,
                       collision_search_time_limit=0.005,
                       collision_search_time_forward=2/90.0)


@pytest.fixture
def plot_motion(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_ball_motion as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'position', param_str, 'png'])
    plot(0, pool_physics,
         title=request.node.name + ' (position)',
         coords=(0,2),
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_motion_x_position(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_ball_motion as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'x-position', param_str, 'png'])
    plot(0, pool_physics,
         title=request.node.name + " ($x$ position)",
         coords=(0,),
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_motion_z_position(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_ball_motion as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'z-position', param_str, 'png'])
    plot(0, pool_physics,
         title=request.node.name + " ($z$ position)",
         coords=(2,),
         collision_depth=1,
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
         show=show_plots)



@pytest.fixture
def plot_energy(pool_physics, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_energy as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'energy', param_str, 'png'])
    plot(pool_physics, title=request.node.name + ' (energy)',
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_motion_timelapse(pool_physics, pool_table, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_motion_timelapse as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'timelapse', param_str, 'png'])
    plot(pool_physics, table=pool_table,
         title=request.node.name + ' (timelapse)',
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_initial_positions(pool_physics, pool_table, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_motion_timelapse as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'initial-positions', param_str, 'png'])
    plot(pool_physics, table=pool_table,
         nt=0, t_0=0.0, t_1=0.0,
         title=request.node.name + ' (initial positions)',
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
         show=show_plots)


@pytest.fixture
def plot_final_positions(pool_physics, pool_table, request):
    show_plots, save_plots = request.config.getoption('--show-plots'), request.config.getoption('--save-plots')
    if not (show_plots or save_plots):
        yield
        return
    from utils import plot_motion_timelapse as plot
    yield
    param_str = request.node.name[len(request.node.originalname)+1:-1]
    filename = '.'.join([request.node.originalname, 'final-positions', param_str, 'png'])
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
         title=request.node.name + ' (final positions)',
         filename=os.path.join(PLOTS_DIR, filename) if save_plots else None,
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
