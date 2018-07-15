import logging
_logger = logging.getLogger(__name__)
import os.path
import numpy as np
import pytest


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')


@pytest.fixture
def pool_table():
    from poolvr.table import PoolTable
    return PoolTable()


@pytest.fixture()
def pool_physics(request, pool_table):
    from poolvr.physics import PoolPhysics
    ball_collision_model = getattr(request, 'param', 'simple')
    _logger.debug("getattr(request, 'param', None) = %s", getattr(request, 'param', None))
    return PoolPhysics(initial_positions=np.array(pool_table.calc_racked_positions(), dtype=np.float64),
                       ball_collision_model=ball_collision_model)


@pytest.fixture
def plot_motion(pool_physics, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_ball_motion as plot
    yield
    test_name = request.function.__name__
    plot(0, pool_physics,
         title=test_name + ' (position)',
         coords=(0,2),
         filename=os.path.join(PLOTS_DIR, test_name + '.png'),
         show=True)


@pytest.fixture
def plot_motion_x_position(pool_physics, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_ball_motion as plot
    yield
    test_name = request.function.__name__
    plot(0, pool_physics,
         title=test_name + " ($x$ position)",
         coords=(0,),
         filename=os.path.join(PLOTS_DIR, test_name + '_x.png'),
         show=True)


@pytest.fixture
def plot_motion_z_position(pool_physics, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_ball_motion as plot
    yield
    test_name = request.function.__name__
    plot(0, pool_physics,
         title=test_name + " ($z$ position)",
         coords=(2,),
         collision_depth=1,
         filename=os.path.join(PLOTS_DIR, test_name + '_z.png'),
         show=True)


@pytest.fixture
def plot_energy(pool_physics, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_energy as plot
    yield
    test_name = request.function.__name__
    plot(pool_physics, title=test_name + ' (energy)',
         filename=os.path.join(PLOTS_DIR, test_name + '_energy.png'),
         show=True)


@pytest.fixture
def plot_motion_timelapse(pool_physics, pool_table, request):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    from .utils import plot_motion_timelapse as plot
    yield
    test_name = request.function.__name__
    plot(pool_physics, table=pool_table,
         title=test_name + ' (timelapse)',
         filename=os.path.join(PLOTS_DIR, test_name + '_timelapse.png'),
         show=True)
