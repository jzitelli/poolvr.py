import unittest
import logging
import argparse


from tests.physics_tests import *
from tests.ode_physics_tests import *


_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="view OpenGL rendering of test outcomes",
                        action="store_true")
    args = parser.parse_args()

    FORMAT = '\n[%(levelname)s] POOLVR.PY 0.0.1  ###  %(asctime)11s  ***  %(name)s  ---  %(funcName)s:\n%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    _logger.debug('verbose logging enabled')

    if args.v:
        _logger.debug('OpenGL enabled')
        PhysicsTests.show = True
        ODEPhysicsTests.show = True

    unittest.main()
