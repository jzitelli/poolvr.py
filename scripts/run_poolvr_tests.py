import unittest
import logging
import argparse
import sys


import tests.physics_tests
from tests.physics_tests import *
from tests.physics_tests import PhysicsTests


_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    FORMAT = '\n[%(levelname)s] POOLVR.PY 0.0.1  ###  %(asctime)11s  ***  %(name)s  ---  %(funcName)s:\n%(message)s'

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="view OpenGL rendering of test outcomes",
                        action="store_true")
    # parser.add_argument('-v', "--verbose", help="enable verbose logging",
    #                     action="store_true")
    # parser.add_argument("--novr", help="non-VR mode", action="store_true")
    args = parser.parse_args()

    # if args.verbose:
    #     print('verbose logging enabled')
    #     logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    # else:
    #     print('verbose logging disabled')
    #     logging.basicConfig(format=FORMAT, level=logging.WARNING)
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    _logger.debug('verbose logging enabled')

    if args.v:
        _logger.debug('OpenGL enabled')
        PhysicsTests.show = True

    unittest.main()
