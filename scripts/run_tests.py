import unittest
import logging
import argparse


import tests.physics_tests
from tests.physics_tests import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",
                        help="show a real-time OpenGL rendering of the test outcomes",
                        action="store_true")
    args = parser.parse_args()
    FORMAT = '\n[%(levelname)s] POOLVR.PY 0.0.1  ###  %(asctime)11s  ***  %(name)s  ---  %(funcName)s:\n%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    tests.physics_tests.show = args.show
    unittest.main()
