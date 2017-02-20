import unittest
import logging


from tests.physics_tests import *


if __name__ == "__main__":
    FORMAT = '[%(levelname)8s] POOLVR.PY 0.0.1  |  %(asctime)11s  |  %(name)s - %(funcName)s:\n%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    unittest.main()
