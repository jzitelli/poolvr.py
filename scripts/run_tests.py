import unittest
import logging


from tests.physics_tests import *


if __name__ == "__main__":
    FORMAT = '%(levelname)s *** %(asctime)s --- %(name)s - %(funcName)s:\n%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    unittest.main()
