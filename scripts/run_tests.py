import unittest
import logging


from tests.physics_tests import *


if __name__ == "__main__":
    FORMAT = '  POOLVR.PY 0.0.1  | %(asctime)s | %(name)s --- %(levelname)s *** %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    unittest.main()
