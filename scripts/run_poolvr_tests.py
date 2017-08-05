import unittest
import logging
import argparse


_logger = logging.getLogger(__name__)


def main(args):
    # from tests.physics_tests import PhysicsTests
    # from tests.ode_physics_tests import ODEPhysicsTests
    # from tests.opengl_tests import OpenGLTests
    from tests.gltf_tests import GLTFTests
    if args.v:
        _logger.debug('OpenGL enabled')
        GLTFTests.show = True
        # OpenGLTests.show = True
        # ODEPhysicsTests.show = True
        # PhysicsTests.show = True
    unittest.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="view OpenGL rendering of test outcomes",
                        action="store_true")
    # parser.add_argument('--module', dest='module', default=None)
    args = parser.parse_args()

    FORMAT = '\n[%(levelname)s] ## %(asctime)11s ** %(name)s -- %(funcName)s:\n%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    _logger.debug('verbose logging enabled')

    from tests.gltf_tests import GLTFTests

    # main(args)
    if args.v:
        _logger.debug('OpenGL enabled')
        # PhysicsTests.show = True
        # ODEPhysicsTests.show = True
        # OpenGLTests.show = True
        GLTFTests.show = True

    unittest.main()
