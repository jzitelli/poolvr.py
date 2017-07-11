import logging
import sys
import argparse
import os.path
_here = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_here, os.path.pardir, 'utils')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action="store_true", help='verbose logging')
    args = parser.parse_args()

    FORMAT = '%(asctime)s %(name)12s: %(message)s'
    logging.basicConfig(level=logging.INFO if not args.v else logging.DEBUG, format=FORMAT)
    import texutils
    texutils.gen_sphere_billboards(fp_alpha='textures/sphere_bb_alpha.png',
                                   fp_normal='textures/sphere_bb_normal.png')

    import bpyutils
    if bpyutils is not None:
        filename = 'WebVRDesk1.blend'
        bpyutils.load_scene(os.path.join(_here, os.path.pardir, 'models', filename))
        bpyutils.create_material()
        bpyutils.bake_all_faces(os.path.join(_here, os.path.pardir, 'models', 'textures',
                                             '%s-baked_faces.png' % filename[:-6]))
