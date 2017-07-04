import logging
import sys
import os.path
_here = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_here, os.path.pardir, 'utils')))


if __name__ == "__main__":
    FORMAT = '%(asctime)s %(name)12s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    import texutils
    texutils.gen_sphere_billboards(fp_alpha='textures/sphere_bb_alpha.png',
                                   fp_normal='textures/sphere_bb_normal.png')

    import bpyutils
    if bpyutils is not None:
        bpyutils.create_material()
        bpyutils.bake_all_faces('textures/baked_faces.png')
