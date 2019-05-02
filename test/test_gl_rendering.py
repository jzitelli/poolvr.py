import logging
import os.path
_here = os.path.dirname(__file__)
_logger = logging.getLogger(__name__)


from poolvr.gl_rendering import Material, Mesh
from poolvr.gl_techniques import LAMBERT_TECHNIQUE


def test_frag_box(render_meshes):
    if render_meshes is None:
        return
    import poolvr
    from poolvr.gl_rendering import FragBox
    from poolvr.table import PoolTable
    import numpy as np
    ball_positions = np.array(PoolTable().calc_racked_positions(), dtype=np.float32)
    with open(os.path.join(os.path.dirname(poolvr.__file__), 'shaders', 'sphere_projection_fs.glsl')) as f:
        fs_src = f.read()
    def on_use(material, **frame_data):
        material.values['u_view'] = frame_data['view_matrix']
        material.values['u_camera'] = frame_data['camera_matrix']
        material.values['u_projection_lrbt'] = frame_data['projection_lrbt']
        material.values['u_znear'] = frame_data['znear']
        material.values['iResolution'] = frame_data['window_size']
        material.values['ball_positions'] = ball_positions
    mesh = FragBox(fs_src, on_use=on_use)
    render_meshes.append(mesh)


def test_cone_mesh(render_meshes):
    from poolvr.gl_primitives import ConeMesh
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
    mesh = ConeMesh(material, radius=0.15, height=0.3)
    for prim in mesh.primitives[material]:
        prim.attributes['a_position'] = prim.attributes['vertices']
    mesh.world_matrix[3,2] = -3
    render_meshes.append(mesh)


def test_sphere_mesh(render_meshes):
    from poolvr.gl_primitives import SpherePrimitive
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 1.0, 1.0, 0.0]})
    prim = SpherePrimitive(radius=0.1)
    prim.attributes['a_position'] = prim.attributes['vertices']
    mesh = Mesh({material: [prim]})
    mesh.world_matrix[3,2] = -3
    render_meshes.append(mesh)


def test_cylinder_mesh(render_meshes):
    from poolvr.gl_primitives import CylinderMesh
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
    mesh = CylinderMesh(material=material, radius=0.15, height=0.5)
    for prim in mesh.primitives[material]:
        prim.attributes['a_position'] = prim.attributes['vertices']
    mesh.world_matrix[3,2] = -3
    render_meshes.append(mesh)


def test_arrow_mesh(render_meshes):
    from poolvr.gl_primitives import ArrowMesh
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
    mesh = ArrowMesh(material=material)
    for prim in mesh.primitives[material]:
        prim.attributes['a_position'] = prim.attributes['vertices']
    mesh.world_matrix[3,2] = -3
    render_meshes.append(mesh)
