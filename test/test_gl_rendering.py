import logging
import os.path
_here = os.path.dirname(__file__)
_logger = logging.getLogger(__name__)


from poolvr.gl_rendering import Material, Mesh
from poolvr.gl_techniques import LAMBERT_TECHNIQUE


def test_frag_box(gl_rendering, meshes):
    import poolvr
    from poolvr.gl_rendering import FragBox
    with open(os.path.join(os.path.dirname(poolvr.__file__), 'shaders', 'sphere_projection_fs.glsl')) as f:
        fs_src = f.read()
    def on_use(material, **frame_data):
        if 'iTime' not in material.values:
            material.values['iTime'] = 0.0
        material.values['iTime'] += frame_data.get('dt', 1.0/60)
        material.values['u_view_matrix'] = frame_data.get('u_view_matrix')
        material.values['camera_position'] = frame_data.get('camera_position')
    mesh = FragBox(fs_src, on_use=on_use)
    meshes.append(mesh)


def test_cone_mesh(gl_rendering, meshes):
    from poolvr.gl_primitives import ConeMesh
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
    mesh = ConeMesh(material, radius=0.15, height=0.3)
    for prim in mesh.primitives[material]:
        prim.attributes['a_position'] = prim.attributes['vertices']
    mesh.world_matrix[3,2] = -3
    meshes.append(mesh)


def test_sphere_mesh(gl_rendering, meshes):
    from poolvr.gl_primitives import SpherePrimitive
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [0.0, 1.0, 1.0, 0.0]})
    prim = SpherePrimitive(radius=0.1)
    prim.attributes['a_position'] = prim.attributes['vertices']
    mesh = Mesh({material: [prim]})
    mesh.world_matrix[3,2] = -3
    meshes.append(mesh)


def test_cylinder_mesh(gl_rendering, meshes):
    from poolvr.gl_primitives import CylinderMesh
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
    mesh = CylinderMesh(material=material, radius=0.15, height=0.5)
    for prim in mesh.primitives[material]:
        prim.attributes['a_position'] = prim.attributes['vertices']
    mesh.world_matrix[3,2] = -3
    meshes.append(mesh)


def test_arrow_mesh(gl_rendering, meshes):
    from poolvr.gl_primitives import ArrowMesh
    material = Material(LAMBERT_TECHNIQUE, values={'u_color': [1.0, 1.0, 0.0, 0.0]})
    mesh = ArrowMesh(material=material)
    for prim in mesh.primitives[material]:
        prim.attributes['a_position'] = prim.attributes['vertices']
    mesh.world_matrix[3,2] = -3
    meshes.append(mesh)
