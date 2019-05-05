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
    ball_quaternions = np.zeros((16,4), dtype=np.float32)
    ball_quaternions[:,3] = 1
    ball_angular_velocities = np.array(np.random.rand(16,3), dtype=np.float32)
    with open(os.path.join(os.path.dirname(poolvr.__file__),
                           'shaders',
                           'sphere_projection_fs.glsl')) as f:
        fs_src = f.read()
    def on_use(material, dt=None, **frame_data):
        if dt is not None:
            for q, omega in zip(ball_quaternions, ball_angular_velocities):
                q_w = q[3]
                q[3] -= 0.5 * dt * omega.dot(q[:3])
                q[:3] += 0.5 * dt * (q_w * omega + np.cross(omega, q[:3]))
                q /= np.sqrt(np.dot(q, q))
        material.values['u_camera'] = frame_data['camera_matrix']
        material.values['u_projection_lrbt'] = frame_data['projection_lrbt']
        material.values['u_znear'] = frame_data['znear']
        material.values['iResolution'] = frame_data['window_size']
        material.values['ball_positions'] = ball_positions
        material.values['ball_quaternions'] = ball_quaternions
    mesh = FragBox(fs_src, on_use=on_use)
    ball_colors = np.array([[float(c & 0xff0000) / 0xff0000,
                             float(c & 0x00ff00) / 0x00ff00,
                             float(c & 0x0000ff) / 0x0000ff]
                            for c in PoolTable.BALL_COLORS], dtype=np.float32)
    mesh.material.values['ball_colors'] = ball_colors
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
