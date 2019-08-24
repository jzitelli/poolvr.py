import os.path
import numpy as np


from .gl_rendering import Material
from .gl_primitives import SkyBoxMesh, PlaneMesh
from .gl_techniques import PHONG_NORMAL_DIFFUSE_ROUGHNESS_TECHNIQUE


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.pardir,
                            'textures')


skybox_mesh = SkyBoxMesh([os.path.join(TEXTURES_DIR, 'cube', 'skybox', '%s.png' % suffix)
                          for suffix in 'px nx py ny pz nz'.split()])


light_position = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
u_light_position = light_position.copy()
floor_material = Material(PHONG_NORMAL_DIFFUSE_ROUGHNESS_TECHNIQUE,
                          values={'u_light_position': u_light_position},
                          on_use=lambda self, **frame_data: frame_data['view_matrix'].T.dot(light_position, out=u_light_position))
floor_mesh = PlaneMesh(floor_material, width=3, height=0, depth=3)
floor_mesh.alias('vertices', 'a_position')
floor_mesh.alias('uvs', 'a_texcoord')
floor_mesh.alias('normals', 'a_normal')
floor_mesh.alias('tangents', 'a_tangent')
floor_mesh.primitive.attributes['uvs'] *= 16
