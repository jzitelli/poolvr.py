import os.path
import numpy as np


from .gl_rendering import CubeTexture, Material, Mesh
from .primitives import BoxPrimitive, PlanePrimitive
from .techniques import SKYBOX_TECHNIQUE, PHONG_NORMAL_DIFFUSE_ROUGHNESS_TECHNIQUE


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.pardir,
                            'textures')


skybox_geom = BoxPrimitive(500,500,500)
skybox_geom.attributes['a_position'] = skybox_geom.attributes['vertices']
skybox_material = Material(SKYBOX_TECHNIQUE,
                           textures={'u_map': CubeTexture([os.path.join(TEXTURES_DIR, 'cube',
                                                                        'skybox', '%s.jpg' % fn)
                                                           for fn in 'px nx py ny pz nz'.split()])})
skybox_mesh = Mesh({skybox_material: [skybox_geom]})

floor_geom = PlanePrimitive(width=3, height=0, depth=3)
floor_geom.attributes['a_position'] = floor_geom.attributes['vertices']
floor_geom.attributes['a_texcoord'] = floor_geom.attributes['uvs']
floor_geom.attributes['a_texcoord'] *= 16
floor_geom.attributes['a_normal'] = floor_geom.attributes['normals']
floor_geom.attributes['a_tangent'] = floor_geom.attributes['tangents']

light_position = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
u_light_position = light_position.copy()

# floor_material = Material(PHONG_NORMAL_DIFFUSE_ROUGHNESS_TECHNIQUE,
#                           values={'u_light_position': u_light_position})
# floor_mesh = Mesh({floor_material: [floor_geom]},
#                   before_draw=lambda self, frame_data: frame_data['view_matrix'].dot(light_position, out=u_light_position))

floor_material = Material(PHONG_NORMAL_DIFFUSE_ROUGHNESS_TECHNIQUE,
                          values={'u_light_position': u_light_position},
                          on_use=lambda self, frame_data: frame_data['view_matrix'].T.dot(light_position, out=u_light_position))
floor_mesh = Mesh({floor_material: [floor_geom]})
