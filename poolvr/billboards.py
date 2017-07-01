import pkgutil
import os.path
from ctypes import c_void_p
import numpy as np
import OpenGL.GL as gl
import OpenGL.error


from .gl_rendering import Node, Technique, Program, DTYPE_COMPONENT_TYPE, Texture
from .primitives import PlanePrimitive


NULL_PTR = c_void_p(0)


# TODO: pkgutils way
TEXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.pardir,
                            'textures')


class BillboardParticles(Node):
    technique = Technique(Program(pkgutil.get_data('poolvr', 'shaders/bb_particles_vs.glsl').decode(),
                                  pkgutil.get_data('poolvr', 'shaders/bb_particles_fs.glsl').decode()),
                          attributes={'position': {'type': gl.GL_FLOAT_VEC3},
                                      'uv': {'type': gl.GL_FLOAT_VEC2},
                                      'translate': {'type': gl.GL_FLOAT_VEC3},
                                      'color': {'type': gl.GL_FLOAT_VEC3}},
                          uniforms={'map': {'type': gl.GL_SAMPLER_2D},
                                    'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                    'u_projection': {'type': gl.GL_FLOAT_MAT4}})
    _modelview = np.eye(4, dtype=np.float32)
    def __init__(self, texture, num_particles=1, scale=1.0, color=None, translate=None):
        Node.__init__(self)
        self.texture = texture
        self.num_particles = num_particles
        if color is None:
            color = np.array([num_particles*[1.0, 1.0, 1.0]], dtype=np.float32)
        if translate is None:
            translate = np.array([[1.1*scale*i, 0.2, 0.0] for i in range(num_particles)], dtype=np.float32)
        self.primitive = PlanePrimitive(width=scale, height=scale,
                                        color=color, translate=translate,
                                        attribute_usage={'color': gl.GL_STATIC_DRAW,
                                                         'translate': gl.GL_DYNAMIC_DRAW})
        self.primitive.attributes['position'] = self.primitive.attributes['vertices']
        self.primitive.attributes['uv'] = self.primitive.attributes['uvs']
        self._initialized = False
    def init_gl(self, force=False):
        if self._initialized:
            if not force: return
        self.texture.init_gl(force=force)
        self.technique.init_gl(force=force)
        self.primitive.init_gl(force=force)
        self._initialized = True
    def update_gl(self):
        if not self._initialized: self.init_gl()
        translate = self.primitive.attributes['translate']
        values = translate.tobytes()
        try:
            gl.glNamedBufferSubData(self.primitive.buffers['translate'], 0, len(values), values)
        except OpenGL.error.NullFunctionError as e:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.primitive.buffers['translate'])
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, len(values), values)
    def draw(self, view=None, projection=None, frame_data=None):
        self.technique.use()
        if view is not None:
            self.world_matrix.dot(view, out=self._modelview)
            gl.glUniformMatrix4fv(self.technique.uniform_locations['u_modelview'], 1, False, self._modelview)
        if projection is not None:
            gl.glUniformMatrix4fv(self.technique.uniform_locations['u_projection'], 1, False, projection)
        gl.glActiveTexture(gl.GL_TEXTURE0+0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.texture_id)
        gl.glBindSampler(0, self.texture.sampler_id)
        gl.glUniform1i(self.technique.uniform_locations['map'], 0)
        for attribute_name, location in self.technique.attribute_locations.items():
            attribute = self.primitive.attributes[attribute_name]
            gl.glEnableVertexAttribArray(location)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.primitive.buffers[attribute_name])
            gl.glVertexAttribPointer(location, attribute.shape[-1],
                                     DTYPE_COMPONENT_TYPE[attribute.dtype], False,
                                     attribute.dtype.itemsize * attribute.shape[-1],
                                     NULL_PTR)
            if attribute_name == 'translate' or attribute_name == 'color':
                gl.glVertexAttribDivisor(location, 1)
            else:
                gl.glVertexAttribDivisor(location, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.primitive.index_buffer)
        gl.glDrawElementsInstanced(self.primitive.mode, self.primitive.indices.size,
                                   DTYPE_COMPONENT_TYPE[self.primitive.indices.dtype], NULL_PTR, self.num_particles)
        # for location in self.technique.attribute_locations.values():
        #     gl.glDisableVertexAttribArray(location)
        self.technique.release()
