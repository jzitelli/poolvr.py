import re
import copy
from ctypes import c_float, POINTER, c_void_p
from contextlib import contextmanager
import logging
import numpy as np
import PIL.Image as Image
import OpenGL.GL as gl


c_float_p = POINTER(c_float)


NULL_PTR = c_void_p(0)


CHECK_GL_ERRORS = False


DTYPE_COMPONENT_TYPE = {
    np.dtype(np.uint16): gl.GL_UNSIGNED_SHORT,
    np.dtype(np.uint32): gl.GL_UNSIGNED_INT,
    np.dtype(np.int16): gl.GL_SHORT,
    np.dtype(np.int32): gl.GL_INT,
    np.dtype(np.float32): gl.GL_FLOAT
}
DTYPE_COMPONENT_TYPE_INV = {v: k for k, v in DTYPE_COMPONENT_TYPE.items()}


GLSL_TYPE_SPEC = {
    'float': gl.GL_FLOAT,
    'vec2': gl.GL_FLOAT_VEC2,
    'vec3': gl.GL_FLOAT_VEC3,
    'vec4': gl.GL_FLOAT_VEC4,
    'mat4': gl.GL_FLOAT_MAT4,
    'mat3': gl.GL_FLOAT_MAT3,
    'sampler2D': gl.GL_SAMPLER_2D,
    'samplerCube': gl.GL_SAMPLER_CUBE
}


_logger = logging.getLogger(__name__)


class Program(object):
    """
    GLSL program
    """
    ATTRIBUTE_DECL_RE = re.compile(r"attribute\s+(?P<type_spec>\w+)\s+(?P<attribute_name>\w+)\s*;")
    UNIFORM_DECL_RE = re.compile(r"uniform\s+(?P<type_spec>\w+)\s+(?P<uniform_name>\w+)\s*(=\s*(?P<initialization>.*)\s*;|;)")
    _current = None
    def __init__(self, vs_src, fs_src, parse_attributes=True, parse_uniforms=True):
        self.vs_src = vs_src
        self.fs_src = fs_src
        self.program_id = None
        if parse_attributes:
            attributes = {}
            for line in vs_src.split('\n'):
                m = self.ATTRIBUTE_DECL_RE.match(line)
                if m:
                    attribute_name, type_spec = m.group('attribute_name'), m.group('type_spec')
                    attributes[attribute_name] = {'type': GLSL_TYPE_SPEC[type_spec]}
            self.attributes = attributes
        else:
            self.attributes = {}
        if parse_uniforms:
            uniforms = {}
            for line in vs_src.split('\n') + fs_src.split('\n'):
                m = self.UNIFORM_DECL_RE.match(line)
                if m:
                    uniform_name, type_spec, initialization = m.group('uniform_name'), m.group('type_spec'), m.group('initialization')
                    uniforms[uniform_name] = {'type': GLSL_TYPE_SPEC[type_spec]}
                    if initialization:
                        uniforms[uniform_name]['initialization'] = initialization
            self.uniforms = uniforms
        else:
            self.uniforms = {}
    def init_gl(self, force=False):
        if force:
            Program._current = None
        if self.program_id is not None:
            if not force: return
        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, self.vs_src)
        gl.glCompileShader(vs)
        if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
            raise Exception('failed to compile vertex shader:\n%s' % gl.glGetShaderInfoLog(vs).decode())
        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, self.fs_src)
        gl.glCompileShader(fs)
        if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
            raise Exception('failed to compile fragment shader:\n%s' % gl.glGetShaderInfoLog(fs).decode())
        program_id = gl.glCreateProgram()
        gl.glAttachShader(program_id, vs)
        gl.glAttachShader(program_id, fs)
        gl.glLinkProgram(program_id)
        if not gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS):
            raise Exception('failed to link program')
        gl.glDetachShader(program_id, vs)
        gl.glDetachShader(program_id, fs)
        self.program_id = program_id
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)
    def use(self):
        if Program._current is self:
            return
        if self.program_id is None:
            self.init_gl()
        gl.glUseProgram(self.program_id)
        Program._current = self
    def release(self):
        Program._current = None


class Technique(object):
    """
    GL rendering technique (based off of Technique defined by glTF schema)
    """
    _current = None
    def __init__(self, program, attributes=None, uniforms=None, states=None, attribute_divisors=None,
                 front_face=gl.GL_CCW):
        self.program = program
        if attributes is None:
            attributes = copy.deepcopy(program.attributes)
        self.attributes = attributes
        if uniforms is None:
            uniforms = copy.deepcopy(program.uniforms)
        else:
            for uniform_name, uniform in program.uniforms.items():
                if uniform_name not in uniforms:
                    uniforms[uniform_name] = uniform
                else:
                    uniforms[uniform_name].update(uniform)
        self.uniforms = uniforms
        if states is None:
            states = []
        self.states = states
        if attribute_divisors is None:
            attribute_divisors = {}
        self.attribute_divisors = attribute_divisors
        self.front_face = front_face
    def init_gl(self, force=False):
        if force:
            Technique._current = None
        self.program.init_gl(force=force)
        program_id = self.program.program_id
        self.attribute_locations = {name: gl.glGetAttribLocation(program_id, name) for name in self.attributes}
        self.uniform_locations = {name: gl.glGetUniformLocation(program_id, name) for name in self.uniforms}
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)
    def use(self):
        if Technique._current is self:
            return
        self.program.use()
        gl.glFrontFace(self.front_face)
        Technique._current = self
    def release(self):
        Technique._current = None


class Texture(object):
    def __init__(self, uri):
        """

        An OpenGL Texture / Sampler2D which is loaded from an image file

        """
        self.uri = uri
        self.texture_id = None
        self.sampler_id = None
    def init_gl(self, force=False):
        """
        Perform initialization for the texture on the current GL context

        :param force: if True, force reinitialization of the GL context for this
                      Texture and all of the GL entities that it depends on
        """
        if self.texture_id is not None:
            if not force: return
        image = Image.open(self.uri)
        texture_id = gl.glGenTextures(1)
        self.texture_id = texture_id
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        sampler_id = gl.glGenSamplers(1)
        self.sampler_id = sampler_id
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_LINEAR)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0,
                        gl.GL_RGB if image.mode == 'RGB' else gl.GL_RGBA,
                        image.width, image.height, 0,
                        gl.GL_RGB if image.mode == 'RGB' else gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        np.array(list(image.getdata()), dtype=np.ubyte))
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init texture: %s' % err)
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)


class CubeTexture(Texture):
    TARGETS = (gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
               gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
               gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
               gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
               gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
               gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
    def __init__(self, uris):
        """

        An OpenGL Cube Texture / SamplerCube which is loaded from 6 image files

        """
        self.uris = uris
        self.texture_id = None
        self.sampler_id = None
    def init_gl(self, force=False):
        """
        Perform initialization for the texture on the current GL context

        :param force: if True, force reinitialization of the GL context for this
                      Texture and all of the GL entities that it depends on
        """
        if self.texture_id is not None:
            if not force: return
        texture_id = gl.glGenTextures(1)
        self.texture_id = texture_id
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture_id)
        sampler_id = gl.glGenSamplers(1)
        self.sampler_id = sampler_id
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR);
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        for uri, target in zip(self.uris, self.TARGETS):
            image = Image.open(uri)
            gl.glTexImage2D(target, 0,
                            gl.GL_RGBA,
                            image.width, image.height, 0,
                            gl.GL_RGB,
                            gl.GL_UNSIGNED_BYTE,
                            np.array(list(image.getdata()), dtype=np.ubyte))
        gl.glGenerateMipmap(gl.GL_TEXTURE_CUBE_MAP)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init cube texture: %s' % err)
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)


class Material(object):
    _current = None
    def __init__(self, technique, values=None, textures=None):
        """
        A Material object is a customization of a :ref:`Technique` with a
        particular set of uniform values and textures.
        """
        self.technique = technique
        if values is None:
            values = {}
        if textures is None:
            textures = {}
        for uniform_name, uniform in self.technique.uniforms.items():
            if uniform['type'] in [gl.GL_SAMPLER_2D, gl.GL_SAMPLER_CUBE]:
                if uniform_name not in textures and 'texture' in uniform:
                    textures[uniform_name] = uniform['texture']
            else:
                if uniform_name not in values and 'value' in uniform:
                    values[uniform_name] = uniform['value']
        self.values = values
        self.textures = textures
        self._initialized = False
    def init_gl(self, force=False):
        if force:
            self._initialized = False
        if self._initialized:
            return
        Material._current = None
        self.technique.init_gl(force=force)
        for texture in self.textures.values():
            texture.init_gl(force=force)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init material: %s' % err)
        self._initialized = True
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)
    def use(self,
            u_view=None,
            u_modelview=None,
            u_projection=None,
            u_normal=None,
            u_modelview_inverse=None,
            u_model=None):
        if Material._current is self:
            return
        if not self._initialized:
            self.init_gl()
        self.technique.use()
        for uniform_name, location in self.technique.uniform_locations.items():
            uniform = self.technique.uniforms[uniform_name]
            uniform_type = uniform['type']
            if uniform_type == gl.GL_SAMPLER_2D:
                texture = self.textures[uniform_name]
                gl.glActiveTexture(gl.GL_TEXTURE0+0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture.texture_id)
                gl.glBindSampler(0, texture.sampler_id)
                gl.glUniform1i(location, 0)
            elif uniform_type == gl.GL_SAMPLER_CUBE:
                texture = self.textures[uniform_name]
                gl.glActiveTexture(gl.GL_TEXTURE0+0)
                gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture.texture_id)
                gl.glBindSampler(0, texture.sampler_id)
                gl.glUniform1i(location, 0)
            elif uniform_name in self.values:
                value = self.values[uniform_name]
                if uniform_type == gl.GL_FLOAT:
                    gl.glUniform1f(location, value)
                elif uniform_type == gl.GL_FLOAT_VEC2:
                    gl.glUniform2f(location, *value)
                elif uniform_type == gl.GL_FLOAT_VEC3:
                    gl.glUniform3f(location, *value)
                elif uniform_type == gl.GL_FLOAT_VEC4:
                    gl.glUniform4f(location, *value)
                else:
                    raise Exception('unhandled uniform type: %d' % uniform_type)
            else:
                if u_modelview is not None and uniform_name == 'u_modelview':
                    gl.glUniformMatrix4fv(location, 1, False, u_modelview)
                elif u_modelview_inverse is not None and uniform_name == 'u_modelview_inverse':
                    gl.glUniformMatrix4fv(location, 1, False, u_modelview_inverse)
                elif u_model is not None and uniform_name == 'u_model':
                    gl.glUniformMatrix4fv(location, 1, False, u_model)
                elif u_view is not None and uniform_name == 'u_view':
                    gl.glUniformMatrix4fv(location, 1, False, u_view)
                elif u_projection is not None and uniform_name == 'u_projection':
                    gl.glUniformMatrix4fv(location, 1, False, u_projection)
                elif u_normal is not None and uniform_name == 'u_normal':
                    gl.glUniformMatrix3fv(location, 1, False, u_normal)
            if CHECK_GL_ERRORS:
                err = gl.glGetError()
                if err != gl.GL_NO_ERROR:
                    raise Exception('error setting material state: %d' % err)
        Material._current = self
    def release(self):
        Material._current = None


class Primitive(object):
    def __init__(self, mode, indices, index_buffer=None, attribute_usage=None, attribute_divisors=None,
                 **attributes):
        """

        A class for specifying GL vertex attribute objects and providing vertex buffer data

        :param **attributes: all other passed keywords are interpreted as providing
                             array data for the named (by keyword) attribute:
                             ``<attribute_name>=<ndarray of data>``

        """
        self.mode = mode
        self.indices = indices
        self.index_buffer = index_buffer
        self.attribute_usage = attribute_usage
        self.attributes = attributes
        self.buffers = None
        self.vaos = {}
    def init_gl(self, force=False):
        if self.buffers is not None:
            if not force: return
        self.buffers = {}
        for name, values in self.attributes.items():
            values = values.tobytes()
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            if self.attribute_usage and name in self.attribute_usage:
                usage = self.attribute_usage[name]
            else:
                usage = gl.GL_STATIC_DRAW
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(values), values, usage)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise Exception('failed to init gl buffer')
            self.buffers[name] = vbo
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        if force or (self.index_buffer is None and self.indices is not None):
            indices = self.indices.tobytes()
            vao = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, vao)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(indices), indices, gl.GL_STATIC_DRAW)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise Exception('failed to init gl buffer')
            self.index_buffer = vao
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)


class Node(object):
    def __init__(self, matrix=None):
        """
        A node in a scenegraph.

        Each node has a TRS transformation which is expressed in
        coordinates local to the parent node (or in global coordinates in the case of the root node).
        """
        if matrix is None:
            matrix = np.eye(4, dtype=np.float32)
        self.matrix = matrix
        self.world_matrix = matrix.copy()
        self.world_position = self.world_matrix[3,:3]
        self.children = []
    def update_world_matrices(self, world_matrix=None):
        """
        Update all world transformations for the subtree rooted at this node

        :param world_matrix: Matrix which transforms the node's local coordinates to world coordinates
        """
        if world_matrix is None:
            self.world_matrix[...] = self.matrix
        else:
            world_matrix.dot(self.matrix, out=self.world_matrix)
        world_matrix = self.world_matrix
        for child in self.children:
            child.update_world_matrices(world_matrix=world_matrix)


class Mesh(Node):
    _modelview = np.eye(4, dtype=np.float32)
    _normal = np.eye(3, dtype=np.float32)
    def __init__(self, primitives, matrix=None):
        """
        A drawable collection of :ref:`Primitive`s, with a :ref:`Material` assigned for each one.

        :param primitives: should be a dict which maps :ref:`Material` to list of :ref:`Primitive` which use that material
        """
        Node.__init__(self, matrix=matrix)
        self.primitives = primitives
        self._initialized = False
    def init_gl(self, force=False):
        if self._initialized and not force:
            return
        self._initialized = False
        for material, prims in self.primitives.items():
            material.init_gl(force=force)
            technique = material.technique
            for prim in prims:
                prim.init_gl(force=force)
                if technique in prim.vaos:
                    continue
                vao = gl.glGenVertexArrays(1)
                prim.vaos[technique] = vao
                gl.glBindVertexArray(vao)
                for attribute_name, location in technique.attribute_locations.items():
                    attribute = prim.attributes[attribute_name]
                    gl.glEnableVertexAttribArray(location)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, prim.buffers[attribute_name])
                    gl.glVertexAttribPointer(location, attribute.shape[-1],
                                             DTYPE_COMPONENT_TYPE[attribute.dtype], False,
                                             attribute.dtype.itemsize * attribute.shape[-1],
                                             NULL_PTR)
                    if attribute_name in technique.attribute_divisors:
                        gl.glVertexAttribDivisor(location, technique.attribute_divisors[attribute_name])
                gl.glBindVertexArray(0)
                for location in technique.attribute_locations.values():
                    gl.glDisableVertexAttribArray(location)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init primitive: %s' % err)
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)
        self._initialized = True
    def draw(self, view=None, projection=None):
        if view is not None:
            self.world_matrix.dot(view, out=self._modelview)
            self._normal[:] = np.linalg.inv(self._modelview[:3,:3].T)
        for material, prims in self.primitives.items():
            material.use(u_view=view, u_projection=projection, u_modelview=self._modelview,
                         u_normal=self._normal, u_model=self.world_matrix)
            technique = material.technique
            for prim in prims:
                gl.glBindVertexArray(prim.vaos[technique])
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, prim.index_buffer)
                gl.glDrawElements(prim.mode, prim.indices.size, DTYPE_COMPONENT_TYPE[prim.indices.dtype],
                                  c_void_p(0))
                if CHECK_GL_ERRORS:
                    err = gl.glGetError()
                    if err != gl.GL_NO_ERROR:
                        raise Exception('error drawing primitive elements: %d' % err)
            gl.glBindVertexArray(0)
            # for location in technique.attribute_locations.values():
            #     gl.glDisableVertexAttribArray(location)
            material.release()
            material.technique.release()


class OpenGLRenderer(object):
    def __init__(self, multisample=0, znear=0.1, zfar=1000, window_size=(960,1080)):
        """

        Renderer for non-VR OpenGL renderering

        """
        self.window_size = window_size
        self.znear = znear
        self.zfar = zfar
        self.camera_matrix = np.eye(4, dtype=np.float32)
        self.camera_position = self.camera_matrix[3,:3]
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = np.empty((4,4), dtype=np.float32)
        self.update_projection_matrix()
        self._gl_states = {}
    def update_projection_matrix(self):
        window_size, znear, zfar = self.window_size, self.znear, self.zfar
        self.projection_matrix[:] = calc_projection_matrix(np.pi / 180 * 60, window_size[0] / window_size[1], znear, zfar).T
    def init_gl(self, clear_color=(0.0, 0.0, 0.0, 0.0)):
        gl.glClearColor(*clear_color)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, self.window_size[0], self.window_size[1])
    @contextmanager
    def render(self, meshes=None):
        """
        Render the given meshes.

        This method returns a managed context for drawing the frame.
        If used in a :code:`with`-statement, the rendering takes place when the
        :code:`with` block is exited.

        :param meshes *optional*: iterable collection of :ref:`Mesh`-like objects
        """
        self.view_matrix[3,:3] = -self.camera_matrix[3,:3]
        self.view_matrix[:3,:3] = self.camera_matrix[:3,:3].T
        yield {
            'camera_world_matrix': self.camera_matrix,
            'camera_position': self.camera_position,
            'view_matrix': self.view_matrix
        }
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if meshes is not None:
            for mesh in meshes:
                mesh.draw(projection=self.projection_matrix,
                          view=self.view_matrix)
    def process_input(self, **kwargs):
        pass
    def shutdown(self):
        pass


def calc_projection_matrix(yfov, aspectRatio, znear, zfar):
    """
    Calculates a standard OpenGL perspective projection matrix, it might be transposed, i forget.

    :param yfov: the field of view measured vertically, in radians
    :param aspectRatio: the frustum width to height ratio (<width> divided by <height>)
    :param znear: the :math:`z` coordinate of the near clipping plane
    :param zfar: the :math:`z` coordinate of the far clipping plane
    """
    f = 1.0 / np.tan(yfov / 2)
    return np.array([[f/aspectRatio, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (znear + zfar) / (znear - zfar), 2 * znear * zfar / (znear - zfar)],
                     [0, 0, -1, 0]], dtype=np.float32)


def set_matrix_from_quaternion(quat, out):
    w, x, y, z = quat
    y2 = y**2
    x2 = x**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    out[0] = [1.0 - 2.0 * (y2 + z2),
              2.0 * (xy - wz),
              2.0 * (xz + wy)]
    out[1] = [2.0 * (xy + wz),
              1.0 - 2.0 * (x2 + z2),
              2.0 * (yz - wx)]
    out[2] = [2.0 * (xz - wy),
              2.0 * (yz + wx),
              1.0 - 2.0 * (x2 + y2)]
    return out

def set_quaternion_from_matrix(U, out):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    # assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
    trace = U.trace()
    if trace > 0:
        s = 0.5 / np.sqrt( trace + 1.0 );
        _w = 0.25 / s
        _x = (U[2,1] - U[1,2]) * s
        _y = (U[0,2] - U[2,0]) * s
        _z = (U[1,0] - U[0,1]) * s
    elif U[0,0] > U[1,1] and U[0,0] > U[2,2]:
        s = 2.0 * np.sqrt(1.0 + U[0,0] - U[1,1] - U[2,2])
        _w = (U[2,1] - U[1,2]) / s
        _x = 0.25 * s
        _y = (U[0,1] + U[1,0]) / s
        _z = (U[0,2] + U[2,0]) / s
    elif U[1,1] > U[2,2]:
        s = 2.0 * np.sqrt( 1.0 + U[1,1] - U[0,0] - U[2,2])
        _w = (U[0,2] - U[2,0]) / s
        _x = (U[0,1] + U[1,0]) / s
        _y = 0.25 * s
        _z = (U[1,2] + U[2,1]) / s
    else:
        s = 2.0 * np.sqrt( 1.0 + U[2,2] - U[0,0] - U[1,1])
        _w = (U[1,0] - U[0,1]) / s
        _x = (U[0,2] + U[2,0]) / s
        _y = (U[1,2] + U[2,1]) / s
        _z = 0.25 * s
    out[:] = np.array([_x, _y, _z, _w])
    return out
