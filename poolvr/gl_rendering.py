import re
import copy
from ctypes import c_float, POINTER, c_void_p
from contextlib import contextmanager
import logging
import numpy as np
import PIL.Image as Image
import OpenGL.GL as gl


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
    'int': gl.GL_INT,
    'float': gl.GL_FLOAT,
    'vec2': gl.GL_FLOAT_VEC2,
    'vec3': gl.GL_FLOAT_VEC3,
    'vec4': gl.GL_FLOAT_VEC4,
    'mat4': gl.GL_FLOAT_MAT4,
    'mat3': gl.GL_FLOAT_MAT3,
    'sampler2D': gl.GL_SAMPLER_2D,
    'samplerCube': gl.GL_SAMPLER_CUBE
}


TYPE_TO_UNIFORM_FN = {
    gl.GL_INT: gl.glUniform1i,
    gl.GL_INT_VEC2: lambda location, value: gl.glUniform2i(location, *value),
    gl.GL_INT_VEC3: lambda location, value: gl.glUniform3i(location, *value),
    gl.GL_INT_VEC4: lambda location, value: gl.glUniform4i(location, *value),
    gl.GL_FLOAT: gl.glUniform1f,
    gl.GL_FLOAT_VEC2: lambda location, value: gl.glUniform2f(location, *value),
    gl.GL_FLOAT_VEC3: lambda location, value: gl.glUniform3f(location, *value),
    gl.GL_FLOAT_VEC4: lambda location, value: gl.glUniform4f(location, *value),
    gl.GL_FLOAT_MAT2: lambda location, value: gl.glUniformMatrix2fv(location, 1, False, value),
    gl.GL_FLOAT_MAT3: lambda location, value: gl.glUniformMatrix3fv(location, 1, False, value),
    gl.GL_FLOAT_MAT4: lambda location, value: gl.glUniformMatrix4fv(location, 1, False, value),
}
ARRAY_TYPE_TO_UNIFORM_FN = {
    gl.GL_INT: gl.glUniform1iv,
    gl.GL_INT_VEC2: gl.glUniform2iv,
    gl.GL_INT_VEC3: gl.glUniform3iv,
    gl.GL_INT_VEC4: gl.glUniform4iv,
    gl.GL_FLOAT: gl.glUniform1fv,
    gl.GL_FLOAT_VEC2: gl.glUniform2fv,
    gl.GL_FLOAT_VEC3: gl.glUniform3fv,
    gl.GL_FLOAT_VEC4: gl.glUniform4fv,
    gl.GL_FLOAT_MAT2: lambda location, count, value: gl.glUniformMatrix2fv(location, 1, False, value),
    gl.GL_FLOAT_MAT3: lambda location, count, value: gl.glUniformMatrix3fv(location, 1, False, value),
    gl.GL_FLOAT_MAT4: lambda location, count, value: gl.glUniformMatrix4fv(location, 1, False, value),
}


DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi
c_float_p = POINTER(c_float)
NULL_PTR = c_void_p(0)
_logger = logging.getLogger(__name__)


class GLRendering(object):
    def __init__(self, *args, name=None, **kwargs):
        self.name = name


class Program(GLRendering):
    ATTRIBUTE_DECL_RE = re.compile(r"\s*attribute\s+(?P<type_spec>\w+)\s+(?P<attribute_name>\w+)\s*;")
    UNIFORM_DECL_RE = re.compile(r"\s*uniform\s+(?P<type_spec>\w+)(?P<array_spec>\[\d+\])?\s+(?P<uniform_name>\w+)\s*(=\s*(?P<initialization>.*)\s*;|;)")
    _current = None
    def __init__(self, vs_src, fs_src, parse_attributes=True, parse_uniforms=True, name=None):
        """
        GLSL program
        """
        super().__init__(name=name)
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
                    # _logger.debug('array_spec = %s', array_spec)
                    uniforms[uniform_name] = {'type': GLSL_TYPE_SPEC[type_spec]}
                    array_spec = m.group('array_spec')
                    if array_spec:
                        uniforms[uniform_name]['array_size'] = int(array_spec[1:-1])
                    if initialization:
                        uniforms[uniform_name]['initialization'] = initialization
            self.uniforms = uniforms
        else:
            self.uniforms = {}
        self._initialized = False
        _logger.debug('self.uniforms:\n%s', '\n'.join('%s: %s' % it for it in self.uniforms.items()))
    def init_gl(self, force=False):
        if force:
            self.program_id = None
        if self.program_id is not None:
            return self.program_id
        Program._current = None
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
        _logger.debug('%s.init_gl: OK', self.__class__.__name__)
    def use(self):
        if Program._current is self:
            return
        if self.program_id is None:
            self.init_gl()
        gl.glUseProgram(self.program_id)
        Program._current = self
    def release(self):
        Program._current = None


class Technique(GLRendering):
    """
    GL rendering technique (based off of Technique defined by glTF schema)
    """
    _current = None
    def __init__(self, program,
                 attributes=None, uniforms=None,
                 states=None, attribute_divisors=None,
                 front_face=gl.GL_CCW, on_use=None, on_release=None, name=None):
        super().__init__(name=name)
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
        self._on_use = on_use
        self._on_release = on_release
        self._initialized = False
    def init_gl(self, force=False):
        if force:
            self._initialized = False
        if self._initialized:
            return
        self.program.init_gl(force=force)
        program_id = self.program.program_id
        _logger.debug(self.attributes)
        self.attribute_locations = {name: gl.glGetAttribLocation(program_id, name) for name in self.attributes.keys()}
        self.uniform_locations = {name: gl.glGetUniformLocation(program_id, name) for name in self.uniforms.keys()}
        self._initialized = True
        _logger.debug('%s.init_gl: OK', self.__class__.__name__)
    def use(self):
        if Technique._current is self:
            return
        self.program.use()
        gl.glFrontFace(self.front_face)
        if self._on_use:
            self._on_use(self)
        Technique._current = self
    def release(self):
        if self._on_release:
            self._on_release(self)
        Technique._current = None


class Texture(GLRendering):
    def __init__(self, uri, name=None, min_filter=gl.GL_NEAREST_MIPMAP_LINEAR,
                 mag_filter=gl.GL_LINEAR, wrap_s=gl.GL_REPEAT, wrap_t=gl.GL_REPEAT, **kwargs):
        """

        An OpenGL Texture / Sampler2D which is loaded from an image file

        """
        super().__init__(name=name)
        self.uri = uri
        self.texture_id = None
        self.sampler_id = None
        self.min_filter = min_filter
        self.mag_filter = mag_filter
        self.wrap_s = wrap_s
        self.wrap_t = wrap_t
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
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MIN_FILTER, self.min_filter)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_S, self.wrap_s)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_T, self.wrap_t)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0,
                        gl.GL_RGB if image.mode == 'RGB' else gl.GL_RGBA,
                        image.width, image.height, 0,
                        gl.GL_RGB if image.mode == 'RGB' else gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        np.array(image.getdata(), dtype=np.ubyte))
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init texture: 0x%02x' % err)
        _logger.debug('%s.init_gl: OK%s', self.__class__.__name__,
                     ' (%s)' % self.name if self.name else '')


class CubeTexture(Texture):
    TARGETS = (gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
               gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
               gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
               gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
               gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
               gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
    def __init__(self, uris, name=None):
        """

        An OpenGL Cube Texture / SamplerCube which is loaded from 6 image files

        """
        GLRendering.__init__(self, name=name)
        self.uris = uris
        self.texture_id = None
        self.sampler_id = None
    def init_gl(self, force=False):
        """
        Perform initialization for the texture within the current GL context

        :param force: if True, force reinitialization of the GL context for this
                      Texture and all of the GL entities that it references
        """
        if self.texture_id is not None:
            if not force: return
        texture_id = gl.glGenTextures(1)
        self.texture_id = texture_id
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture_id)
        sampler_id = gl.glGenSamplers(1)
        self.sampler_id = sampler_id
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glSamplerParameteri(sampler_id, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        for uri, target in zip(self.uris, self.TARGETS):
            image = Image.open(uri)
            gl.glTexImage2D(target, 0,
                            gl.GL_RGB if image.mode == 'RGB' else gl.GL_RGBA,
                            image.width, image.height, 0,
                            gl.GL_RGB if image.mode == 'RGB' else gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE,
                            np.array(image.getdata(), dtype=np.ubyte))
        gl.glGenerateMipmap(gl.GL_TEXTURE_CUBE_MAP)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init cube texture: %s' % err)
        _logger.debug('%s.init_gl: OK', self.__class__.__name__)


class Material(GLRendering):
    _current = None
    def __init__(self, technique, values=None, textures=None, on_use=None, on_release=None, name=None):
        """
        A Material object is a customization of a :ref:`Technique` with a
        particular set of uniform values and textures.
        """
        super().__init__(name=name)
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
        self._on_use = on_use
        self._on_release = on_release
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
        _logger.debug('%s.init_gl: OK', self.__class__.__name__)
    def use(self,
            **frame_data):
        # if Material._current is self:
        #     return
        if not self._initialized:
            self.init_gl()
        self.technique.use()
        if self._on_use:
            self._on_use(self, **frame_data)
        tex_unit = 0
        for uniform_name, location in self.technique.uniform_locations.items():
            uniform = self.technique.uniforms[uniform_name]
            uniform_type = uniform['type']
            if uniform_type in (gl.GL_SAMPLER_2D, gl.GL_SAMPLER_CUBE):
                texture = self.textures[uniform_name]
                gl.glActiveTexture(gl.GL_TEXTURE0+tex_unit)
                gl.glBindTexture(uniform_type, texture.texture_id)
                gl.glBindSampler(tex_unit, texture.sampler_id)
                gl.glUniform1i(location, tex_unit)
                tex_unit += 1
                continue
            elif uniform_name in self.values:
                value = self.values[uniform_name]
            elif uniform_name in frame_data:
                value = frame_data[uniform_name]
            else:
                continue
            if 'array_size' in uniform:
                ARRAY_TYPE_TO_UNIFORM_FN[uniform_type](location, uniform['array_size'], value)
            else:
                TYPE_TO_UNIFORM_FN[uniform_type](location, value)
        if CHECK_GL_ERRORS:
            err = gl.glGetError()
            if err != gl.GL_NO_ERROR:
                raise Exception('error setting material state: %d' % err)
        Material._current = self
    def release(self):
        if self._on_release:
            self._on_release(self)
        Material._current = None


class Primitive(GLRendering):
    def __init__(self, mode, indices=None, index_buffer=None,
                 attribute_usage=None, attribute_divisors=None,
                 name=None, **attributes):
        """

        A class for specifying GL vertex attribute objects and providing vertex buffer data

        :param **attributes: all other passed keywords are interpreted as providing
                             array data for the named (by keyword) attribute:
                             ``<attribute_name>=<ndarray of data>``

        """
        super().__init__(name=name)
        self.mode = mode
        self.indices = indices
        self.index_buffer = index_buffer
        if attribute_usage is None:
            attribute_usage = {}
        self.attribute_usage = attribute_usage
        self.attributes = attributes
        self.buffers = None
        self.vaos = {}
    def init_gl(self, force=False):
        if not force and self.buffers is not None:
            return
        self.buffers = {}
        for name, values in self.attributes.items():
            values = values.tobytes()
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            if name in self.attribute_usage:
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
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
                            len(indices), indices, gl.GL_STATIC_DRAW)
            if gl.glGetError() != gl.GL_NO_ERROR:
                raise Exception('failed to init gl buffer')
            self.index_buffer = vao
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        _logger.debug('%s.init_gl: OK', self.__class__.__name__)
    def alias(self, attribute_name, alias):
        if attribute_name not in self.attributes:
            raise Exception('attribute "%s" is not defined' % attribute_name)
        self.attributes[alias] = self.attributes[attribute_name]
    def update_buffer_data(self, name, buffer_data):
        vbo = self.buffers[name]
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        buffer_data = buffer_data.tobytes()
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, len(buffer_data), buffer_data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


class Node(GLRendering):
    def __init__(self, matrix=None, name=None):
        """
        A node in a scenegraph.

        Each node has a TRS transformation which is expressed in
        coordinates local to the parent node (or in global coordinates in the case of the root node).
        """
        super().__init__(name=name)
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
    def draw(self, **frame_data):
        for child in self.children:
            child.draw(**frame_data)


class Mesh(Node):
    _modelview = np.eye(4, dtype=np.float32)
    _normal = np.eye(3, dtype=np.float32)
    def __init__(self, primitives, matrix=None, before_draw=None, after_draw=None, name=None):
        """
        A drawable collection of :ref:`Primitive`s, with a :ref:`Material` assigned for each one.

        :param primitives: should be a dict which maps :ref:`Material` to list of :ref:`Primitive` which use that material
        """
        Node.__init__(self, matrix=matrix, name=name)
        self.primitives = primitives
        self.visible = True
        self._before_draw = before_draw
        self._after_draw = after_draw
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
        _logger.debug('%s.init_gl: OK' % self.__class__.__name__)
        self._initialized = True
    def draw(self, **frame_data):
        if not self.visible:
            return super().draw(**frame_data)
        view = frame_data.get('view_matrix', None)
        projection = frame_data.get('projection_matrix', None)
        if self._before_draw:
            self._before_draw(self, **frame_data)
        if view is not None:
            self.world_matrix.dot(view, out=self._modelview)
            self._normal[:] = np.linalg.inv(self._modelview[:3,:3].T)
        for material, prims in self.primitives.items():
            material.use(u_view=view, u_projection=projection, u_modelview=self._modelview,
                         u_modelview_inverse_transpose=self._normal, u_model=self.world_matrix,
                         **frame_data)
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
        if self._after_draw:
            self._after_draw(self, **frame_data)
        super().draw(**frame_data)


class FragBox(Node):
    _VS_SRC = r"""#version 410 core
const vec2 quadVertices[4] = vec2[4](vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0), vec2( 1.0,  1.0));
void main() { gl_Position = vec4(quadVertices[gl_VertexID], 0.0, 0.0); }
"""
    def __init__(self, fs_src, on_use=None, **kwargs):
        try:
            with open(fs_src) as f:
                fs_src = f.read()
        except:
            pass
        super().__init__(**kwargs)
        program = Program(self._VS_SRC, fs_src, parse_uniforms=True)
        self.material = Material(Technique(program),
                                 on_use=on_use)
        self._initialized = False
    def init_gl(self, force=False):
        if force:
            self._initialized = False
        if self._initialized:
            return
        self.material.init_gl(force=force)
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            raise Exception('failed to init %s: %s' % (self.__class__.__name__, err))
        _logger.info('%s.init_gl: OK' % self.__class__.__name__)
        self._initialized = True
    def update_world_matrices(self, world_matrix=None):
        self.world_matrix = np.eye(4)
        if world_matrix is None:
            world_matrix = self.world_matrix
        for child in self.children:
            child.update_world_matrices(world_matrix=world_matrix)
    def draw(self, **frame_data):
        super().draw(**frame_data)
        self.material.use(**frame_data)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        try:
            if CHECK_GL_ERRORS:
                err = gl.glGetError()
                if err != gl.GL_NO_ERROR:
                    raise Exception('error drawing %s: glGetError returned %d' % (self.__class__.__name__, err))
        finally:
            self.material.release()
            self.material.technique.release()


class OpenGLRenderer(object):
    def __init__(self, multisample=0, znear=0.1, zfar=1000, window_size=(960,1080)):
        """

        Renderer for non-VR OpenGL renderering

        """
        self.window_size = np.array(window_size, dtype=np.float32)
        self.znear = znear
        self.zfar = zfar
        self.camera_matrix = np.eye(4, dtype=np.float32)
        self.camera_position = self.camera_matrix[3,:3]
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = np.empty((4,4), dtype=np.float32)
        self.projection_lrbt = np.zeros(4, dtype=np.float32)
        self.update_projection_matrix()
        self._gl_states = {}
        self._nframes = 0
    def update_projection_matrix(self):
        window_size, znear, zfar = self.window_size, self.znear, self.zfar
        self.projection_matrix[:] = calc_projection_matrix(np.pi / 180 * 60, window_size[0] / window_size[1], znear, zfar).T
        yfov = 60.0 * DEG2RAD
        aspect = window_size[0] / window_size[1]
        tan = np.tan(0.5*yfov)
        self.projection_lrbt[:] = (-aspect*tan, aspect*tan,
                                          -tan,        tan)
    def init_gl(self, clear_color=(0.0, 0.0, 0.0, 0.0)):
        gl.glClearColor(*clear_color)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, self.window_size[0], self.window_size[1])
    @contextmanager
    def render(self, meshes=None, **kwargs):
        """
        Render the given meshes.

        This method returns a managed context for drawing the frame.
        If used in a :code:`with`-statement, the rendering takes place when the
        :code:`with` block is exited.

        :param meshes *optional*: iterable collection of :ref:`Mesh`-like objects
        """
        camera = self.camera_matrix
        view = self.view_matrix
        view[:3,:3] = camera[:3,:3].T
        np.dot(camera[:3,:3], camera[3,:3],
               out=view[3,:3])
        view[3,:3] *= -1
        frame_data = {
            'camera_matrix': camera,
            'view_matrix': view,
            'projection_matrix': self.projection_matrix,
            'projection_lrbt': self.projection_lrbt,
            'znear': self.znear,
            'zfar': self.zfar,
            'window_size': self.window_size,
        }
        frame_data.update(kwargs)
        # if self._nframes % 90 == 0:
        #     _logger.debug('rendering with frame_data:\n%s',
        #                   '\n'.join('%s:\n%s' % it for it in frame_data.items()))
        yield frame_data
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if meshes is not None:
            for mesh in meshes:
                mesh.draw(**frame_data)
        self._nframes += 1
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


def set_matrix_from_quaternion(quat, out=None):
    """
    Set the values of a 3x3 matrix to those of a rotation matrix.
    """
    if out is None:
        out = np.empty((3,3), dtype=quat.dtype)
    x, y, z, w = quat
    yy = y**2
    xx = x**2
    zz = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    out[0] = [1.0 - 2.0 * (yy + zz),
              2.0 * (xy - wz),
              2.0 * (xz + wy)]
    out[1] = [2.0 * (xy + wz),
              1.0 - 2.0 * (xx + zz),
              2.0 * (yz - wx)]
    out[2] = [2.0 * (xz - wy),
              2.0 * (yz + wx),
              1.0 - 2.0 * (xx + yy)]
    return out


def set_quaternion_from_matrix(U, out=None):
    """
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm

    assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)
    """
    if out is None:
        out = np.empty(4, dtype=U.dtype)
    trace = U.trace()
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (U[2,1] - U[1,2]) * s
        y = (U[0,2] - U[2,0]) * s
        z = (U[1,0] - U[0,1]) * s
    elif U[0,0] > U[1,1] and U[0,0] > U[2,2]:
        s = 2.0 * np.sqrt(1.0 + U[0,0] - U[1,1] - U[2,2])
        w = (U[2,1] - U[1,2]) / s
        x = 0.25 * s
        y = (U[0,1] + U[1,0]) / s
        z = (U[0,2] + U[2,0]) / s
    elif U[1,1] > U[2,2]:
        s = 2.0 * np.sqrt(1.0 + U[1,1] - U[0,0] - U[2,2])
        w = (U[0,2] - U[2,0]) / s
        x = (U[0,1] + U[1,0]) / s
        y = 0.25 * s
        z = (U[1,2] + U[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + U[2,2] - U[0,0] - U[1,1])
        w = (U[1,0] - U[0,1]) / s
        x = (U[0,2] + U[2,0]) / s
        y = (U[1,2] + U[2,1]) / s
        z = 0.25 * s
    out[:] = np.array([x, y, z, w])
    return out
