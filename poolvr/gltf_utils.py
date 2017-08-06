import logging
import os.path
import base64
from copy import deepcopy
from ctypes import c_void_p
try: # python 3.3 or later
    from types import MappingProxyType
except ImportError as err:
    MappingProxyType = dict
import numpy as np
import OpenGL.GL as gl


_logger = logging.getLogger(__name__)


from .gl_rendering import (Program, Primitive, Mesh,
                           set_matrix_from_quaternion, CHECK_GL_ERRORS)


GLTF_BUFFERVIEW_TYPE_SIZES = MappingProxyType({
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16
})


class GLTFDict(dict):
    pass


class GLTFProgram(GLTFDict, Program):
    def __init__(self, gltf, program_id, uri_path):
        GLTFDict.__init__(self, gltf['programs'][program_id])
        shader_src = read_shaders(gltf, uri_path)
        Program.__init__(self,
                         shader_src[self['vertexShader']],
                         shader_src[self['fragmentShader']],
                         name=program_id)


class GLTFTechnique(GLTFDict):
    def __init__(self, gltf, technique_id, uri_path):
        GLTFDict.__init__(self, gltf['techniques'][technique_id])
        self.program = GLTFProgram(gltf, self['program'], uri_path)
        self._unif2param = dict(self['uniforms'])
        self._unif2semantic = {unif: self['parameters'][param]['semantic']
                               for unif, param in self._unif2param.items()
                               if 'semantic' in self['parameters'][param]}
        self._semantic2unif = {v: k for k, v in self._unif2semantic.items()}
        self._attr2param = dict(self['attributes'])
        self._attr2semantic = {attr: self['parameters'][param]['semantic']
                               for attr, param in self._attr2param.items()
                               if 'semantic' in self['parameters'][param]}
        self._semantic2attr = {v: k for k, v in self._attr2semantic.items()}
        self._initialized = False
    def init_gl(self, force=False):
        if self._initialized and not force:
            return
        self.program.init_gl(force=force)
        self._attr_locations = {attr: gl.glGetAttribLocation(self.program.program_id, attr)
                                for attr in self['attributes']}
        self._unif_locations = {unif: gl.glGetUniformLocation(self.program.program_id, unif)
                                for unif in self['uniforms']}
        _logger.info('%s.init_gl: OK', self.__class__.__name__)
        self._initialized = True
    def use(self, **frame_data):
        # TODO: some decorater-y state management / caching stuff
        self.program.use(**frame_data)


class GLTFMaterial(GLTFDict):
    def __init__(self, gltf, material_id, uri_path):
        GLTFDict.__init__(self, gltf['materials'][material_id])
        self.technique = GLTFTechnique(gltf, self['technique'], uri_path)
    def init_gl(self, force=False):
        self.technique.init_gl(force=force)
        _logger.info('%s.init_gl: OK', self.__class__.__name__)


class GLTFBufferView(GLTFDict):
    def __init__(self, gltf, buffer_view_id, uri_path):
        GLTFDict.__init__(self, gltf['bufferViews'][buffer_view_id])
        self._gltf = gltf
        self._uri_path = uri_path
        self._initialized = False
    def init_gl(self, force=False):
        if self._initialized and not force:
            return
        self.id = gl.glGenBuffers(1)
        gl.glBindBuffer(self['target'], self.id)
        buffer_data = self._load_buffer(self._gltf, self['buffer'], self._uri_path)
        gl.glBufferData(self['target'], len(buffer_data), buffer_data, gl.GL_STATIC_DRAW)
        if gl.glGetError() != gl.GL_NO_ERROR:
            raise Exception('failed to init gl buffer')
        gl.glBindBuffer(self['target'], 0)
        self._initialized = True
        _logger.info('%s.init_gl: OK', self.__class__.__name__)
    @classmethod
    def _load_buffer(cls, gltf, buffer_id, uri_path):
        uri = gltf['buffers'][buffer_id]['uri']
        if uri.startswith('data:application/octet-stream;base64,'):
            data = base64.b64decode(uri.split(',')[1])
        else:
            filename = os.path.join(uri_path, uri)
            if gltf['buffers'][buffer_id]['type'] == 'arraybuffer':
                with open(filename, 'rb') as f:
                    data = f.read()
            elif gltf['buffers'][buffer_id]['type'] == 'text':
                raise Exception('TODO')
            _logger.info('loaded buffer "%s" from "%s"', buffer_id, filename)
        return data


class GLTFPrimitive(GLTFDict):
    def __init__(self, gltf, primitive, uri_path):
        GLTFDict.__init__(self, primitive)
        self.material = GLTFMaterial(gltf, self['material'], uri_path)
        self.index_buffer_view = GLTFBufferView(gltf, gltf['accessors'][self['indices']]['bufferView'], uri_path)
        self.attribute_buffer_views = {semantic: GLTFBufferView(gltf, gltf['accessors'][accessor_id]['bufferView'], uri_path)
                                       for semantic, accessor_id in self['attributes'].items()}
        self._initialized = False
    def init_gl(self, force=False):
        if self._initialized and not force:
            return
        self.material.init_gl(force=force)
        self.index_buffer_view.init_gl(force=force)
        for buffer_view in self.attribute_buffer_views.values():
            buffer_view.init_gl(force=force)
        _logger.info('%s.init_gl: OK', self.__class__.__name__)
        self._initialized = True
    def draw(self, **frame_data):
        view = frame_data.get('view_matrix', None)
        projection = frame_data.get('projection_matrix', None)
        self.material.use(**frame_data)


class GLTFMesh(GLTFDict):
    def __init__(self, gltf, mesh_id):
        mesh_json = gltf['meshes'][mesh_id]
        GLTFDict.__init__(self, mesh_json)
        primitives = [GLTFPrimitive(gltf, prim)
                      for prim in self['primitives']]
        accessors = gltf['accessors']
        buffers = gltf['buffers']
        buffer_views = gltf['bufferViews']



class GLTFNode(GLTFDict):
    def __init__(self, gltf, node_id):
        node_json = gltf['nodes'][node_id]
        super().__init__(node_json)
        self.meshes = [gltf['meshes'][mesh_id]
                       for mesh_id in self.get('meshes', [])]
        self.children = [GLTFNode(gltf, child)
                         for child in self.get('children', [])]
    def init_gl(self, force=False):
        for mesh in self.meshes:
            mesh.init_gl(force=force)
        for child in self.children:
            child.init_gl(force=force)
    def draw(self, **frame_data):
        for mesh in self.meshes:
            mesh.draw(**frame_data)
        for child in self.children:
            child.draw(**frame_data)


class GLTFScene(GLTFDict):
    def __init__(self, gltf, scene_id):
        GLTFDict.__init__(self, gltf['scenes'][scene_id])
        node_ids = self.get('nodes', [])
        self.nodes = [GLTFNode(gltf, node_id) for node_id in node_ids]
    def init_gl(self, force=False):
        for node in self.nodes:
            node.init_gl(force=force)
    def draw(self, **frame_data):
        for node in self.nodes:
            node.draw(**frame_data)
    @classmethod
    def load_from(cls, gltf, scene_id=None):
        if scene_id is None:
            scene_id = gltf.get('scene', None)
        # nodes = {node_id: GLTFNode(gltf, node_id) for node_id in gltf['nodes']}
        # if scene_id:
        #     nodes = {k: v for k, v in nodes.items() if k in gltf['scenes'][scene_id]['nodes']}
        # return GLTFScene(list(nodes.values()))
        return cls(gltf, scene_id)


# TODO: caching decorators
def read_shaders(gltf, uri_path):
    shaders = gltf['shaders']
    shader_src = {}
    for shader_name, shader in shaders.items():
        uri = shader['uri']
        if uri.startswith('data:text/plain;base64,'):
            shader_str = base64.urlsafe_b64decode(uri.split(',')[1]).decode()
            _logger.info('decoded shader "%s":\n%s', shader_name, shader_str)
        else:
            filename = os.path.join(uri_path, shader['uri'])
            with open(filename) as f:
                shader_str = f.read()
            _logger.info('loaded shader "%s" from "%s"', shader_name, filename)
        shader_src[shader_name] = shader_str
    return shader_src


def setup_programs(gltf, uri_path):
    shader_src = read_shaders(gltf, uri_path)
    programs = {}
    for program_name, program in gltf['programs'].items():
        programs[program_name] = Program(shader_src[program['vertexShader']],
                                         shader_src[program['fragmentShader']],
                                         name=program_name)
    return programs


def read_buffers(gltf, uri_path):
    buffers = gltf['buffers']
    data_buffers = {}
    for buffer_name, buffer in buffers.items():
        uri = buffer['uri']
        if uri.startswith('data:application/octet-stream;base64,'):
            data_buffers[buffer_name] = base64.b64decode(uri.split(',')[1])
        else:
            filename = os.path.join(uri_path, buffer['uri'])
            if buffer['type'] == 'arraybuffer':
                data_buffers[buffer_name] = open(filename, 'rb').read()
            elif buffer['type'] == 'text':
                raise Exception('TODO')
                #data_buffers[buffer_name] = open(filename, 'r').read()
            _logger.info('loaded buffer "%s" (from %s)', buffer_name, filename)
    return data_buffers


def setup_buffers(gltf, uri_path):
    data_buffers = read_buffers(gltf, uri_path)
    buffer_ids = {}
    for bufferView_name, bufferView in gltf['bufferViews'].items():
        buffer_id = gl.glGenBuffers(1)
        byteOffset, byteLength = bufferView['byteOffset'], bufferView['byteLength']
        gl.glBindBuffer(bufferView['target'], buffer_id)
        gl.glBufferData(bufferView['target'], bufferView['byteLength'],
                        data_buffers[bufferView['buffer']][byteOffset:], gl.GL_STATIC_DRAW)
        if gl.glGetError() != gl.GL_NO_ERROR:
            raise Exception('failed to create buffer "%s"' % bufferView_name)
        gl.glBindBuffer(bufferView['target'], 0)
        _logger.info('created buffer "%s"', bufferView_name)
        buffer_ids[bufferView_name] = buffer_id
    return buffer_ids


def setup_meshes(gltf, uri_path, buffer_ids, materials):
    meshes = gltf['meshes']
    accessors = gltf['accessors']
    bufferViews = gltf['bufferViews']
    meshes = {}
    for mesh_name, mesh in gltf['meshes'].items():
        primitives = {}
        for primitive in mesh['primitives']:
            index_accessor = accessors[primitive['indices']]
            vao = buffer_ids[index_accessor['bufferView']]
            material = materials[primitive['material']]
            primitives[material] = []
            attribute_buffers = {}
            for attribute_name, accessor_name in primitive['attributes'].items():
                vbo = buffer_ids[accessors[accessor_name]['bufferView']]
                attribute_buffers[attribute_name] = vbo
            primitives[material].append(Primitive(primitive['mode'], index_buffer=vao, attribute_buffers=attribute_buffers))
        meshes[mesh_name] = Mesh(primitives)
    return meshes


def set_technique_state(technique_name, gltf):
    if set_technique_state.current_technique is not None and set_technique_state.current_technique == technique_name:
        return
    set_technique_state.current_technique = technique_name
    technique = gltf['techniques'][technique_name]
    program = gltf['programs'][technique['program']]
    gl.glUseProgram(program['id'])
    enabled_states = technique.get('states', {}).get('enable', [])
    for state, is_enabled in list(set_technique_state.states.items()):
        if state in enabled_states:
            if not is_enabled:
                gl.glEnable(state)
                set_technique_state.states[state] = True
        elif is_enabled:
            gl.glDisable(state)
            set_technique_state.states[state] = False
    for state in enabled_states:
        if state not in set_technique_state.states:
            gl.glEnable(state)
            set_technique_state.states[state] = True
set_technique_state.current_technique = None
set_technique_state.states = {}


def set_material_state(material_name, gltf):
    if set_material_state.current_material == material_name:
        return
    set_material_state.current_material = material_name
    material = gltf['materials'][material_name]
    set_technique_state(material['technique'], gltf)
    technique = gltf['techniques'][material['technique']]
    program = gltf['programs'][technique['program']]
    textures = gltf.get('textures', {})
    samplers = gltf.get('samplers', {})
    material_values = material.get('values', {})
    for uniform_name, parameter_name in technique['uniforms'].items():
        parameter = technique['parameters'][parameter_name]
        if 'semantic' in parameter:
            continue
        value = material_values.get(parameter_name, parameter.get('value'))
        if value:
            if uniform_name in program['uniform_locations']:
                location = program['uniform_locations'][uniform_name]
            else:
                location = gl.glGetUniformLocation(program['id'], uniform_name)
                program['uniform_locations'][uniform_name] = location
            if parameter['type'] == gl.GL_SAMPLER_2D:
                texture = textures[value]
                gl.glActiveTexture(gl.GL_TEXTURE0+0)
                gl.glBindTexture(texture['target'], texture['id'])
                gl.glBindSampler(0, samplers[texture['sampler']]['id'])
                gl.glUniform1i(location, 0)
            elif parameter['type'] == gl.GL_FLOAT:
                gl.glUniform1f(location, value)
            elif parameter['type'] == gl.GL_FLOAT_VEC2:
                gl.glUniform2f(location, *value)
            elif parameter['type'] == gl.GL_FLOAT_VEC3:
                gl.glUniform3f(location, *value)
            elif parameter['type'] == gl.GL_FLOAT_VEC4:
                gl.glUniform4f(location, *value)
            else:
                raise Exception('unhandled parameter type: %s' % parameter['type'])
        else:
            raise Exception('no value provided for parameter "%s"' % parameter_name)
    if CHECK_GL_ERRORS:
        if gl.glGetError() != gl.GL_NO_ERROR:
            raise Exception('error setting material state')
set_material_state.current_material = None


def set_draw_state(primitive, gltf,
                   modelview_matrix=None,
                   projection_matrix=None,
                   view_matrix=None,
                   normal_matrix=None):
    set_material_state(primitive['material'], gltf)
    material = gltf['materials'][primitive['material']]
    technique = gltf['techniques'][material['technique']]
    program = gltf['programs'][technique['program']]
    accessors = gltf['accessors']
    bufferViews = gltf['bufferViews']
    accessor_names = primitive['attributes']
    for uniform_name, parameter_name in technique['uniforms'].items():
        parameter = technique['parameters'][parameter_name]
        if 'semantic' in parameter:
            location = gl.glGetUniformLocation(program['id'], uniform_name)
            if parameter['semantic'] == 'MODELVIEW':
                if 'node' in parameter and view_matrix is not None:
                    world_matrix = gltf['nodes'][parameter['node']]['world_matrix']
                    world_matrix.dot(view_matrix, out=set_draw_state.modelview_matrix)
                    gl.glUniformMatrix4fv(location, 1, False, set_draw_state.modelview_matrix)
                elif modelview_matrix is not None:
                    gl.glUniformMatrix4fv(location, 1, False, modelview_matrix)
            elif parameter['semantic'] == 'PROJECTION':
                if 'node' in parameter:
                    raise Exception('TODO')
                elif projection_matrix is not None:
                    gl.glUniformMatrix4fv(location, 1, False, projection_matrix)
            elif parameter['semantic'] == 'MODELVIEWINVERSETRANSPOSE':
                if 'node' in parameter:
                    raise Exception('TODO')
                elif normal_matrix is not None:
                    gl.glUniformMatrix3fv(location, 1, True, normal_matrix)
            else:
                raise Exception('unhandled semantic: %s' % parameter['semantic'])
    if 'vao' not in primitive:
        enabled_locations = []
        buffer_id = None
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        for attribute_name, parameter_name in technique['attributes'].items():
            parameter = technique['parameters'][parameter_name]
            if 'semantic' in parameter:
                semantic = parameter['semantic']
                if semantic in accessor_names:
                    accessor = accessors[accessor_names[semantic]]
                    bufferView = bufferViews[accessor['bufferView']]
                    location = program['attribute_locations'][attribute_name]
                    gl.glEnableVertexAttribArray(location)
                    enabled_locations.append(location)
                    if buffer_id != bufferView['id']:
                        buffer_id = bufferView['id']
                        gl.glBindBuffer(bufferView['target'], buffer_id)
                    gl.glVertexAttribPointer(location, GLTF_BUFFERVIEW_TYPE_SIZES[accessor['type']],
                                             accessor['componentType'], False, accessor['byteStride'], c_void_p(accessor['byteOffset']))
                else:
                    raise Exception('expected a semantic property for attribute "%s"' % attribute_name)
        primitive['vao'] = vao
        gl.glBindVertexArray(0)
        for location in enabled_locations:
            gl.glDisableVertexAttribArray(location)
    gl.glBindVertexArray(primitive['vao'])
    if CHECK_GL_ERRORS:
        if gl.glGetError() != gl.GL_NO_ERROR:
            raise Exception('error setting draw state')
set_draw_state.modelview_matrix = np.empty((4,4), dtype=np.float32)
set_draw_state.vaos = {}


def draw_primitive(primitive, gltf,
                   modelview_matrix=None,
                   projection_matrix=None,
                   view_matrix=None,
                   normal_matrix=None):
    set_draw_state(primitive, gltf,
                   modelview_matrix=modelview_matrix,
                   projection_matrix=projection_matrix,
                   view_matrix=view_matrix,
                   normal_matrix=normal_matrix)
    index_accessor = gltf['accessors'][primitive['indices']]
    index_bufferView = gltf['bufferViews'][index_accessor['bufferView']]
    gl.glBindBuffer(index_bufferView['target'], index_bufferView['id'])
    gl.glDrawElements(primitive['mode'], index_accessor['count'], index_accessor['componentType'],
                      c_void_p(index_accessor['byteOffset']))
    global num_draw_calls
    num_draw_calls += 1
    if CHECK_GL_ERRORS:
        if gl.glGetError() != gl.GL_NO_ERROR:
            raise Exception('error drawing elements')
num_draw_calls = 0


def draw_mesh(mesh, gltf,
              modelview_matrix=None,
              projection_matrix=None,
              view_matrix=None,
              normal_matrix=None):
    for i, primitive in enumerate(mesh['primitives']):
        draw_primitive(primitive, gltf,
                       modelview_matrix=(modelview_matrix if i == 0 else None),
                       projection_matrix=(projection_matrix if i == 0 else None),
                       view_matrix=(view_matrix if i == 0 else None),
                       normal_matrix=(normal_matrix if i == 0 else None))


def draw_node(node, gltf,
              projection_matrix=None, view_matrix=None):
    node['world_matrix'].dot(view_matrix, out=draw_node.modelview_matrix)
    normal_matrix = np.linalg.inv(draw_node.modelview_matrix[:3,:3])
    meshes = node.get('meshes', [])
    for mesh_name in meshes:
        draw_mesh(gltf['meshes'][mesh_name], gltf,
                  modelview_matrix=draw_node.modelview_matrix,
                  projection_matrix=projection_matrix, view_matrix=view_matrix, normal_matrix=normal_matrix)
    for child in node['children']:
        draw_node(gltf['nodes'][child], gltf,
                  projection_matrix=projection_matrix, view_matrix=view_matrix)
draw_node.modelview_matrix = np.empty((4,4), dtype=np.float32)


def update_world_matrices(node, gltf, world_matrix=None):
    if 'matrix' not in node:
        matrix = np.empty((4,4), dtype=np.float32)
        set_matrix_from_quaternion(np.array(node['rotation']), matrix)
        matrix[:3, 0] *= node['scale'][0]
        matrix[:3, 1] *= node['scale'][1]
        matrix[:3, 2] *= node['scale'][2]
        matrix[:3, 3] = node['translation']
    else:
        matrix = np.array(node['matrix'], dtype=np.float32).reshape((4, 4)).T
    if world_matrix is None:
        world_matrix = matrix
    else:
        world_matrix = world_matrix.dot(matrix)
    node['world_matrix'] = world_matrix.T
    for child in [gltf['nodes'][n] for n in node['children']]:
        update_world_matrices(child, gltf, world_matrix=world_matrix)
