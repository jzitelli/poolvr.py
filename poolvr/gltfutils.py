import logging
import os.path
import base64
from ctypes import c_void_p
try: # python 3.3 or later
    from types import MappingProxyType
except ImportError as err:
    MappingProxyType = dict

import numpy as np
import OpenGL.GL as gl


_logger = logging.getLogger(__name__)


from .gl_rendering import (Program, Technique, Material, Texture, CubeTexture,
                           Primitive, Mesh, Node, set_matrix_from_quaternion, CHECK_GL_ERRORS)


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
    # def __init__(self, gltf):
    #     self.update(gltf)


class GLTFNode(GLTFDict):
    pass


class GLTFScene(GLTFNode):
    @classmethod
    def load_from(cls, gltf):
        pass


def setup_programs(gltf, uri_path):
    shaders = gltf['shaders']
    shader_src = {}
    for shader_name, shader in shaders.items():
        uri = shader['uri']
        if uri.startswith('data:text/plain;base64,'):
            shader_str = base64.urlsafe_b64decode(uri.split(',')[1]).decode()
            _logger.info('decoded shader "%s":\n%s' % (shader_name, shader_str))
        else:
            filename = os.path.join(uri_path, shader['uri'])
            shader_str = open(filename).read()
            _logger.info('loaded shader "%s" (from %s):\n%s' % (shader_name, filename, shader_str))
        shader_src[shader_name] = shader_str
    programs = {}
    for program_name, program in gltf['programs'].items():
        programs[program_name] = Program(shader_src[program['vertexShader']],
                                         shader_src[program['fragmentShader']],
                                         name=program_name)
    return programs


def setup_textures(gltf, uri_path):
    # TODO: support data URIs
    textures = {}
    for texture_name, texture in gltf.get('textures', {}).items():
        sampler = gltf['samplers'][texture['sampler']]
        textures[texture_name] = Texture(os.path.join(uri_path, gltf['images'][texture['source']]['uri']),
                                         name=texture_name, min_filter=sampler.get('minFilter', 9986),
                                         mag_filter=sampler.get('magFilter', 9729),
                                         wrap_s=sampler.get('wrapS', 10497), wrap_t=sampler.get('wrapT', 10497))
    return textures


def setup_techniques(gltf, uri_path, programs):
    techniques = {}
    for technique_name, technique in gltf['techniques'].items():
        techniques[technique_name] = Technique(programs[technique['program']])
    return techniques


def setup_materials(gltf, uri_path, techniques, textures):
    materials = {}
    for material_name, material in gltf['materials'].items():
        # technique = techniques[material['technique']]
        technique = gltf['techniques'][material['technique']]
        texture_params = {param_name: param for param_name, param in technique['parameters'].items()
                          if param['type'] == gl.GL_SAMPLER_2D}
        tech_textures = {param_name: texture for param_name, texture in textures.items()
                         if param_name in texture_params}
        materials[material_name] = Material(techniques[material['technique']],
                                            values={param_name: v
                                                    for param_name, v in material['values'].items()
                                                    if param_name not in tech_textures},
                                            textures=tech_textures)
    return materials


def load_buffers(gltf, uri_path):
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
    data_buffers = load_buffers(gltf, uri_path)
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
