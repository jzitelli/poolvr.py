import sys
import os.path
import json
import argparse
import functools
import logging
from collections import defaultdict

import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import cyglfw3 as glfw


_logger = logging.getLogger(__name__)


from poolvr.app import setup_glfw
from poolvr.gl_rendering import OpenGLRenderer
try:
    from poolvr.pyopenvr_renderer import openvr, OpenVRRenderer
except ImportError as err:
    _logger.warning('could not import pyopenvr_renderer:\n%s', err)
    _logger.warning('\n\n\n**** VR FEATURES ARE NOT AVAILABLE! ****\n\n\n')
    OpenVRRenderer = None
from poolvr.keyboard_controls import init_keyboard, set_on_keydown
from poolvr.mouse_controls import init_mouse
import poolvr.gltf_utils as gltfu


def view_gltf(gltf, uri_path, scene_name=None, openvr=False,
              window_size=None, multisample=0):
    if scene_name is None:
        scene_name = gltf['scene']
    if window_size is None:
        window_size = [800, 600]
    window, renderer = setup_glfw(width=window_size[0], height=window_size[1],
                                  double_buffered=not openvr, title='gltf viewer')
    if openvr and OpenVRRenderer is not None:
        renderer = OpenVRRenderer(window_size=window_size, multisample=multisample)

    gl.glClearColor(0.01, 0.01, 0.17, 1.0);

    programs = gltfu.setup_programs(gltf, uri_path)
    _logger.info('programs = %s', programs)
    for name, program in programs.items():
        program.init_gl()
    techniques = gltfu.setup_techniques(gltf, uri_path, programs)
    _logger.info('techniques = %s', techniques)
    for name, technique in techniques.items():
        technique.init_gl()
    textures = gltfu.setup_textures(gltf, uri_path)
    _logger.info('textures = %s', textures)
    for name, texture in textures.items():
        texture.init_gl()
    materials = gltfu.setup_materials(gltf, uri_path, techniques, textures)
    _logger.info('materials = %s', materials)
    for name, material in materials.items():
        material.init_gl()
    buffer_ids = gltfu.setup_buffers(gltf, uri_path)
    _logger.info('buffer_ids = %s', buffer_ids)
    meshes = gltfu.setup_meshes(gltf, uri_path, buffer_ids, materials)
    _logger.info('meshes = %s', meshes)
    scene = gltf['scenes'][scene_name]
    nodes = [gltf['nodes'][n] for n in scene['nodes']]
    _logger.info('nodes = %s', nodes)

    raise Exception('asdf')

    for node in nodes:
        gltfu.update_world_matrices(node, gltf)
    camera_world_matrix = np.eye(4, dtype=np.float32)
    # projection_matrix = np.array(matrix44.create_perspective_projection_matrix(np.rad2deg(55), window_size[0]/window_size[1], 0.1, 1000),
    #                              dtype=np.float32)
    for node in nodes:
        if 'camera' in node:
            camera = gltf['cameras'][node['camera']]
            if 'perspective' in camera:
                perspective = camera['perspective']
                # projection_matrix = np.array(matrix44.create_perspective_projection_matrix(np.rad2deg(perspective['yfov']), perspective['aspectRatio'],
                #                                                                            perspective['znear'], perspective['zfar']),
                #                              dtype=np.float32)
            elif 'orthographic' in camera:
                raise Exception('TODO')
            camera_world_matrix = node['world_matrix']
            break
    camera_position = camera_world_matrix[3, :3]
    camera_rotation = camera_world_matrix[:3, :3]
    dposition = np.zeros(3, dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)
    key_state = defaultdict(bool)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.SetWindowShouldClose(window, gl.GL_TRUE)
        elif action == glfw.PRESS:
            key_state[key] = True
        elif action == glfw.RELEASE:
            key_state[key] = False
    glfw.SetKeyCallback(window, on_keydown)
    def on_mousedown(window, button, action, mods):
        pass
    glfw.SetMouseButtonCallback(window, on_mousedown)
    move_speed = 2.0
    turn_speed = 0.5
    def process_input(dt):
        glfw.PollEvents()
        dposition[:] = 0.0
        if key_state[glfw.KEY_W]:
            dposition[2] -= dt * move_speed
        if key_state[glfw.KEY_S]:
            dposition[2] += dt * move_speed
        if key_state[glfw.KEY_A]:
            dposition[0] -= dt * move_speed
        if key_state[glfw.KEY_D]:
            dposition[0] += dt * move_speed
        if key_state[glfw.KEY_Q]:
            dposition[1] += dt * move_speed
        if key_state[glfw.KEY_Z]:
            dposition[1] -= dt * move_speed
        theta = 0.0
        if key_state[glfw.KEY_LEFT]:
            theta -= dt * turn_speed
        if key_state[glfw.KEY_RIGHT]:
            theta += dt * turn_speed
        rotation[0,0] = np.cos(theta)
        rotation[2,2] = rotation[0,0]
        rotation[0,2] = np.sin(theta)
        rotation[2,0] = -rotation[0,2]
        camera_rotation[...] = rotation.dot(camera_world_matrix[:3,:3])
        camera_position[:] += camera_rotation.T.dot(dposition)

    # sort nodes from front to back to avoid overdraw (assuming opaque objects):
    nodes = sorted(nodes, key=lambda node: np.linalg.norm(camera_position - node['world_matrix'][3, :3]))

    meshes = list(meshes.values())

    _logger.info('entering render loop...')
    sys.stdout.flush()

    nframes = 0
    max_frame_time = 0.0
    lt = glfw.GetTime()
    while not glfw.WindowShouldClose(window):
        t = glfw.GetTime()
        dt = t - lt
        lt = t
        process_input(dt)
        with renderer.render(meshes=meshes) as frame_data:
            pass

        max_frame_time = max(max_frame_time, dt)
        if nframes == 0:
            st = glfw.GetTime()
        nframes += 1
        glfw.SwapBuffers(window)

    if nframes > 1:
        _logger.info('...exited render loop: average FPS: %f, maximum frame time: %f, average frame time: %f',
                     (nframes - 1) / (t - st), max_frame_time, (t - st) / (nframes - 1))

    renderer.shutdown()
    _logger.info('...shut down renderer')
    glfw.DestroyWindow(window)
    glfw.Terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='path of glTF file to view')
    parser.add_argument("--openvr", help="view in VR", action="store_true")
    parser.add_argument('--multisample', help="set multisampling level for VR rendering",
                        type=int, default=0)
    parser.add_argument("-v", help="verbose logging", action="store_true")

    args = parser.parse_args()
    FORMAT = '  gltfview.py  | %(asctime)s | %(name)s --- %(levelname)s *** %(message)s'
    if args.v:
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FORMAT, level=logging.WARNING)

    uri_path = os.path.dirname(args.filename)
    if args.openvr and OpenVRRenderer is None:
        raise Exception('error importing OpenVRRenderer')
    global gltf
    try:
        gltf = json.loads(open(args.filename).read())
        _logger.info('loaded "%s"', args.filename)
    except Exception as err:
        raise Exception('failed to load %s:\n%s', args.filename, err)
    view_gltf(gltf, uri_path, openvr=args.openvr)
    global view
    view = functools.partial(view_gltf, gltf, uri_path, openvr=args.openvr)


if __name__ == "__main__":
    main()
