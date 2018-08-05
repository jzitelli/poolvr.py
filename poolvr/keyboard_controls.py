from collections import defaultdict
import cyglfw3 as glfw
import numpy as np
import OpenGL.GL as gl


KB_TURN_SPEED = 1.3
KB_MOVE_SPEED = 0.5
KB_CUE_MOVE_SPEED = 0.2
KB_CUE_ROTATE_SPEED = 0.1

_on_keydown_cb = None

key_state = defaultdict(bool)


def __on_keydown(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.SetWindowShouldClose(window, gl.GL_TRUE)
    elif key == glfw.KEY_R and action == glfw.PRESS:
        glfw
    elif action == glfw.PRESS:
        key_state[key] = True
    elif action == glfw.RELEASE:
        key_state[key] = False
    global _on_keydown_cb
    if _on_keydown_cb:
        _on_keydown_cb(window, key, scancode, action, mods)


def set_on_keydown(window, cb):
    global _on_keydown_cb
    _on_keydown_cb = cb


def init_keyboard(window):
    glfw.SetKeyCallback(window, __on_keydown)
    theta = 0.0
    def process_keyboard_input(dt, camera_world_matrix, cue=None):
        nonlocal theta
        theta += KB_TURN_SPEED * dt * (key_state[glfw.KEY_LEFT] - key_state[glfw.KEY_RIGHT])
        sin, cos = np.sin(theta), np.cos(theta)
        camera_world_matrix[0,0] = cos
        camera_world_matrix[0,2] = -sin
        camera_world_matrix[2,0] = sin
        camera_world_matrix[2,2] = cos
        camera_world_matrix[3,0] += dt * KB_MOVE_SPEED * (key_state[glfw.KEY_D] - key_state[glfw.KEY_A])
        camera_world_matrix[3,1] += dt * KB_MOVE_SPEED * (key_state[glfw.KEY_Q] - key_state[glfw.KEY_Z])
        camera_world_matrix[3,2] += dt * KB_MOVE_SPEED * (key_state[glfw.KEY_S] - key_state[glfw.KEY_W])
        if cue is not None:
            fb = key_state[glfw.KEY_I] - key_state[glfw.KEY_K]
            lr = key_state[glfw.KEY_L] - key_state[glfw.KEY_J]
            ud = key_state[glfw.KEY_U] - key_state[glfw.KEY_M]
            # cue.world_matrix[:3,:3] = cue.rotation.T
            cue.velocity = KB_CUE_MOVE_SPEED * (lr * cue.world_matrix[0,:3] +
                                                ud * cue.world_matrix[1,:3] +
                                                fb * cue.world_matrix[2,:3])
            cue.position += cue.velocity * dt
    return process_keyboard_input
