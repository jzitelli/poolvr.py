from collections import defaultdict
import cyglfw3 as glfw
import numpy as np
import OpenGL.GL as gl


KB_TURN_SPEED = 1.2
KB_MOVE_SPEED = 0.3
KB_CUE_MOVE_SPEED = 0.3
KB_CUE_ROTATE_SPEED = 0.1


def init_keyboard(window):
    key_state = defaultdict(bool)
    def on_keydown(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.SetWindowShouldClose(window, gl.GL_TRUE)
        elif action == glfw.PRESS:
            key_state[key] = True
        elif action == glfw.RELEASE:
            key_state[key] = False
    glfw.SetKeyCallback(window, on_keydown)
    theta = 0.0
    def process_keyboard_input(dt, camera_world_matrix, cue):
        nonlocal theta
        theta += KB_TURN_SPEED * dt * (key_state[glfw.KEY_LEFT] - key_state[glfw.KEY_RIGHT])
        sin, cos = np.sin(theta), np.cos(theta)
        camera_position = camera_world_matrix[3,:3]
        camera_world_matrix[0,0] = cos
        camera_world_matrix[0,2] = -sin
        camera_world_matrix[2,0] = sin
        camera_world_matrix[2,2] = cos
        fb = KB_MOVE_SPEED * dt * (-key_state[glfw.KEY_W] + key_state[glfw.KEY_S])
        lr = KB_MOVE_SPEED * dt * (key_state[glfw.KEY_D] - key_state[glfw.KEY_A])
        ud = KB_MOVE_SPEED * dt * (key_state[glfw.KEY_Q] - key_state[glfw.KEY_Z])
        camera_position[:] += fb * camera_world_matrix[2,:3] + lr * camera_world_matrix[0,:3] + ud * camera_world_matrix[1,:3]
        fb = KB_CUE_MOVE_SPEED * (-key_state[glfw.KEY_I] + key_state[glfw.KEY_K])
        lr = KB_CUE_MOVE_SPEED * (key_state[glfw.KEY_L] - key_state[glfw.KEY_J])
        ud = KB_CUE_MOVE_SPEED * (key_state[glfw.KEY_U] - key_state[glfw.KEY_M])
        cue.world_matrix[:3,:3] = cue.rotation.T
        cue.velocity = -fb * cue.world_matrix[1,:3] + lr * cue.world_matrix[0,:3] + ud * cue.world_matrix[2,:3]
        cue.position += cue.velocity * dt
    return process_keyboard_input
