from collections import defaultdict
import glfw


MOUSE_MOVE_SPEED = 0.07
MOUSE_CUE_MOVE_SPEED = 0.06
MOUSE_CUE_ROTATE_SPEED = 0.03


def init_mouse(window):
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    mouse_button_state = defaultdict(int)
    def on_mousedown(window, button, action, mods):
        if action == glfw.PRESS:
            mouse_button_state[button] = True
        elif action == glfw.RELEASE:
            mouse_button_state[button] = False
    glfw.set_mouse_button_callback(window, on_mousedown)
    cursor_pos = glfw.get_cursor_pos(window)
    theta_x, theta_y = 0.0, 0.0
    def process_mouse_input(dt, cue):
        pos = glfw.get_cursor_pos(window)
        nonlocal cursor_pos
        lr, fb = pos[0] - cursor_pos[0], pos[1] - cursor_pos[1]
        cursor_pos = pos
        nonlocal theta_x, theta_y
        cue.velocity[2] = fb * MOUSE_CUE_MOVE_SPEED
        cue.velocity[0] = lr * MOUSE_CUE_MOVE_SPEED
        cue.position[:] += dt * cue.velocity
    return process_mouse_input
