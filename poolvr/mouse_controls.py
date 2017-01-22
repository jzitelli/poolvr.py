from collections import defaultdict
import cyglfw3 as glfw


MOUSE_MOVE_SPEED = 0.07
MOUSE_CUE_MOVE_SPEED = 0.06


def init_mouse(window):
    glfw.SetInputMode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    mouse_button_state = defaultdict(int)
    def on_mousedown(window, button, action, mods):
        if action == glfw.PRESS:
            mouse_button_state[button] = True
        elif action == glfw.RELEASE:
            mouse_button_state[button] = False
    glfw.SetMouseButtonCallback(window, on_mousedown)
    cursor_pos = glfw.GetCursorPos(window)
    def process_mouse_input(dt, cue):
        pos = glfw.GetCursorPos(window)
        nonlocal cursor_pos
        lr, fb = pos[0] - cursor_pos[0], pos[1] - cursor_pos[1]
        cursor_pos = pos
        cue.velocity[2] = fb * MOUSE_CUE_MOVE_SPEED
        cue.velocity[0] = lr * MOUSE_CUE_MOVE_SPEED
        cue.position[:] += dt * cue.velocity
    return process_mouse_input
