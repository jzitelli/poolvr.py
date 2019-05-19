from collections import defaultdict
from cyglfw3 import (SetKeyCallback, SetWindowShouldClose,
                     PRESS, RELEASE,
                     KEY_ESCAPE, KEY_R,
                     KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN,
                     KEY_W, KEY_S, KEY_A, KEY_D, KEY_Q, KEY_Z)
import OpenGL.GL as gl


key_state = defaultdict(bool)


_on_keydown_cb = None


def init_keyboard(window):
    SetKeyCallback(window, __on_keydown)


def set_on_keydown_callback(window, cb):
    global _on_keydown_cb
    _on_keydown_cb = cb


def __on_keydown(window, key, scancode, action, mods):
    if key == KEY_ESCAPE and action == PRESS:
        SetWindowShouldClose(window, gl.GL_TRUE)
    elif action == PRESS:
        key_state[key] = True
    elif action == RELEASE:
        key_state[key] = False
    global _on_keydown_cb
    if _on_keydown_cb:
        _on_keydown_cb(window, key, scancode, action, mods)
