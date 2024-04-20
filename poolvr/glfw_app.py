import logging
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import glfw


_logger = logging.getLogger('poolvr')


from .gl_rendering import OpenGLRenderer


def setup_glfw(title="poolvr.py 0.0.1",
               window_size=(800,600),
               double_buffered=False,
               multisample=4,
               fullscreen=False):
    if not glfw.init():
        raise Exception('failed to initialize glfw')
    if not double_buffered:
        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        glfw.swap_interval(0)
    if multisample:
        glfw.window_hint(glfw.SAMPLES, multisample)
    width, height = window_size
    if fullscreen:
        window = glfw.create_window(width, height, title, glfw.get_primary_monitor(), None)
    else:
        window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception('failed to create glfw window')
    glfw.make_context_current(window)
    _logger.info('GL_VERSION: %s', gl.glGetString(gl.GL_VERSION))
    renderer = OpenGLRenderer(window_size=(width, height), znear=0.1, zfar=1000)
    def on_resize(renderer, window, width, height):
        gl.glViewport(0, 0, width, height)
        renderer.window_size[:] = (width, height)
        renderer.update_projection_matrix()
    from functools import partial
    on_resize = partial(on_resize, renderer)
    glfw.set_window_size_callback(window, on_resize)
    renderer.init_gl()
    on_resize(window, window_size[0], window_size[1])
    return window, renderer


def capture_window(window,
                   filename='screenshot.png'):
    import PIL
    import os.path
    if not filename.endswith('.png'):
        filename += '.png'
    _logger.info('saving screen capture...')
    mWidth, mHeight = glfw.get_window_size(window)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    pixels = gl.glReadPixels(0, 0, mWidth, mHeight, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    pil_image = PIL.Image.frombytes('RGB', (mWidth, mHeight), pixels)
    pil_image = pil_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    dire = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dire):
        from os import makedirs
        makedirs(dire)
    pil_image.save(filename)
    _logger.info('...saved screen capture to "%s"', filename)
