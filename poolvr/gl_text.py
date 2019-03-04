import pkgutil
import os.path
from ctypes import c_void_p
import numpy as np
import PIL.Image as Image


from .gl_rendering import gl, Program, Technique, Texture, Material, Mesh
from .gl_primitives import PlanePrimitive


TEXT_TECHNIQUE = Technique(Program(pkgutil.get_data('poolvr', 'shaders/gltext_vs.glsl').decode(),
                                   pkgutil.get_data('poolvr', 'shaders/gltext_fs.glsl').decode()),
                           attributes={'a_position': {'type': gl.GL_FLOAT_VEC2},
                                       'a_texcoord0': {'type': gl.GL_FLOAT_VEC2}},
                           uniforms={'u_modelview': {'type': gl.GL_FLOAT_MAT4},
                                     'u_projection': {'type': gl.GL_FLOAT_MAT4},
                                     'advance': {'type': gl.GL_FLOAT},
                                     'u_color': {'type': gl.GL_FLOAT_VEC4,
                                                 'value': [0.0, 1.0, 1.0, 0.0]},
                                     'u_fonttex': {'type': gl.GL_SAMPLER_2D}})
DEFAULT_FONT_IMAGE = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'stb_font_consolas_32_usascii.png')
TEXT_TEXTURE = Texture(DEFAULT_FONT_IMAGE)
TEXT_MATERIAL = Material(TEXT_TECHNIQUE, textures={'u_fonttex': TEXT_TEXTURE})


STB_FONT_consolas_32_usascii_BITMAP_WIDTH = 256
STB_FONT_consolas_32_usascii_BITMAP_HEIGHT = 138
STB_FONT_consolas_32_usascii_BITMAP_HEIGHT_POW2 = 256
STB_FONT_consolas_32_usascii_FIRST_CHAR = 32
STB_FONT_consolas_32_usascii_NUM_CHARS = 95
STB_FONT_consolas_32_usascii_LINE_SPACING = 21
stb__consolas_32_usascii_x = np.array([0,6,4,0,1,0,0,7,4,4,2,1,3,4,
                                       6,1,1,2,2,2,0,2,1,1,1,1,6,3,2,2,3,4,0,0,2,1,1,3,3,1,1,2,2,2,
                                       3,0,1,0,2,0,2,1,1,1,0,0,0,0,1,5,2,4,1,0,0,2,2,2,1,1,0,1,2,2,
                                       2,2,2,1,2,1,2,1,3,2,0,2,1,0,1,0,2,2,7,3,1])
stb__consolas_32_usascii_y = np.array([23,0,0,2,-1,0,1,0,-1,-1,0,6,17,13,
                                       18,0,2,2,2,2,2,2,2,2,2,2,7,7,5,10,5,0,0,2,2,2,2,2,2,2,2,2,2,2,
                                       2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,2,27,0,7,0,7,0,7,0,7,0,0,
                                       0,0,0,7,7,7,7,7,7,7,2,7,7,7,7,7,7,0,-3,0,11])
stb__consolas_32_usascii_w = np.array([0,5,10,17,15,18,18,4,10,9,14,16,9,10,
                                       6,15,16,14,14,14,17,14,15,15,15,15,6,9,13,14,13,11,18,18,14,15,16,12,12,15,15,13,12,15,
                                       13,17,15,17,14,18,15,15,16,15,18,17,18,18,15,9,14,9,15,18,11,14,14,13,15,15,17,16,14,14,
                                       12,15,14,16,14,16,14,15,14,13,16,14,16,18,16,17,14,13,4,13,16])
stb__consolas_32_usascii_h = np.array([0,24,9,21,28,24,23,9,31,31,15,17,12,3,
                                       6,27,22,21,21,22,21,22,22,21,22,21,17,22,19,9,19,24,30,21,21,22,21,21,21,22,21,21,22,21,
                                       21,21,21,22,21,27,21,22,21,22,21,21,21,21,21,30,27,30,11,3,8,17,24,17,24,17,23,23,23,23,
                                       30,23,23,16,16,17,23,23,16,17,22,17,16,16,16,23,16,30,33,30,7])
stb__consolas_32_usascii_s = np.array([241,223,178,141,107,173,1,204,6,17,137,
                                       91,152,245,238,142,200,209,157,88,193,185,217,240,233,240,249,1,15,189,1,
                                       229,78,174,159,11,124,111,98,72,66,52,43,19,226,1,224,167,211,123,82,
                                       27,35,56,172,191,119,138,103,97,158,68,162,162,209,45,208,60,192,29,101,
                                       84,69,54,55,20,241,152,201,74,135,119,186,123,150,108,216,233,169,36,137,
                                       27,1,41,221])
stb__consolas_32_usascii_t = np.array([25,1,121,82,1,1,35,121,1,1,121,
                                       104,121,121,121,1,35,59,59,59,82,35,35,59,35,82,35,59,104,121,104,
                                       1,1,82,82,59,82,82,82,59,82,82,59,82,82,82,59,35,82,1,82,
                                       59,82,59,59,59,59,59,59,1,1,1,121,133,121,104,1,104,1,104,35,
                                       35,35,35,1,35,1,104,104,104,35,35,104,104,35,104,104,104,104,35,104,
                                       1,1,1,121])
stb__consolas_32_usascii_a = np.array([282,282,282,282,282,282,282,282,
                                       282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
                                       282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
                                       282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
                                       282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
                                       282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,
                                       282,282,282,282,282,282,282])

recip_width = 1.0 / STB_FONT_consolas_32_usascii_BITMAP_WIDTH
recip_height = 1.0 / STB_FONT_consolas_32_usascii_BITMAP_HEIGHT
s0f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
t0f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
s1f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
t1f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
x0f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
y0f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
x1f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
y1f = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
advance = np.empty(STB_FONT_consolas_32_usascii_NUM_CHARS, dtype=np.float32)
for i in range(len(s0f)):
    s0f[i] = (stb__consolas_32_usascii_s[i] - 0.5) * recip_width
    t0f[i] = (stb__consolas_32_usascii_t[i] - 0.5) * recip_height
    s1f[i] = (stb__consolas_32_usascii_s[i] + stb__consolas_32_usascii_w[i] + 0.5) * recip_width
    t1f[i] = (stb__consolas_32_usascii_t[i] + stb__consolas_32_usascii_h[i] + 0.5) * recip_height
    x0f[i] = stb__consolas_32_usascii_x[i] - 0.5
    y0f[i] = stb__consolas_32_usascii_y[i] - 0.5
    x1f[i] = stb__consolas_32_usascii_x[i] + stb__consolas_32_usascii_w[i] + 0.5
    y1f[i] = stb__consolas_32_usascii_y[i] + stb__consolas_32_usascii_h[i] + 0.5
    advance[i] = stb__consolas_32_usascii_a[i] / 16.0


class TexturedText(Mesh):
    def __init__(self):
        Mesh.__init__(self, {TEXT_MATERIAL: [PlanePrimitive(width=)]})


class TextDrawer(object):
    def __init__(self, font_image=DEFAULT_FONT_IMAGE):

        self._advance = advance
        buffer_ids = gl.glGenBuffers(STB_FONT_consolas_32_usascii_NUM_CHARS)
        for i, buffer_id in enumerate(buffer_ids):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
            buffer_data = np.array([x0f[i], y0f[i],
                                    x0f[i], y1f[i],
                                    x1f[i], y0f[i],
                                    x1f[i], y1f[i],
                                    s0f[i], t1f[i],
                                    s0f[i], t0f[i],
                                    s1f[i], t1f[i],
                                    s1f[i], t0f[i]]).tobytes()
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(buffer_data), buffer_data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        if gl.glGetError() != gl.GL_NO_ERROR:
            raise Exception('failed to create buffers')
        self._buffer_ids = buffer_ids
        self._matrix = np.eye(4, dtype=np.float32)
        self._matrix[:3, :3] *= 0.01;
        self._modelview_matrix = np.eye(4, dtype=np.float32)
    def set_text(self, text):
        self.text = text
    def draw_text(self, text, color=(1.0, 1.0, 1.0, 0.0),
                  view_matrix=None, projection_matrix=None):
        gl.glUseProgram(self._program_id)
        gl.glActiveTexture(gl.GL_TEXTURE0+0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glBindSampler(0, self._sampler_id)
        gl.glUniform1i(self._uniform_locations['u_fonttex'], 0)
        gl.glUniform4f(self._uniform_locations['u_color'], *color)
        if view_matrix is not None:
            self._matrix.dot(view_matrix, out=self._modelview_matrix)
            gl.glUniformMatrix4fv(self._uniform_locations['u_modelview'], 1, False, self._modelview_matrix)
        if projection_matrix is not None:
            gl.glUniformMatrix4fv(self._uniform_locations['u_projection'], 1, False, projection_matrix)
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        x = 0.0
        text = [ord(c) - 32 for c in text]
        for i in text:
            if i >= 0 and i < 95:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffer_ids[i])
                gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, c_void_p(0))
                gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 0, c_void_p(8*4))
                gl.glUniform1f(self._uniform_locations['advance'], x)
                gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                x += self._advance[i]
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisable(gl.GL_BLEND)
    def draw(self, view=None, projection=None, color=(1.0, 1.0, 1.0, 0.0)):
        gl.glUseProgram(self._program_id)
        gl.glActiveTexture(gl.GL_TEXTURE0+0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glBindSampler(0, self._sampler_id)
        gl.glUniform1i(self._uniform_locations['u_fonttex'], 0)
        gl.glUniform4f(self._uniform_locations['u_color'], *color)
