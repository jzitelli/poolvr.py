import numpy as np
import OpenGL.GL as gl
from .gl_rendering import Primitive


def triangulate_quad(quad_face, flip_normals=False):
    if flip_normals:
        return [[quad_face[0], quad_face[2], quad_face[1]], [quad_face[0], quad_face[3], quad_face[2]]]
    else:
        return [[quad_face[0], quad_face[1], quad_face[2]], [quad_face[0], quad_face[2], quad_face[3]]]


class HexaPrimitive(Primitive):
    faces = [[0,1,2,3][::-1], # bottom
             [4,5,6,7], # top
             [0,1,5,4], # front
             [1,2,6,5], # right
             [2,3,7,6], # rear
             [7,3,0,4]] # left
    indices = np.array([triangulate_quad(quad) for quad in faces], dtype=np.uint16).reshape(-1)
    index_buffer = None
    def __init__(self, vertices=None):
        Primitive.__init__(self, gl.GL_TRIANGLES, HexaPrimitive.indices, index_buffer=HexaPrimitive.index_buffer,
                           vertices=vertices)
    def init_gl(self):
        Primitive.init_gl(self)
        if HexaPrimitive.index_buffer is None:
            HexaPrimitive.index_buffer = self.index_buffer


class BoxPrimitive(HexaPrimitive):
    def __init__(self, width=1.0, height=1.0, length=1.0):
        w, h, l = width, height, length
        vertices = np.array([[-0.5*w, -0.5*h,  0.5*l],
                             [ 0.5*w, -0.5*h,  0.5*l],
                             [ 0.5*w, -0.5*h, -0.5*l],
                             [-0.5*w, -0.5*h, -0.5*l],
                             [-0.5*w,  0.5*h,  0.5*l],
                             [ 0.5*w,  0.5*h,  0.5*l],
                             [ 0.5*w,  0.5*h, -0.5*l],
                             [-0.5*w,  0.5*h, -0.5*l]], dtype=np.float32)
        HexaPrimitive.__init__(self, vertices=vertices)


class CylinderPrimitive(Primitive):
    def __init__(self, radius=0.5, height=1.0, num_radial=12):
        self.radius = radius
        self.height = height
        vertices = np.array([[radius*np.cos(theta), -0.5*height, radius*np.sin(theta),
                              radius*np.cos(theta), 0.5*height, radius*np.sin(theta)]
                             for theta in np.linspace(0, 2*np.pi, num_radial+1)[:-1]] + [[0.0, -0.5*height, 0.0,
                                                                                          0.0,  0.5*height, 0.0]],
                            dtype=np.float32).reshape(-1, 3)
        normals = np.array([2*[np.cos(theta), 0.0, np.sin(theta)]
                            for theta in np.linspace(0, 2*np.pi, num_radial+1)[:-1]] + [[0.0, -1.0, 0.0,
                                                                                         0.0,  1.0, 0.0]],
                           dtype=np.float32).reshape(-1, 3)
        indices = np.array([[i, i+1, i+2, i+2, i+1, i+3] for i in range(0, 2*(num_radial-1), 2)] +
                           [[2*num_radial-2, 2*num_radial-1, 0, 0, 2*num_radial-1, 1]],
                           dtype=np.uint16).ravel()
        indices = np.concatenate([indices,
                                  np.array([(len(vertices)-2, i, i+2) for i in range(0, 2*(num_radial-1), 2)], dtype=np.uint16).reshape(-1),
                                  np.array([(len(vertices)-2, 2*num_radial-2, 0)], dtype=np.uint16).reshape(-1),
                                  np.array([(len(vertices)-1, i+1, i+3) for i in range(0, 2*(num_radial-1), 2)], dtype=np.uint16).reshape(-1),
                                  np.array([(len(vertices)-1, 2*num_radial-1, 1)], dtype=np.uint16).reshape(-1)])
        Primitive.__init__(self, gl.GL_TRIANGLES, indices, vertices=vertices, normals=normals)


class CirclePrimitive(Primitive):
    def __init__(self, radius=0.5, num_radial=12):
        vertices = np.concatenate([np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                                   np.array([[radius*np.cos(theta), 0.0, radius*np.sin(theta)]
                                             for theta in np.linspace(0, 2*np.pi, num_radial+1)[:-1]], dtype=np.float32)]).reshape(-1, 3)
        normals = np.array(len(vertices) * [0.0, 1.0, 0.0], dtype=np.float32)
        indices = np.array(list(range(len(vertices)))+[1], dtype=np.uint16)
        Primitive.__init__(self, gl.GL_TRIANGLE_FAN, indices, vertices=vertices, normals=normals)


class QuadPrimitive(Primitive):
    indices = np.array([0,1,3,2], dtype=np.uint16)
    index_buffer = None
    def __init__(self, vertices, **attributes):
        Primitive.__init__(self, gl.GL_TRIANGLE_STRIP, QuadPrimitive.indices, index_buffer=QuadPrimitive.index_buffer,
                           vertices=vertices, **attributes)
    def init_gl(self):
        Primitive.init_gl(self)
        if QuadPrimitive.index_buffer is None:
            QuadPrimitive.index_buffer = self.index_buffer


class PlanePrimitive(QuadPrimitive):
    def __init__(self, width=1.0, height=1.0, **kwargs):
        vertices = np.array([[-0.5*width, -0.5*height, 0.0],
                             [0.5*width, -0.5*height, 0.0],
                             [0.5*width, 0.5*height, 0.0],
                             [-0.5*width, 0.5*height, 0.0]], dtype=np.float32)
        uvs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0]], dtype=np.float32)
        QuadPrimitive.__init__(self, vertices, uvs=uvs, **kwargs)
