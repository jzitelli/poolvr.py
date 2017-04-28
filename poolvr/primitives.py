import itertools
import numpy as np
import OpenGL.GL as gl


from .gl_rendering import Primitive, Mesh


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
    tri_faces = list(itertools.chain.from_iterable(itertools.chain.from_iterable([triangulate_quad(quad) for quad in faces])))
    indices = np.array([triangulate_quad(quad) for quad in faces], dtype=np.uint16).reshape(-1)
    index_buffer = None
    def __init__(self, vertices=None):
        Primitive.__init__(self, gl.GL_TRIANGLES, HexaPrimitive.indices, index_buffer=HexaPrimitive.index_buffer,
                           vertices=vertices)
    def init_gl(self, force=False):
        Primitive.init_gl(self, force=force)
        if HexaPrimitive.index_buffer is None:
            HexaPrimitive.index_buffer = self.index_buffer


class BoxPrimitive(HexaPrimitive):
    def __init__(self, width=1.0, height=1.0, length=1.0):
        self.width = width
        self.height = height
        self.length = length
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
        self.num_radial = num_radial
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
    def __init__(self, radius=0.5, num_radial=12, basis=None):
        self.radius = radius
        self.num_radial = num_radial
        vertices = np.concatenate([np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                                   np.array([[radius*np.cos(theta), 0.0, radius*np.sin(theta)]
                                             for theta in np.linspace(0, 2*np.pi, num_radial+1)[:-1]], dtype=np.float32)]).reshape(-1, 3)
        normals = np.array(len(vertices) * [0.0, 1.0, 0.0], dtype=np.float32)
        indices = np.array(list(range(len(vertices)))+[1], dtype=np.uint16)
        if basis is not None:
            i, j, k = basis
            for iv, v in enumerate(vertices):
                vertices[iv] = np.dot(v, i) * i + np.dot(v, j) * j + np.dot(v, k) * k
            normals.reshape(-1,3)[:] = j
        Primitive.__init__(self, gl.GL_TRIANGLE_FAN, indices, vertices=vertices, normals=normals)


class ConePrimitive(Primitive):
    def __init__(self, radius=0.5, height=1.0, num_radial=12):
        self.radius = radius
        self.height = height
        self.num_radial = num_radial
        vertices = np.concatenate([np.array([[0.0, height, 0.0]], dtype=np.float32),
                                   np.array([[radius*np.cos(theta), 0.0, radius*np.sin(theta)]
                                             for theta in np.linspace(0, 2*np.pi, num_radial+1)[:-1]], dtype=np.float32)]).reshape(-1, 3)
        indices = np.array(list(range(len(vertices)))+[1], dtype=np.uint16)
        Primitive.__init__(self, gl.GL_TRIANGLE_FAN, indices, vertices=vertices)


class ConeMesh(Mesh):
    def __init__(self, material, radius=0.5, height=1.0, num_radial=12, closed=True):
        primitives = [ConePrimitive(radius=radius, height=height, num_radial=num_radial)]
        if closed:
            basis = np.eye(3); basis[1,1] *= -1; basis[2,2] *= -1
            primitives.append(CirclePrimitive(radius=radius, num_radial=num_radial, basis=basis))
        Mesh.__init__(self, {material: primitives})


class QuadPrimitive(Primitive):
    indices = np.array([0,1,3,2], dtype=np.uint16)
    index_buffer = None
    def __init__(self, vertices, **attributes):
        n1 = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(vertices[3] - vertices[2], vertices[0] - vertices[3])
        n2 /= np.linalg.norm(n2)
        if not np.allclose(n1, n2):
            raise Exception('quad vertices are not co-planar')
        uvs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0]], dtype=np.float32)
        Primitive.__init__(self, gl.GL_TRIANGLE_STRIP, QuadPrimitive.indices, index_buffer=QuadPrimitive.index_buffer,
                           vertices=vertices, uvs=uvs, **attributes)
    def init_gl(self, force=False):
        Primitive.init_gl(self, force=force)
        if QuadPrimitive.index_buffer is None:
            QuadPrimitive.index_buffer = self.index_buffer


class PlanePrimitive(QuadPrimitive):
    def __init__(self, width=1.0, height=1.0, depth=0.0, **attributes):
        self.width = width
        self.height = height
        self.depth = depth
        vertices = np.array([[-0.5*width, -0.5*height, 0.5*depth],
                             [0.5*width, -0.5*height, 0.5*depth],
                             [0.5*width, 0.5*height, -0.5*depth],
                             [-0.5*width, 0.5*height, -0.5*depth]], dtype=np.float32)
        QuadPrimitive.__init__(self, vertices, **attributes)


class SpherePrimitive(Primitive):
    """
    Sphere geometry based on three.js implementation:
    https://github.com/mrdoob/three.js/blob/44ec6fa7a277a3ee0d2883d9686978655bdac235/src/geometries/SphereGeometry.js
    """
    def __init__(self,
                 radius=0.5,
                 widthSegments=16,
                 heightSegments=12,
                 phiStart=0.0,
                 phiLength=2*np.pi,
                 thetaStart=0.0,
                 thetaLength=np.pi):
        self.radius = radius
        self.widthSegments = widthSegments
        self.heightSegments = heightSegments
        self.phiStart = phiStart
        self.phiLength = phiLength
        self.thetaStart = thetaStart
        self.thetaLength = thetaLength
        thetaEnd = thetaStart + thetaLength
        index = 0
        vertices = []
        uvs = []
        positions = []
        for iy in range(0, heightSegments+1):
            verticesRow = []
            v = iy / heightSegments
            for ix in range(0, widthSegments+1):
                u = ix / widthSegments
                px = -radius * np.cos(phiStart + u * phiLength) * np.sin(thetaStart + v * thetaLength)
                py = radius * np.cos(thetaStart + v * thetaLength)
                pz = radius * np.sin(phiStart + u * phiLength) * np.sin(thetaStart + v * thetaLength)
                positions.append([px, py, pz])
                uvs.append([u, 1-v])
                verticesRow.append(index)
                index += 1
            vertices.append(verticesRow)
        indices = []
        for iy in range(heightSegments):
            for ix in range(widthSegments):
                v1 = vertices[iy][ix+1]
                v2 = vertices[iy][ix]
                v3 = vertices[iy+1][ix]
                v4 = vertices[iy+1][ix+1]
                if iy != 0 or thetaStart > 0:
                    indices.append([v1, v2, v4])
                if iy != heightSegments - 1 or thetaEnd < np.pi:
                    indices.append([v2, v3, v4])
        indices = np.array(indices, dtype=np.uint16).ravel()
        vertices = np.array(positions, dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32)
        Primitive.__init__(self, gl.GL_TRIANGLE_STRIP, indices,
                           vertices=vertices, uvs=uvs)


class RoundedRectanglePrimitive(Primitive):
    def __init__(self, width, height, radius=None):
        if radius is None:
            radius = 0.15 * min(width, height)
        self.width = width
        self.height = height
        self.radius = radius
