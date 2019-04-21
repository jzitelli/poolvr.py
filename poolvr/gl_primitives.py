import itertools
import logging
import numpy as np
import OpenGL.GL as gl

try:
    import ode
except Exception:
    import poolvr.fake_ode as ode


from .gl_rendering import Primitive, Mesh


_logger = logging.getLogger(__name__)


def triangulate_quad(quad_face, flip_normals=False):
    if flip_normals:
        return [[quad_face[0], quad_face[2], quad_face[1]], [quad_face[0], quad_face[3], quad_face[2]]]
    else:
        return [[quad_face[0], quad_face[1], quad_face[2]], [quad_face[0], quad_face[2], quad_face[3]]]


class HexaPrimitive(Primitive):
    faces = np.array(
        [[0,1,2,3][::-1], # bottom
         [4,5,6,7],       # top
         [0,1,5,4],       # front
         [1,2,6,5],       # right
         [2,3,7,6],       # rear
         [7,3,0,4]]       # left
        , dtype=np.int64)
    tri_faces = list(itertools.chain.from_iterable(itertools.chain.from_iterable([triangulate_quad(quad) for quad in faces])))
    indices = np.array([triangulate_quad(quad) for quad in faces], dtype=np.uint16).reshape(-1)
    _INDEX_BUFFER = None
    def __init__(self, vertices=None):
        "A :ref:`Primitive` with the geometry of a hexahedron (six-sided three-dimension solid)."
        Primitive.__init__(self, gl.GL_TRIANGLES, HexaPrimitive.indices, index_buffer=HexaPrimitive._INDEX_BUFFER,
                           vertices=vertices)

    def init_gl(self, force=False):
        Primitive.init_gl(self, force=force)
        if HexaPrimitive._INDEX_BUFFER is None:
            HexaPrimitive._INDEX_BUFFER = self.index_buffer


class BoxPrimitive(HexaPrimitive):
    # *** TODO *** use unambigous variable names (e.g. "length_x", "l_x", "size_x", ..., instead of "width")
    def __init__(self, width=1.0, height=1.0, length=1.0):
        "A :ref:`Primitive` with the geometry of a box."
        self.width = width
        self.height = height
        self.length = length
        self.lengths = (width, height, length)
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

    def create_ode_mass(self, total_mass):
        mass = ode.Mass()
        mass.setSphereTotal(total_mass, self.width, self.height, self.length)
        return mass

    def create_ode_geom(self, space):
        return ode.GeomBox(space=space, lengths=self.lengths)


class HexaMesh(Mesh):
    def __init__(self, material, *args, **kwargs):
        self.primitive = HexaPrimitive(*args, **kwargs)
        self.primitives = [self.primitive]
        Mesh.__init__(self, {material: self.primitives})


class BoxMesh(Mesh):
    def __init__(self, material, *args, **kwargs):
        self.primitive = BoxPrimitive(*args, **kwargs)
        self.primitives = [self.primitive]
        Mesh.__init__(self, {material: self.primitives})

    def create_ode_body(self, world, space, total_mass):
        body = ode.Body(world)
        body.setMass(self.primitive.create_ode_mass(total_mass))
        body.shape = "box"
        body.boxsize = tuple(self.primitive.lengths)
        geom = self.primitive.create_ode_geom(space)
        geom.setBody(body)
        return body


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

    def create_ode_mass(self, total_mass, direction=3):
        mass = ode.Mass()
        mass.setCylinderTotal(total_mass, direction, self.radius, self.height)
        return mass

    def create_ode_geom(self, space):
        return ode.GeomCylinder(space=space, radius=self.radius, length=self.height)


class CylinderMesh(Mesh):
    def __init__(self, material=None, radius=0.5, height=1.0, num_radial=12,
                 bottom_closed=True, top_closed=True):
        primitives = [CylinderPrimitive(radius=radius, height=height, num_radial=num_radial)]
        if bottom_closed:
            basis = np.eye(3); basis[1,1] *= -1; basis[2,2] *= -1
            circle = CirclePrimitive(radius=radius, num_radial=num_radial, basis=basis)
            circle.attributes['vertices'][:,1] -= 0.5*height
            primitives.append(circle)
        if top_closed:
            circle = CirclePrimitive(radius=radius, num_radial=num_radial)
            circle.attributes['vertices'][:,1] += 0.5*height
            primitives.append(circle)
        Mesh.__init__(self, {material: primitives})


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
    INDICES = np.array([0,1,3,2], dtype=np.uint16)
    INDEX_BUFFER = None
    def __init__(self, vertices, **attributes):
        n1 = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(vertices[3] - vertices[2], vertices[0] - vertices[3])
        n2 /= np.linalg.norm(n2)
        if not np.allclose(n1, n2):
            raise Exception('quad vertices are not co-planar')
        normals = np.array(4 * [(n1 + n2) / 2], dtype=np.float32)
        tangents = np.array([vertices[1] - vertices[0],
                             vertices[1] - vertices[0],
                             vertices[2] - vertices[3],
                             vertices[2] - vertices[3]], dtype=np.float32)
        for i, t in enumerate(tangents):
            tangents[i] /= np.linalg.norm(t)
        uvs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0]], dtype=np.float32)
        Primitive.__init__(self, gl.GL_TRIANGLE_STRIP, QuadPrimitive.INDICES, index_buffer=QuadPrimitive.INDEX_BUFFER,
                           vertices=vertices, uvs=uvs, normals=normals, tangents=tangents, **attributes)

    def init_gl(self, force=False):
        Primitive.init_gl(self, force=force)
        if QuadPrimitive.INDEX_BUFFER is None:
            QuadPrimitive.INDEX_BUFFER = self.index_buffer


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
        # *** TODO *** vectorize:
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

    def create_ode_mass(self, total_mass):
        mass = ode.Mass()
        mass.setSphereTotal(total_mass, self.radius)
        return mass

    def create_ode_geom(self, space):
        return ode.GeomSphere(space=space, radius=self.radius)


class SphereMesh(Mesh):
    def __init__(self, material, *args, **kwargs):
        self.primitive = SpherePrimitive(*args, **kwargs)
        self.primitives = [self.primitive]
        Mesh.__init__(self, {material: self.primitives})

    def create_ode_body(self, world, space, total_mass):
        body = ode.Body(world)
        body.setMass(self.primitive.create_ode_mass(total_mass))
        body.shape = "sphere"
        body.boxsize = tuple(self.primitive.lengths)
        geom = self.primitive.create_ode_geom(space)
        geom.setBody(body)
        return body


class RoundedRectanglePrimitive(Primitive):
    def __init__(self, width, height, radius=None):
        if radius is None:
            radius = 0.15 * min(width, height)
        self.width = width
        self.height = height
        self.radius = radius


class ProjectedMesh(Mesh):
    def __init__(self, mesh, material, primitives=None,
                 normal=(0.0, 1.0, 0.0), c=1.02*0.77,
                 light_position=(0.0, 100.2, 0.0, 0.1)):
        self.mesh = mesh
        self.material = material
        if primitives is None:
            primitives = itertools.chain.from_iterable(mesh.primitives.values())
        self.primitives = list(primitives)
        Mesh.__init__(self, {material: self.primitives})
        self.normal = np.array(normal)
        self.c = c
        self._plane = np.array(list(self.normal[:]) + [-self.c])
        self._shadow_matrix = np.eye(4, dtype=np.float32)
        self.light_position = light_position

    def update(self, c=None):
        if c is not None:
            self.c = c
            self._plane = np.array(list(self.normal[:]) + [-self.c])
        light_position = self.light_position
        v = self._plane.dot(light_position)
        shadow_matrix = self._shadow_matrix
        shadow_matrix[:,0] = -light_position[0] * self._plane
        shadow_matrix[0,0] += v
        shadow_matrix[:,1] = -light_position[1] * self._plane
        shadow_matrix[1,1] += v
        shadow_matrix[:,2] = -light_position[2] * self._plane
        shadow_matrix[2,2] += v
        shadow_matrix[:,3] = -light_position[3] * self._plane
        shadow_matrix[3,3] += v
        self.mesh.world_matrix.dot(shadow_matrix, out=self.world_matrix)


class ArrowMesh(Mesh):
    def __init__(self, material=None, head_radius=0.05, tail_radius=0.02,
                 head_length=0.1, tail_length=1, num_radial=12):
        head = ConeMesh(material=material, radius=head_radius, height=head_length, num_radial=num_radial)
        tail = CylinderMesh(material=material, radius=tail_radius, height=tail_length, num_radial=num_radial,
                            top_closed=False)
        for prim in tail.primitives[material]:
            prim.attributes['vertices'][:,1] -= 0.5*tail_length
        Mesh.__init__(self, {material: head.primitives[material] + tail.primitives[material]})
        self.material = material
