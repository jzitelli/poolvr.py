import logging
_logger = logging.getLogger(__name__)
import numpy as np


from .gl_rendering import Material, Mesh
from .gl_primitives import CylinderPrimitive, ProjectedMesh
from .gl_techniques import LAMBERT_TECHNIQUE, EGA_TECHNIQUE


class PoolCue(Mesh):
    rotation = np.array([[1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0],
                         [0.0, -1.0, 0.0]], dtype=np.float32)
    def __init__(self, radius=0.007, length=1.15, mass=0.54):
        self.radius = radius
        self.length = length
        self.mass = mass
        cylinder = CylinderPrimitive(radius=radius, height=length)
        cylinder.attributes['a_position'] = cylinder.attributes['vertices']
        Mesh.__init__(self, {Material(LAMBERT_TECHNIQUE, values={'u_color': [0.5, 0.5, 0.0, 0.0],
                                                                 'u_lightpos': [1.0, 15.0, 1.5]})
                             : [cylinder]})
        self.update_world_matrices()
        self.shadow_mesh = ProjectedMesh(self,
                                         Material(EGA_TECHNIQUE, values={'u_color': [0.0, 0x12/0xff, 0.0, 0.0]}))
        self.position = self.world_matrix[3,:3]
        self.velocity = np.zeros(3, dtype=np.float64)
        self.quaternion = np.zeros(4, dtype=np.float64); self.quaternion[3] = 1
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.bb = np.array([[-radius, -0.5*length, -radius],
                            [radius, 0.5*length, radius]], dtype=np.float32)
        self._positions = None
        self.y_local = self.world_matrix[1,:3]
    def aabb_check(self, positions, ball_radius):
        """
        Perform axis-aligned bounding-box (AABB) check for each ball position specified in world coordinates.
        Returns a list of index / cue-local position pairs of the balls that intersect the AABB.
        """
        if self._positions is None:
            self._positions = np.empty(positions.shape, dtype=positions.dtype)
        (positions - self.position).dot(self.world_matrix[:3,:3].T, out=self._positions)
        aabb = self.bb
        separate = ((aabb[0] > self._positions + ball_radius) | (aabb[1] < self._positions - ball_radius)).any(axis=-1)
        intersect = ~separate
        return [(i, self._positions[i]) for i, inter in enumerate(intersect) if inter]
    def contact(self, position, ball_radius):
        """
        Find (if it exists, otherwise return None) the point of contact in world coordinates,
        given a ball position in cue local coordinates.
        """
        x, y, z = position
        r_sqrd = x**2 + z**2
        poc = None
        if abs(y) <= 0.5*self.length:
            # potential contact on the side of the cue:
            if r_sqrd > self.radius**2 and r_sqrd <= (self.radius + ball_radius)**2:
                n = position.copy()
                n[1] = 0.0
                n /= np.sqrt(r_sqrd)
                poc = position - ball_radius * n
        elif abs(y) <= 0.5*self.length + ball_radius:
            # potential contact on flat end of the cue:
            if r_sqrd <= self.radius**2:
                poc = position.copy()
                if y >= 0.0:
                    poc[1] -= ball_radius
                else:
                    poc[1] += ball_radius
            else:
                r = np.sqrt(r_sqrd)
                if (r - self.radius)**2 + (abs(y) - 0.5*self.length)**2 <= ball_radius**2:
                    # contact on the ring edge of the flat end:
                    if y >= 0.0:
                        n = np.array([0.0, -(y - 0.5*self.length), 0.0])
                    else:
                        n = np.array([0.0, -y - 0.5*self.length, 0.0])
                    n[::2] += -(r - self.radius) / r * position[::2]
                    n /= np.linalg.norm(n)
                    poc = position + ball_radius * n
        if poc is not None:
            return self.world_matrix[:3,:3].T.dot(poc) + self.position
