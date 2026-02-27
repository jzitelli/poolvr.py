import * as THREE from 'three';

/**
 * Build a hexahedral prism (cushion/rail shape) from bottom and top quads.
 * bottomQuad: array of 4 [x,y,z], topQuad: array of 4 [x,y,z]
 */
function buildHexaGeometry(bottomQuad, topQuad) {
  // 6 faces: bottom, top, and 4 sides
  const vertices = [];
  const normals = [];

  function addFace(a, b, c, d) {
    const va = new THREE.Vector3(...a);
    const vb = new THREE.Vector3(...b);
    const vc = new THREE.Vector3(...c);
    const vd = new THREE.Vector3(...d);
    const e1 = new THREE.Vector3().subVectors(vb, va);
    const e2 = new THREE.Vector3().subVectors(vc, va);
    const n = new THREE.Vector3().crossVectors(e1, e2).normalize();
    // two triangles: a,b,c and a,c,d
    vertices.push(...a, ...b, ...c, ...a, ...c, ...d);
    for (let k = 0; k < 6; k++) normals.push(n.x, n.y, n.z);
  }

  const b = bottomQuad, t = topQuad;
  // bottom face (outward-facing down)
  addFace(b[0], b[3], b[2], b[1]);
  // top face
  addFace(t[0], t[1], t[2], t[3]);
  // 4 side faces
  addFace(b[0], b[1], t[1], t[0]);
  addFace(b[1], b[2], t[2], t[1]);
  addFace(b[2], b[3], t[3], t[2]);
  addFace(b[3], b[0], t[0], t[3]);

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geom;
}

/**
 * Create the table mesh group from the geometry returned by the API.
 */
export function createTableMesh(geom) {
  const { corners, pocket_positions, L, W, H, w, h, ball_radius, width_rail, R_c, R_s } = geom;
  const group = new THREE.Group();

  // --- Surface ---
  const surfaceGeom = new THREE.PlaneGeometry(W + 2 * w, L + 2 * w);
  surfaceGeom.rotateX(-Math.PI / 2);
  const surfaceMat = new THREE.MeshPhongMaterial({ color: 0x00aa00 });
  const surface = new THREE.Mesh(surfaceGeom, surfaceMat);
  surface.position.y = H;
  surface.receiveShadow = true;
  group.add(surface);

  // --- Cushions ---
  // corners is 24x2 (XZ coords). Groups of 4: indices 0-3, 4-7, 8-11, 12-15, 16-19, 20-23
  const cushionMat = new THREE.MeshPhongMaterial({ color: 0x028844 });
  for (let ci = 0; ci < 6; ci++) {
    const base = ci * 4;
    const c = corners;
    const bottomQuad = [
      [c[base][0],     H, c[base][1]],
      [c[base + 1][0], 0.95 * h + H, c[base + 1][1]],
      [c[base + 2][0], 0.95 * h + H, c[base + 2][1]],
      [c[base + 3][0], H, c[base + 3][1]],
    ];
    const topQuad = [
      [c[base][0],     1.3 * h + H, c[base][1]],
      [c[base + 1][0], h + H, c[base + 1][1]],
      [c[base + 2][0], h + H, c[base + 2][1]],
      [c[base + 3][0], 1.3 * h + H, c[base + 3][1]],
    ];
    const cushionGeom = buildHexaGeometry(bottomQuad, topQuad);
    const cushion = new THREE.Mesh(cushionGeom, cushionMat);
    group.add(cushion);
  }

  // --- Rails ---
  const railMat = new THREE.MeshPhongMaterial({ color: 0xdda400 });
  const w_r = width_rail;
  const c = corners;

  // Rail 0: behind cushion 0 (bottom short cushion)
  function addRail(bottomQuad, topQuad) {
    const g = buildHexaGeometry(bottomQuad, topQuad);
    group.add(new THREE.Mesh(g, railMat));
  }

  // Rail pair 1: bottom short rail (indices 0,3)
  addRail(
    [[c[0][0], H, c[0][1]], [c[3][0], H, c[3][1]], [c[3][0], H, c[3][1] - w_r], [c[0][0], H, c[0][1] - w_r]],
    [[c[0][0], 1.3*h+H, c[0][1]], [c[3][0], 1.3*h+H, c[3][1]], [c[3][0], 1.3*h+H, c[3][1] - w_r], [c[0][0], 1.3*h+H, c[0][1] - w_r]]
  );
  // Rail pair 2: right side rail (indices 4,7)
  addRail(
    [[c[4][0], H, c[4][1]], [c[7][0], H, c[7][1]], [c[7][0]+w_r, H, c[7][1]], [c[4][0]+w_r, H, c[4][1]]],
    [[c[4][0], 1.3*h+H, c[4][1]], [c[7][0], 1.3*h+H, c[7][1]], [c[7][0]+w_r, 1.3*h+H, c[7][1]], [c[4][0]+w_r, 1.3*h+H, c[4][1]]]
  );
  // Rail pair 3: right side rail upper (indices 8,11)
  addRail(
    [[c[8][0], H, c[8][1]], [c[11][0], H, c[11][1]], [c[11][0]+w_r, H, c[11][1]], [c[8][0]+w_r, H, c[8][1]]],
    [[c[8][0], 1.3*h+H, c[8][1]], [c[11][0], 1.3*h+H, c[11][1]], [c[11][0]+w_r, 1.3*h+H, c[11][1]], [c[8][0]+w_r, 1.3*h+H, c[8][1]]]
  );

  // Mirror rails (negate X and Z)
  // Rail pair 4: top short rail (mirror of rail 1)
  addRail(
    [[-c[0][0], H, -c[0][1]], [-c[3][0], H, -c[3][1]], [-c[3][0], H, -c[3][1] + w_r], [-c[0][0], H, -c[0][1] + w_r]],
    [[-c[0][0], 1.3*h+H, -c[0][1]], [-c[3][0], 1.3*h+H, -c[3][1]], [-c[3][0], 1.3*h+H, -c[3][1] + w_r], [-c[0][0], 1.3*h+H, -c[0][1] + w_r]]
  );
  // Rail pair 5: left side rail lower (mirror of rail 2)
  addRail(
    [[-c[4][0], H, -c[4][1]], [-c[7][0], H, -c[7][1]], [-c[7][0]-w_r, H, -c[7][1]], [-c[4][0]-w_r, H, -c[4][1]]],
    [[-c[4][0], 1.3*h+H, -c[4][1]], [-c[7][0], 1.3*h+H, -c[7][1]], [-c[7][0]-w_r, 1.3*h+H, -c[7][1]], [-c[4][0]-w_r, 1.3*h+H, -c[4][1]]]
  );
  // Rail pair 6: left side rail upper (mirror of rail 3)
  addRail(
    [[-c[8][0], H, -c[8][1]], [-c[11][0], H, -c[11][1]], [-c[11][0]-w_r, H, -c[11][1]], [-c[8][0]-w_r, H, -c[8][1]]],
    [[-c[8][0], 1.3*h+H, -c[8][1]], [-c[11][0], 1.3*h+H, -c[11][1]], [-c[11][0]-w_r, 1.3*h+H, -c[11][1]], [-c[8][0]-w_r, 1.3*h+H, -c[8][1]]]
  );

  // --- Pockets ---
  const pocketMat = new THREE.MeshPhongMaterial({ color: 0x000000 });
  for (let i = 0; i < 6; i++) {
    const R = (i === 2 || i === 5) ? R_s : R_c;
    const pocketGeom = new THREE.CylinderGeometry(R, R, 0.002, 24);
    const pocket = new THREE.Mesh(pocketGeom, pocketMat);
    // pocket_positions[i] is [x, y, z]
    pocket.position.set(pocket_positions[i][0], H + 0.001, pocket_positions[i][2]);
    group.add(pocket);
  }

  // --- Legs (simple boxes under the table) ---
  const legMat = new THREE.MeshPhongMaterial({ color: 0x4a3520 });
  const legW = 0.05, legD = 0.05;
  const legGeom = new THREE.BoxGeometry(legW, H, legD);
  const offX = W / 2 + w * 0.5;
  const offZ = L / 2 + w * 0.5;
  for (const [lx, lz] of [[-offX, -offZ], [offX, -offZ], [-offX, offZ], [offX, offZ]]) {
    const leg = new THREE.Mesh(legGeom, legMat);
    leg.position.set(lx, H / 2, lz);
    group.add(leg);
  }

  return group;
}
