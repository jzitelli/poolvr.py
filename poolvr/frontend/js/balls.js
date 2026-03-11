import * as THREE from 'three';

const BALL_COLORS = [
  0xddddde, // 0: cue ball (white)
  0xeeee00, // 1: yellow
  0x0000ee, // 2: blue
  0xee0000, // 3: red
  0xee00ee, // 4: purple
  0xee7700, // 5: orange
  0x00ee00, // 6: green
  0xbb2244, // 7: maroon
  0x111111, // 8: black
  0xeeee00, // 9: yellow stripe
  0x0000ee, // 10: blue stripe
  0xee0000, // 11: red stripe
  0xee00ee, // 12: purple stripe
  0xee7700, // 13: orange stripe
  0x00ee00, // 14: green stripe
  0xbb2244, // 15: maroon stripe
];

function createStripeTexture(color, size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  // Fill with white
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  // Draw colored band on top and bottom thirds
  const c = '#' + color.toString(16).padStart(6, '0');
  ctx.fillStyle = c;
  ctx.fillRect(0, 0, size, Math.floor(size / 3));
  ctx.fillRect(0, Math.floor(2 * size / 3), size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createBallMeshes(numBalls, ballRadius, H) {
  const sphereGeom = new THREE.SphereGeometry(ballRadius, 24, 16);
  const shadowGeom = new THREE.CircleGeometry(ballRadius, 16);
  shadowGeom.rotateX(-Math.PI / 2);
  const shadowMat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
  });

  const balls = [];
  for (let i = 0; i < numBalls; i++) {
    const color = BALL_COLORS[i] !== undefined ? BALL_COLORS[i] : 0xcccccc;
    let mat;
    if (i >= 9 && i <= 15) {
      // Striped ball
      const tex = createStripeTexture(color);
      mat = new THREE.MeshPhongMaterial({ map: tex, flatShading: true, transparent: true });
    } else {
      mat = new THREE.MeshPhongMaterial({ color, flatShading: true, transparent: true });
    }
    const mesh = new THREE.Mesh(sphereGeom, mat);
    mesh.castShadow = true;

    const shadow = new THREE.Mesh(shadowGeom, shadowMat.clone());
    shadow.position.y = H + 0.001;

    balls.push({ mesh, shadow });
  }
  return balls;
}
