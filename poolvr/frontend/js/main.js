import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createSimulation, getTableGeometry, strike, getSimulation } from './api.js';
import { createTableMesh } from './table.js';
import { createBallMeshes } from './balls.js';
import { AnimationEngine } from './animation.js';
import { AimingController } from './controls.js';

const hud = document.getElementById('hud');

function updateHUD(text) {
  if (hud) hud.textContent = text;
}

async function main() {
  updateHUD('Initializing...');

  // --- Three.js setup ---
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  document.getElementById('app').appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 50);

  // --- Lighting ---
  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambient);
  const pointLight = new THREE.PointLight(0xffffff, 1.0, 20);
  scene.add(pointLight);

  // --- Create simulation ---
  updateHUD('Creating simulation...');
  const simData = await createSimulation({});
  const simId = simData.sim_id;
  const initialPositions = simData.ball_positions;
  const numBalls = initialPositions.length;

  // --- Fetch table geometry ---
  const tableGeom = await getTableGeometry(simId);
  const H = tableGeom.H;
  const ballRadius = tableGeom.ball_radius;

  // Position camera and light above table center
  camera.position.set(0, H + 1.8, 1.5);
  camera.lookAt(0, H, 0);
  pointLight.position.set(0, H + 1.5, 0);

  // --- OrbitControls ---
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, H, 0);
  controls.maxPolarAngle = Math.PI / 2 - 0.05;
  controls.minDistance = 0.5;
  controls.maxDistance = 5;
  controls.update();

  // --- Table mesh ---
  const tableMesh = createTableMesh(tableGeom);
  scene.add(tableMesh);

  // --- Floor under table ---
  const floorGeom = new THREE.PlaneGeometry(6, 6);
  floorGeom.rotateX(-Math.PI / 2);
  const floorMat = new THREE.MeshPhongMaterial({ color: 0x2a2a3e });
  const floor = new THREE.Mesh(floorGeom, floorMat);
  floor.receiveShadow = true;
  scene.add(floor);

  // --- Ball meshes ---
  const ballObjs = createBallMeshes(numBalls, ballRadius, H);
  for (let i = 0; i < numBalls; i++) {
    scene.add(ballObjs[i].mesh);
    scene.add(ballObjs[i].shadow);
    const pos = initialPositions[i];
    ballObjs[i].mesh.position.set(pos[0], pos[1], pos[2]);
    ballObjs[i].shadow.position.set(pos[0], H + 0.001, pos[2]);
  }

  // --- Animation engine ---
  const animEngine = new AnimationEngine();
  animEngine.setInitialPositions(initialPositions);

  // --- Aiming controller ---
  const aimController = new AimingController({
    camera,
    domElement: renderer.domElement,
    ballMeshes: ballObjs,
    ballRadius,
    tableH: H,
    orbitControls: controls,
    onStrike: async (params) => {
      updateHUD('Striking...');
      try {
        const result = await strike(simId, params);
        animEngine.addEvents(result.events, result.balls_at_rest_time);
        const eventCount = result.events.length;
        const restTime = result.balls_at_rest_time;
        updateHUD(`Simulating ${eventCount} events (${restTime?.toFixed(2) ?? '?'}s)`);
      } catch (err) {
        console.error('Strike failed:', err);
        updateHUD('Strike failed: ' + err.message);
        aimController.setIdle();
      }
    },
  });
  scene.add(aimController.aimLine);

  // --- Resize handler ---
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // --- Render loop ---
  const clock = new THREE.Clock();
  const tmpVec = new THREE.Vector3();
  const tmpOmega = new THREE.Vector3();

  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    controls.update();

    const prevSimTime = animEngine.simTime;
    const justCompleted = animEngine.update(delta);
    const simDt = animEngine.simTime - prevSimTime;

    // Update ball positions and rotations
    for (let i = 0; i < numBalls; i++) {
      if (animEngine.evalPosition(i, animEngine.simTime, tmpVec)) {
        ballObjs[i].mesh.position.copy(tmpVec);
        ballObjs[i].shadow.position.set(tmpVec.x, H + 0.001, tmpVec.z);
      }
      // Quaternion integration (small time-step approximation from game.py)
      if (simDt > 0) {
        animEngine.evalAngularVelocity(i, animEngine.simTime, tmpOmega);
        const q = ballObjs[i].mesh.quaternion;
        const qw = q.w;
        const qx = q.x, qy = q.y, qz = q.z;
        const ox = tmpOmega.x, oy = tmpOmega.y, oz = tmpOmega.z;
        // q.w -= 0.5 * dt * dot(omega, q.xyz)
        q.w = qw - 0.5 * simDt * (ox * qx + oy * qy + oz * qz);
        // q.xyz += 0.5 * dt * (qw * omega + cross(omega, q.xyz))
        q.x = qx + 0.5 * simDt * (qw * ox + (oy * qz - oz * qy));
        q.y = qy + 0.5 * simDt * (qw * oy + (oz * qx - ox * qz));
        q.z = qz + 0.5 * simDt * (qw * oz + (ox * qy - oy * qx));
        q.normalize();
      }
    }

    if (animEngine.isPlaying) {
      updateHUD(`t = ${animEngine.simTime.toFixed(3)}s`);
    } else if (aimController.isCharging) {
      updateHUD('Hold SPACE + move mouse up/down to set power. Release SPACE to strike.');
    } else if (aimController.isAiming) {
      updateHUD('Move mouse left/right to aim. Hold SPACE to set power. ESC to cancel.');
    }

    if (justCompleted) {
      onAnimationComplete();
    }

    renderer.render(scene, camera);
  }

  async function onAnimationComplete() {
    updateHUD('Checking pocketed balls...');
    try {
      const simState = await getSimulation(simId);
      const onTable = new Set(simState.balls_on_table);
      for (let i = 0; i < numBalls; i++) {
        const visible = onTable.has(i);
        ballObjs[i].mesh.visible = visible;
        ballObjs[i].shadow.visible = visible;
      }
      updateHUD(`${onTable.size} balls on table. Click cue ball to aim.`);
    } catch (err) {
      console.error('Failed to check pocketed balls:', err);
      updateHUD('Click cue ball to aim.');
    }
    aimController.setIdle();
  }

  updateHUD('Click cue ball to aim.');
  animate();
}

main().catch((err) => {
  console.error('Initialization error:', err);
  updateHUD('Error: ' + err.message);
});
