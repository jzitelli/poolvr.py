import * as THREE from 'three';

const State = { IDLE: 0, AIMING: 1, ANIMATING: 2 };

export class AimingController {
  constructor({ camera, domElement, ballMeshes, ballRadius, tableH, onStrike }) {
    this.camera = camera;
    this.domElement = domElement;
    this.ballMeshes = ballMeshes;
    this.ballRadius = ballRadius;
    this.tableH = tableH;
    this.onStrike = onStrike;
    this.state = State.IDLE;

    this._raycaster = new THREE.Raycaster();
    this._mouse = new THREE.Vector2();
    this._aimStart = new THREE.Vector3();
    this._aimEnd = new THREE.Vector3();
    this._tablePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(tableH + ballRadius));

    // Aiming line visual
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(),
    ]);
    this._aimLine = new THREE.Line(
      lineGeom,
      new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 }),
    );
    this._aimLine.visible = false;
    this._aimLine.frustumCulled = false;

    // Power indicator (shown in HUD)
    this._power = 0;
    this._maxPower = 3.0;
    this._minPower = 0.3;

    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    domElement.addEventListener('mousedown', this._onMouseDown);
    domElement.addEventListener('mousemove', this._onMouseMove);
    domElement.addEventListener('mouseup', this._onMouseUp);
  }

  get aimLine() {
    return this._aimLine;
  }

  get power() {
    return this._power;
  }

  setAnimating() {
    this.state = State.ANIMATING;
    this._aimLine.visible = false;
  }

  setIdle() {
    this.state = State.IDLE;
    this._aimLine.visible = false;
  }

  _updateMouse(event) {
    const rect = this.domElement.getBoundingClientRect();
    this._mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  _getTableIntersection(event) {
    this._updateMouse(event);
    this._raycaster.setFromCamera(this._mouse, this.camera);
    const target = new THREE.Vector3();
    const hit = this._raycaster.ray.intersectPlane(this._tablePlane, target);
    return hit ? target : null;
  }

  _onMouseDown(event) {
    if (this.state !== State.IDLE || event.button !== 0) return;
    this._updateMouse(event);
    this._raycaster.setFromCamera(this._mouse, this.camera);

    // Check if we clicked the cue ball (ball 0)
    const cueBall = this.ballMeshes[0]?.mesh;
    if (!cueBall) return;
    const intersects = this._raycaster.intersectObject(cueBall);
    if (intersects.length > 0) {
      this.state = State.AIMING;
      this._aimStart.copy(cueBall.position);
      this._aimLine.visible = true;
      event.preventDefault();
      event.stopPropagation();
    }
  }

  _onMouseMove(event) {
    if (this.state !== State.AIMING) return;
    const tablePoint = this._getTableIntersection(event);
    if (!tablePoint) return;
    this._aimEnd.copy(tablePoint);

    // Direction from cue ball toward mouse
    const dir = new THREE.Vector3().subVectors(this._aimEnd, this._aimStart);
    dir.y = 0;
    const dist = dir.length();
    if (dist < 0.001) return;

    // Power from distance (clamped)
    this._power = Math.min(this._maxPower, Math.max(this._minPower, dist * 3));

    // Update aim line: from cue ball, extending in the shot direction
    const normDir = dir.clone().normalize();
    const lineEnd = this._aimStart.clone().add(normDir.clone().multiplyScalar(Math.min(dist, 1.0)));
    const positions = this._aimLine.geometry.attributes.position;
    positions.setXYZ(0, this._aimStart.x, this._aimStart.y, this._aimStart.z);
    positions.setXYZ(1, lineEnd.x, lineEnd.y, lineEnd.z);
    positions.needsUpdate = true;

    // Update power bar in HUD
    const powerBar = document.getElementById('power-fill');
    if (powerBar) {
      const pct = ((this._power - this._minPower) / (this._maxPower - this._minPower)) * 100;
      powerBar.style.width = pct + '%';
    }
  }

  _onMouseUp(event) {
    if (this.state !== State.AIMING) return;
    const tablePoint = this._getTableIntersection(event);
    if (!tablePoint) {
      this.state = State.IDLE;
      this._aimLine.visible = false;
      return;
    }

    const dir = new THREE.Vector3().subVectors(tablePoint, this._aimStart);
    dir.y = 0;
    if (dir.length() < 0.01) {
      this.state = State.IDLE;
      this._aimLine.visible = false;
      return;
    }
    dir.normalize();

    const speed = this._power;
    const cueBallPos = this._aimStart;

    // Strike parameters
    const cue_velocity = [dir.x * speed, 0, dir.z * speed];
    const contact_point = [
      cueBallPos.x - dir.x * this.ballRadius,
      cueBallPos.y,
      cueBallPos.z - dir.z * this.ballRadius,
    ];

    this.state = State.ANIMATING;
    this._aimLine.visible = false;

    if (this.onStrike) {
      this.onStrike({
        ball_index: 0,
        cue_velocity,
        contact_point,
        cue_mass: 0.54,
      });
    }
  }

  dispose() {
    this.domElement.removeEventListener('mousedown', this._onMouseDown);
    this.domElement.removeEventListener('mousemove', this._onMouseMove);
    this.domElement.removeEventListener('mouseup', this._onMouseUp);
  }
}
