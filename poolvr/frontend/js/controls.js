import * as THREE from 'three';

// IDLE: orbit camera freely. Click cue ball → AIMING.
// AIMING: aim line visible. Mouse left/right rotates shot direction. Space down → CHARGING.
// CHARGING: power bar visible. Mouse up/down sets power. Space up → strike.
// ANIMATING: waiting for physics to finish.
const State = { IDLE: 0, AIMING: 1, CHARGING: 2, ANIMATING: 3 };

export class AimingController {
  constructor({ camera, domElement, ballMeshes, ballRadius, tableH, orbitControls, onStrike }) {
    this.camera = camera;
    this.domElement = domElement;
    this.ballMeshes = ballMeshes;
    this.ballRadius = ballRadius;
    this.tableH = tableH;
    this.orbitControls = orbitControls;
    this.onStrike = onStrike;
    this.state = State.IDLE;

    this._raycaster = new THREE.Raycaster();
    this._mouse = new THREE.Vector2();

    // Aim angle (radians on XZ plane, 0 = +X direction)
    this._aimAngle = 0;
    this._aimSensitivity = 0.005;

    // Power
    this._power = 1.5;
    this._maxPower = 4.0;
    this._minPower = 0.2;
    this._powerSensitivity = 0.008;

    // Aim line visual
    this._aimLineLength = 0.6;
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(),
    ]);
    this._aimLine = new THREE.Line(
      lineGeom,
      new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 }),
    );
    this._aimLine.visible = false;
    this._aimLine.frustumCulled = false;

    // Power bar DOM
    this._powerBar = document.getElementById('power-bar');
    this._powerFill = document.getElementById('power-fill');
    if (this._powerBar) this._powerBar.style.display = 'none';

    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    domElement.addEventListener('mousedown', this._onMouseDown);
    domElement.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
  }

  get aimLine() {
    return this._aimLine;
  }

  setAnimating() {
    this.state = State.ANIMATING;
    this._aimLine.visible = false;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;
  }

  setIdle() {
    this.state = State.IDLE;
    this._aimLine.visible = false;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;
  }

  _hidePowerBar() {
    if (this._powerBar) this._powerBar.style.display = 'none';
  }

  _showPowerBar() {
    if (this._powerBar) this._powerBar.style.display = '';
    this._updatePowerBar();
  }

  _updatePowerBar() {
    if (this._powerFill) {
      const pct = ((this._power - this._minPower) / (this._maxPower - this._minPower)) * 100;
      this._powerFill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    }
  }

  _cueBallPos() {
    const cueBall = this.ballMeshes[0]?.mesh;
    return cueBall ? cueBall.position : null;
  }

  _aimDir() {
    return new THREE.Vector3(Math.sin(this._aimAngle), 0, Math.cos(this._aimAngle));
  }

  _updateAimLine() {
    const pos = this._cueBallPos();
    if (!pos) return;
    const dir = this._aimDir();
    const start = pos;
    const end = new THREE.Vector3().copy(pos).addScaledVector(dir, this._aimLineLength);
    const positions = this._aimLine.geometry.attributes.position;
    positions.setXYZ(0, start.x, start.y, start.z);
    positions.setXYZ(1, end.x, end.y, end.z);
    positions.needsUpdate = true;
    this._aimLine.visible = true;
  }

  _initAimAngle() {
    // Initialize aim angle to point from cue ball toward camera projected on XZ
    const pos = this._cueBallPos();
    if (!pos) return;
    const camDir = new THREE.Vector3().subVectors(this.camera.position, pos);
    camDir.y = 0;
    if (camDir.length() > 0.001) {
      camDir.normalize();
      this._aimAngle = Math.atan2(camDir.x, camDir.z);
    }
  }

  _onMouseDown(event) {
    if (event.button !== 0) return;

    if (this.state === State.AIMING || this.state === State.CHARGING) {
      // Click while aiming cancels
      this._cancel();
      return;
    }

    if (this.state !== State.IDLE) return;

    // Raycast to check cue ball click
    const rect = this.domElement.getBoundingClientRect();
    this._mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._mouse, this.camera);

    const cueBall = this.ballMeshes[0]?.mesh;
    if (!cueBall) return;
    const intersects = this._raycaster.intersectObject(cueBall);
    if (intersects.length > 0) {
      this.state = State.AIMING;
      if (this.orbitControls) this.orbitControls.enabled = false;
      this._initAimAngle();
      this._updateAimLine();
    }
  }

  _onMouseMove(event) {
    if (this.state === State.AIMING) {
      // Left/right mouse movement rotates aim
      this._aimAngle -= event.movementX * this._aimSensitivity;
      this._updateAimLine();
    } else if (this.state === State.CHARGING) {
      // Up/down mouse movement adjusts power (mouse up = more power)
      this._power -= event.movementY * this._powerSensitivity;
      this._power = Math.max(this._minPower, Math.min(this._maxPower, this._power));
      this._updatePowerBar();
    }
  }

  _onKeyDown(event) {
    if (event.code === 'Escape') {
      if (this.state === State.AIMING || this.state === State.CHARGING) {
        this._cancel();
      }
      return;
    }

    if (event.code !== 'Space') return;
    event.preventDefault();

    if (this.state === State.AIMING) {
      this.state = State.CHARGING;
      this._showPowerBar();
    }
  }

  _onKeyUp(event) {
    if (event.code !== 'Space') return;
    event.preventDefault();

    if (this.state === State.CHARGING) {
      this._fire();
    }
  }

  _cancel() {
    this.state = State.IDLE;
    this._aimLine.visible = false;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;
  }

  _fire() {
    const pos = this._cueBallPos();
    if (!pos) { this._cancel(); return; }

    const dir = this._aimDir();
    const speed = this._power;

    const cue_velocity = [dir.x * speed, 0, dir.z * speed];
    const contact_point = [
      pos.x - dir.x * this.ballRadius,
      pos.y,
      pos.z - dir.z * this.ballRadius,
    ];

    this.state = State.ANIMATING;
    this._aimLine.visible = false;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;

    if (this.onStrike) {
      this.onStrike({
        ball_index: 0,
        cue_velocity,
        contact_point,
        cue_mass: 0.54,
      });
    }
  }

  get isAiming() {
    return this.state === State.AIMING;
  }

  get isCharging() {
    return this.state === State.CHARGING;
  }

  dispose() {
    this.domElement.removeEventListener('mousedown', this._onMouseDown);
    this.domElement.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
  }
}
