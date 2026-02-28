import * as THREE from 'three';

// IDLE: orbit camera freely. Click cue ball → AIMING.
// AIMING: aim line visible. Mouse left/right rotates shot direction. Space down → CHARGING.
// CHARGING: mouse up/down moves cue stick back/forward. Cue contacts ball → strike.
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

    // Cue stick movement (charging mode)
    this._cuePullBack = 0;           // meters behind default rest position
    this._cueMoveSensitivity = 0.002; // mouse pixels to meters
    this._cueVelocityScale = 0.4;    // mouse pixels to m/s for strike
    this._cuePullBackMax = 0.4;      // max pull-back distance

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

    // Cue stick visual (simple tapered cylinder)
    const cueLength = 1.45;
    const cueTipRadius = 0.005;
    const cueButtRadius = 0.013;
    const cueGeom = new THREE.CylinderGeometry(cueButtRadius, cueTipRadius, cueLength, 12);
    // Shift so tip end is at local origin, body extends along +Y
    cueGeom.translate(0, cueLength / 2, 0);
    // Rotate so body extends along -Z (tip at origin)
    cueGeom.rotateX(-Math.PI / 2);
    // Offset tip away from group center by ball radius + small gap
    cueGeom.translate(0, 0, -(this.ballRadius + 0.02));
    this._cueStick = new THREE.Mesh(
      cueGeom,
      new THREE.MeshPhongMaterial({ color: 0xC19A6B }),
    );
    this._cueStick.visible = false;

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

  get cueStick() {
    return this._cueStick;
  }

  setAnimating() {
    this.state = State.ANIMATING;
    this._aimLine.visible = false;
    this._cueStick.visible = false;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;
  }

  setIdle() {
    this.state = State.IDLE;
    this._aimLine.visible = false;
    this._cueStick.visible = false;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;
  }

  _hidePowerBar() {
    if (this._powerBar) this._powerBar.style.display = 'none';
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
    this._updateCueStick();
  }

  _updateCueStick() {
    const pos = this._cueBallPos();
    if (!pos) return;
    const dir = this._aimDir();
    // Offset along -aimDir by _cuePullBack (positive = further from ball)
    this._cueStick.position.set(
      pos.x - dir.x * this._cuePullBack,
      pos.y,
      pos.z - dir.z * this._cuePullBack,
    );
    this._cueStick.rotation.y = this._aimAngle;
    this._cueStick.visible = true;
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
      // Mouse down (positive movementY) = push cue forward toward ball
      const prev = this._cuePullBack;
      this._cuePullBack -= event.movementY * this._cueMoveSensitivity;
      this._cuePullBack = Math.min(this._cuePullBack, this._cuePullBackMax);
      // Contact when tip gap (0.02m built into geometry) is closed
      const contactThreshold = -0.02;
      if (this._cuePullBack <= contactThreshold && prev > contactThreshold) {
        const speed = Math.max(event.movementY * this._cueVelocityScale, 0.2);
        this._fire(speed);
        return;
      }
      this._cuePullBack = Math.max(this._cuePullBack, contactThreshold);
      this._updateCueStick();
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
    }
  }

  _onKeyUp(event) {
    if (event.code !== 'Space') return;
    event.preventDefault();

    if (this.state === State.CHARGING) {
      // Released spacebar without contact → return to aiming
      this.state = State.AIMING;
      this._cuePullBack = 0;
      this._updateCueStick();
    }
  }

  _cancel() {
    this.state = State.IDLE;
    this._aimLine.visible = false;
    this._cueStick.visible = false;
    this._cuePullBack = 0;
    this._hidePowerBar();
    if (this.orbitControls) this.orbitControls.enabled = true;
  }

  _fire(speed) {
    const pos = this._cueBallPos();
    if (!pos) { this._cancel(); return; }

    const dir = this._aimDir();
    const cue_velocity = [dir.x * speed, 0, dir.z * speed];
    const contact_point = [
      pos.x - dir.x * this.ballRadius,
      pos.y,
      pos.z - dir.z * this.ballRadius,
    ];

    this.state = State.ANIMATING;
    this._aimLine.visible = false;
    this._cueStick.visible = false;
    this._cuePullBack = 0;
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
