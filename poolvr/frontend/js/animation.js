/**
 * Compute a naive (collision-free) cue ball trajectory client-side.
 * Replicates CueStrikeEvent -> BallSlidingEvent -> BallRollingEvent -> BallRestEvent.
 *
 * @param {number[]} ballPos - [x, y, z] ball position
 * @param {number[]} cueVelocity - [vx, vy, vz] cue velocity
 * @param {number[]} contactPoint - [x, y, z] contact point
 * @param {number} cueMass - cue mass in kg
 * @param {object} physics - { ball_mass, ball_radius, mu_s, mu_r, mu_sp, g }
 * @param {number} t0 - start time (default 0)
 * @returns {{ events: object[], ballsAtRestTime: number }}
 */
export function computeNaiveTrajectory(ballPos, cueVelocity, contactPoint, cueMass, physics, t0 = 0) {
  const { ball_mass: m, ball_radius: R, mu_s, mu_r, mu_sp, g } = physics;
  const I = (2 / 5) * m * R * R;

  // --- CueStrikeEvent physics ---
  const Vx = cueVelocity[0], Vz = cueVelocity[2];
  const V_mag = Math.sqrt(Vx * Vx + Vz * Vz);
  if (V_mag < 1e-10) return { events: [], ballsAtRestTime: t0 };

  const Qx = contactPoint[0] - ballPos[0];
  const Qy = contactPoint[1] - ballPos[1];
  const Qz = contactPoint[2] - ballPos[2];

  // _j = -V/|V| (XZ plane), _k = [0,1,0], _i = cross(_j, _k)
  const jx = -Vx / V_mag, jz = -Vz / V_mag;
  const ix = -jz, iz = jx; // cross([jx,0,jz],[0,1,0]) = [-jz, 0, jx]

  const a_p = Qx * ix + Qz * iz; // dot(Q, _i)
  const b_p = Qy;
  const c_p = Math.sqrt(Math.max(0, R * R - a_p * a_p - b_p * b_p));

  const sinB = b_p / R;
  const cosB = Math.sqrt(Math.max(0, R * R - b_p * b_p)) / R;

  const F_mag = 2 * m * V_mag / (
    1 + m / cueMass + 5 / (2 * R * R) * (
      a_p * a_p + b_p * b_p * cosB * cosB +
      c_p * c_p * sinB * sinB - 2 * b_p * c_p * cosB * sinB
    )
  );

  // Initial ball velocity: v_0 = -F/m * _j
  const v0x = -F_mag / m * jx;
  const v0z = -F_mag / m * jz;

  // Initial angular velocity
  const t1 = (-c_p * F_mag * sinB + b_p * F_mag * cosB) / I;
  const t2 = (a_p * F_mag * sinB) / I;
  const t3 = (-a_p * F_mag * cosB) / I;
  const w0x = t1 * ix + t2 * jx;
  const w0y = t3; // _i[1]=0, _j[1]=0, _k[1]=1
  const w0z = t1 * iz + t2 * jz;

  // --- BallSlidingEvent ---
  const u0x = v0x + R * w0z;
  const u0z = v0z - R * w0x;
  const u0_mag = Math.sqrt(u0x * u0x + u0z * u0z);

  const events = [];
  let r1x, r1y, r1z, v1x, v1z, w1y, t_roll_start;

  if (u0_mag >= 1e-10) {
    const T_s = 2 * u0_mag / (7 * mu_s * g);
    const sAccel = -0.5 * mu_s * g / u0_mag;
    const slide_a = [
      [ballPos[0], ballPos[1], ballPos[2]],
      [v0x, 0, v0z],
      [sAccel * u0x, 0, sAccel * u0z],
    ];
    const sgnW0y = w0y > 0 ? 1 : (w0y < 0 ? -1 : 0);
    const bCoeff = 5 * mu_s * g / (2 * R * u0_mag);
    const slide_b = [
      [w0x, w0y, w0z],
      [bCoeff * u0z, -sgnW0y * 5 * mu_sp * g / (2 * R), -bCoeff * u0x],
    ];

    events.push({
      type: 'BallSlidingEvent', t: t0, T: T_s, ball_index: 0,
      a: slide_a, b: slide_b,
    });

    // End-of-sliding state
    r1x = slide_a[0][0] + slide_a[1][0] * T_s + slide_a[2][0] * T_s * T_s;
    r1y = slide_a[0][1];
    r1z = slide_a[0][2] + slide_a[1][2] * T_s + slide_a[2][2] * T_s * T_s;
    v1x = slide_a[1][0] + 2 * slide_a[2][0] * T_s;
    v1z = slide_a[1][2] + 2 * slide_a[2][2] * T_s;
    w1y = slide_b[0][1] + T_s * slide_b[1][1];
    if (Math.sign(w1y) !== Math.sign(slide_b[0][1])) w1y = 0;
    t_roll_start = t0 + T_s;
  } else {
    // No sliding phase — start rolling from initial state
    r1x = ballPos[0]; r1y = ballPos[1]; r1z = ballPos[2];
    v1x = v0x; v1z = v0z; w1y = w0y;
    t_roll_start = t0;
  }

  // --- BallRollingEvent ---
  const v1_mag = Math.sqrt(v1x * v1x + v1z * v1z);
  if (v1_mag < 1e-10) {
    events.push({ type: 'BallRestEvent', t: t_roll_start, T: null, ball_index: 0,
      r_0: [r1x, r1y, r1z] });
    return { events, ballsAtRestTime: t_roll_start };
  }

  const T_r = v1_mag / (mu_r * g);
  const rAccel = -0.5 * mu_r * g / v1_mag;
  const wr0x = v1z / R, wr0z = -v1x / R;
  const roll_a = [
    [r1x, r1y, r1z],
    [v1x, 0, v1z],
    [rAccel * v1x, 0, rAccel * v1z],
  ];
  const sgnWr0y = w1y > 0 ? 1 : (w1y < 0 ? -1 : 0);
  const roll_b = [
    [wr0x, w1y, wr0z],
    [
      -wr0x / T_r,
      -sgnWr0y * 5 * mu_sp * g / (2 * R),
      -wr0z / T_r,
    ],
  ];

  events.push({
    type: 'BallRollingEvent', t: t_roll_start, T: T_r, ball_index: 0,
    a: roll_a, b: roll_b,
  });

  // End-of-rolling position
  const rEx = roll_a[0][0] + roll_a[1][0] * T_r + roll_a[2][0] * T_r * T_r;
  const rEz = roll_a[0][2] + roll_a[1][2] * T_r + roll_a[2][2] * T_r * T_r;
  const t_end = t_roll_start + T_r;

  // Check for residual spin
  let wrEy = roll_b[0][1] + T_r * roll_b[1][1];
  if (Math.sign(wrEy) !== Math.sign(roll_b[0][1])) wrEy = 0;

  if (Math.abs(wrEy) > 1e-6) {
    const T_sp = Math.abs(wrEy) * 2 * R / (5 * mu_sp * g);
    events.push({
      type: 'BallSpinningEvent', t: t_end, T: T_sp, ball_index: 0,
      r_0: [rEx, r1y, rEz], omega_0_y: wrEy,
    });
    events.push({
      type: 'BallRestEvent', t: t_end + T_sp, T: null, ball_index: 0,
      r_0: [rEx, r1y, rEz],
    });
    return { events, ballsAtRestTime: t_end + T_sp };
  }

  events.push({
    type: 'BallRestEvent', t: t_end, T: null, ball_index: 0,
    r_0: [rEx, r1y, rEz],
  });
  return { events, ballsAtRestTime: t_end };
}


/**
 * AnimationEngine: interpolates ball positions from physics events.
 *
 * Events have motion coefficients: position = a[0] + a[1]*tau + a[2]*tau^2
 * where tau = t - event.t
 */
export class AnimationEngine {
  constructor() {
    this.simTime = 0;
    this.ballsAtRestTime = 0;
    this.playing = false;
    this.speed = 1.0;
    // per-ball event lists, indexed by ball index
    this._ballEvents = {};
    this._numBalls = 0;
    // per-ball pocket times: { ballIndex: time }
    this._pocketTimes = {};
  }

  setInitialPositions(positions) {
    this._numBalls = positions.length;
    this._ballEvents = {};
    for (let i = 0; i < positions.length; i++) {
      this._ballEvents[i] = [{
        type: 'BallRestEvent',
        t: 0,
        T: null,
        ball_index: i,
        r_0: positions[i],
      }];
    }
    this.simTime = 0;
    this.ballsAtRestTime = 0;
    this.playing = false;
    this._pocketTimes = {};
  }

  addEvents(apiEvents, ballsAtRestTime) {
    this.ballsAtRestTime = ballsAtRestTime;
    const motionTypes = new Set([
      'BallSlidingEvent', 'BallRollingEvent',
      'BallRestEvent', 'BallSpinningEvent',
      'BallPocketedEvent',
    ]);
    for (const evt of apiEvents) {
      if (!motionTypes.has(evt.type)) continue;
      const bi = evt.ball_index;
      if (bi == null) continue;
      if (!this._ballEvents[bi]) this._ballEvents[bi] = [];
      this._ballEvents[bi].push(evt);
      if (evt.type === 'BallPocketedEvent') {
        this._pocketTimes[bi] = evt.t;
      }
    }
    // Sort each ball's events by time
    for (const bi in this._ballEvents) {
      this._ballEvents[bi].sort((a, b) => a.t - b.t);
    }
    this.playing = true;
  }

  /**
   * Find the active event for a ball at time t using binary search.
   */
  _findEvent(ballIndex, t) {
    const events = this._ballEvents[ballIndex];
    if (!events || events.length === 0) return null;
    let lo = 0, hi = events.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (events[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    return events[lo];
  }

  /**
   * Evaluate ball position at time t, writing into outVec3.
   */
  evalPosition(ballIndex, t, outVec3) {
    const evt = this._findEvent(ballIndex, t);
    if (!evt) return false;
    const type = evt.type;
    if (type === 'BallRestEvent' || type === 'BallSpinningEvent') {
      outVec3.set(evt.r_0[0], evt.r_0[1], evt.r_0[2]);
    } else if (evt.a) {
      const tau = t - evt.t;
      const a = evt.a;
      outVec3.set(
        a[0][0] + a[1][0] * tau + a[2][0] * tau * tau,
        a[0][1] + a[1][1] * tau + a[2][1] * tau * tau,
        a[0][2] + a[1][2] * tau + a[2][2] * tau * tau,
      );
    } else if (evt.r_0) {
      outVec3.set(evt.r_0[0], evt.r_0[1], evt.r_0[2]);
    } else {
      return false;
    }
    return true;
  }

  /**
   * Evaluate ball angular velocity at time t, writing into outVec3.
   * For sliding/rolling events: omega(tau) = b[0] + tau * b[1],
   * with omega_y clamped to zero when it changes sign.
   */
  evalAngularVelocity(ballIndex, t, outVec3) {
    const evt = this._findEvent(ballIndex, t);
    if (!evt) { outVec3.set(0, 0, 0); return; }
    const type = evt.type;
    if (type === 'BallRestEvent') {
      outVec3.set(0, 0, 0);
    } else if (type === 'BallSpinningEvent') {
      const omega0 = evt.omega_0_y || 0;
      if (evt.T != null) {
        const tau = t - evt.t;
        const frac = Math.max(0, 1 - tau / evt.T);
        outVec3.set(0, omega0 * frac, 0);
      } else {
        outVec3.set(0, omega0, 0);
      }
    } else if (evt.b) {
      // BallSlidingEvent / BallRollingEvent: omega(tau) = b[0] + tau * b[1]
      const tau = t - evt.t;
      const b = evt.b;
      let ox = b[0][0] + tau * b[1][0];
      let oy = b[0][1] + tau * b[1][1];
      let oz = b[0][2] + tau * b[1][2];
      // Clamp omega_y to zero when it changes sign
      if (Math.sign(oy) !== Math.sign(b[0][1])) oy = 0;
      outVec3.set(ox, oy, oz);
    } else {
      outVec3.set(0, 0, 0);
    }
  }

  update(deltaSec) {
    if (!this.playing) return false;
    this.simTime += deltaSec * this.speed;
    if (this.ballsAtRestTime != null && this.simTime >= this.ballsAtRestTime) {
      this.simTime = this.ballsAtRestTime;
      this.playing = false;
      return true; // animation just completed
    }
    return false;
  }

  /**
   * Return the time at which a ball was pocketed, or null if not pocketed.
   */
  getPocketTime(ballIndex) {
    return this._pocketTimes[ballIndex] ?? null;
  }

  get isPlaying() {
    return this.playing;
  }

  /**
   * Save a snapshot of the current ball events state (for restoring after
   * naive trajectory is replaced by backend events).
   */
  snapshot() {
    const copy = {};
    for (const bi in this._ballEvents) {
      copy[bi] = [...this._ballEvents[bi]];
    }
    return {
      ballEvents: copy,
      ballsAtRestTime: this.ballsAtRestTime,
      pocketTimes: { ...this._pocketTimes },
    };
  }

  /**
   * Restore ball events from a snapshot, keeping the current simTime
   * and playing state.
   */
  restore(snap) {
    this._ballEvents = {};
    for (const bi in snap.ballEvents) {
      this._ballEvents[bi] = [...snap.ballEvents[bi]];
    }
    this.ballsAtRestTime = snap.ballsAtRestTime;
    this._pocketTimes = { ...snap.pocketTimes };
  }
}
