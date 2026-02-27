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
  }

  addEvents(apiEvents, ballsAtRestTime) {
    this.ballsAtRestTime = ballsAtRestTime;
    const motionTypes = new Set([
      'BallSlidingEvent', 'BallRollingEvent',
      'BallRestEvent', 'BallSpinningEvent',
    ]);
    for (const evt of apiEvents) {
      if (!motionTypes.has(evt.type)) continue;
      const bi = evt.ball_index;
      if (bi == null) continue;
      if (!this._ballEvents[bi]) this._ballEvents[bi] = [];
      this._ballEvents[bi].push(evt);
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

  get isPlaying() {
    return this.playing;
  }
}
