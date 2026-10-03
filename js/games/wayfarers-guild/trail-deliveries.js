(function (root, factory) {
  'use strict';
  const api = factory(typeof module === 'object' && module.exports ? require('./numbers.js') : root.WayfarersNumbers);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersTrailDeliveries = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N) {
  'use strict';
  const WORK = 30;
  const HOLD = 1.2;
  const EPS = 1e-8;
  const KEYS = ['version', 'phase', 'work', 'hold', 'cargo', 'deliveries', 'lifetimeCoins', 'lastReward', 'sequence', 'lastKind', 'landmarkRunId', 'landmarkIndex'];
  function established(state) {
    const x = state.expedition;
    return !!x && (x.version === 3 ? x.cleared >= 0 : !!x.areas?.greenway?.established);
  }
  function initial(state, historical) {
    const x = state.expedition;
    const cleared = x ? (x.version === 3 ? x.cleared : x.index - (x.completed ? 0 : 1)) : -1;
    return { version: 1, phase: 'travel', work: 0, hold: 0, cargo: N.zero(), deliveries: 0,
      lifetimeCoins: N.zero(), lastReward: N.zero(), sequence: 0, lastKind: 'none',
      landmarkRunId: state.run.id, landmarkIndex: historical ? Math.max(-1, cleared) : -1 };
  }
  function validate(value, state) {
    if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== KEYS.length || !KEYS.every(key => Object.hasOwn(value, key))) return false;
    if (value.version !== 1 || !['travel', 'arrived'].includes(value.phase) || !['none', 'delivery', 'landmark'].includes(value.lastKind)) return false;
    if (!['cargo', 'lifetimeCoins', 'lastReward'].every(key => N.valid(value[key]))) return false;
    if (!['deliveries', 'sequence', 'landmarkRunId'].every(key => Number.isSafeInteger(value[key]) && value[key] >= 0)) return false;
    if (value.sequence < value.deliveries || value.landmarkRunId > state.run.id || !Number.isSafeInteger(value.landmarkIndex) || value.landmarkIndex < -1) return false;
    if (!Number.isFinite(value.work) || !Number.isFinite(value.hold) || value.work < 0 || value.work > WORK || value.hold < 0 || value.hold > HOLD) return false;
    if (value.phase === 'travel' ? value.work >= WORK || value.hold !== 0 : value.work !== WORK || value.hold <= 0 || value.cargo.m !== 0) return false;
    return N.cmp(value.lastReward, value.lifetimeCoins) <= 0;
  }
  function speed(rates) { return Math.max(1, Math.min(8, Math.sqrt(Math.max(0, Number(rates.travel) || 0)))); }
  function active(state, rates) { return established(state) && Number(rates.travel) > 0; }
  function nextEvent(state, rates) {
    if (!active(state, rates)) return Infinity;
    const x = state.trailDeliveries;
    return x.phase === 'arrived' ? x.hold : (WORK - x.work) / speed(rates);
  }
  function credit(state, amount, kind, count = 1) {
    const x = state.trailDeliveries;
    const total = N.mul(amount, count);
    state.resources.coins = N.add(state.resources.coins, total);
    state.lifetime.coins = N.add(state.lifetime.coins, total);
    x.lifetimeCoins = N.add(x.lifetimeCoins, total);
    x.lastReward = N.from(amount);
    x.lastKind = kind;
    x.sequence += count;
    return total;
  }
  function tick(state, seconds, rates) {
    if (!Number.isFinite(seconds) || seconds <= 0 || !active(state, rates)) return;
    const x = state.trailDeliveries;
    let remaining = seconds;
    // Preserve fractional intervals from other simulation events. Repeatedly
    // discarding a tiny tail would make foreground trips lag offline trips.
    while (remaining > 0) {
      // Core splits at every rate/spending boundary. Inside this constant-rate
      // interval, whole empty-start trips can be settled in constant time.
      // Preserve the individual last award and exact final partial-trip phase.
      if (x.phase === 'travel' && x.work === 0 && x.cargo.m === 0) {
        const duration = WORK / speed(rates), cycle = duration + HOLD;
        const cycles = Math.floor(remaining / cycle);
        if (cycles >= 2) {
          credit(state, N.max(6, N.mul(rates.coins, duration * 0.4)), 'delivery', cycles);
          x.deliveries += cycles;
          remaining = Math.max(0, remaining - cycles * cycle);
          continue;
        }
      }
      const boundary = nextEvent(state, rates);
      const dt = Math.min(remaining, boundary);
      remaining -= dt;
      if (x.phase === 'arrived') {
        x.hold = Math.max(0, x.hold - dt);
        if (x.hold < EPS) { x.phase = 'travel'; x.work = 0; x.hold = 0; }
      } else {
        x.work = Math.min(WORK, x.work + speed(rates) * dt);
        x.cargo = N.add(x.cargo, N.mul(rates.coins, dt * 0.4));
        if (WORK - x.work < EPS) {
          credit(state, N.max(6, x.cargo), 'delivery');
          x.cargo = N.zero(); x.deliveries += 1; x.work = WORK; x.phase = 'arrived'; x.hold = HOLD;
        }
      }
    }
  }
  function landmark(state, index, areaId, rates) {
    const x = state.trailDeliveries;
    if (areaId !== 'greenway' || !Number.isSafeInteger(index) || index < 0 || x.landmarkRunId === state.run.id && index <= x.landmarkIndex) return N.zero();
    x.landmarkRunId = state.run.id;
    x.landmarkIndex = index;
    return credit(state, N.max(24, N.mul(rates.coins, 30)), 'landmark');
  }
  function resetTrip(state) {
    const x = state.trailDeliveries;
    x.work = 0; x.hold = 0; x.cargo = N.zero(); x.phase = 'travel';
  }
  function view(state, rates) {
    const x = state.trailDeliveries;
    const enabled = active(state, rates);
    const eta = enabled ? nextEvent(state, rates) : null;
    return { active: enabled, phase: x.phase, progress: x.work / WORK, eta, deliveries: x.deliveries,
      reward: N.max(6, N.add(x.cargo, N.mul(rates.coins, (WORK - x.work) / speed(rates) * 0.4))),
      lastReward: N.from(x.lastReward), sequence: x.sequence, lastKind: x.lastKind, destination: 'Trail outpost' };
  }
  return { initial, validate, nextEvent, tick, landmark, resetTrip, view };
});
