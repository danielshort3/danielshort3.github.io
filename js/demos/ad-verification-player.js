/* One clock and one commit boundary for BOTH views. No independent animation timers. */
(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./ad-verification-core.js'));
  else root.AdVerificationPlayer = factory(root.AdVerificationCore);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (core) {
  'use strict';
  const EVENT_MS = 1900;
  const GAP_MS = 320;
  function createPlayer(options = {}) {
    const raf = options.requestFrame || requestAnimationFrame;
    const cancel = options.cancelFrame || cancelAnimationFrame;
    const notify = options.onChange || (() => {});
    let ledger = null;
    let generation = 0;
    let frameId = null;
    let lastTime = null;
    let gap = 0;
    let queue = [];
    let active = null;
    let sequence = 0;
    let group = 0;
    let groups = [];
    let running = false;
    let ready = false;
    let speed = 1;
    let continuous = true;
    let scenario = 'mixed';
    let error = '';
    let limit = false;
    let destroyed = false;
    const snapshot = (includeProof = true) => ({ ready, running, speed, continuous, scenario, group, groups: core.clone(groups),
      active: active ? { draft: core.clone(active.draft), id: active.id, height: active.height, progress: active.progress,
        phase: active.progress < .4 ? 'recording' : 'verifying' } : null,
      proof: ledger && includeProof ? ledger.snapshot() : null, hasWork: Boolean(active || queue.length), sequence, error, limit });
    function emit(kind, extra = {}) { notify({ kind, ...extra }, snapshot(kind !== 'frame')); }
    function stopClock() {
      running = false;
      if (frameId !== null) cancel(frameId);
      frameId = null;
      lastTime = null;
    }
    function fail(reason) { stopClock(); error = reason.message || String(reason); emit('error'); }
    function addGroup() {
      const plan = core.createPlan(scenario, group + 1, group === 0);
      if (ledger.snapshot().blocks.length + plan.events.length > core.MAX_BLOCKS) {
        limit = true;
        stopClock();
        emit('limit');
        return false;
      }
      group += 1;
      groups.push({ group, scenario, travelers: plan.travelers });
      queue = plan.events.slice();
      emit('group');
      return true;
    }
    function beginEvent() {
      const draft = queue.shift();
      const height = ledger.snapshot().blocks.length + 1;
      const target = { draft, height, id: 'event-' + height, progress: 0, elapsed: 0, ticket: null };
      const epoch = generation;
      const owner = ledger;
      active = target;
      emit('begin');
      owner.prepare(draft).then((ticket) => {
        if (destroyed || epoch !== generation || active !== target || owner !== ledger) return;
        target.ticket = ticket;
      }).catch((reason) => {
        if (!destroyed && epoch === generation && active === target && owner === ledger) fail(reason);
      });
    }
    function tick(time) {
      frameId = null;
      if (!running || destroyed) return;
      const delta = lastTime === null ? 0 : Math.min(80, Math.max(0, time - lastTime)) * speed;
      lastTime = time;
      if (active) {
        active.elapsed += delta;
        // Slow real signing/verification never lets the animation falsely finish.
        if (!active.ticket) active.elapsed = Math.min(active.elapsed, EVENT_MS * .78);
        active.progress = Math.min(active.elapsed / EVENT_MS, 1);
        if (active.progress >= 1 && active.ticket) {
          try {
            const completed = { id: active.id, draft: core.clone(active.draft), height: active.height };
            const block = ledger.commit(active.ticket);
            sequence += 1;
            active = null;
            gap = GAP_MS;
            // One synchronous notification updates the traveler and ledger in this frame.
            emit('commit', { completed, block });
          } catch (reason) { fail(reason); return; }
        } else emit('frame');
      } else if (gap > 0) {
        gap = Math.max(0, gap - delta);
      } else {
        if (!queue.length) {
          if (group && !continuous) { stopClock(); emit('complete'); return; }
          if (!addGroup()) return;
        }
        beginEvent();
      }
      if (running) frameId = raf(tick);
    }
    function play() {
      if (!ready || destroyed || error || limit || running) return;
      if (!active && !queue.length && group && !continuous && !addGroup()) return;
      running = true;
      lastTime = null;
      emit('play');
      frameId = raf(tick);
    }
    function pause() { if (running) { stopClock(); emit('pause'); } }
    async function reset() {
      generation += 1;
      const epoch = generation;
      stopClock();
      if (ledger) ledger.dispose();
      ledger = null;
      active = null;
      queue = [];
      groups = [];
      sequence = 0;
      group = 0;
      gap = 0;
      error = '';
      limit = false;
      ready = false;
      emit('reset');
      try {
        const next = await (options.createLedger || core.createLedger)();
        if (epoch !== generation || destroyed) { next.dispose(); return; }
        ledger = next;
        ready = true;
        emit('ready');
      } catch (reason) { if (epoch === generation && !destroyed) fail(reason); }
    }
    function setSpeed(value) {
      if (![1, 2, 4].includes(value)) return;
      speed = value;
      lastTime = null;
      emit('speed');
    }
    function setScenario(value) { if (Object.hasOwn(core.SCENARIOS, value)) { scenario = value; emit('scenario'); } }
    function setContinuous(value) { continuous = Boolean(value); emit('continuous'); }
    function destroy() { destroyed = true; generation += 1; stopClock(); if (ledger) ledger.dispose(); }
    return Object.freeze({ play, pause, reset, snapshot, setSpeed, setScenario, setContinuous, destroy });
  }
  return Object.freeze({ EVENT_MS, GAP_MS, createPlayer });
});
