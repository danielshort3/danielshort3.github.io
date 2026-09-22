/* Parallel traveler clocks, one ordered block writer, one shared animation frame. */
(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./ad-verification-core.js'));
  else root.AdVerificationPlayer = factory(root.AdVerificationCore);
})(globalThis, function (core) {
  'use strict';
  const LANES = 5;
  const SEAL_MS = 520;
  const HOLD_MS = 1200;
  function createPlayer(options = {}) {
    const raf = options.requestFrame || requestAnimationFrame;
    const cancel = options.cancelFrame || cancelAnimationFrame;
    const notify = options.onChange || (() => {});
    let ledger = null;
    let epoch = 0;
    let frame = null;
    let lastTime = null;
    let time = 0;
    let speed = 1;
    let scenario = 'mixed';
    let continuous = true;
    let ready = false;
    let running = false;
    let started = false;
    let retired = false;
    let capacityClosed = false;
    let error = '';
    let count = 0;
    let admitted = 0;
    let completed = 0;
    let reserved = 0;
    let lanes = Array(LANES).fill(null);
    let queue = [];
    let writer = null;
    function laneView(lane, slot) {
      if (!lane) return { slot, id: null, phase: 'waiting', progress: 0, records: {} };
      return { slot, id: lane.id, number: lane.number, path: lane.path, phase: lane.phase,
        type: lane.events[lane.step]?.type || 'summary', key: lane.events[lane.step]?.key || lane.id + '/summary',
        progress: lane.progress, records: { ...lane.records },
        opacity: lane.phase === 'done' && continuous && !capacityClosed ? Math.max(.1, Math.min(1, (lane.replaceAt - time) / 300)) : Math.min(1, (time - lane.born + 200) / 400) };
    }
    const snapshot = (includeProof = true) => ({ ready, running, started, speed, scenario, continuous, time,
      count, admitted, completed, capacityClosed, error, lanes: lanes.map(laneView),
      queue: queue.map((draft) => ({ key: draft.key, type: draft.type, travelerId: draft.travelerId })),
      writer: writer ? { key: writer.draft.key, type: writer.draft.type, travelerId: writer.draft.travelerId, height: count + 1, progress: writer.progress } : null,
      proof: includeProof && ledger ? ledger.snapshot() : null });
    const emit = (kind, extra = {}, includeProof = false) => notify({ kind, ...extra }, snapshot(includeProof));
    function stop() { running = false; if (frame !== null) cancel(frame); frame = null; lastTime = null; }
    function fail(reason) { stop(); error = reason.message || String(reason); emit('error'); }
    function admit(slot, first = false) {
      if (capacityClosed) return false;
      const plan = core.createTraveler(scenario, admitted + 1);
      // Reserve every remaining event: the cap cannot strand half-finished travelers.
      if (count + reserved + plan.events.length > core.MAX_BLOCKS) { capacityClosed = true; return false; }
      admitted += 1;
      reserved += plan.events.length;
      lanes[slot] = { ...plan, slot, step: 0, records: {}, phase: 'waiting', progress: 0, born: time,
        nextAt: time + (first ? slot * 410 : 120), beganAt: 0, duration: 0, replaceAt: Infinity };
      return true;
    }
    const duration = (lane) => lane.events[lane.step].type === 'summary' ? 450 : 1700 + (lane.number * 137 + lane.step * 211) % 950;
    function nextDue(lane) {
      const type = lane.events[lane.step]?.type;
      if (type === 'summary') return Math.max(time + 650, lane.born + 9400 + (lane.number * 2311) % 9000);
      return time + (type === 'destination' ? 1900 + (lane.number * 919) % 3000 : 650 + (lane.number * 367) % 1350);
    }
    function startWriter() {
      if (writer || !queue.length) return;
      queue.sort((a, b) => a.observedAtMs - b.observedAtMs || a.key.localeCompare(b.key));
      const draft = queue.shift();
      const target = { draft, progress: 0, elapsed: 0, ticket: null };
      const owner = ledger;
      const generation = epoch;
      writer = target;
      const lane = lanes.find((item) => item?.id === draft.travelerId);
      if (lane) { lane.phase = 'verifying'; lane.progress = 0; }
      owner.prepare(draft).then((ticket) => {
        if (!retired && generation === epoch && owner === ledger && writer === target) target.ticket = ticket;
      }).catch((reason) => {
        if (!retired && generation === epoch && owner === ledger && writer === target) fail(reason);
      });
    }
    function tick(timestamp) {
      frame = null;
      if (!running || retired) return;
      const delta = lastTime === null ? 0 : Math.max(0, Math.min(80, timestamp - lastTime)) * speed;
      lastTime = timestamp;
      time += delta;
      const changes = [];
      if (count >= 2) {
        lanes.forEach((lane, slot) => {
          if (!lane || (lane.phase === 'done' && time >= lane.replaceAt && continuous)) {
            if (admit(slot, !lane)) changes.push({ type: 'enter', slot, travelerId: lanes[slot].id, replaced: lane?.id || null });
          }
        });
        lanes.forEach((lane) => {
          if (!lane) return;
          if (lane.phase === 'waiting' && time >= lane.nextAt) {
            lane.phase = 'recording'; lane.beganAt = lane.nextAt; lane.duration = duration(lane);
            changes.push({ type: 'observe', travelerId: lane.id, key: lane.events[lane.step].key });
          }
          if (lane.phase === 'recording') {
            lane.progress = Math.min(1, (time - lane.beganAt) / lane.duration);
            if (lane.progress >= 1) {
              queue.push({ ...core.clone(lane.events[lane.step]), observedAtMs: Math.round(lane.beganAt + lane.duration) });
              lane.phase = 'queued';
            }
          }
        });
      }
      let block = null;
      if (writer) {
        writer.elapsed += delta;
        if (!writer.ticket) writer.elapsed = Math.min(writer.elapsed, SEAL_MS * .8);
        writer.progress = Math.min(1, writer.elapsed / SEAL_MS);
        const lane = lanes.find((item) => item?.id === writer.draft.travelerId);
        if (lane) lane.progress = writer.progress;
        if (writer.ticket && writer.progress >= 1) {
          try {
            block = ledger.commit(writer.ticket);
            count += 1; reserved -= 1;
            const event = block.transactions[0].event;
            if (lane) {
              lane.records[event.type] = { height: count, key: event.key };
              if (event.type === 'summary') {
                completed += 1; lane.phase = 'done'; lane.replaceAt = time + HOLD_MS;
              } else {
                lane.step += 1; lane.phase = 'waiting'; lane.progress = 0; lane.nextAt = nextDue(lane);
              }
            }
            changes.push({ type: 'commit', key: event.key, travelerId: event.travelerId, height: count });
            writer = null;
          } catch (reason) { fail(reason); return; }
        }
      }
      // The writer is serial; observation above keeps running in all five lanes.
      if (!writer && queue.length) { startWriter(); changes.push({ type: 'write', key: writer.draft.key }); }
      if (started && admitted >= LANES && !writer && !queue.length && lanes.every((lane) => !lane || lane.phase === 'done') && (!continuous || capacityClosed)) stop();
      emit(block ? 'commit' : 'frame', { changes, block }, Boolean(block));
      if (running) frame = raf(tick);
    }
    function play() {
      if (!ready || retired || error || running || (capacityClosed && lanes.every((lane) => !lane || lane.phase === 'done'))) return;
      if (!started) {
        started = true; reserved = 2;
        queue.push({ key: 'campaign', type: 'campaign', travelerId: null, observedAtMs: 0, data: { destination: 'Cedar Valley Tourism', name: 'A little closer to nature', synthetic: true } },
          { key: 'purchase', type: 'purchase', travelerId: null, observedAtMs: 1, data: { agency: 'Example Media', placement: 'Display and streaming video', synthetic: true } });
      }
      running = true; lastTime = null; emit('play'); frame = raf(tick);
    }
    function pause() { if (running) { stop(); emit('pause'); } }
    async function reset() {
      epoch += 1;
      const generation = epoch;
      stop(); if (ledger) ledger.dispose(); ledger = null;
      lanes = Array(LANES).fill(null); queue = []; writer = null;
      time = 0; count = 0; admitted = 0; completed = 0; reserved = 0;
      ready = false; started = false; capacityClosed = false; error = '';
      emit('reset');
      try {
        const next = await (options.createLedger || core.createLedger)();
        if (generation !== epoch || retired) { next.dispose(); return; }
        ledger = next; ready = true; emit('ready', {}, true);
      } catch (reason) { if (generation === epoch && !retired) fail(reason); }
    }
    function setScenario(value) { if (Object.hasOwn(core.SCENARIOS, value)) { scenario = value; emit('settings'); } }
    function setSpeed(value) { if ([1, 2, 4].includes(value)) { speed = value; lastTime = null; emit('settings'); } }
    function setContinuous(value) { continuous = Boolean(value); emit('settings'); }
    function destroy() { retired = true; epoch += 1; stop(); if (ledger) ledger.dispose(); }
    return Object.freeze({ play, pause, reset, snapshot, setScenario, setSpeed, setContinuous, destroy });
  }
  return Object.freeze({ LANES, SEAL_MS, HOLD_MS, createPlayer });
});
