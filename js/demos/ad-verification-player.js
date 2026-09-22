/* Parallel activity is independent of batching. Only the final append changes recorded checks. */
(function (root, factory) {
  'use strict';
  const core = typeof module === 'object' && module.exports ? require('./ad-verification-core.js') : root.AdVerificationCore;
  const api = factory(core);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AdVerificationPlayer = api;
})(globalThis, function (core) {
  'use strict';
  const LANES = 5;
  const OBSERVE_MS = 1150;
  const SEAL_MS = 850;
  const WAIT_MS = 1600;
  const HOLD_MS = 1800;
  function createPlayer(options = {}) {
    const notify = options.onChange || (() => {});
    const raf = options.requestFrame || requestAnimationFrame;
    const cancel = options.cancelFrame || cancelAnimationFrame;
    let session = null;
    let epoch = 0;
    let writerToken = 0;
    let running = false;
    let ready = false;
    let busy = false;
    let error = '';
    let frame = null;
    let last = null;
    let time = 0;
    let speed = 1;
    let scenario = 'mixed';
    let continuous = true;
    let admitted = 0;
    let completed = 0;
    let recordCount = 0;
    let blockCount = 0;
    let reserved = 2;
    let limited = false;
    let lanes = Array(LANES).fill(null);
    let queue = [];
    let messages = [];
    let writer = null;
    const annotations = new Map();
    function viewLane(lane, slot) {
      if (!lane) return { slot, id: null, phase: 'waiting', recorded: {}, observed: {}, active: null, opacity: 1 };
      const active = lane.stages.find((stage) => time >= lane.born + stage.at && time < lane.born + stage.at + OBSERVE_MS);
      return { slot, id: lane.id, number: lane.number, path: lane.path, phase: lane.phase,
        recorded: core.clone(lane.recorded), observed: { ...lane.observed }, active: active?.type || null,
        progress: active ? Math.min(1, (time - lane.born - active.at) / OBSERVE_MS) : 0,
        opacity: lane.phase === 'done' && continuous && !limited ? Math.max(.15, Math.min(1, (lane.replaceAt - time) / 250)) : 1 };
    }
    function snapshot(full = true) {
      return { ready, running, busy, error, time, speed, scenario, continuous, admitted, completed,
        recordCount, blockCount, limited, lanes: lanes.map(viewLane), queued: queue.length,
        writer: writer ? { ids: writer.records.map((r) => r.receipt.id), count: writer.records.length, progress: writer.progress, height: blockCount + 1 } : null,
        copies: session ? session.copyStates() : [], proof: full && session ? session.snapshot() : null };
    }
    const emit = (kind, data = {}, full = false) => notify({ kind, ...data }, snapshot(full));
    function pause() {
      running = false; if (frame !== null) cancel(frame); frame = null; last = null;
      emit('pause');
    }
    function fail(reason) { pause(); error = reason.message || String(reason); emit('error'); }
    function admit(slot) {
      if (limited) return;
      const plan = core.traveler(scenario, admitted + 1);
      if (recordCount + reserved + plan.stages.length > core.LIMIT - 12) { limited = true; return; }
      admitted += 1; reserved += plan.stages.length;
      lanes[slot] = { ...plan, born: time + (admitted <= LANES ? slot * 380 : 0), phase: 'live',
        recorded: {}, observed: {}, requested: new Set(), refs: {}, sourceTail: Promise.resolve(), replaceAt: Infinity };
    }
    function requestObservation(lane, stage) {
      const generation = epoch; const owner = session;
      lane.requested.add(stage.type); lane.observed[stage.type] = true;
      // Source work is ordered per traveler; it never waits for a blockchain append.
      lane.sourceTail = lane.sourceTail.then(async () => {
        const record = await owner.observe(stage.type, lane.id, stage.day, lane.refs, Math.round(lane.born + stage.at + OBSERVE_MS));
        lane.refs[stage.type] = record.receipt.id;
        if (generation === epoch) messages.push({ record, traveler: lane.id, type: stage.type });
      }).catch((reason) => { if (generation === epoch) fail(reason); });
    }
    function prepareBatch() {
      const generation = epoch; const token = ++writerToken; const owner = session;
      const records = queue.splice(0, core.BATCH_SIZE).map((item) => item.record);
      const item = { records, elapsed: 0, progress: 0, ticket: null };
      writer = item;
      owner.prepare(records).then((ticket) => {
        if (generation === epoch && token === writerToken && writer === item) item.ticket = ticket;
      }).catch((reason) => { if (generation === epoch && token === writerToken) fail(reason); });
    }
    function publish(block, manual = false) {
      blockCount += 1; recordCount += block.records.length;
      for (const { receipt } of block.records) {
        if (!['report', 'correction'].includes(receipt.type)) reserved -= 1;
        const annotation = annotations.get(receipt.id);
        const lane = lanes.find((item) => item?.id === annotation?.traveler);
        if (lane) lane.recorded[receipt.type] = { id: receipt.id, block: block.header.height };
      }
      // A traveler exits only after all its receipts, including the closing record, are committed.
      for (const lane of lanes) {
        if (lane && lane.phase !== 'done' && lane.stages.every((stage) => lane.recorded[stage.type])) {
          lane.phase = 'done'; lane.replaceAt = time + HOLD_MS; completed += 1;
        }
      }
      emit(manual ? 'manual' : 'block', { block }, true);
    }
    function tick(timestamp) {
      frame = null; if (!running || busy) return;
      const delta = last === null ? 0 : Math.min(80, Math.max(0, timestamp - last)) * speed;
      last = timestamp; time += delta;
      for (const message of messages.splice(0)) {
        annotations.set(message.record.receipt.id, { traveler: message.traveler, type: message.type });
        queue.push({ record: message.record, since: time });
      }
      if (recordCount >= 2) {
        for (let slot = 0; slot < LANES; slot += 1) {
          const lane = lanes[slot];
          if (!lane || (lane.phase === 'done' && continuous && time >= lane.replaceAt)) admit(slot);
        }
        for (const lane of lanes) {
          if (!lane || lane.phase === 'done') continue;
          for (const stage of lane.stages) {
            if (!lane.requested.has(stage.type) && time >= lane.born + stage.at + OBSERVE_MS) requestObservation(lane, stage);
          }
        }
      }
      if (!writer && queue.length && (queue.length >= core.BATCH_SIZE || time - queue[0].since >= WAIT_MS)) prepareBatch();
      if (writer) {
        writer.elapsed += delta;
        if (!writer.ticket) writer.elapsed = Math.min(writer.elapsed, SEAL_MS * .8);
        writer.progress = Math.min(1, writer.elapsed / SEAL_MS);
        if (writer.ticket && writer.progress >= 1) {
          try { const block = session.commit(writer.ticket); writer = null; publish(block); }
          catch (reason) { fail(reason); return; }
        }
      }
      if (admitted >= LANES && lanes.every((lane) => lane.phase === 'done') && !writer && !queue.length && (!continuous || limited)) running = false;
      emit('frame');
      if (running) frame = raf(tick);
    }
    function play() {
      if (!ready || busy || error || running || (limited && lanes.every((lane) => !lane || lane.phase === 'done'))) return;
      running = true; last = null; emit('play'); frame = raf(tick);
    }
    function abandonCandidate() {
      writerToken += 1;
      if (writer) queue.unshift(...writer.records.map((record) => ({ record, since: time })));
      writer = null;
    }
    async function reset() {
      epoch += 1; const generation = epoch;
      pause(); if (session) session.dispose(); session = null; abandonCandidate();
      ready = false; busy = false; error = ''; time = 0; admitted = 0; completed = 0;
      recordCount = 0; blockCount = 0; reserved = 2; limited = false;
      lanes = Array(LANES).fill(null); queue = []; messages = []; annotations.clear(); emit('reset');
      try {
        const next = await (options.createSession || core.createSession)();
        if (generation !== epoch) { next.dispose(); return; }
        session = next;
        const campaign = await session.issue('campaign', { notice: 'Fictional campaign authorization' }, { rule: core.RULE.id, window: core.RULE.days });
        const purchase = await session.issue('purchase', { notice: 'Fictional media authorization' }, { authorized: true });
        if (generation !== epoch) return;
        queue = [campaign, purchase].map((record) => ({ record, since: 0 })); ready = true; emit('ready', {}, true);
      } catch (reason) { if (generation === epoch) fail(reason); }
    }
    async function action(name, value, extra) {
      if (!ready || busy) return null;
      pause(); abandonCandidate(); busy = true; emit('busy');
      const generation = epoch; const owner = session;
      try {
        let result;
        if (name === 'report') { result = await owner.report(); if (generation === epoch) publish(result, true); }
        else if (name === 'correct') { result = await owner.correct(value); if (generation === epoch) publish(result, true); }
        else if (name === 'copy') result = await owner.manageCopy(value, extra);
        else if (name === 'withhold') { owner.withhold(value, extra); result = await owner.audit(value); }
        else throw new Error('Unknown action.');
        return generation === epoch ? result : null;
      } finally { if (generation === epoch) { busy = false; emit('action', {}, true); } }
    }
    const audit = (id) => session.audit(id);
    const describe = (id) => session.describe(id);
    function setSpeed(value) { if ([1, 2, 4].includes(value)) { speed = value; last = null; emit('setting'); } }
    function setScenario(value) { if (Object.hasOwn(core.SCENARIOS, value)) { scenario = value; emit('setting'); } }
    function setContinuous(value) { continuous = Boolean(value); emit('setting'); }
    function destroy() { epoch += 1; pause(); if (session) session.dispose(); }
    return Object.freeze({ reset, play, pause, snapshot, action, audit, describe, setSpeed, setScenario, setContinuous, destroy });
  }
  return Object.freeze({ createPlayer, LANES, OBSERVE_MS, SEAL_MS, WAIT_MS, HOLD_MS });
});
