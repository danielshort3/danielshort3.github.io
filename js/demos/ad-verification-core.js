/* Educational single-browser blockchain. Signatures verify records, not real-world claims. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AdVerificationCore = api;
})(globalThis, function () {
  'use strict';
  const VERSION = 4;
  const MAX_BLOCKS = 256;
  const ZERO = '0'.repeat(64);
  const RULE = Object.freeze({ id: 'example-exposure-window-v1', windowDays: 30 });
  const TYPES = Object.freeze(Object.fromEntries(Object.entries({
    campaign: { actor: 'advertiser', title: 'Campaign started', source: 'Advertiser' },
    purchase: { actor: 'agency', title: 'Ad space purchased', source: 'Media agency' },
    ad: { actor: 'publisher', title: 'Ad served', source: 'Ad platform' },
    website: { actor: 'web-analytics', title: 'Website visit', source: 'Website analytics' },
    destination: { actor: 'measurement', title: 'Visit reported', source: 'Location measurement' },
    attribution: { actor: 'attribution', title: 'Attribution recorded', source: 'Attribution service' },
    summary: { actor: 'measurement', title: 'Measurement ended', source: 'Measurement service' }
  }).map(([key, value]) => [key, Object.freeze(value)])));
  const SCENARIOS = Object.freeze({ mixed: 'Mixed outcomes', none: 'Neither visit', website: 'Website only', destination: 'Destination only', both: 'Website + destination', late: 'Visit outside the window' });
  const ACTORS = [...new Set(Object.values(TYPES).map((type) => type.actor))];
  const VALIDATORS = ['campaign-validator', 'delivery-validator', 'audit-validator'];
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const isHash = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
  const exact = (value, keys) => value && Object.getPrototypeOf(value) === Object.prototype && Object.keys(value).sort().join(',') === keys.slice().sort().join(',');
  const dayValid = (day) => Number.isSafeInteger(day) && day >= 0 && day <= 1000000;
  const encoder = new TextEncoder();
  const hex = (buffer) => Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, '0')).join('');
  function subtle() {
    if (!globalThis.crypto?.subtle) throw new Error('Web Crypto requires HTTPS or localhost.');
    return globalThis.crypto.subtle;
  }
  function canonical(value, depth = 0) {
    if (depth > 12) throw new TypeError('JSON is too deeply nested.');
    if (value === null || ['string', 'boolean'].includes(typeof value)) return JSON.stringify(value);
    if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map((item) => canonical(item, depth + 1)).join(',') + ']';
    if (value && Object.getPrototypeOf(value) === Object.prototype) return '{' + Object.keys(value).sort().map((key) => JSON.stringify(key) + ':' + canonical(value[key], depth + 1)).join(',') + '}';
    throw new TypeError('Only finite JSON values are supported.');
  }
  async function hash(text) { return hex(await subtle().digest('SHA-256', encoder.encode(text))); }
  const eventMessage = (event) => 'AD-VERIFY/EVENT/v4\n' + canonical(event);
  const approvalMessage = (digest) => 'AD-VERIFY/APPROVAL/v4\n' + digest;
  const headerHash = (header) => hash('AD-VERIFY/HEADER/v4\n' + canonical(header));
  const algorithm = { name: 'ECDSA', hash: 'SHA-256' };
  async function sign(key, text) { return hex(await subtle().sign(algorithm, key, encoder.encode(text))); }
  async function verify(key, signature, text) {
    if (typeof signature !== 'string' || !/^[a-f0-9]{128}$/.test(signature)) return false;
    try { return await subtle().verify(algorithm, key, Uint8Array.from(signature.match(/../g), (pair) => parseInt(pair, 16)), encoder.encode(text)); }
    catch (_) { return false; }
  }
  async function merkleRoot(transactions) {
    if (!Array.isArray(transactions) || !transactions.length || transactions.length > 64) throw new TypeError('Invalid transaction count.');
    let level = await Promise.all(transactions.map((tx) => hash('AD-VERIFY/LEAF/v4\n' + canonical(tx))));
    while (level.length > 1) {
      const next = [];
      for (let index = 0; index < level.length; index += 2) next.push(await hash('AD-VERIFY/BRANCH/v4\n' + level[index] + (level[index + 1] || level[index])));
      level = next;
    }
    return level[0];
  }
  const eventOf = (block) => block.transactions[0].event;
  const reference = (block) => ({ eventId: eventOf(block).id, blockHeight: block.header.height, blockHash: block.hash });

  // Attribution is calculated from earlier evidence, never from a scenario's claimed credit.
  // Example days are authored fictional occurrence dates, distinct from receipt/recording time.
  function attributionFor(blocks, travelerId) {
    const records = blocks.filter((block) => eventOf(block).travelerId === travelerId);
    const exposure = records.find((block) => eventOf(block).type === 'ad');
    const visit = records.find((block) => eventOf(block).type === 'destination');
    if (!exposure || !visit) throw new Error('Attribution requires an earlier ad and a reported visit.');
    if (eventOf(exposure).campaignId !== eventOf(visit).campaignId) throw new Error('Evidence belongs to different campaigns.');
    const from = eventOf(exposure).data.exampleDay;
    const to = eventOf(visit).data.exampleDay;
    if (!dayValid(from) || !dayValid(to)) throw new Error('Invalid example occurrence dates.');
    const elapsedDays = to - from;
    const withinWindow = elapsedDays >= 0 && elapsedDays <= RULE.windowDays;
    const notPreviouslyCounted = !blocks.some((block) => eventOf(block).type === 'attribution' &&
      eventOf(block).data.credited === true && eventOf(block).data.visit?.eventId === eventOf(visit).id);
    return { ruleId: RULE.id, windowDays: RULE.windowDays, exposure: reference(exposure), visit: reference(visit),
      elapsedDays, withinWindow, notPreviouslyCounted, credited: withinWindow && notPreviouslyCounted, synthetic: true };
  }
  function checkJourney(event, index, journeys) {
    if (index < 2) return event.type === (index ? 'purchase' : 'campaign') && event.travelerId === null && event.previousTravelerEvent === null && event.key === event.type;
    if (!/^T\d{3,4}$/.test(event.travelerId) || event.key !== event.travelerId + '/' + event.type) return false;
    const prior = journeys.get(event.travelerId);
    if (event.type === 'ad') {
      if (prior || event.previousTravelerEvent !== null || !dayValid(event.data.exampleDay) || event.data.impressions !== 1) return false;
      journeys.set(event.travelerId, { last: event.id, time: event.observedAtMs, website: false, destination: false, attribution: false, closed: false });
      return true;
    }
    if (!prior || prior.closed || event.previousTravelerEvent !== prior.last || event.observedAtMs < prior.time) return false;
    if (['website', 'destination'].includes(event.type)) {
      if (prior[event.type] || prior.attribution || !dayValid(event.data.exampleDay)) return false;
      prior[event.type] = true;
    } else if (event.type === 'attribution') {
      if (!prior.destination || prior.attribution) return false;
      prior.attribution = true;
    } else if (event.type === 'summary') {
      if (event.data.websiteRecorded !== prior.website || event.data.destinationRecorded !== prior.destination || (prior.destination && !prior.attribution)) return false;
      prior.closed = true;
    } else return false;
    prior.last = event.id;
    prior.time = event.observedAtMs;
    return true;
  }
  async function verifyChain(blocks, trust) {
    const rows = [];
    const failure = (reason) => ({ valid: false, reason, blocks: rows, lengthValid: false, headValid: false });
    if (!Array.isArray(blocks) || blocks.length > MAX_BLOCKS || !trust || trust.version !== VERSION || typeof trust.chainId !== 'string' || typeof trust.campaignId !== 'string' || !Number.isInteger(trust.expectedLength) || trust.expectedLength < 0 || trust.expectedLength > MAX_BLOCKS || !isHash(trust.headHash)) return failure('Invalid proof or checkpoint.');
    const keys = {};
    try {
      for (const id of [...ACTORS, ...VALIDATORS]) keys[id] = await subtle().importKey('jwk', trust.publicKeys[id], { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
    } catch (_) { return failure('The original public-key registry is unavailable.'); }
    const journeys = new Map();
    let ancestryValid = true;
    for (let index = 0; index < blocks.length; index += 1) {
      const block = blocks[index];
      const checks = { structure: false, journey: false, attribution: false, eventSignature: false, merkleRoot: false, blockHash: false, previousHash: false, approvals: false };
      let computedHash = '';
      try {
        const header = block.header;
        const tx = block.transactions?.[0];
        const event = tx?.event;
        checks.structure = Boolean(exact(block, ['header', 'transactions', 'hash', 'approvals']) &&
          exact(header, ['version', 'chainId', 'height', 'previousHash', 'merkleRoot', 'recordedAt']) &&
          Array.isArray(block.transactions) && block.transactions.length === 1 && exact(tx, ['event', 'signature']) &&
          exact(event, ['version', 'chainId', 'campaignId', 'id', 'key', 'type', 'actor', 'travelerId', 'previousTravelerEvent', 'observedAtMs', 'data']) &&
          Object.hasOwn(TYPES, event.type) && event.version === VERSION && header.version === VERSION &&
          event.chainId === trust.chainId && header.chainId === trust.chainId && event.campaignId === trust.campaignId &&
          event.id === 'event-' + (index + 1) && event.actor === TYPES[event.type].actor && header.height === index + 1 &&
          Number.isSafeInteger(event.observedAtMs) && event.observedAtMs >= 0 && typeof header.recordedAt === 'string' &&
          Number.isFinite(Date.parse(header.recordedAt)) && event.data && Object.getPrototypeOf(event.data) === Object.prototype &&
          canonical(event.data).length <= 4096 && isHash(block.hash) && isHash(header.previousHash) && isHash(header.merkleRoot));
        if (checks.structure) {
          checks.journey = checkJourney(event, index, journeys);
          checks.attribution = event.type !== 'attribution' || canonical(event.data) === canonical(attributionFor(blocks.slice(0, index), event.travelerId));
          const root = await merkleRoot(block.transactions);
          computedHash = await headerHash({ ...header, merkleRoot: root });
          checks.eventSignature = await verify(keys[event.actor], tx.signature, eventMessage(event));
          checks.merkleRoot = root === header.merkleRoot;
          checks.blockHash = await headerHash(header) === block.hash;
          checks.previousHash = header.previousHash === (index ? blocks[index - 1].hash : ZERO);
          const approvals = block.approvals;
          if (Array.isArray(approvals) && approvals.length === VALIDATORS.length && approvals.every((item) => exact(item, ['validator', 'signature']) && VALIDATORS.includes(item.validator)) && new Set(approvals.map((item) => item.validator)).size === VALIDATORS.length) {
            checks.approvals = (await Promise.all(approvals.map((item) => verify(keys[item.validator], item.signature, approvalMessage(block.hash))))).every(Boolean);
          }
        }
      } catch (_) { /* An edited or malformed record fails closed. */ }
      const localValid = Object.values(checks).every(Boolean);
      rows.push({ index, checks, computedHash, localValid, ancestryValid, state: !localValid ? 'changed' : ancestryValid ? 'verified' : 'dependent' });
      ancestryValid = ancestryValid && localValid;
    }
    const lengthValid = blocks.length === trust.expectedLength;
    const headValid = (blocks.length ? blocks.at(-1)?.hash : ZERO) === trust.headHash;
    const valid = ancestryValid && lengthValid && headValid;
    return { valid, blocks: rows, lengthValid, headValid, reason: valid ? 'Records match their signed history.' : 'A record, signature, rule, link, or checkpoint does not match.' };
  }
  // Call only for accepted records (or after verifyChain). Recomputed, never animation counters.
  function campaignTotals(blocks) {
    const totals = { exposures: 0, websiteVisits: 0, reportedVisits: 0, attributedVisits: 0, completed: 0 };
    const counted = new Set();
    for (const block of blocks) {
      const event = eventOf(block);
      if (event.type === 'ad') totals.exposures += 1;
      if (event.type === 'website') totals.websiteVisits += 1;
      if (event.type === 'destination') totals.reportedVisits += 1;
      if (event.type === 'summary') totals.completed += 1;
      if (event.type === 'attribution' && event.data.credited && !counted.has(event.data.visit.eventId)) {
        counted.add(event.data.visit.eventId); totals.attributedVisits += 1;
      }
    }
    return totals;
  }
  async function verifyResults(blocks, trust) {
    const report = await verifyChain(blocks, trust);
    return { report, totals: report.valid ? campaignTotals(blocks) : null };
  }
  async function createLedger() {
    const identities = {};
    const publicKeys = {};
    for (const id of [...ACTORS, ...VALIDATORS]) {
      const pair = await subtle().generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
      identities[id] = pair.privateKey;
      publicKeys[id] = await subtle().exportKey('jwk', pair.publicKey);
    }
    const chainId = 'cedar-valley-' + hex(globalThis.crypto.getRandomValues(new Uint8Array(12)));
    const campaignId = 'CV-SPRING-2027';
    const originals = [];
    const tickets = new WeakMap();
    let retired = false;
    const checkpoint = () => ({ version: VERSION, chainId, campaignId, expectedLength: originals.length, headHash: originals.at(-1)?.hash || ZERO, publicKeys: clone(publicKeys) });
    const snapshot = () => ({ blocks: clone(originals), trust: checkpoint() });
    async function prepare(draft) {
      if (retired || originals.length >= MAX_BLOCKS) throw new Error('Start a new demo campaign.');
      if (!draft || !Object.hasOwn(TYPES, draft.type)) throw new TypeError('Unknown event.');
      const baseLength = originals.length;
      const previous = originals.findLast((block) => draft.travelerId && eventOf(block).travelerId === draft.travelerId);
      const data = draft.type === 'attribution' ? attributionFor(originals, draft.travelerId) : clone(draft.data);
      const event = { version: VERSION, chainId, campaignId, id: 'event-' + (baseLength + 1), key: draft.key, type: draft.type,
        actor: TYPES[draft.type].actor, travelerId: draft.travelerId, previousTravelerEvent: previous ? eventOf(previous).id : null,
        observedAtMs: draft.observedAtMs, data };
      if (canonical(event.data).length > 4096) throw new TypeError('Event data exceeds the demo limit.');
      const transactions = [{ event, signature: await sign(identities[event.actor], eventMessage(event)) }];
      const header = { version: VERSION, chainId, height: baseLength + 1, previousHash: checkpoint().headHash, merkleRoot: await merkleRoot(transactions), recordedAt: new Date().toISOString() };
      const digest = await headerHash(header);
      const approvals = await Promise.all(VALIDATORS.map(async (validator) => ({ validator, signature: await sign(identities[validator], approvalMessage(digest)) })));
      const block = { header, transactions, hash: digest, approvals };
      const report = await verifyChain([...originals, block], { ...checkpoint(), expectedLength: baseLength + 1, headHash: digest });
      if (retired || baseLength !== originals.length || !report.valid) throw new Error('Candidate verification failed or the chain changed.');
      const ticket = Object.freeze({ block: clone(block) });
      tickets.set(ticket, { block, baseLength });
      return ticket;
    }
    function commit(ticket) {
      const candidate = tickets.get(ticket);
      if (retired || !candidate || candidate.baseLength !== originals.length || candidate.block.header.previousHash !== checkpoint().headHash) throw new Error('Stale or unverified candidate.');
      tickets.delete(ticket); originals.push(candidate.block);
      return clone(candidate.block);
    }
    function dispose() { retired = true; for (const id of Object.keys(identities)) delete identities[id]; }
    return Object.freeze({ prepare, commit, snapshot, dispose });
  }
  function createTraveler(scenario, number) {
    if (!Object.hasOwn(SCENARIOS, scenario) || !Number.isInteger(number) || number < 1 || number > 9999) throw new TypeError('Invalid traveler.');
    const path = scenario === 'mixed' ? ['both', 'none', 'website', 'destination', 'late', 'both', 'none', 'website'][(number - 1) % 8] : scenario;
    const id = 'T' + String(number).padStart(3, '0');
    const web = ['website', 'both', 'late'].includes(path);
    const visit = ['destination', 'both', 'late'].includes(path);
    const day = number * 2;
    const events = [{ type: 'ad', data: { publisher: 'Example Media', impressions: 1, exampleDay: day } }];
    if (web) events.push({ type: 'website', data: { page: '/hiking-guides', exampleDay: day + 1 } });
    if (visit) {
      events.push({ type: 'destination', data: { place: 'Cedar Valley', exampleDay: day + (path === 'late' ? 35 : 5 + number % 9) } });
      events.push({ type: 'attribution', data: {} });
    }
    events.push({ type: 'summary', data: { websiteRecorded: web, destinationRecorded: visit, note: 'No observation is not proof of no visit.' } });
    return { id, number, path, events: events.map((event) => ({ ...event, travelerId: id, key: id + '/' + event.type, data: { ...event.data, synthetic: true } })) };
  }
  return Object.freeze({ VERSION, MAX_BLOCKS, ZERO, RULE, TYPES, SCENARIOS, createLedger, createTraveler, attributionFor,
    verifyChain, campaignTotals, verifyResults, canonical, hash, merkleRoot, headerHash, clone });
});
