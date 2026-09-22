/* Real cryptography, fictional events, one browser. No independent network consensus. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AdVerificationCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const VERSION = 2;
  const MAX_BLOCKS = 256;
  const ZERO = '0'.repeat(64);
  const ACTORS = ['advertiser', 'agency', 'publisher', 'web-analytics', 'measurement'];
  const VALIDATORS = ['campaign-validator', 'delivery-validator', 'audit-validator'];
  const TYPES = Object.freeze({
    campaign: { actor: 'advertiser', title: 'Campaign created' },
    purchase: { actor: 'agency', title: 'Ad purchased' },
    ad: { actor: 'publisher', title: 'Ad served' },
    website: { actor: 'web-analytics', title: 'Website visit recorded' },
    destination: { actor: 'measurement', title: 'Destination visit attributed' },
    summary: { actor: 'measurement', title: 'Observation window closed' }
  });
  const encoder = new TextEncoder();
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const isHash = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
  const exact = (value, keys) => value && Object.getPrototypeOf(value) === Object.prototype &&
    Object.keys(value).sort().join(',') === keys.slice().sort().join(',');
  const hex = (value) => Array.from(new Uint8Array(value), (byte) => byte.toString(16).padStart(2, '0')).join('');
  function subtle() {
    if (!globalThis.crypto || !globalThis.crypto.subtle) throw new Error('Web Crypto requires HTTPS or localhost.');
    return globalThis.crypto.subtle;
  }
  function canonical(value, depth = 0) {
    if (depth > 12) throw new TypeError('JSON is too deeply nested.');
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
    if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map((item) => canonical(item, depth + 1)).join(',') + ']';
    if (value && Object.getPrototypeOf(value) === Object.prototype) {
      return '{' + Object.keys(value).sort().map((key) => JSON.stringify(key) + ':' + canonical(value[key], depth + 1)).join(',') + '}';
    }
    throw new TypeError('Only finite JSON values are supported.');
  }
  async function hash(text) { return hex(await subtle().digest('SHA-256', encoder.encode(text))); }
  const eventMessage = (event) => 'AD-VERIFY/EVENT/v2\n' + canonical(event);
  const approvalMessage = (digest) => 'AD-VERIFY/APPROVAL/v2\n' + digest;
  const headerHash = (header) => hash('AD-VERIFY/HEADER/v2\n' + canonical(header));
  const algorithm = { name: 'ECDSA', hash: 'SHA-256' };
  async function sign(key, text) { return hex(await subtle().sign(algorithm, key, encoder.encode(text))); }
  async function verify(key, signature, text) {
    if (typeof signature !== 'string' || !/^[a-f0-9]{128}$/.test(signature)) return false;
    try {
      const bytes = Uint8Array.from(signature.match(/../g), (pair) => parseInt(pair, 16));
      return await subtle().verify(algorithm, key, bytes, encoder.encode(text));
    } catch (_) { return false; }
  }
  async function merkleRoot(transactions) {
    if (!Array.isArray(transactions) || !transactions.length || transactions.length > 64) throw new TypeError('Invalid transaction count.');
    let level = await Promise.all(transactions.map((tx) => hash('AD-VERIFY/LEAF/v2\n' + canonical(tx))));
    while (level.length > 1) {
      const next = [];
      for (let index = 0; index < level.length; index += 2) next.push(await hash('AD-VERIFY/BRANCH/v2\n' + level[index] + (level[index + 1] || level[index])));
      level = next;
    }
    return level[0];
  }

  // These are business-event links within ONE linear chain, not blockchain forks.
  function checkJourney(event, index, journeys) {
    if (event.type === 'campaign' || event.type === 'purchase') {
      return event.type === (index === 0 ? 'campaign' : index === 1 ? 'purchase' : '') &&
        event.travelerId === null && event.group === 0 && event.previousTravelerEvent === null;
    }
    if (index < 2 || !/^T\d{3,4}$/.test(event.travelerId) || !Number.isInteger(event.group) || event.group < 1) return false;
    const prior = journeys.get(event.travelerId);
    if (event.type === 'ad') {
      if (prior || event.previousTravelerEvent !== null) return false;
      journeys.set(event.travelerId, { last: event.id, group: event.group, website: false, destination: false, closed: false });
      return true;
    }
    if (!prior || prior.closed || event.group !== prior.group || event.previousTravelerEvent !== prior.last) return false;
    if (event.type === 'website' || event.type === 'destination') {
      if (prior[event.type]) return false;
      prior[event.type] = true;
    } else if (event.type === 'summary') {
      if (event.data.websiteRecorded !== prior.website || event.data.destinationRecorded !== prior.destination) return false;
      prior.closed = true;
    } else return false;
    prior.last = event.id;
    return true;
  }

  async function verifyChain(blocks, trust) {
    const rows = [];
    const failure = (reason) => ({ valid: false, reason, blocks: rows, lengthValid: false, headValid: false });
    if (!Array.isArray(blocks) || blocks.length > MAX_BLOCKS || !trust || trust.version !== VERSION ||
        typeof trust.chainId !== 'string' || typeof trust.campaignId !== 'string' ||
        !Number.isInteger(trust.expectedLength) || trust.expectedLength < 0 || trust.expectedLength > MAX_BLOCKS || !isHash(trust.headHash)) return failure('Invalid proof or checkpoint.');
    const keys = {};
    try {
      for (const id of [...ACTORS, ...VALIDATORS]) keys[id] = await subtle().importKey('jwk', trust.publicKeys[id], { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
    } catch (_) { return failure('The original public-key registry is unavailable.'); }
    const journeys = new Map();
    let ancestryValid = true;
    for (let index = 0; index < blocks.length; index += 1) {
      const block = blocks[index];
      const checks = { structure: false, journey: false, eventSignature: false, merkleRoot: false, blockHash: false, previousHash: false, approvals: false };
      let computedHash = '';
      try {
        const header = block.header;
        const tx = block.transactions && block.transactions[0];
        const event = tx && tx.event;
        checks.structure = Boolean(exact(block, ['header', 'transactions', 'hash', 'approvals']) &&
          exact(header, ['version', 'chainId', 'height', 'previousHash', 'merkleRoot', 'recordedAt']) &&
          Array.isArray(block.transactions) && block.transactions.length === 1 && exact(tx, ['event', 'signature']) &&
          exact(event, ['version', 'chainId', 'campaignId', 'id', 'type', 'actor', 'group', 'travelerId', 'previousTravelerEvent', 'reportedAt', 'data']) &&
          Object.hasOwn(TYPES, event.type) && event.version === VERSION && header.version === VERSION &&
          event.chainId === trust.chainId && header.chainId === trust.chainId && event.campaignId === trust.campaignId &&
          event.id === 'event-' + (index + 1) && event.actor === TYPES[event.type].actor && header.height === index + 1 &&
          typeof event.reportedAt === 'string' && Number.isFinite(Date.parse(event.reportedAt)) && header.recordedAt === event.reportedAt &&
          event.data && Object.getPrototypeOf(event.data) === Object.prototype && canonical(event.data).length <= 4096 &&
          isHash(block.hash) && isHash(header.previousHash) && isHash(header.merkleRoot));
        if (checks.structure) {
          checks.journey = checkJourney(event, index, journeys);
          const root = await merkleRoot(block.transactions);
          computedHash = await headerHash({ ...header, merkleRoot: root });
          checks.eventSignature = await verify(keys[event.actor], tx.signature, eventMessage(event));
          checks.merkleRoot = root === header.merkleRoot;
          checks.blockHash = await headerHash(header) === block.hash;
          checks.previousHash = header.previousHash === (index ? blocks[index - 1].hash : ZERO);
          const approvals = block.approvals;
          if (Array.isArray(approvals) && approvals.length === VALIDATORS.length &&
              approvals.every((item) => exact(item, ['validator', 'signature']) && VALIDATORS.includes(item.validator)) &&
              new Set(approvals.map((item) => item.validator)).size === VALIDATORS.length) {
            checks.approvals = (await Promise.all(approvals.map((item) => verify(keys[item.validator], item.signature, approvalMessage(block.hash))))).every(Boolean);
          }
        }
      } catch (_) { /* Malformed or edited records fail closed. */ }
      const localValid = Object.values(checks).every(Boolean);
      rows.push({ index, checks, computedHash, localValid, ancestryValid, state: !localValid ? 'changed' : ancestryValid ? 'verified' : 'dependent' });
      ancestryValid = ancestryValid && localValid;
    }
    const lengthValid = blocks.length === trust.expectedLength;
    const headValid = (blocks.length ? blocks[blocks.length - 1]?.hash : ZERO) === trust.headHash;
    const valid = ancestryValid && lengthValid && headValid;
    return { valid, blocks: rows, lengthValid, headValid, reason: valid ? 'Recorded history matches its checkpoint.' : 'A record, signature, link, or checkpoint does not match.' };
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
    const checkpoint = () => ({ version: VERSION, chainId, campaignId, expectedLength: originals.length,
      headHash: originals.length ? originals[originals.length - 1].hash : ZERO, publicKeys: clone(publicKeys) });
    const snapshot = () => ({ blocks: clone(originals), trust: checkpoint() });
    async function prepare(draft) {
      if (retired || originals.length >= MAX_BLOCKS) throw new Error('Start a new demo campaign.');
      if (!draft || !Object.hasOwn(TYPES, draft.type)) throw new TypeError('Unknown event.');
      const baseLength = originals.length;
      const previous = originals.filter((block) => draft.travelerId && block.transactions[0].event.travelerId === draft.travelerId).at(-1);
      const event = { version: VERSION, chainId, campaignId, id: 'event-' + (baseLength + 1), type: draft.type,
        actor: TYPES[draft.type].actor, group: draft.group, travelerId: draft.travelerId,
        previousTravelerEvent: previous ? previous.transactions[0].event.id : null,
        reportedAt: new Date(Date.UTC(2027, 3, 25, 10, 0, baseLength * 15)).toISOString(), data: clone(draft.data) };
      if (canonical(event.data).length > 4096) throw new TypeError('Event data exceeds the demo limit.');
      const transactions = [{ event, signature: await sign(identities[event.actor], eventMessage(event)) }];
      const header = { version: VERSION, chainId, height: baseLength + 1, previousHash: checkpoint().headHash,
        merkleRoot: await merkleRoot(transactions), recordedAt: event.reportedAt };
      const digest = await headerHash(header);
      const approvals = await Promise.all(VALIDATORS.map(async (validator) => ({ validator,
        signature: await sign(identities[validator], approvalMessage(digest)) })));
      const block = { header, transactions, hash: digest, approvals };
      const report = await verifyChain([...originals, block], { ...checkpoint(), expectedLength: baseLength + 1, headHash: digest });
      if (retired || baseLength !== originals.length || !report.valid) throw new Error('Candidate verification failed or the chain changed.');
      const ticket = Object.freeze({ block: clone(block) });
      // The commit uses this private verified copy, never the caller's editable copy.
      tickets.set(ticket, { block, baseLength });
      return ticket;
    }
    function commit(ticket) {
      const candidate = tickets.get(ticket);
      if (retired || !candidate || candidate.baseLength !== originals.length || candidate.block.header.previousHash !== checkpoint().headHash) throw new Error('Stale or unverified candidate.');
      tickets.delete(ticket);
      originals.push(candidate.block);
      return clone(candidate.block);
    }
    function dispose() {
      retired = true;
      for (const id of Object.keys(identities)) delete identities[id];
    }
    return Object.freeze({ prepare, commit, snapshot, dispose });
  }

  const SCENARIOS = Object.freeze({ mixed: 'Mixed results (all outcomes)', none: 'No website or destination visit',
    website: 'Website visit only', destination: 'Destination visit only', both: 'Website + destination visit' });
  function createPlan(scenario, group, includeCampaign = false) {
    if (!Object.hasOwn(SCENARIOS, scenario) || !Number.isInteger(group) || group < 1 || group > 50) throw new TypeError('Invalid scenario or group.');
    const paths = scenario === 'mixed' ? ['both', 'website', 'destination', 'none'] : Array(4).fill(scenario);
    const travelers = paths.map((path, index) => ({ id: 'T' + String((group - 1) * 4 + index + 1).padStart(3, '0'), group, avatar: index, path }));
    const events = [];
    const add = (type, traveler, data) => events.push({ type, travelerId: traveler ? traveler.id : null, group: traveler ? group : 0, data });
    if (includeCampaign) {
      add('campaign', null, { destination: 'Cedar Valley Tourism', name: 'A little closer to nature', budget: 100000, synthetic: true });
      add('purchase', null, { agency: 'Example Media', placement: 'Display and streaming video', synthetic: true });
    }
    travelers.forEach((traveler) => add('ad', traveler, { publisher: 'Example Travel Journal', impressions: 1, synthetic: true }));
    travelers.filter((traveler) => ['both', 'website'].includes(traveler.path)).forEach((traveler) => add('website', traveler, { page: traveler.avatar % 2 ? '/events' : '/things-to-do', source: 'Example website analytics', synthetic: true }));
    // Interleave paths. A destination visit does not require a website event.
    [...travelers].reverse().filter((traveler) => ['both', 'destination'].includes(traveler.path)).forEach((traveler) => add('destination', traveler,
      { place: 'Cedar Valley Visitor Center', source: 'Example location measurement', attribution: 'Simulated provider inference', synthetic: true }));
    travelers.forEach((traveler) => add('summary', traveler, { websiteRecorded: ['both', 'website'].includes(traveler.path),
      destinationRecorded: ['both', 'destination'].includes(traveler.path), note: 'Only observed events in this simulated window; no record is not proof of no visit.', synthetic: true }));
    return { scenario, group, travelers, events };
  }
  return Object.freeze({ VERSION, MAX_BLOCKS, ZERO, TYPES, SCENARIOS, createLedger, createPlan, verifyChain, canonical, hash, merkleRoot, headerHash, clone });
});
