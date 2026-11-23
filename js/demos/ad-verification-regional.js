/* Local regional attribution exercise. Only the signed aggregate ledger is exportable. */
(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./ad-verification-core'));
  else root.AdVerificationRegional = factory(root.AdVerificationCore);
})(globalThis, function (core) {
  'use strict';
  const { canonical, clone, hash } = core;
  const VERSION = 'regional-aggregate-v1';
  const REGION = 'High Country';
  const MIN_VISIBLE_PERCENT = 50;
  const MIN_CONTINUOUS_SECONDS = 1;
  const RULE = 'confirmed-view/min-' + MIN_VISIBLE_PERCENT + 'pct/continuous-' + MIN_CONTINUOUS_SECONDS +
    's/registered-destination/0-30-days/unique-per-campaign/v1';
  const ZERO = '0'.repeat(64);
  const DAY = 1440;
  const VALIDATORS = ['network', 'reviewer'];
  const CURVE = { name: 'ECDSA', namedCurve: 'P-256' };
  const SIGNATURE = { name: 'ECDSA', hash: 'SHA-256' };
  const HMAC = { name: 'HMAC', hash: 'SHA-256', length: 256 };
  const encoder = new TextEncoder();
  const SOURCES = {
    'field-guide': { type: 'view', label: 'Field Guide app' },
    'weekend-reader': { type: 'view', label: 'Weekend Reader app' },
    downtown: { type: 'stop', label: 'Downtown district', source: 'Downtown gateway' },
    trailhead: { type: 'stop', label: 'Regional trailhead', source: 'Trailhead gateway' },
    lodging: { type: 'stop', label: 'Lodging district', source: 'Lodging gateway' },
    interchange: { type: 'stop', label: 'Transit interchange', source: 'Interchange gateway', transit: true }
  };
  const CASES = [
    ['three-stops', 'Three-stop traveler'], ['day-30', 'Day-30 arrival'],
    ['missing-view', 'Missing app receipt'], ['transit', 'Transit passenger'],
    ['late', 'Late arrival'], ['unlinked', 'Unlinked pass']
  ];
  const BATCH_COUNT = 3;
  const zeroTotals = () => ({ views: 0, observations: 0, matched: 0 });
  const isHex = (value, length) => typeof value === 'string' && value.length === length && /^[a-f0-9]+$/.test(value);
  const toHex = (buffer) => Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, '0')).join('');
  const fromHex = (value) => Uint8Array.from(value.match(/../g), (pair) => parseInt(pair, 16));
  const isInt = (value, min, max) => Number.isSafeInteger(value) && value >= min && value <= max;
  const exact = (value, keys) => value !== null && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));
  const validCampaign = (value) => typeof value === 'string' && /^regional-[a-f0-9]{24}$/.test(value);
  const message = (purpose, value) => 'REGIONAL/' + purpose + '/v1\n' + canonical(value);

  function cryptoAPI() {
    if (!globalThis.crypto?.subtle || !globalThis.crypto.getRandomValues) {
      throw new Error('The regional demo needs Web Crypto on HTTPS or localhost.');
    }
    return globalThis.crypto.subtle;
  }
  function randomHex(size) {
    return toHex(globalThis.crypto.getRandomValues(new Uint8Array(size)));
  }
  async function sign(key, text) {
    return toHex(await cryptoAPI().sign(SIGNATURE, key, encoder.encode(text)));
  }
  async function verify(key, signature, text) {
    return isHex(signature, 128) && cryptoAPI().verify(SIGNATURE, key, fromHex(signature), encoder.encode(text));
  }
  async function makeKeys(names) {
    const pairs = await Promise.all(names.map(() => cryptoAPI().generateKey(CURVE, false, ['sign', 'verify'])));
    return Object.fromEntries(names.map((name, index) => [name, pairs[index]]));
  }
  async function encodeValidators(keys) {
    const entries = await Promise.all(VALIDATORS.map(async (name) =>
      [name, toHex(await cryptoAPI().exportKey('raw', keys[name].publicKey))]));
    return Object.fromEntries(entries);
  }
  async function deriveCampaignToken(key, campaign, credential, authorized) {
    if (authorized !== true) return null;
    if (!validCampaign(campaign) || !isHex(credential, 64) || key?.type !== 'secret' ||
        key.extractable || key.algorithm.name !== 'HMAC' || key.algorithm.hash.name !== 'SHA-256') {
      throw new Error('Invalid private enrollment or campaign matching key.');
    }
    const input = message('join', { campaign, credential });
    return toHex(await cryptoAPI().sign('HMAC', key, encoder.encode(input)));
  }

  function validStatement(statement, campaign) {
    if (!statement || !Object.hasOwn(SOURCES, statement.issuer)) return false;
    const source = SOURCES[statement.issuer];
    const common = ['campaign', 'region', 'id', 'issuer', 'type', 'day', 'minute', 'authorized', 'token'];
    const fields = source.type === 'view'
      ? ['received', 'viewable', 'creative', 'placement', 'visiblePercent', 'seconds'] : ['dwellMinutes'];
    if (!exact(statement, common.concat(fields)) || statement.campaign !== campaign || statement.region !== REGION ||
        !isHex(statement.id, 32) || statement.type !== source.type || !isInt(statement.day, 0, 36500) ||
        !isInt(statement.minute, 0, DAY - 1) || typeof statement.authorized !== 'boolean' ||
        !(statement.authorized ? isHex(statement.token, 64) : statement.token === null)) return false;
    if (source.type === 'stop') return isInt(statement.dwellMinutes, 1, DAY);
    // The app signs the minimum percentage of the ad area displayed throughout this uninterrupted interval.
    return statement.received === true && statement.viewable === true &&
      typeof statement.creative === 'string' && statement.creative.length > 0 && statement.creative.length <= 80 &&
      typeof statement.placement === 'string' && statement.placement.length > 0 && statement.placement.length <= 80 &&
      Number.isFinite(statement.visiblePercent) && statement.visiblePercent >= MIN_VISIBLE_PERCENT && statement.visiblePercent <= 100 &&
      Number.isFinite(statement.seconds) && statement.seconds >= MIN_CONTINUOUS_SECONDS && statement.seconds <= 3600;
  }

  function withinWindow(view, stop) {
    const elapsed = (stop.day - view.day) * DAY + stop.minute - view.minute;
    return elapsed >= 0 && elapsed <= 30 * DAY;
  }
  function measure(statements) {
    const views = new Map();
    const eligibleStops = [];
    const totals = zeroTotals();
    for (const event of statements) {
      if (event.type === 'view') {
        totals.views += 1;
        if (event.authorized) {
          if (!views.has(event.token)) views.set(event.token, []);
          views.get(event.token).push(event);
        }
      } else {
        totals.observations += 1;
        if (event.authorized && !SOURCES[event.issuer].transit) eligibleStops.push(event);
      }
    }
    const matched = new Set();
    for (const stop of eligibleStops) {
      if (views.get(stop.token)?.some((view) => withinWindow(view, stop))) matched.add(stop.token);
    }
    totals.matched = matched.size;
    return { totals, matched };
  }

  // The private auditor supplies its trusted source directory. No source key is taken from a receipt.
  async function reviewEvidence(campaign, sourceKeys, batches) {
    if (!validCampaign(campaign) || !Array.isArray(batches) || batches.length > BATCH_COUNT) {
      throw new Error('Invalid regional evidence scope.');
    }
    batches = clone(batches);
    sourceKeys = { ...sourceKeys };
    const seen = new Set();
    const statements = [];
    const summaries = [];
    for (const records of batches) {
      if (!Array.isArray(records) || !records.length || records.length > 1000) throw new Error('Invalid source batch.');
      for (const record of records) {
        if (!exact(record, ['statement', 'signature']) || !validStatement(record.statement, campaign)) {
          throw new Error('A source receipt has invalid fields or scope.');
        }
        const event = record.statement;
        if (seen.has(event.id)) throw new Error('Repeated source receipt.');
        if (!Object.hasOwn(sourceKeys, event.issuer) ||
            !(await verify(sourceKeys[event.issuer], record.signature, message('source', event)))) {
          throw new Error('A source signature could not be verified.');
        }
        seen.add(event.id);
        statements.push(event);
      }
      summaries.push({
        fingerprint: await hash(message('private-batch', records)),
        totals: measure(statements).totals
      });
    }
    return summaries;
  }

  function validScope(proof) {
    return exact(proof, ['version', 'campaign', 'region', 'rule', 'validators', 'batches', 'checkpoint']) &&
      proof.version === VERSION && validCampaign(proof.campaign) && proof.region === REGION && proof.rule === RULE &&
      exact(proof.validators, VALIDATORS) &&
      VALIDATORS.every((name) => isHex(proof.validators[name], 130) && proof.validators[name].startsWith('04')) &&
      proof.validators.network !== proof.validators.reviewer && Array.isArray(proof.batches) && proof.batches.length <= BATCH_COUNT;
  }
  function ledgerScope(proof) {
    return { version: proof.version, campaign: proof.campaign, region: proof.region, rule: proof.rule, validators: proof.validators };
  }
  function batchHeader(batch) {
    return { number: batch.number, previousHash: batch.previousHash, fingerprint: batch.fingerprint, totals: batch.totals };
  }
  function validTotals(totals, previous) {
    return exact(totals, ['views', 'observations', 'matched']) &&
      Object.keys(previous).every((key) => isInt(totals[key], previous[key], 3000)) &&
      totals.matched <= Math.min(totals.views, totals.observations) &&
      totals.views + totals.observations > previous.views + previous.observations;
  }
  async function checkpointFor(scope, height, head, key) {
    return { height, head, signature: await sign(key, message('checkpoint', { scope, height, head })) };
  }
  async function verifyProof(proof) {
    try {
      if (!validScope(proof)) return { valid: false };
      proof = clone(proof);
      const scope = ledgerScope(proof);
      const keys = {};
      for (const name of VALIDATORS) {
        keys[name] = await cryptoAPI().importKey('raw', fromHex(scope.validators[name]), CURVE, false, ['verify']);
      }
      let previousHash = ZERO;
      let previousTotals = zeroTotals();
      for (const [index, batch] of proof.batches.entries()) {
        if (!exact(batch, ['number', 'previousHash', 'fingerprint', 'totals', 'hash', 'approvals']) ||
            batch.number !== index + 1 || batch.previousHash !== previousHash || !isHex(batch.fingerprint, 64) ||
            !isHex(batch.hash, 64) || !validTotals(batch.totals, previousTotals) || !exact(batch.approvals, VALIDATORS)) {
          return { valid: false };
        }
        if (await hash(message('block', { scope, header: batchHeader(batch) })) !== batch.hash) return { valid: false };
        for (const name of VALIDATORS) {
          if (!(await verify(keys[name], batch.approvals[name], message('approval', batch.hash)))) return { valid: false };
        }
        previousHash = batch.hash;
        previousTotals = batch.totals;
      }
      const checkpoint = proof.checkpoint;
      if (!exact(checkpoint, ['height', 'head', 'signature']) || checkpoint.height !== proof.batches.length ||
          checkpoint.head !== previousHash) return { valid: false };
      const text = message('checkpoint', { scope, height: checkpoint.height, head: checkpoint.head });
      return { valid: await verify(keys.reviewer, checkpoint.signature, text) };
    } catch (_) {
      return { valid: false };
    }
  }

  // Case labels exist only for the teaching display. The matcher gets signed statements without case labels.
  function fixture(batch) {
    const view = (row, issuer, day) => ({ row, issuer, day });
    const stop = (row, issuer, day, dwellMinutes) => ({ row, issuer, day, dwellMinutes });
    if (batch === 1) return [
      view(0, 'field-guide', 1), view(0, 'weekend-reader', 2), view(1, 'field-guide', 1),
      view(3, 'weekend-reader', 1), view(4, 'field-guide', 1), view(5, 'weekend-reader', 1)
    ];
    if (batch === 2) return [
      stop(0, 'downtown', 8, 44), stop(0, 'trailhead', 9, 68), stop(2, 'lodging', 12, 55),
      stop(3, 'interchange', 4, 12), stop(5, 'downtown', 7, 36)
    ];
    return [stop(0, 'lodging', 10, 50), stop(1, 'trailhead', 31, 39), stop(4, 'downtown', 36, 25)];
  }
  async function issue(entry, campaign, enrollments, matchKey, keys) {
    const { row, issuer, day } = entry;
    const credential = enrollments[row];
    const token = await deriveCampaignToken(matchKey, campaign, credential, credential !== null);
    const type = SOURCES[issuer].type;
    const statement = {
      campaign, region: REGION, id: randomHex(16), issuer, type, day, minute: 720,
      authorized: token !== null, token
    };
    if (type === 'view') {
      const reader = issuer === 'weekend-reader';
      Object.assign(statement, {
        received: true, viewable: true, creative: 'Explore High Country',
        placement: reader ? 'Weekend edition' : 'Trip planner', visiblePercent: reader ? 85 : 100, seconds: reader ? 3 : 4
      });
    } else statement.dwellMinutes = entry.dwellMinutes;
    return { row, record: { statement, signature: await sign(keys[issuer].privateKey, message('source', statement)) } };
  }

  function describe(events, credited) {
    const views = events.filter((e) => e.type === 'view');
    const stops = events.filter((e) => e.type === 'stop');
    if (!events.length) return ['Awaiting receipts', 'No signed evidence has arrived for this example.'];
    if (!stops.length) return ['Awaiting observation', 'The receiving app confirmed a view. No destination observation has arrived.'];
    if (!views.length) return ['Missing signed view', 'A gateway signed a presence observation. No receiving app signed a view for this device.'];
    if (!events.some((e) => e.authorized)) return ['Unlinked', 'The receipts carry no authorized campaign token. The matcher cannot join them.'];
    if (stops.every((e) => SOURCES[e.issuer].transit)) return ['Transit only', 'The registered interchange signed a pass-through. It does not qualify as a destination stop.'];
    if (credited) return ['Matched', 'A signed destination stop followed a confirmed app view within 30 days. This device counts once for the campaign.'];
    return ['Outside window', 'No signed destination stop fell in the 30 days following a confirmed app view.'];
  }
  function makeJourneys(stored) {
    const entries = stored.flat();
    const { matched } = measure(entries.map((item) => item.record.statement));
    return CASES.map(([id, label], index) => {
      const events = entries.filter((item) => item.row === index).map((item) => item.record.statement);
      const viewReceipts = events.filter((e) => e.type === 'view').map((e) => ({
        id: e.id, app: SOURCES[e.issuer].label, day: e.day, minute: e.minute, placement: e.placement, creative: e.creative,
        visiblePercent: e.visiblePercent, seconds: e.seconds, tokenPreview: e.token?.slice(0, 12) || ''
      }));
      const stops = events.filter((e) => e.type === 'stop').map((e) => ({
        id: e.id, label: SOURCES[e.issuer].label, source: SOURCES[e.issuer].source,
        day: e.day, minute: e.minute, dwellMinutes: e.dwellMinutes, tokenPreview: e.token?.slice(0, 12) || ''
      }));
      const token = events.find((event) => event.authorized)?.token;
      const [outcome, reason] = describe(events, matched.has(token));
      return { id, label, views: viewReceipts.length, viewReceipts, stops, outcome, reason,
        tokenPreview: token?.slice(0, 12) || '' };
    });
  }

  async function createSession() {
    cryptoAPI();
    const [sources, validators, matchKey] = await Promise.all([
      makeKeys(Object.keys(SOURCES)), makeKeys(VALIDATORS), cryptoAPI().generateKey(HMAC, false, ['sign'])
    ]);
    const sourceKeys = Object.fromEntries(Object.entries(sources).map(([id, pair]) => [id, pair.publicKey]));
    const scope = {
      version: VERSION, campaign: 'regional-' + randomHex(12), region: REGION, rule: RULE,
      validators: await encodeValidators(validators)
    };
    // Fictional enrolled devices share their credential only with this private simulation. Each source derives its own receipt token.
    const enrollments = CASES.map(([id]) => id === 'unlinked' ? null : randomHex(32));
    let stored = [];
    let ledger = [];
    let rejectedReplays = 0;
    let checkpoint = await checkpointFor(scope, 0, ZERO, validators.reviewer.privateKey);
    let pending;
    const proof = () => clone({ ...scope, batches: ledger, checkpoint });
    function snapshot() {
      return clone({
        completedBatches: ledger.length, totalBatches: BATCH_COUNT, totals: ledger.at(-1)?.totals || zeroTotals(),
        journeys: makeJourneys(stored), batches: ledger.map((b) => ({ ...batchHeader(b), hash: b.hash })),
        rejectedReplays
      });
    }
    async function advance() {
      const number = ledger.length + 1;
      const previousHash = checkpoint.head;
      const incoming = await Promise.all(fixture(number).map((entry) => issue(entry, scope.campaign, enrollments, matchKey, sources)));
      // A simulated retry resends a previously signed gateway receipt in the final batch.
      if (number === 3) incoming.push(stored[1][1]);
      const seen = new Map(stored.flat().map(({ record }) => [record.statement.id, record]));
      const accepted = [];
      let replays = 0;
      for (const item of incoming) {
        const id = item.record.statement.id;
        if (seen.has(id)) {
          if (canonical(seen.get(id)) !== canonical(item.record)) throw new Error('Conflicting source receipt identifier.');
          replays += 1;
        } else {
          seen.set(id, item.record);
          accepted.push(item);
        }
      }
      const draft = stored.concat([accepted]);
      const evidence = draft.map((batch) => batch.map((item) => item.record));
      const latest = (await reviewEvidence(scope.campaign, sourceKeys, evidence)).at(-1);
      const header = { number, previousHash, fingerprint: latest.fingerprint, totals: latest.totals };
      const blockHash = await hash(message('block', { scope, header }));
      const approvals = {};
      for (const name of VALIDATORS) approvals[name] = await sign(validators[name].privateKey, message('approval', blockHash));
      const nextCheckpoint = await checkpointFor(scope, number, blockHash, validators.reviewer.privateKey);
      stored = draft;
      ledger = ledger.concat({ ...header, hash: blockHash, approvals });
      checkpoint = nextCheckpoint;
      rejectedReplays += replays;
      return snapshot();
    }
    function nextBatch() {
      if (pending) return pending;
      if (ledger.length === BATCH_COUNT) return Promise.resolve(snapshot());
      pending = advance().finally(() => { pending = undefined; });
      return pending;
    }
    async function audit(candidate = proof()) {
      // Capture one immutable revision before awaiting source or ledger cryptography.
      const evidence = stored.map((batch) => batch.map((item) => item.record));
      const trustedHead = checkpoint.head;
      try {
        candidate = clone(candidate);
        if (!(await verifyProof(candidate)).valid || canonical(ledgerScope(candidate)) !== canonical(scope) ||
            candidate.checkpoint.head !== trustedHead || candidate.batches.length !== evidence.length) return { valid: false, matched: 0 };
        const expected = await reviewEvidence(scope.campaign, sourceKeys, evidence);
        const valid = expected.every((batch, i) => batch.fingerprint === candidate.batches[i].fingerprint &&
          canonical(batch.totals) === canonical(candidate.batches[i].totals));
        return { valid, matched: valid ? expected.at(-1)?.totals.matched || 0 : 0 };
      } catch (_) {
        return { valid: false, matched: 0 };
      }
    }
    return { nextBatch, snapshot, proof, audit };
  }

  return { createSession, verifyProof, reviewEvidence, deriveCampaignToken };
});
