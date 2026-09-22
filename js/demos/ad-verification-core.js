/* Educational audit layer. Real signatures; fictional data and local participants. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AdVerificationCore = api;
})(globalThis, function () {
  'use strict';
  const VERSION = 5;
  const LIMIT = 192;
  const BATCH_SIZE = 4;
  const ZERO = '0'.repeat(64);
  const RULE = Object.freeze({ id: 'demo-window-v1', days: 30 });
  const ROLES = Object.freeze({
    campaign: 'advertiser', purchase: 'agency', ad: 'ad-platform', website: 'analytics',
    visit: 'location-partner', attribution: 'attribution-service', end: 'location-partner',
    report: 'attribution-service', correction: 'attribution-service'
  });
  const SOURCES = Object.freeze({
    advertiser: 'Advertiser', agency: 'Media agency', 'ad-platform': 'Ad platform',
    analytics: 'Website analytics', 'location-partner': 'Location partner',
    'attribution-service': 'Attribution service'
  });
  const TITLES = Object.freeze({
    campaign: 'Campaign opened', purchase: 'Media authorized', ad: 'Ad exposure reported',
    website: 'Website session reported', visit: 'Destination visit reported',
    attribution: 'Attribution decision', end: 'Observation window closed',
    report: 'Campaign report signed', correction: 'Correction appended'
  });
  const COPIES = Object.freeze(['Advertiser', 'Agency', 'Measurement partner']);
  const VALIDATORS = COPIES.map((_, index) => 'validator-' + index);
  const SCENARIOS = Object.freeze({
    mixed: 'Mixed outcomes', none: 'Neither visit', website: 'Website only',
    destination: 'Destination only', both: 'Website + destination', late: 'Outside the window'
  });
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const encoder = new TextEncoder();
  const hex = (value) => Array.from(new Uint8Array(value), (n) => n.toString(16).padStart(2, '0')).join('');
  const random = () => hex(globalThis.crypto.getRandomValues(new Uint8Array(24)));
  const isHash = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
  const object = (value) => value && Object.getPrototypeOf(value) === Object.prototype;
  const exact = (value, keys) => object(value) && Object.keys(value).sort().join(',') === keys.slice().sort().join(',');
  function canonical(value, depth = 0) {
    if (depth > 12) throw new TypeError('JSON is too deeply nested.');
    if (value === null || ['string', 'boolean'].includes(typeof value)) return JSON.stringify(value);
    if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map((v) => canonical(v, depth + 1)).join(',') + ']';
    if (object(value)) return '{' + Object.keys(value).sort().map((key) => JSON.stringify(key) + ':' + canonical(value[key], depth + 1)).join(',') + '}';
    throw new TypeError('Only finite JSON values are supported.');
  }
  function cryptoAPI() {
    if (!globalThis.crypto?.subtle) throw new Error('This demo needs Web Crypto on HTTPS or localhost.');
    return globalThis.crypto.subtle;
  }
  const hash = async (text) => hex(await cryptoAPI().digest('SHA-256', encoder.encode(text)));
  const receiptMessage = (r) => 'CAMPAIGN/RECEIPT/v5\n' + canonical(r);
  const headerHash = (h) => hash('CAMPAIGN/BLOCK/v5\n' + canonical(h));
  const approvalMessage = (h) => 'CAMPAIGN/APPROVAL/v5\n' + h;
  const evidenceHash = (e) => hash('CAMPAIGN/EVIDENCE/v5\n' + canonical(e));
  const algorithm = { name: 'ECDSA', hash: 'SHA-256' };
  const sign = async (key, message) => hex(await cryptoAPI().sign(algorithm, key, encoder.encode(message)));
  async function signatureValid(key, signature, message) {
    try {
      if (!/^[a-f0-9]{128}$/.test(signature)) return false;
      return await cryptoAPI().verify(algorithm, key,
        Uint8Array.from(signature.match(/../g), (v) => parseInt(v, 16)), encoder.encode(message));
    } catch (_) { return false; }
  }
  async function merkleRoot(records) {
    if (!Array.isArray(records) || !records.length || records.length > BATCH_SIZE) throw new Error('Invalid batch size.');
    let level = await Promise.all(records.map((r) => hash('CAMPAIGN/LEAF/v5\n' + canonical(r))));
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i < level.length; i += 2) next.push(await hash('CAMPAIGN/PAIR/v5\n' + level[i] + (level[i + 1] || level[i])));
      level = next;
    }
    return level[0];
  }
  const receipts = (blocks) => blocks.flatMap((block) => block.records.map((record) => record.receipt));
  function totals(records) {
    const result = { exposures: 0, websites: 0, visits: 0, attributed: 0, corrections: 0 };
    const decisions = new Map();
    for (const r of records) {
      if (r.type === 'ad') result.exposures += r.data.count;
      if (r.type === 'website') result.websites += r.data.count;
      if (r.type === 'visit') result.visits += r.data.count;
      if (r.type === 'attribution') decisions.set(r.id, r.data.credited);
      if (r.type === 'correction') {
        decisions.set(r.refs[0], r.data.credited);
        result.corrections += 1;
      }
    }
    result.attributed = [...decisions.values()].filter(Boolean).length;
    return result;
  }
  function decision(ad, visit) {
    if (!ad || !visit || ad.type !== 'ad' || visit.type !== 'visit' ||
        ad.traveler !== visit.traveler || ad.campaign !== visit.campaign) throw new Error('Evidence cannot be matched.');
    if (![ad.day, visit.day].every((day) => Number.isSafeInteger(day) && day >= 0)) throw new Error('Invalid occurrence day.');
    const days = visit.day - ad.day;
    return { credited: days >= 0 && days <= RULE.days, days, rule: RULE.id, window: RULE.days };
  }
  function receiptPolicy(r, previous, block, indexInBlock, blockPrefix) {
    const find = (id) => previous.find((old) => old.id === id);
    if (!exact(r, ['version', 'chain', 'campaign', 'id', 'type', 'source', 'evidenceDigest', 'refs', 'data']) ||
        r.version !== VERSION || !/^receipt-[a-f0-9]{48}$/.test(r.id) || !Object.hasOwn(ROLES, r.type) ||
        r.source !== ROLES[r.type] || !isHash(r.evidenceDigest) || !Array.isArray(r.refs) || r.refs.length > 8 ||
        new Set(r.refs).size !== r.refs.length || !r.refs.every((id) => typeof id === 'string' && find(id)) ||
        previous.some((old) => old.id === r.id) || !object(r.data)) return false;
    if (!previous.length) return r.type === 'campaign' && !r.refs.length &&
      exact(r.data, ['rule', 'window']) && r.data.rule === RULE.id && r.data.window === RULE.days;
    if (r.type === 'campaign') return false;
    if (r.type === 'purchase') return previous.length === 1 && !r.refs.length &&
      exact(r.data, ['authorized']) && r.data.authorized === true;
    if (previous.length < 2) return false;
    if (['ad', 'website', 'visit'].includes(r.type)) return !r.refs.length && exact(r.data, ['count']) && r.data.count === 1;
    if (r.type === 'attribution') return r.refs.length === 2 && find(r.refs[0]).type === 'ad' &&
      find(r.refs[1]).type === 'visit' && exact(r.data, ['credited', 'rule', 'window']) &&
      typeof r.data.credited === 'boolean' && r.data.rule === RULE.id && r.data.window === RULE.days &&
      !previous.some((old) => old.type === 'attribution' && old.refs[1] === r.refs[1]);
    if (r.type === 'end') return !r.refs.length && exact(r.data, ['closed']) && r.data.closed === true;
    if (r.type === 'correction') return r.refs.length === 1 && find(r.refs[0]).type === 'attribution' &&
      find(r.refs[0]).data.credited === true && !previous.some((old) => old.type === 'correction' && old.refs[0] === r.refs[0]) &&
      exact(r.data, ['credited', 'reason']) && r.data.credited === false && r.data.reason === 'duplicate-visit';
    if (r.type === 'report') return block.records.length === 1 && indexInBlock === 0 && !r.refs.length &&
      exact(r.data, ['throughBlock', 'throughHash', 'totals']) &&
      r.data.throughBlock === blockPrefix.length && r.data.throughHash === (blockPrefix.at(-1)?.hash || ZERO) &&
      canonical(r.data.totals) === canonical(totals(previous));
    return false;
  }
  async function verifyProof(proof) {
    const failures = [];
    try {
      if (!exact(proof, ['version', 'blocks', 'trust']) || proof.version !== VERSION ||
          !Array.isArray(proof.blocks) || proof.blocks.length > LIMIT ||
          !exact(proof.trust, ['chain', 'campaign', 'length', 'head', 'keys']) ||
          !Number.isSafeInteger(proof.trust.length) || proof.trust.length < 0 ||
          proof.trust.length > LIMIT || !isHash(proof.trust.head) ||
          typeof proof.trust.chain !== 'string' || typeof proof.trust.campaign !== 'string') throw new Error('Malformed proof.');
      const { blocks, trust } = proof;
      const keys = {};
      for (const id of [...new Set(Object.values(ROLES)), ...VALIDATORS]) keys[id] = await cryptoAPI().importKey('jwk', trust.keys[id], { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
      const history = [];
      for (let i = 0; i < blocks.length; i += 1) {
        const block = blocks[i];
        const check = (valid, reason) => { if (!valid) failures.push({ block: i + 1, reason }); };
        if (!exact(block, ['header', 'records', 'hash', 'approvals']) ||
            !exact(block.header, ['version', 'chain', 'height', 'previous', 'root', 'recordedAt']) ||
            !Array.isArray(block.records) || block.records.length < 1 || block.records.length > BATCH_SIZE ||
            !Array.isArray(block.approvals)) throw new Error('Malformed block.');
        const h = block.header;
        check(h.version === VERSION && h.chain === trust.chain && h.height === i + 1 &&
          typeof h.recordedAt === 'string' && Number.isFinite(Date.parse(h.recordedAt)), 'Block header');
        check(h.previous === (blocks[i - 1]?.hash || ZERO), 'Previous block link');
        check(await merkleRoot(block.records) === h.root, 'Batch fingerprint');
        check(await headerHash(h) === block.hash, 'Block fingerprint');
        for (let j = 0; j < block.records.length; j += 1) {
          const signed = block.records[j];
          if (!exact(signed, ['receipt', 'signature'])) throw new Error('Malformed signed receipt.');
          const r = signed.receipt;
          check(r.chain === trust.chain && r.campaign === trust.campaign &&
            receiptPolicy(r, history, block, j, blocks.slice(0, i)), 'Receipt policy');
          check(await signatureValid(keys[r.source], signed.signature, receiptMessage(r)), 'Provider signature');
          history.push(r);
        }
        check(block.approvals.length === VALIDATORS.length &&
          block.approvals.every((a) => exact(a, ['validator', 'signature']) && VALIDATORS.includes(a.validator)) &&
          new Set(block.approvals.map((a) => a.validator)).size === VALIDATORS.length &&
          (await Promise.all(block.approvals.map((a) => signatureValid(keys[a.validator], a.signature, approvalMessage(block.hash))))).every(Boolean), 'Ledger approvals');
      }
      if (history.length > LIMIT) failures.push({ block: 0, reason: 'Receipt limit' });
      if (blocks.length !== trust.length || (blocks.at(-1)?.hash || ZERO) !== trust.head) failures.push({ block: 0, reason: 'Original checkpoint' });
      return { valid: failures.length === 0, failures, totals: failures.length ? null : totals(history) };
    } catch (_) { return { valid: false, failures: [...failures, { block: 0, reason: 'Malformed proof or unavailable trusted key' }], totals: null }; }
  }
  async function createSession() {
    const privateKeys = {};
    const keys = {};
    for (const id of [...new Set(Object.values(ROLES)), ...VALIDATORS]) {
      const pair = await cryptoAPI().generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
      privateKeys[id] = pair.privateKey;
      keys[id] = await cryptoAPI().exportKey('jwk', pair.publicKey);
    }
    const chain = 'local-' + random();
    const campaign = 'CV-DEMO';
    const blocks = [];
    // Private provider evidence and the UI's fictional associations are NOT in public receipts.
    const evidence = new Map();
    const issued = new Map();
    const hidden = new Set();
    const tickets = new WeakMap();
    const replicas = COPIES.map((name) => ({ name, blocks: [], online: true, status: 'Up to date' }));
    let disposed = false;
    let replicaRevision = 0;
    const live = () => { if (disposed) throw new Error('This session has ended.'); };
    const snapshot = () => ({ version: VERSION, blocks: clone(blocks), trust: { chain, campaign, length: blocks.length, head: blocks.at(-1)?.hash || ZERO, keys: clone(keys) } });
    async function issue(type, body, data, refs = []) {
      live();
      if (!Object.hasOwn(ROLES, type) || issued.size >= LIMIT) throw new Error('Receipt limit reached.');
      const id = 'receipt-' + random();
      const packet = { salt: random(), body: { ...clone(body), campaign, type } };
      const receipt = { version: VERSION, chain, campaign, id, type, source: ROLES[type], evidenceDigest: await evidenceHash(packet), refs: refs.slice(), data: clone(data) };
      const signed = { receipt, signature: await sign(privateKeys[receipt.source], receiptMessage(receipt)) };
      live(); evidence.set(id, packet); issued.set(id, signed);
      return clone(signed);
    }
    async function observe(type, traveler, day, prior = {}, receiptTime = 0) {
      if (!['ad', 'website', 'visit', 'attribution', 'end'].includes(type) || !/^T\d{3,4}$/.test(traveler) || !Number.isSafeInteger(day)) throw new Error('Invalid fictional observation.');
      const body = { traveler, day, receiptTime };
      if (type === 'website') body.page = '/hiking-guides';
      if (type === 'visit') body.place = 'Cedar Valley';
      if (type === 'attribution') {
        const ad = evidence.get(prior.ad)?.body;
        const visit = evidence.get(prior.visit)?.body;
        const result = decision(ad, visit);
        if (ad.traveler !== traveler) throw new Error('Different traveler evidence.');
        return issue(type, { ...body, calculation: result }, { credited: result.credited, rule: RULE.id, window: RULE.days }, [prior.ad, prior.visit]);
      }
      return issue(type, body, type === 'end' ? { closed: true } : { count: 1 });
    }
    async function prepare(records) {
      live();
      if (!Array.isArray(records) || !records.length || records.length > BATCH_SIZE) throw new Error('Invalid batch.');
      const baseLength = blocks.length;
      const revision = replicaRevision;
      const safe = clone(records);
      const header = { version: VERSION, chain, height: baseLength + 1, previous: blocks.at(-1)?.hash || ZERO,
        root: await merkleRoot(safe), recordedAt: new Date().toISOString() };
      const digest = await headerHash(header);
      const approvals = await Promise.all(VALIDATORS.map(async (validator) => ({ validator, signature: await sign(privateKeys[validator], approvalMessage(digest)) })));
      const block = { header, records: safe, hash: digest, approvals };
      const proof = snapshot(); proof.blocks.push(block); proof.trust.length += 1; proof.trust.head = digest;
      const checked = await verifyProof(proof);
      live();
      if (blocks.length !== baseLength || !checked.valid) throw new Error('Batch verification failed.');
      const nextCopies = [];
      for (let index = 0; index < replicas.length; index += 1) {
        const replica = replicas[index];
        const prefixStatus = await checkCopy(replica.blocks);
        const own = clone(replica.blocks);
        if (replica.online && prefixStatus === 'Up to date') own.push(clone(block));
        const nextStatus = own.length === proof.blocks.length && prefixStatus === 'Up to date'
          ? ((await verifyProof({ ...proof, blocks: own })).valid ? 'Up to date' : 'Mismatch')
          : prefixStatus === 'Mismatch' ? 'Mismatch' : 'Behind';
        nextCopies.push({ blocks: own, status: nextStatus });
      }
      live();
      if (revision !== replicaRevision || baseLength !== blocks.length) throw new Error('Participant state changed.');
      const ticket = Object.freeze({ block: clone(block) });
      tickets.set(ticket, { block, baseLength, revision, nextCopies });
      return ticket;
    }
    async function checkCopy(own) {
      const proof = snapshot();
      proof.blocks = clone(own);
      // A valid prefix is behind, not current. Its anchor comes from canonical history.
      proof.trust.length = own.length;
      proof.trust.head = blocks[own.length - 1]?.hash || ZERO;
      const verified = await verifyProof(proof);
      return !verified.valid ? 'Mismatch' : own.length < blocks.length ? 'Behind' : 'Up to date';
    }
    function commit(ticket) {
      live(); const item = tickets.get(ticket);
      if (!item || item.baseLength !== blocks.length || item.revision !== replicaRevision ||
          item.block.header.previous !== (blocks.at(-1)?.hash || ZERO)) throw new Error('Stale or unverified batch.');
      tickets.delete(ticket);
      blocks.push(item.block);
      // All cryptographic checks finished in prepare. This state change is synchronous.
      replicas.forEach((replica, index) => Object.assign(replica, item.nextCopies[index]));
      return clone(item.block);
    }
    async function append(records) { return commit(await prepare(records)); }
    async function report() {
      const proof = snapshot();
      const data = { throughBlock: blocks.length, throughHash: proof.trust.head, totals: totals(receipts(blocks)) };
      return append([await issue('report', { coveredReceipts: receipts(blocks).map((r) => r.id) }, data)]);
    }
    async function correct(targetId) {
      const all = receipts(blocks); const target = all.find((r) => r.id === targetId);
      if (!target || target.type !== 'attribution' || !target.data.credited || all.some((r) => r.type === 'correction' && r.refs[0] === targetId)) throw new Error('Choose an uncorrected credited decision.');
      return append([await issue('correction', { reason: 'Example provider identified a duplicate visit', target: targetId }, { credited: false, reason: 'duplicate-visit' }, [targetId])]);
    }
    async function audit(id) {
      const proof = snapshot(); const checked = await verifyProof(proof);
      const record = receipts(blocks).find((r) => r.id === id);
      if (!checked.valid || !record) return { record: false, evidence: 'not-checked' };
      const packet = evidence.get(id);
      if (!packet || hidden.has(id)) return { record: true, evidence: 'unavailable' };
      if (await evidenceHash(packet) !== record.evidenceDigest) return { record: true, evidence: 'mismatch' };
      let calculation = null;
      if (record.type === 'attribution') {
        const packets = record.refs.map((ref) => evidence.get(ref));
        if (record.refs.some((ref, index) => hidden.has(ref) || !packets[index])) return { record: true, evidence: 'unavailable' };
        for (let index = 0; index < packets.length; index += 1) {
          const ref = receipts(blocks).find((r) => r.id === record.refs[index]);
          if (await evidenceHash(packets[index]) !== ref.evidenceDigest) return { record: true, evidence: 'mismatch' };
        }
        calculation = decision(packets[0].body, packets[1].body);
        if (calculation.credited !== record.data.credited || calculation.window !== record.data.window || calculation.rule !== record.data.rule) return { record: true, evidence: 'mismatch' };
      }
      return { record: true, evidence: 'checked', body: clone(packet.body), calculation,
        caveat: 'Checks consistency with provider evidence, not whether the underlying observation or causal claim is true.' };
    }
    const describe = (id) => {
      const signed = issued.get(id); const packet = evidence.get(id);
      return signed ? { signed: clone(signed), traveler: packet?.body.traveler || null, hidden: hidden.has(id) } : null;
    };
    async function manageCopy(index, action) {
      live(); const copy = replicas[index]; if (!copy) throw new Error('Unknown participant.');
      if (action === 'pause') copy.online = false;
      else if (action === 'alter') {
        if (!copy.blocks.length) throw new Error('No records to alter yet.');
        const record = copy.blocks.at(-1).records[0].receipt;
        record.data = { ...record.data, unauthorizedChange: true };
      } else if (action === 'restore') {
        if (!(await verifyProof(snapshot())).valid) throw new Error('The source history did not verify.');
        copy.blocks = clone(blocks); copy.online = true;
      } else throw new Error('Unknown copy action.');
      replicaRevision += 1;
      copy.status = await checkCopy(copy.blocks);
      return copy.status;
    }
    const copyStates = () => replicas.map(({ name, status, online, blocks: own }) => ({ name, status, online, length: own.length }));
    const withhold = (id, value) => { if (!evidence.has(id)) throw new Error('Unknown evidence.'); if (value) hidden.add(id); else hidden.delete(id); };
    function dispose() { disposed = true; Object.keys(privateKeys).forEach((key) => delete privateKeys[key]); }
    return Object.freeze({ snapshot, observe, issue, prepare, commit, append, report, correct, audit, describe, manageCopy, copyStates, withhold, dispose });
  }
  function traveler(scenario, number) {
    if (!Object.hasOwn(SCENARIOS, scenario) || !Number.isSafeInteger(number) || number < 1 || number > 9999) throw new Error('Invalid traveler.');
    const path = scenario === 'mixed' ? ['both', 'none', 'website', 'destination', 'late', 'website', 'both', 'none'][(number - 1) % 8] : scenario;
    const id = 'T' + String(number).padStart(3, '0');
    const day = number * 2;
    const stages = [{ type: 'ad', day, at: 300 }];
    if (['both', 'website', 'late'].includes(path)) stages.push({ type: 'website', day: day + 1, at: 3400 + number % 3 * 400 });
    if (['both', 'destination', 'late'].includes(path)) {
      stages.push({ type: 'visit', day: day + (path === 'late' ? 35 : 6), at: 7800 + number % 3 * 600 });
      stages.push({ type: 'attribution', day: day + 36, at: 10300 + number % 3 * 600 });
    }
    stages.push({ type: 'end', day: day + 40, at: 13000 + number % 4 * 1800 });
    return { id, number, path, stages };
  }
  return Object.freeze({ VERSION, LIMIT, BATCH_SIZE, ZERO, RULE, ROLES, SOURCES, TITLES, COPIES, SCENARIOS,
    clone, canonical, hash, merkleRoot, headerHash, evidenceHash, receipts, totals, decision, verifyProof, createSession, traveler });
});
