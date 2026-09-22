/* Educational, single-browser permissioned blockchain. No network or private-key export. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AdVerificationCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const ACTORS = Object.freeze(['advertiser', 'agency', 'publisher', 'measurement']);
  const VALIDATORS = Object.freeze(['campaign-validator', 'delivery-validator', 'audit-validator']);
  const ZERO_HASH = '0'.repeat(64);
  const encoder = new TextEncoder();
  const ALGORITHM = Object.freeze({ name: 'ECDSA', hash: 'SHA-256' });
  const clone = (value) => JSON.parse(JSON.stringify(value));

  function subtle() {
    if (!globalThis.crypto || !globalThis.crypto.subtle) {
      throw new Error('This demo needs a browser with Web Crypto on HTTPS or localhost.');
    }
    return globalThis.crypto.subtle;
  }

  // This demo accepts JSON only. Sorting keys makes serialization deterministic.
  function canonical(value) {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
    if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
    if (value && Object.getPrototypeOf(value) === Object.prototype) {
      return '{' + Object.keys(value).sort().map((key) => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
    }
    throw new TypeError('Only finite JSON values are supported.');
  }

  const hex = (buffer) => Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, '0')).join('');
  function unhex(value, bytes) {
    if (typeof value !== 'string' || !/^[a-f0-9]+$/.test(value) || value.length !== bytes * 2) {
      throw new TypeError('Malformed cryptographic value.');
    }
    return Uint8Array.from(value.match(/../g), (pair) => parseInt(pair, 16));
  }
  async function hash(text) { return hex(await subtle().digest('SHA-256', encoder.encode(text))); }
  async function sign(key, text) { return hex(await subtle().sign(ALGORITHM, key, encoder.encode(text))); }
  async function verify(key, signature, text) {
    try { return await subtle().verify(ALGORITHM, key, unhex(signature, 64), encoder.encode(text)); }
    catch (_) { return false; }
  }
  const eventMessage = (event) => 'AD-VERIFY/EVENT/v1\n' + canonical(event);
  const blockMessage = (blockHash) => 'AD-VERIFY/BLOCK/v1\n' + blockHash;
  const headerHash = (header) => hash('AD-VERIFY/HEADER/v1\n' + canonical(header));

  async function merkleRoot(transactions) {
    if (!Array.isArray(transactions) || !transactions.length || transactions.length > 64) {
      throw new TypeError('Expected 1 to 64 transactions.');
    }
    let level = await Promise.all(transactions.map((tx) => hash('AD-VERIFY/LEAF/v1\n' + canonical(tx))));
    while (level.length > 1) {
      const next = [];
      for (let index = 0; index < level.length; index += 2) {
        next.push(await hash('AD-VERIFY/BRANCH/v1\n' + level[index] + (level[index + 1] || level[index])));
      }
      level = next;
    }
    return level[0];
  }

  async function createIdentity() {
    const keys = await subtle().generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
    return { privateKey: keys.privateKey, publicKey: await subtle().exportKey('jwk', keys.publicKey) };
  }

  async function createDemo() {
    const identities = {};
    for (const id of [...ACTORS, ...VALIDATORS]) identities[id] = await createIdentity();
    const chainId = 'cedar-valley-' + hex(globalThis.crypto.getRandomValues(new Uint8Array(12)));
    const events = [
      { name: 'Campaign created', reportedAt: '2027-04-25T10:00:12.000Z', data: { destination: 'Cedar Valley Tourism', campaign: 'A little closer to nature', creative: 'cedar-valley-spring-display-v1' } },
      { name: 'Placement authorized', reportedAt: '2027-04-25T10:00:15.000Z', data: { agency: 'Example Media Agency', publisher: 'Example Travel Journal', creative: 'cedar-valley-spring-display-v1' } },
      { name: 'Delivery reported', reportedAt: '2027-04-25T10:00:21.000Z', data: { publisher: 'Example Travel Journal', creative: 'cedar-valley-spring-display-v1', impressions: 10000 } },
      { name: 'Measurement recorded', reportedAt: '2027-04-25T10:00:28.000Z', data: { provider: 'Example Measurement', reportedImpressions: 10000, note: 'Synthetic measurement, not proof of human attention.' } }
    ];
    const blocks = [];
    for (let index = 0; index < events.length; index += 1) {
      const actor = ACTORS[index];
      const event = { version: 1, chainId, campaignId: 'cedar-valley-spring-2027', step: index + 1, actor, ...events[index] };
      const transactions = [{ event, signature: await sign(identities[actor].privateKey, eventMessage(event)) }];
      const header = {
        version: 1, chainId, height: index + 1,
        previousHash: index ? blocks[index - 1].hash : ZERO_HASH,
        merkleRoot: await merkleRoot(transactions), recordedAt: event.reportedAt
      };
      const blockHash = await headerHash(header);
      const approvals = [];
      for (const validator of VALIDATORS) {
        approvals.push({ validator, signature: await sign(identities[validator].privateKey, blockMessage(blockHash)) });
      }
      blocks.push({ header, transactions, hash: blockHash, approvals });
    }
    // The registry and checkpoint are held separately from the editable ledger.
    // All signers are local; this is NOT distributed consensus or external identity proof.
    const trust = {
      chainId, expectedLength: ACTORS.length, headHash: blocks[blocks.length - 1].hash,
      publicKeys: Object.fromEntries(Object.entries(identities).map(([id, identity]) => [id, identity.publicKey]))
    };
    return { blocks, trust }; // Private keys never leave this function.
  }

  function exactKeys(value, keys) {
    return value && Object.getPrototypeOf(value) === Object.prototype &&
      Object.keys(value).sort().join(',') === [...keys].sort().join(',');
  }

  async function verifyChain(blocks, trust) {
    subtle();
    const rows = [];
    const validTrust = trust && typeof trust.chainId === 'string' && trust.expectedLength === ACTORS.length &&
      typeof trust.headHash === 'string' && /^[a-f0-9]{64}$/.test(trust.headHash) && trust.publicKeys;
    if (!validTrust || !Array.isArray(blocks) || blocks.length > 64) {
      return { valid: false, reason: 'Invalid ledger or trust registry.', blocks: rows, lengthValid: false, headValid: false };
    }
    const keys = {};
    try {
      for (const id of [...ACTORS, ...VALIDATORS]) {
        keys[id] = await subtle().importKey('jwk', trust.publicKeys[id], { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
      }
    } catch (_) {
      return { valid: false, reason: 'A trusted public key is missing or invalid.', blocks: rows, lengthValid: false, headValid: false };
    }
    let ancestryValid = true;
    for (let index = 0; index < blocks.length; index += 1) {
      const block = blocks[index];
      const checks = { structure: false, eventSignature: false, merkleRoot: false, blockHash: false, previousHash: false, approvals: false };
      let computedHash = '';
      let computedRoot = '';
      try {
        const header = block.header;
        const tx = block.transactions && block.transactions[0];
        const event = tx && tx.event;
        checks.structure = Boolean(index < ACTORS.length &&
          exactKeys(block, ['header', 'transactions', 'hash', 'approvals']) &&
          exactKeys(header, ['version', 'chainId', 'height', 'previousHash', 'merkleRoot', 'recordedAt']) &&
          Array.isArray(block.transactions) && block.transactions.length === 1 &&
          exactKeys(tx, ['event', 'signature']) &&
          exactKeys(event, ['version', 'chainId', 'campaignId', 'step', 'actor', 'name', 'reportedAt', 'data']) &&
          event.version === 1 && header.version === 1 && header.chainId === trust.chainId && event.chainId === trust.chainId &&
          header.height === index + 1 && event.step === index + 1 && event.actor === ACTORS[index] &&
          event.campaignId === 'cedar-valley-spring-2027' && typeof event.name === 'string' &&
          typeof event.reportedAt === 'string' && Number.isFinite(Date.parse(event.reportedAt)) &&
          header.recordedAt === event.reportedAt && /^[a-f0-9]{64}$/.test(block.hash) &&
          /^[a-f0-9]{64}$/.test(header.previousHash) && /^[a-f0-9]{64}$/.test(header.merkleRoot));
        if (checks.structure) {
          computedRoot = await merkleRoot(block.transactions);
          computedHash = await headerHash({ ...header, merkleRoot: computedRoot });
          checks.eventSignature = await verify(keys[ACTORS[index]], tx.signature, eventMessage(event));
          checks.merkleRoot = computedRoot === header.merkleRoot;
          checks.blockHash = await headerHash(header) === block.hash;
          checks.previousHash = header.previousHash === (index ? blocks[index - 1].hash : ZERO_HASH);
          const approvals = block.approvals;
          if (Array.isArray(approvals) && approvals.length === VALIDATORS.length &&
              approvals.every((a) => exactKeys(a, ['validator', 'signature']) && VALIDATORS.includes(a.validator)) &&
              new Set(approvals.map((a) => a.validator)).size === VALIDATORS.length) {
            checks.approvals = (await Promise.all(approvals.map((a) => verify(keys[a.validator], a.signature, blockMessage(block.hash))))).every(Boolean);
          }
        }
      } catch (_) { /* Malformed records fail closed, rather than crashing the UI. */ }
      const localValid = Object.values(checks).every(Boolean);
      const state = !localValid ? 'changed' : !ancestryValid ? 'dependent' : 'verified';
      rows.push({ index, state, localValid, ancestryValid, checks, computedHash, computedRoot });
      ancestryValid = ancestryValid && localValid;
    }
    const lengthValid = blocks.length === trust.expectedLength;
    const head = blocks[blocks.length - 1];
    const headValid = Boolean(head && head.hash === trust.headHash);
    const valid = ancestryValid && lengthValid && headValid;
    return { valid, reason: valid ? 'All four records match their signed history.' : 'A record, signature, link, or checkpoint does not match.', blocks: rows, lengthValid, headValid };
  }

  function simulateChange(blocks) {
    const changed = clone(blocks);
    changed[2].transactions[0].event.data.impressions = 12500;
    return changed;
  }

  return Object.freeze({ createDemo, verifyChain, simulateChange, canonical, hash, merkleRoot, headerHash, clone });
});
