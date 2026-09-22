/**
 * Educational, single-browser permissioned block ledger. Not a distributed network.
 * Public verification context must be retained separately from the untrusted ledger.
 * No dependencies, storage, network requests, tokens, or exportable private keys.
 */
const NETWORK = 'cedar-valley-ad-demo-v1';
const CAMPAIGN = 'CVT-AUTUMN-2026';
const MAX_BYTES = 65536;
const SIGNING = Object.freeze({ name: 'ECDSA', hash: 'SHA-256' });
const encoder = new TextEncoder();
export const STEPS = Object.freeze([
  Object.freeze({ signer: 'advertiser', title: 'Advertiser', event: 'campaign_created', label: 'Campaign Created', time: '10:00:12' }),
  Object.freeze({ signer: 'agency', title: 'Agency / DSP', event: 'bid_submitted', label: 'Bid Request Sent', time: '10:00:15' }),
  Object.freeze({ signer: 'publisher', title: 'Publisher', event: 'delivery_reported', label: 'Ad Served', time: '10:00:21' }),
  Object.freeze({ signer: 'measurement', title: 'Measurement', event: 'impressions_reported', label: 'Impression Recorded', time: '10:00:28' })
]);
const VALIDATORS = Object.freeze(['demo-auditor-a', 'demo-auditor-b', 'demo-auditor-c']);

function subtle() {
  if (!globalThis.crypto?.subtle) throw new Error('Web Crypto is unavailable. Open this demo over HTTPS or localhost in a current browser.');
  return globalThis.crypto.subtle;
}

/** Deterministic JSON for this deliberately small schema; not a general JCS implementation. */
export function canonical(value, depth = 0) {
  if (depth > 12) throw new Error('Record nesting is too deep.');
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(item => canonical(item, depth + 1)).join(',')}]`;
  if (value && Object.getPrototypeOf(value) === Object.prototype) {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key], depth + 1)}`).join(',')}}`;
  }
  throw new Error('Records must contain only finite JSON values.');
}

function bytes(value) {
  const result = encoder.encode(value);
  if (result.byteLength > MAX_BYTES) throw new Error('Record exceeds the demo size limit.');
  return result;
}
function hex(buffer) {
  return Array.from(new Uint8Array(buffer), value => value.toString(16).padStart(2, '0')).join('');
}
function unhex(value) {
  if (typeof value !== 'string' || !/^[0-9a-f]{128}$/.test(value)) throw new Error('Invalid P-256 signature encoding.');
  return Uint8Array.from(value.match(/../g), pair => Number.parseInt(pair, 16));
}
async function digest(value) {
  return hex(await subtle().digest('SHA-256', bytes(value)));
}
export async function fingerprint(value) {
  return digest(canonical(value));
}
async function sign(key, message) {
  return hex(await subtle().sign(SIGNING, key, bytes(message)));
}
async function checkSignature(key, signature, message) {
  try {
    return await subtle().verify(SIGNING, key, unhex(signature), bytes(message));
  } catch {
    return false;
  }
}
function keyPair() {
  return subtle().generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
}

/** Domain-separated Merkle leaves and nodes; odd final leaves are duplicated. */
export async function merkleRoot(transactions) {
  if (!Array.isArray(transactions) || !transactions.length || transactions.length > 32) throw new Error('A block must contain 1–32 records.');
  let nodes = await Promise.all(transactions.map(tx => digest(`leaf\0${canonical(tx)}`)));
  while (nodes.length > 1) {
    const next = [];
    for (let index = 0; index < nodes.length; index += 2) {
      next.push(await digest(`node\0${nodes[index]}${nodes[index + 1] || nodes[index]}`));
    }
    nodes = next;
  }
  return nodes[0];
}
export function blockHash(header) {
  return digest(`block\0${canonical(header)}`);
}

/** Four illustrative batch records, with three LOCAL approval keys per block. */
export async function createDemo() {
  const pairs = await Promise.all([...STEPS, ...VALIDATORS].map(() => keyPair()));
  const sessionId = globalThis.crypto.randomUUID();
  const genesisHash = await fingerprint({ network: NETWORK, sessionId, genesis: true });
  const creativeHash = await fingerprint({ destination: 'Cedar Valley Tourism', creative: 'Take the scenic route.', format: 'display' });
  const chain = [];
  let previousHash = genesisHash;
  for (const [index, step] of STEPS.entries()) {
    const payload = {
      id: `${sessionId}:${index + 1}`,
      campaignId: CAMPAIGN,
      signer: step.signer,
      event: step.event,
      timestamp: `2026-09-22T${step.time}.000Z`,
      details: {
        destination: 'Cedar Valley Tourism',
        publisher: 'cedar-trails.example',
        creativeHash,
        ...(index >= 2 ? { impressions: 12500 } : { format: 'Display campaign' }),
        simulated: true
      }
    };
    const transactions = [{ payload, signature: await sign(pairs[index].privateKey, `event\0${canonical(payload)}`) }];
    const header = { version: 1, network: NETWORK, sessionId, height: index + 1, previousHash, merkleRoot: await merkleRoot(transactions) };
    const hash = await blockHash(header);
    const approvals = await Promise.all(VALIDATORS.map(async (validator, offset) => ({
      validator,
      signature: await sign(pairs[STEPS.length + offset].privateKey, `approval\0${hash}`)
    })));
    chain.push({ header, transactions, hash, approvals });
    previousHash = hash;
  }
  // Private CryptoKeys are intentionally not returned or serialized.
  const trust = Object.freeze({
    network: NETWORK, sessionId, genesisHash, headHash: previousHash, length: STEPS.length,
    signers: Object.freeze(Object.fromEntries(STEPS.map((step, index) => [step.signer, pairs[index].publicKey]))),
    validators: Object.freeze(Object.fromEntries(VALIDATORS.map((id, index) => [id, pairs[STEPS.length + index].publicKey])))
  });
  return { chain, trust };
}

/** Fail closed on malformed data, reordered/deleted blocks, wrong keys or approvals. */
export async function verifyChain(chain, trust) {
  subtle();
  if (!trust || trust.network !== NETWORK || trust.length !== STEPS.length || !trust.signers || !trust.validators) throw new Error('The separately retained verification context is required.');
  const lengthValid = Array.isArray(chain) && chain.length === trust.length;
  const results = [];
  let previousHash = trust.genesisHash;
  let parentTrusted = true;
  for (const [index, step] of STEPS.entries()) {
    const result = { index, hashValid: false, signatureValid: false, approvalsValid: false, linkValid: false, sequenceValid: false, trusted: false, reasons: [] };
    try {
      const block = chain[index];
      if (!block?.header || !Array.isArray(block.transactions) || block.transactions.length !== 1) throw new Error('Missing or malformed block.');
      // Bound canonicalization before further verification work.
      bytes(canonical(block));
      const tx = block.transactions[0];
      const payload = tx.payload;
      const root = await merkleRoot(block.transactions);
      const computedHash = await blockHash({ ...block.header, merkleRoot: root });
      result.computedHash = computedHash;
      result.recordedHash = block.hash;
      result.hashValid = root === block.header.merkleRoot && computedHash === block.hash;
      result.signatureValid = await checkSignature(trust.signers[step.signer], tx.signature, `event\0${canonical(payload)}`);
      result.linkValid = block.header.previousHash === previousHash;
      result.sequenceValid = block.header.version === 1 && block.header.network === NETWORK &&
        block.header.sessionId === trust.sessionId && block.header.height === index + 1 &&
        payload.id === `${trust.sessionId}:${index + 1}` && payload.signer === step.signer &&
        payload.event === step.event && payload.campaignId === CAMPAIGN;
      const approvals = block.approvals;
      if (Array.isArray(approvals) && approvals.length === VALIDATORS.length && new Set(approvals.map(item => item.validator)).size === VALIDATORS.length) {
        const approved = await Promise.all(VALIDATORS.map(async id => {
          const approval = approvals.find(item => item.validator === id);
          return Boolean(approval && await checkSignature(trust.validators[id], approval.signature, `approval\0${computedHash}`));
        }));
        result.approvalsValid = approved.every(Boolean);
      }
      if (!result.hashValid) result.reasons.push('Record fingerprint changed.');
      if (!result.signatureValid) result.reasons.push('Event signature does not match this record.');
      if (!result.approvalsValid) result.reasons.push('The three local approvals do not match this block.');
      if (!result.linkValid) result.reasons.push('Previous-block link does not match.');
      if (!result.sequenceValid) result.reasons.push('Unexpected event, signer, session, or sequence.');
      if (!parentTrusted) result.reasons.push('An earlier step is no longer trusted.');
      if (!lengthValid) result.reasons.push('The retained four-block checkpoint does not match the ledger length.');
      if (index === STEPS.length - 1 && computedHash !== trust.headHash) result.reasons.push('Final block does not match the retained checkpoint.');
      result.trusted = result.reasons.length === 0;
      previousHash = computedHash;
    } catch {
      result.reasons.push('Missing, malformed, or oversized block.');
      previousHash = '';
    }
    parentTrusted = result.trusted;
    results.push(result);
  }
  return { valid: lengthValid && results.every(item => item.trusted), count: results.filter(item => item.trusted).length, blocks: results };
}

export function simulateChange(chain) {
  const changed = structuredClone(chain);
  if (!changed[2]?.transactions?.[0]?.payload?.details) throw new Error('No publisher record to modify.');
  changed[2].transactions[0].payload.details.impressions = 13000;
  return changed;
}
