'use strict';

const crypto = require('node:crypto');
const zlib = require('node:zlib');

const MAGIC = Buffer.from('DSUPD001', 'ascii');
const MAX_APK_BYTES = 256 * 1024 * 1024;
const MAX_OPERATIONS = 100000;
const HEADER_BYTES = 88;
const DEFAULT_BLOCK_BYTES = 32 * 1024;
const ADLER_MOD = 65521;

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function checkApk(bytes, label) {
  if (!Buffer.isBuffer(bytes) || bytes.length < 1 || bytes.length > MAX_APK_BYTES) {
    throw new Error(`${label} must contain 1 to ${MAX_APK_BYTES} bytes`);
  }
}

function adler(bytes, offset, length) {
  let a = 1;
  let b = 0;
  for (let i = offset; i < offset + length; i += 1) {
    a = (a + bytes[i]) % ADLER_MOD;
    b = (b + a) % ADLER_MOD;
  }
  return { a, b };
}

function adlerKey(checksum) {
  return checksum.b * 65536 + checksum.a;
}

// Match exact blocks after rolling-checksum filtering, including when inserts shift
// all later bytes. Coalesce adjacent matches so APKs need few copy operations.
function createOperations(base, target, blockBytes) {
  const index = new Map();
  for (let offset = 0; offset + blockBytes <= base.length; offset += blockBytes) {
    const key = adlerKey(adler(base, offset, blockBytes));
    const candidates = index.get(key) || new Map();
    const digest = sha256(base.subarray(offset, offset + blockBytes));
    if (!candidates.has(digest)) candidates.set(digest, offset);
    index.set(key, candidates);
  }
  const operations = [];
  function append(operation) {
    const previous = operations.at(-1);
    if (operation.tag === 0 && previous?.tag === 0 && previous.offset + previous.length === operation.offset) {
      previous.length += operation.length;
    } else {
      operations.push(operation);
      if (operations.length > MAX_OPERATIONS) throw new Error('Patch has too many operations');
    }
  }
  let offset = 0;
  let literalStart = 0;
  let checksum = target.length >= blockBytes ? adler(target, 0, blockBytes) : null;
  while (offset + blockBytes <= target.length) {
    const candidates = index.get(adlerKey(checksum));
    const source = candidates?.get(sha256(target.subarray(offset, offset + blockBytes)));
    if (source !== undefined) {
      if (offset > literalStart) append({ tag: 1, bytes: target.subarray(literalStart, offset) });
      append({ tag: 0, offset: source, length: blockBytes });
      offset += blockBytes;
      literalStart = offset;
      checksum = offset + blockBytes <= target.length ? adler(target, offset, blockBytes) : null;
    } else {
      if (offset + blockBytes < target.length) {
        const removed = target[offset];
        checksum.a = (checksum.a - removed + target[offset + blockBytes] + ADLER_MOD) % ADLER_MOD;
        checksum.b = (checksum.b - blockBytes * removed + checksum.a - 1) % ADLER_MOD;
        if (checksum.b < 0) checksum.b += ADLER_MOD;
      }
      offset += 1;
    }
  }
  if (literalStart < target.length) append({ tag: 1, bytes: target.subarray(literalStart) });
  return operations;
}

function encodePatch(base, target, { blockBytes = DEFAULT_BLOCK_BYTES } = {}) {
  checkApk(base, 'Base APK');
  checkApk(target, 'Target APK');
  if (!Number.isSafeInteger(blockBytes) || blockBytes < 1 || blockBytes > MAX_APK_BYTES) {
    throw new Error('Invalid block size');
  }
  const header = Buffer.alloc(HEADER_BYTES);
  MAGIC.copy(header);
  Buffer.from(sha256(base), 'hex').copy(header, 8);
  Buffer.from(sha256(target), 'hex').copy(header, 40);
  header.writeBigInt64BE(BigInt(base.length), 72);
  header.writeBigInt64BE(BigInt(target.length), 80);
  const parts = [header];
  for (const operation of createOperations(base, target, blockBytes)) {
    if (operation.tag === 0) {
      const record = Buffer.alloc(13);
      record[0] = 0;
      record.writeBigInt64BE(BigInt(operation.offset), 1);
      record.writeUInt32BE(operation.length, 9);
      parts.push(record);
    } else {
      const record = Buffer.alloc(5);
      record[0] = 1;
      record.writeUInt32BE(operation.bytes.length, 1);
      parts.push(record, operation.bytes);
    }
  }
  parts.push(Buffer.from([255]));
  return zlib.gzipSync(Buffer.concat(parts), { level: 9, mtime: 0 });
}

function applyPatch(base, compressed) {
  checkApk(base, 'Base APK');
  if (!Buffer.isBuffer(compressed) || compressed.length > MAX_APK_BYTES) throw new Error('Invalid patch size');
  const data = zlib.gunzipSync(compressed, { maxOutputLength: MAX_APK_BYTES + MAX_OPERATIONS * 13 + HEADER_BYTES + 1 });
  let cursor = 0;
  function read(length) {
    if (length > data.length - cursor) throw new Error('Truncated patch');
    const bytes = data.subarray(cursor, cursor + length);
    cursor += length;
    return bytes;
  }
  function readSize() {
    const value = read(8).readBigInt64BE();
    if (value < 1n || value > BigInt(MAX_APK_BYTES)) throw new Error('Invalid APK size');
    return Number(value);
  }
  if (!read(8).equals(MAGIC)) throw new Error('Unknown patch format');
  const baseHash = read(32).toString('hex');
  const targetHash = read(32).toString('hex');
  const baseSize = readSize();
  const targetSize = readSize();
  if (baseSize !== base.length || baseHash !== sha256(base)) throw new Error('Base APK does not match');
  const output = Buffer.alloc(targetSize);
  let written = 0;
  let count = 0;
  while (true) {
    const tag = read(1)[0];
    if (tag === 255) break;
    count += 1;
    if (count > MAX_OPERATIONS) throw new Error('Patch has too many operations');
    if (tag !== 0 && tag !== 1) throw new Error('Unknown patch operation');
    const source = tag === 0 ? read(8).readBigInt64BE() : 0n;
    const length = read(4).readUInt32BE();
    if (length === 0 || length > targetSize - written) throw new Error('Invalid operation length');
    if (tag === 0) {
      if (source < 0n || source > BigInt(base.length - length)) throw new Error('Copy exceeds base APK');
      base.copy(output, written, Number(source), Number(source) + length);
    } else {
      read(length).copy(output, written);
    }
    written += length;
  }
  if (cursor !== data.length) throw new Error('Trailing patch data');
  if (written !== targetSize || sha256(output) !== targetHash) throw new Error('Target APK does not match');
  return output;
}

module.exports = { MAGIC, HEADER_BYTES, MAX_APK_BYTES, MAX_OPERATIONS, DEFAULT_BLOCK_BYTES, sha256, encodePatch, applyPatch };
