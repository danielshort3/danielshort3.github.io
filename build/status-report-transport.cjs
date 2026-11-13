'use strict';

const fs = require('node:fs');
const { constants, createCipheriv, createDecipheriv, privateDecrypt, publicEncrypt, randomBytes } = require('node:crypto');

const MAX_BYTES = 16 * 1024;
const ENVELOPE_KEYS = 'data,iv,key,tag';
const ALGORITHM = 'aes-256-gcm';
const RSA_OPTIONS = { padding: constants.RSA_PKCS1_OAEP_PADDING, oaepHash: 'sha256' };

function seal(report, publicKey) {
  const input = Buffer.from(report);
  if (!input.length || input.length > MAX_BYTES) throw new Error('Report is missing or exceeds the transport limit.');
  const key = randomBytes(32);
  const iv = randomBytes(12);
  const cipher = createCipheriv(ALGORITHM, key, iv);
  const data = Buffer.concat([cipher.update(input), cipher.final()]);
  const envelope = {
    data, iv, key: publicEncrypt({ ...RSA_OPTIONS, key: publicKey }, key), tag: cipher.getAuthTag()
  };
  return JSON.stringify(Object.fromEntries(Object.entries(envelope).map(([name, value]) => [name, value.toString('base64')])));
}

function open(envelope, privateKey) {
  if (typeof envelope !== 'string' || envelope.length > MAX_BYTES * 2) throw new Error('Invalid report envelope.');
  const fields = JSON.parse(envelope);
  if (!fields || Object.keys(fields).sort().join(',') !== ENVELOPE_KEYS ||
    !Object.values(fields).every(value => typeof value === 'string' && /^[A-Za-z0-9+/]+={0,2}$/.test(value))) {
    throw new Error('Invalid report envelope.');
  }
  const [data, iv, wrappedKey, tag] = ['data', 'iv', 'key', 'tag'].map(name => Buffer.from(fields[name], 'base64'));
  const key = privateDecrypt({ ...RSA_OPTIONS, key: privateKey }, wrappedKey);
  const decipher = createDecipheriv(ALGORITHM, key, iv);
  decipher.setAuthTag(tag);
  const report = Buffer.concat([decipher.update(data), decipher.final()]);
  if (!report.length || report.length > MAX_BYTES) throw new Error('Invalid report size.');
  return report;
}

if (require.main === module) {
  try {
    const [mode, path, publicKeyPath] = process.argv.slice(2);
    if (mode === 'seal' && path && publicKeyPath) {
      if (fs.statSync(path).size > MAX_BYTES) throw new Error('Report exceeds the transport limit.');
      process.stdout.write(seal(fs.readFileSync(path), fs.readFileSync(publicKeyPath)));
    } else if (mode === 'open' && path && !publicKeyPath) {
      const report = open(process.env.SEALED_REPORT, process.env.AWS_STATUS_REPORT_PRIVATE_KEY);
      fs.writeFileSync(path, report, { flag: 'wx', mode: 0o600 });
    } else {
      throw new Error('Invalid report transport arguments.');
    }
  } catch {
    process.stderr.write('Could not process the encrypted status report.\n');
    process.exitCode = 1;
  }
}

module.exports = { seal, open };
