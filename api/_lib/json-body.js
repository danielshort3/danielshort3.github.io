/* Bounded UTF-8 JSON input for parsed bodies, strings, buffers and streams. */
'use strict';

const DEFAULT_MAX_BYTES = 512 * 1024;

function tooLarge(maxBytes){
  const err = new Error(`JSON request body exceeds ${maxBytes} bytes.`);
  err.code = 'BODY_TOO_LARGE';
  err.statusCode = 413;
  return err;
}

async function readJson(req, options = {}){
  const configured = Number(options.maxBytes);
  const maxBytes = Number.isFinite(configured) && configured > 0
    ? Math.max(1, Math.floor(configured)) : DEFAULT_MAX_BYTES;
  const declared = Number(req?.headers?.['content-length']);
  if (Number.isFinite(declared) && declared > maxBytes) throw tooLarge(maxBytes);

  if (req.body !== undefined) {
    if (Buffer.isBuffer(req.body) || typeof req.body === 'string') {
      if (Buffer.byteLength(req.body, 'utf8') > maxBytes) throw tooLarge(maxBytes);
      const raw = req.body.toString();
      return raw.trim() ? JSON.parse(raw) : {};
    }
    const raw = JSON.stringify(req.body);
    if (typeof raw !== 'string') throw new TypeError('Invalid JSON body');
    if (Buffer.byteLength(raw, 'utf8') > maxBytes) throw tooLarge(maxBytes);
    return req.body;
  }

  return new Promise((resolve, reject) => {
    let chunks = [];
    let size = 0;
    let settled = false;
    const fail = (err) => {
      if (settled) return;
      settled = true;
      chunks = [];
      reject(err);
    };
    req.on('data', chunk => {
      if (settled) return;
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk), 'utf8');
      size += bytes.length;
      if (size > maxBytes) { fail(tooLarge(maxBytes)); return; }
      chunks.push(bytes);
    });
    req.on('end', () => {
      if (settled) return;
      const raw = Buffer.concat(chunks).toString('utf8');
      chunks = [];
      try {
        const body = raw.trim() ? JSON.parse(raw) : {};
        settled = true;
        resolve(body);
      } catch (err) { fail(err); }
    });
    req.on('error', fail);
    req.on('aborted', () => fail(new Error('Request aborted')));
  });
}

module.exports = { readJson };
