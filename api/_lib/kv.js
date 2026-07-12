/*
  Minimal Upstash/Vercel KV REST client (no deps).
  Supports either Vercel KV env vars or Upstash REST env vars.
*/
'use strict';

const READ_ONLY_COMMANDS = new Set([
  'EXISTS',
  'GET',
  'HGET',
  'HGETALL',
  'HLEN',
  'HMGET',
  'LLEN',
  'LRANGE',
  'MGET',
  'PTTL',
  'SCARD',
  'SISMEMBER',
  'SMEMBERS',
  'TTL',
  'TYPE',
  'ZCARD',
  'ZREVRANGE',
  'ZSCORE'
]);

function cleanEnv(value){
  return typeof value === 'string' ? value.trim() : '';
}

function getKvEnv(options = {}){
  const requireWrite = options === true || Boolean(options && options.requireWrite);
  const vercelUrl = cleanEnv(process.env.KV_REST_API_URL);
  const upstashUrl = cleanEnv(process.env.UPSTASH_REDIS_REST_URL);
  const vercelWriteToken = cleanEnv(process.env.KV_REST_API_TOKEN);
  const upstashWriteToken = cleanEnv(process.env.UPSTASH_REDIS_REST_TOKEN);
  const vercelReadOnlyToken = cleanEnv(process.env.KV_REST_API_READ_ONLY_TOKEN);

  const useVercel = Boolean(
    (vercelUrl && vercelWriteToken) ||
    (!(upstashUrl && upstashWriteToken) && vercelUrl)
  );
  const url = useVercel ? vercelUrl : upstashUrl;
  const writeToken = useVercel ? vercelWriteToken : upstashWriteToken;
  const readOnlyToken = useVercel ? vercelReadOnlyToken : '';
  const token = requireWrite ? writeToken : (writeToken || readOnlyToken);

  if (!url) {
    const err = new Error(
      'Missing KV env vars. Set KV_REST_API_URL + KV_REST_API_TOKEN (Vercel KV) or ' +
      'UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN (Upstash).'
    );
    err.code = 'KV_ENV_MISSING';
    throw err;
  }

  if (requireWrite && !writeToken) {
    const err = new Error('KV writes require a write-capable REST token.');
    err.code = 'KV_READONLY';
    throw err;
  }

  if (!token) {
    const err = new Error('Missing KV REST token.');
    err.code = 'KV_ENV_MISSING';
    throw err;
  }

  return {
    url: url.replace(/\/+$/, ''),
    token,
    readOnly: !writeToken,
    writeCapable: Boolean(writeToken)
  };
}

function normalizeCommand(command){
  const normalized = String(command || '').trim().toUpperCase();
  if (!/^[A-Z][A-Z0-9_-]*$/.test(normalized)) {
    const err = new Error('Invalid KV command.');
    err.code = 'KV_COMMAND_INVALID';
    throw err;
  }
  return normalized;
}

async function kvCall(command, ...args){
  const normalizedCommand = normalizeCommand(command);
  const requireWrite = !READ_ONLY_COMMANDS.has(normalizedCommand);
  const { url, token } = getKvEnv({ requireWrite });
  const payload = [normalizedCommand, ...args.map(value => String(value))];
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  let data;
  try {
    data = await resp.json();
  } catch (err) {
    const error = new Error(`KV request failed (${resp.status})`);
    error.code = 'KV_BAD_RESPONSE';
    error.status = resp.status;
    throw error;
  }

  if (!resp.ok || data.error) {
    const message = data.error || `KV request failed (${resp.status})`;
    const error = new Error(message);
    error.status = resp.status;
    if (resp.status === 401 || resp.status === 403) error.code = 'KV_AUTH';
    if (String(message).toUpperCase().includes('READONLY')) error.code = 'KV_READONLY';
    throw error;
  }

  return data.result;
}

async function kvGet(key){
  return kvCall('get', key);
}

async function kvSet(key, value){
  return kvCall('set', key, value);
}

async function kvDel(...keys){
  return kvCall('del', ...keys);
}

async function kvIncr(key){
  return kvCall('incr', key);
}

async function kvSadd(key, member){
  return kvCall('sadd', key, member);
}

async function kvSrem(key, member){
  return kvCall('srem', key, member);
}

async function kvSmembers(key){
  const result = await kvCall('smembers', key);
  return Array.isArray(result) ? result : [];
}

module.exports = {
  getKvEnv,
  kvCall,
  kvGet,
  kvSet,
  kvDel,
  kvIncr,
  kvSadd,
  kvSrem,
  kvSmembers
};
