/*
  Minimal Upstash/Vercel KV REST client (no deps).
  Supports either Vercel KV env vars or Upstash REST env vars.
*/
'use strict';

function getKvEnv(){
  const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
  const token =
    process.env.KV_REST_API_TOKEN ||
    process.env.KV_REST_API_READ_ONLY_TOKEN ||
    process.env.UPSTASH_REDIS_REST_TOKEN;

  if (!url || !token) {
    const err = new Error(
      'Missing KV env vars. Set KV_REST_API_URL + KV_REST_API_TOKEN (Vercel KV) or ' +
      'UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN (Upstash).'
    );
    err.code = 'KV_ENV_MISSING';
    throw err;
  }

  return { url: url.replace(/\/+$/,'').trim(), token: String(token).trim() };
}

async function kvCall(command, ...args){
  const { url, token } = getKvEnv();
  const parts = [command, ...args].map(value => encodeURIComponent(String(value)));
  const endpoint = `${url}/${parts.join('/')}`;
  const resp = await fetch(endpoint, {
    method: 'GET',
    headers: { Authorization: `Bearer ${token}` }
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

