/*
  Storage backend selection for tools accounts.

  Prefer AWS DynamoDB (tools-store-ddb). Fall back to KV (tools-store-kv) if DynamoDB is not configured.
*/
'use strict';

function hasValue(value){
  return typeof value === 'string' && value.trim().length > 0;
}

function useDynamo(){
  return hasValue(process.env.TOOLS_DDB_TABLE) || hasValue(process.env.TOOLS_DDB_TABLE_NAME);
}

function allowKvCompatibility(){
  if (String(process.env.NODE_ENV || '').trim().toLowerCase() === 'production') return false;
  return String(process.env.TOOLS_ALLOW_KV_COMPAT || '').trim().toLowerCase() === 'true';
}

function useKv(){
  const hasVercelKv =
    hasValue(process.env.KV_REST_API_URL) &&
    hasValue(process.env.KV_REST_API_TOKEN);
  const hasUpstashRedis =
    hasValue(process.env.UPSTASH_REDIS_REST_URL) &&
    hasValue(process.env.UPSTASH_REDIS_REST_TOKEN);
  return hasVercelKv || hasUpstashRedis;
}

if (useDynamo()) {
  module.exports = require('./tools-store-ddb');
} else if (allowKvCompatibility() && useKv()) {
  module.exports = require('./tools-store-kv');
} else {
  module.exports = require('./tools-store-ddb');
}
