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

function useKv(){
  const hasUrl = hasValue(process.env.KV_REST_API_URL) || hasValue(process.env.UPSTASH_REDIS_REST_URL);
  const hasToken =
    hasValue(process.env.KV_REST_API_TOKEN) ||
    hasValue(process.env.KV_REST_API_READ_ONLY_TOKEN) ||
    hasValue(process.env.UPSTASH_REDIS_REST_TOKEN);
  return hasUrl && hasToken;
}

if (useDynamo()) {
  module.exports = require('./tools-store-ddb');
} else if (useKv()) {
  module.exports = require('./tools-store-kv');
} else {
  module.exports = require('./tools-store-ddb');
}
