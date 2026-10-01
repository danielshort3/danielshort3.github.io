'use strict';

const { SESClient, SendEmailCommand } = require('@aws-sdk/client-ses');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, TransactWriteCommand } = require('@aws-sdk/lib-dynamodb');
const { getProxySecret, normalizeOrigin, validatePayload, verifySignedRequest } = require('../../api/_lib/contact-protection');

function positiveInteger(value, fallback) {
  const numeric = Number(value);
  return Number.isSafeInteger(numeric) && numeric > 0 ? numeric : fallback;
}

function reservation(tableName, signed, now, env) {
  const second = Math.floor(now / 1000);
  const ttl = second + 2 * 86400;
  const policies = [
    [signed.actor, 60, positiveInteger(env.CONTACT_PER_MINUTE_LIMIT, 2)],
    [signed.actor, 3600, positiveInteger(env.CONTACT_PER_HOUR_LIMIT, 5)],
    [signed.actor, 86400, positiveInteger(env.CONTACT_PER_DAY_LIMIT, 10)],
    ['GLOBAL', 86400, positiveInteger(env.CONTACT_GLOBAL_DAILY_LIMIT, 100)]
  ];
  return {
    TransactItems: [
      { Put: {
        TableName: tableName,
        Item: { pk: `CONTACT#NONCE#${signed.nonce}`, sk: 'REQUEST', ttl },
        ConditionExpression: 'attribute_not_exists(pk)'
      } },
      ...policies.map(([actor, seconds, limit]) => ({ Update: {
        TableName: tableName,
        Key: { pk: `CONTACT#ACTOR#${actor}`, sk: `${seconds}#${Math.floor(second / seconds)}` },
        ConditionExpression: 'attribute_not_exists(#count) OR #count < :limit',
        UpdateExpression: 'SET #ttl = :ttl ADD #count :one',
        ExpressionAttributeNames: { '#ttl': 'ttl', '#count': 'count' },
        ExpressionAttributeValues: { ':ttl': ttl, ':one': 1, ':limit': limit }
      } }))
    ]
  };
}

function createHandler({ env = process.env, ses, ddb, now = Date.now } = {}) {
  const sesClient = ses || new SESClient({});
  const ddbClient = ddb || DynamoDBDocumentClient.from(new DynamoDBClient({}));
  const allowedOrigins = new Set(String(env.ALLOWED_ORIGINS || 'https://www.danielshort.me,https://danielshort.me')
    .split(',').map(normalizeOrigin).filter(Boolean));
  return async (event = {}) => {
    const headers = Object.fromEntries(Object.entries(event.headers || {}).map(([key, value]) => [key.toLowerCase(), String(value || '')]));
    const origin = normalizeOrigin(headers.origin);
    const response = (statusCode, body, extraHeaders = {}) => ({
      statusCode,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
        ...(allowedOrigins.has(origin) ? { 'Access-Control-Allow-Origin': origin } : {}),
        ...extraHeaders
      },
      body: JSON.stringify(body)
    });
    const method = event.requestContext?.http?.method || event.httpMethod || '';
    if (method !== 'POST') return response(405, { ok: false, error: 'Method Not Allowed' }, { Allow: 'POST' });
    if (!allowedOrigins.has(origin)) return response(403, { ok: false, error: 'Origin not allowed.' });
    let secret;
    try {
      secret = getProxySecret(env);
      if (!env.CONTACT_RATE_LIMIT_TABLE || !env.SENDER_EMAIL || !env.RECIPIENT_EMAIL) throw new Error('Configuration missing');
    } catch {
      return response(503, { ok: false, error: 'Message delivery is temporarily unavailable.' });
    }
    const signed = verifySignedRequest(event, secret, now());
    if (!signed) return response(403, { ok: false, error: 'Trusted website request required.' });
    let payload;
    try { payload = validatePayload(JSON.parse(signed.body)); } catch { payload = null; }
    if (!payload) return response(400, { ok: false, error: 'Please provide a valid name, email, and message.' });
    if (payload.company) return response(200, { ok: true });
    const command = new TransactWriteCommand(reservation(env.CONTACT_RATE_LIMIT_TABLE, signed, now(), env));
    for (let attempt = 0; ; attempt += 1) {
      try {
        await ddbClient.send(command);
        break;
      } catch (error) {
        const reasons = error?.CancellationReasons || [];
        if (reasons.some((reason) => reason?.Code === 'ConditionalCheckFailed')) {
          const seconds = Math.floor(now() / 1000);
          const windows = [60, 60, 3600, 86400, 86400];
          const retryAfter = Math.max(1, ...reasons.map((reason, index) => reason?.Code === 'ConditionalCheckFailed'
            ? windows[index] - seconds % windows[index] : 0));
          return response(429, { ok: false, error: 'Too many messages. Please wait and try again.' }, { 'Retry-After': String(retryAfter) });
        }
        const conflict = error?.name === 'TransactionConflictException' || reasons.some((reason) => reason?.Code === 'TransactionConflict');
        if (!conflict || attempt >= 3) {
          console.error('Contact protection unavailable', { name: String(error?.name || 'Unknown') });
          return response(503, { ok: false, error: 'Message delivery is temporarily unavailable.' });
        }
        await new Promise((resolve) => setTimeout(resolve, 10 * (attempt + 1)));
      }
    }
    try {
      await sesClient.send(new SendEmailCommand({
        Source: env.SENDER_EMAIL,
        Destination: { ToAddresses: env.RECIPIENT_EMAIL.split(',').map((value) => value.trim()).filter(Boolean) },
        ReplyToAddresses: [payload.email],
        Message: {
          Subject: { Data: `Contact form submission from ${payload.name}` },
          Body: { Text: { Data: `Name: ${payload.name}\nEmail: ${payload.email}\n\nMessage:\n${payload.message}` } }
        }
      }));
      return response(200, { ok: true });
    } catch (error) {
      console.error('Contact delivery failed', { name: String(error?.name || 'Unknown') });
      return response(502, { ok: false, error: 'Unable to confirm message delivery.' });
    }
  };
}

exports.handler = createHandler();
exports.createHandler = createHandler;
exports.reservation = reservation;
