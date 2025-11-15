const AWS = require('aws-sdk');
const ses = new AWS.SES({ apiVersion: '2010-12-01' });

const { SENDER_EMAIL, RECIPIENT_EMAIL, ALLOWED_ORIGINS = '' } = process.env;
const allowedOrigins = ALLOWED_ORIGINS.split(',').map(o => o.trim()).filter(Boolean);

const isEmail = (value = '') => /.+@.+\..+/.test(value);
const sanitize = (value = '') => value.replace(/[\r\n\t]+/g, ' ').trim();

const buildHeaders = (origin) => ({
  'Access-Control-Allow-Origin': origin || '*',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
  'Content-Type': 'application/json'
});

const parseBody = (event = {}) => {
  const rawBody = event.isBase64Encoded
    ? Buffer.from(event.body || '', 'base64').toString('utf8')
    : (event.body || '');
  const header = event.headers || {};
  const contentType = (header['content-type'] || header['Content-Type'] || '').toLowerCase();
  if (!rawBody) return {};
  if (contentType.includes('application/x-www-form-urlencoded')) {
    const params = new URLSearchParams(rawBody);
    return Object.fromEntries(params.entries());
  }
  try {
    return JSON.parse(rawBody);
  } catch {
    return {};
  }
};

exports.handler = async (event) => {
  const requestOrigin = event?.headers?.origin || event?.headers?.Origin || '';
  const corsOrigin = allowedOrigins.length === 0
    ? requestOrigin || '*'
    : (allowedOrigins.includes('*') ? (requestOrigin || '*')
      : (allowedOrigins.includes(requestOrigin) ? requestOrigin : allowedOrigins[0]));

  if (event.requestContext?.http?.method === 'OPTIONS') {
    return { statusCode: 204, headers: buildHeaders(corsOrigin) };
  }

  const payload = parseBody(event);
  if (!payload || typeof payload !== 'object') {
    return {
      statusCode: 400,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ error: 'Invalid form payload.' })
    };
  }

  const name = sanitize(payload.name);
  const email = sanitize(payload.email);
  const message = (payload.message || '').toString().trim();
  const honey = sanitize(payload.company || '');

  if (honey) {
    return {
      statusCode: 200,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ ok: true })
    };
  }

  if (!name || name.length > 200 || !email || !isEmail(email) || !message || message.length > 4000) {
    return {
      statusCode: 400,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ error: 'Please provide a valid name, email, and message.' })
    };
  }

  if (!SENDER_EMAIL || !RECIPIENT_EMAIL) {
    return {
      statusCode: 500,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ error: 'Email configuration missing.' })
    };
  }

  const emailParams = {
    Source: SENDER_EMAIL,
    Destination: { ToAddresses: RECIPIENT_EMAIL.split(',').map(sanitize) },
    ReplyToAddresses: [email],
    Message: {
      Subject: { Data: `Contact form submission from ${name}` },
      Body: {
        Text: {
          Data: `Name: ${name}\nEmail: ${email}\n\nMessage:\n${message}`
        }
      }
    }
  };

  try {
    await ses.sendEmail(emailParams).promise();
    return {
      statusCode: 200,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ ok: true })
    };
  } catch (err) {
    console.error('SES error', err);
    return {
      statusCode: 500,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ error: 'Unable to send message right now.' })
    };
  }
};
