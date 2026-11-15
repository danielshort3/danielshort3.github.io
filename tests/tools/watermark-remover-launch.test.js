'use strict';

const assert = require('node:assert/strict');
const { createHmac, randomBytes } = require('node:crypto');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const route = require('../../api/tools/[...slug]');
const { COOKIE_NAME, createSessionFromClaims } = require('../../api/_lib/tools-auth-session');

const SITE_ORIGIN = 'https://www.danielshort.me';
const GPU_ORIGIN = 'https://daniel-pc.tail9686c5.ts.net:8443';
const LAUNCH_API = '/api/tools/watermark-remover';
const LAUNCH_PATH = '/home-tools/watermark/session';

function request({ method = 'POST', origin = SITE_ORIGIN, cookie = '', url = LAUNCH_API } = {}) {
  const req = {
    method,
    url,
    headers: { host: 'www.danielshort.me', 'x-forwarded-proto': 'https', origin, cookie }
  };
  const res = {
    headers: {},
    setHeader(name, value) { this.headers[name] = value; },
    end(value) { this.body = JSON.parse(value); }
  };
  return route(req, res).then(() => res);
}

test('launch endpoint validates the Tools administrator cookie, origin, configuration and scoped ticket', async (t) => {
  const secret = randomBytes(32).toString('base64url');
  const configured = {
    HOME_GPU_WATERMARK_SECRET: secret,
    HOME_GPU_WATERMARK_ORIGIN: GPU_ORIGIN,
    TOOLS_SESSION_SECRETS: randomBytes(32).toString('base64url'),
    TOOLS_ADMIN_EMAILS: 'verified-admin@example.test',
    TOOLS_ADMIN_GROUPS: 'admin',
    NODE_ENV: 'production',
    VERCEL: '1',
    VERCEL_ENV: 'production'
  };
  const original = Object.fromEntries(Object.keys(configured).map((key) => [key, process.env[key]]));
  t.after(() => {
    for (const [key, value] of Object.entries(original)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });
  Object.assign(process.env, configured);
  const cookie = (claims) => `${COOKIE_NAME}=${createSessionFromClaims(claims).value}`;
  const admin = cookie({ sub: 'fixture-admin', 'cognito:groups': ['admin'] });
  const nonAdmin = cookie({ sub: 'fixture-member', email: 'verified-admin@example.test' });

  assert.equal((await request({ method: 'GET' })).statusCode, 405);
  assert.equal((await request({ origin: 'https://unknown.example', cookie: admin })).statusCode, 403);
  assert.equal((await request({ origin: '', cookie: admin })).statusCode, 403);
  assert.equal((await request()).statusCode, 401);
  assert.equal((await request({ cookie: nonAdmin })).statusCode, 403);
  assert.equal((await request({ cookie: admin, url: LAUNCH_API + '/unexpected' })).statusCode, 404);

  const first = await request({ cookie: admin });
  assert.equal(first.statusCode, 200);
  assert.equal(first.headers['Cache-Control'], 'no-store');
  assert.equal(first.body.launchUrl, GPU_ORIGIN + LAUNCH_PATH);
  const [claimsPart, signature] = first.body.ticket.split('.');
  const claims = JSON.parse(Buffer.from(claimsPart, 'base64url'));
  assert.deepEqual(Object.keys(claims).sort(), ['aud', 'exp', 'iat', 'jti', 'origin', 'sub', 'v']);
  assert.equal(claims.sub, 'fixture-admin');
  assert.equal(claims.v, 1);
  assert.equal(claims.aud, 'home-gpu-watermark');
  assert.equal(claims.origin, SITE_ORIGIN);
  assert.equal(claims.exp - claims.iat, 60);
  assert.ok(Math.abs(claims.iat - Math.floor(Date.now() / 1000)) < 2);
  assert.match(claims.jti, /^[\w-]{22}$/);
  assert.equal(signature, createHmac('sha256', secret).update(claimsPart, 'ascii').digest('base64url'));
  assert.notEqual((await request({ cookie: admin })).body.ticket, first.body.ticket);

  for (const origin of [
    'http://127.0.0.1:8788',
    'https://bad.example',
    GPU_ORIGIN + '/path',
    GPU_ORIGIN + '?test=yes',
    'https://example@daniel-pc.tail9686c5.ts.net:8443'
  ]) {
    process.env.HOME_GPU_WATERMARK_ORIGIN = origin;
    assert.equal((await request({ cookie: admin })).statusCode, 503, origin);
  }

  delete process.env.VERCEL;
  delete process.env.VERCEL_ENV;
  process.env.NODE_ENV = 'development';
  process.env.HOME_GPU_WATERMARK_ORIGIN = 'http://127.0.0.1:8788';
  assert.equal((await request({ cookie: admin })).body.launchUrl, 'http://127.0.0.1:8788' + LAUNCH_PATH);
  process.env.HOME_GPU_WATERMARK_SECRET = '';
  assert.equal((await request({ cookie: admin })).statusCode, 503);
});

test('browser exposes the launch only to admins and hands the ticket off in a top-level form POST', async () => {
  const listeners = new Map();
  const submitted = [];
  const button = { addEventListener: (type, listener) => listeners.set(type, listener) };
  const action = {};
  const status = {};
  const controls = { '[data-watermark-open]': button, '[data-watermark-action]': action, '[data-watermark-status]': status };
  const root = {
    querySelector: (selector) => controls[selector],
    append() {}
  };
  let user = null;
  let fetches = 0;
  let launchUrl = GPU_ORIGIN + LAUNCH_PATH;
  const document = {
    readyState: 'complete',
    querySelector: () => root,
    addEventListener: (type, listener) => listeners.set(type, listener),
    createElement(tag) {
      if (tag === 'input') return {};
      return {
        append(input) { this.input = input; },
        submit() { submitted.push({ input: this.input, method: this.method, target: this.target, action: this.action }); },
        remove() {}
      };
    }
  };
  const window = {
    location: new URL(SITE_ORIGIN),
    addEventListener: (type, listener) => listeners.set(type, listener),
    ToolsAuth: {
      getAuth: () => user,
      authIsValid: Boolean,
      isAdmin: (auth) => auth.admin,
      getUser: (auth) => auth,
      fetchWithAuth: async (url, options) => {
        assert.equal(url, LAUNCH_API);
        assert.equal(options.method, 'POST');
        fetches++;
        return { ok: true, json: async () => ({ ok: true, launchUrl, ticket: 'fixture.signature' }) };
      }
    }
  };
  const controller = readFileSync(path.join(__dirname, '../../js/tools/watermark-remover.js'), 'utf8');
  vm.runInNewContext(controller, { document, window, URL, AbortController, setTimeout, clearTimeout });

  assert.equal(action.hidden, true);
  assert.equal(button.disabled, true);
  assert.match(status.textContent, /Sign in/);
  await listeners.get('click')();
  assert.equal(fetches, 0);

  user = { sub: 'member', admin: false };
  listeners.get('tools:auth-changed')();
  assert.equal(action.hidden, true);
  assert.equal(button.disabled, true);
  assert.match(status.textContent, /does not have administrator access/);
  await listeners.get('click')();
  assert.equal(fetches, 0);

  user = { sub: 'admin', admin: true };
  listeners.get('tools:auth-changed')();
  assert.equal(action.hidden, false);
  await Promise.all([listeners.get('click')(), listeners.get('click')()]);
  assert.equal(fetches, 1);
  assert.deepEqual(submitted, [{
    input: { type: 'hidden', name: 'ticket', value: 'fixture.signature' },
    method: 'POST',
    target: '_top',
    action: GPU_ORIGIN + LAUNCH_PATH
  }]);

  listeners.get('pagehide')();
  listeners.get('pageshow')();
  assert.equal(button.disabled, false);
  launchUrl = 'https://bad.example' + LAUNCH_PATH;
  await listeners.get('click')();
  assert.equal(submitted.length, 1);
  assert.match(status.textContent, /Could not open/);
});

test('both launch page aliases opt out of indexing and allow a form only to the expected worker path', () => {
  const config = require('../../vercel.json');
  const routes = ['/tools/watermark-remover', '/tools/watermark-remover.html'];
  for (const source of routes) {
    assert.equal(config.rewrites.find((rule) => rule.source === source).destination, '/pages/watermark-remover');
    const headers = config.headers.find((rule) => rule.source === source).headers;
    assert.equal(headers.find((header) => header.key === 'X-Robots-Tag').value, 'noindex, nofollow');
    const policy = headers.find((header) => header.key === 'Content-Security-Policy').value;
    assert.ok(policy.includes(`form-action 'self' ${GPU_ORIGIN}/home-tools/watermark/;`));
  }
});
