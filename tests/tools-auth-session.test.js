'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const vm = require('vm');

const originalEnv = {
  TOOLS_SESSION_SECRETS: process.env.TOOLS_SESSION_SECRETS,
  TOOLS_SESSION_SECRET: process.env.TOOLS_SESSION_SECRET,
  TOOLS_SESSION_TTL_SECONDS: process.env.TOOLS_SESSION_TTL_SECONDS,
  TOOLS_AUTH_BEARER_FALLBACK: process.env.TOOLS_AUTH_BEARER_FALLBACK
};

const firstKey = crypto.randomBytes(32);
const previousKey = crypto.randomBytes(32);
process.env.TOOLS_SESSION_SECRETS = `${firstKey.toString('base64url')},${previousKey.toString('base64url')}`;
delete process.env.TOOLS_SESSION_SECRET;
process.env.TOOLS_SESSION_TTL_SECONDS = '28800';
process.env.TOOLS_AUTH_BEARER_FALLBACK = 'true';

const sessions = require('../api/_lib/tools-auth-session');

function createResponse(){
  return {
    statusCode: 200,
    headers: {},
    body: '',
    setHeader(name, value){
      this.headers[String(name).toLowerCase()] = value;
    },
    end(value){
      this.body = String(value || '');
    }
  };
}

function listManagedHtmlFiles(root = '.'){
  const skippedDirectories = new Set(['.git', '.vercel', 'node_modules', 'public']);
  const files = [];
  const visit = (directory) => {
    fs.readdirSync(directory, { withFileTypes: true }).forEach((entry) => {
      if (entry.isDirectory() && skippedDirectories.has(entry.name)) return;
      const file = `${directory}/${entry.name}`.replace(/^\.\//, '');
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile() && entry.name.endsWith('.html')) files.push(file);
    });
  };
  visit(root);
  return files;
}

async function run(){
  const now = Math.floor(Date.now() / 1000);
  const claims = {
    sub: 'user-123',
    email: 'person@example.com',
    name: 'Example Person',
    'cognito:username': 'person',
    'cognito:groups': ['Admin', 'Analyst'],
    idToken: 'must-not-be-copied',
    accessToken: 'must-not-be-copied',
    refreshToken: 'must-not-be-copied'
  };
  const payload = sessions.createSessionPayload(claims, now);
  assert.strictEqual(payload.exp - payload.iat, 28800);
  assert.deepStrictEqual(payload.groups, ['admin', 'analyst']);
  assert.deepStrictEqual(Object.keys(payload).sort(), ['email', 'exp', 'groups', 'iat', 'name', 'sub', 'v']);

  const value = sessions.encryptSessionPayload(payload);
  assert(!value.includes(claims.email));
  assert(!value.includes(claims.sub));
  assert.deepStrictEqual(sessions.decryptSessionPayload(value, now), payload);

  const rotatedValue = sessions.encryptSessionPayload(payload, previousKey);
  assert.strictEqual(sessions.decryptSessionPayload(rotatedValue, now).sub, claims.sub);

  process.env.TOOLS_SESSION_SECRETS = crypto.randomBytes(32).toString('base64url');
  assert.throws(() => sessions.decryptSessionPayload(value, now), /Invalid session cookie/);
  process.env.TOOLS_SESSION_SECRETS = `${firstKey.toString('base64url')},${previousKey.toString('base64url')}`;

  const tamperedParts = value.split('.');
  tamperedParts[2] = `${tamperedParts[2][0] === 'a' ? 'b' : 'a'}${tamperedParts[2].slice(1)}`;
  const tampered = tamperedParts.join('.');
  assert.throws(() => sessions.decryptSessionPayload(tampered, now), /Invalid session cookie/);

  const expired = { ...payload, iat: now - 1000, exp: now - 1 };
  const expiredValue = sessions.encryptSessionPayload(expired);
  assert.throws(() => sessions.decryptSessionPayload(expiredValue, now), /Session expired/);

  const created = sessions.createSessionFromClaims(claims);
  assert(created.cookie.includes(`${sessions.COOKIE_NAME}=${created.value}`));
  assert(created.cookie.includes('HttpOnly'));
  assert(created.cookie.includes('Secure'));
  assert(created.cookie.includes('SameSite=Lax'));
  assert(created.cookie.includes('Path=/'));
  assert(!created.cookie.includes('Domain='));
  assert(!created.cookie.includes(claims.email));
  const clearedCookie = sessions.serializeClearedSessionCookie();
  assert(clearedCookie.includes(`${sessions.COOKIE_NAME}=;`));
  assert(clearedCookie.includes('HttpOnly') && clearedCookie.includes('Secure') && clearedCookie.includes('SameSite=Lax'));
  assert(clearedCookie.includes('Max-Age=0') && clearedCookie.includes('Path=/'));

  const cookieRequest = {
    method: 'GET',
    headers: {
      cookie: `${sessions.COOKIE_NAME}=${created.value}`,
      authorization: 'Bearer deliberately-invalid-and-unused'
    }
  };
  let verifierCalls = 0;
  const auth = await sessions.authenticateToolsRequest(cookieRequest, {
    verifyToken: async () => {
      verifierCalls += 1;
      throw new Error('Cookie-first authentication should not verify the Bearer token.');
    }
  });
  assert.strictEqual(auth.source, 'cookie');
  assert.strictEqual(auth.claims.sub, claims.sub);
  assert.strictEqual(verifierCalls, 0);

  const bearerAuth = await sessions.authenticateToolsRequest({
    method: 'GET',
    headers: { authorization: 'Bearer test-token' }
  }, {
    verifyToken: async (token) => {
      assert.strictEqual(token, 'test-token');
      return { ...claims, exp: now + 3600 };
    }
  });
  assert.strictEqual(bearerAuth.source, 'bearer');
  assert.strictEqual(bearerAuth.claims.sub, claims.sub);

  delete process.env.TOOLS_SESSION_SECRETS;
  const bearerWithoutCookieConfig = await sessions.authenticateToolsRequest({
    method: 'GET',
    headers: {
      cookie: `${sessions.COOKIE_NAME}=${created.value}`,
      authorization: 'Bearer test-token'
    }
  }, {
    verifyToken: async () => ({ ...claims, exp: now + 3600 })
  });
  assert.strictEqual(bearerWithoutCookieConfig.source, 'bearer');
  process.env.TOOLS_SESSION_SECRETS = `${firstKey.toString('base64url')},${previousKey.toString('base64url')}`;

  const meResponse = createResponse();
  await require('../api/_lib/tools-endpoints/me')({ method: 'GET', headers: cookieRequest.headers }, meResponse);
  assert.strictEqual(meResponse.statusCode, 200);
  assert.strictEqual(JSON.parse(meResponse.body).user.sub, claims.sub);

  const activityResponse = createResponse();
  await require('../api/_lib/tools-endpoints/activity')({
    method: 'POST',
    headers: { cookie: `${sessions.COOKIE_NAME}=${created.value}` }
  }, activityResponse);
  assert.strictEqual(activityResponse.statusCode, 403);

  const sameOriginMutation = {
    method: 'POST',
    socket: { encrypted: true },
    headers: {
      host: 'www.danielshort.me',
      origin: 'https://www.danielshort.me',
      cookie: `${sessions.COOKIE_NAME}=${created.value}`
    }
  };
  assert.strictEqual((await sessions.authenticateToolsRequest(sameOriginMutation)).source, 'cookie');

  const crossOriginMutation = {
    ...sameOriginMutation,
    headers: {
      ...sameOriginMutation.headers,
      origin: 'https://attacker.example',
      authorization: 'Bearer test-token'
    }
  };
  let crossOriginVerifierCalls = 0;
  await assert.rejects(
    () => sessions.authenticateToolsRequest(crossOriginMutation, {
      verifyToken: async () => {
        crossOriginVerifierCalls += 1;
        return { ...claims, exp: now + 3600 };
      }
    }),
    (err) => err?.code === 'AUTH_ORIGIN_MISMATCH' && err?.statusCode === 403
  );
  assert.strictEqual(crossOriginVerifierCalls, 0);

  const localMutation = {
    method: 'PATCH',
    socket: { encrypted: false },
    headers: {
      host: '127.0.0.1:4177',
      origin: 'http://127.0.0.1:4177',
      cookie: `${sessions.COOKIE_NAME}=${created.value}`
    }
  };
  assert.strictEqual((await sessions.authenticateToolsRequest(localMutation)).source, 'cookie');

  process.env.TOOLS_AUTH_BEARER_FALLBACK = 'false';
  await assert.rejects(
    () => sessions.authenticateToolsRequest({ method: 'GET', headers: { authorization: 'Bearer ignored' } }),
    (err) => err?.code === 'AUTH_UNAUTHORIZED'
  );

  process.env.TOOLS_SESSION_TTL_SECONDS = '999999';
  assert.strictEqual(sessions.getSessionTtlSeconds(), sessions.MAX_SESSION_TTL_SECONDS);
  process.env.TOOLS_SESSION_TTL_SECONDS = '1';
  assert.strictEqual(sessions.getSessionTtlSeconds(), sessions.MIN_SESSION_TTL_SECONDS);
  process.env.TOOLS_SESSION_TTL_SECONDS = '28800';
  assert.throws(
    () => sessions.createSessionPayload({ ...claims, auth_time: now - 28801 }, now),
    (err) => err?.code === 'AUTH_REAUTH_REQUIRED'
  );

  process.env.TOOLS_SESSION_SECRETS = `invalid,${previousKey.toString('base64url')}`;
  assert.throws(
    () => sessions.getSessionKeys(),
    (err) => err?.code === 'TOOLS_SESSION_SECRET_INVALID' && err?.statusCode === 503
  );
  process.env.TOOLS_SESSION_SECRETS = `${firstKey.toString('base64url')},${previousKey.toString('base64url')}`;

  const authEndpointModule = require('../api/_lib/tools-endpoints/auth');
  const authHandler = authEndpointModule.createHandler({
    verifyToken: async (token) => {
      assert.strictEqual(token, 'exchange-token');
      return { ...claims, exp: now + 3600 };
    },
    authenticateRequest: async () => ({
      source: 'cookie',
      claims,
      expiresAt: now + 3600
    })
  });
  const exchangeResponse = createResponse();
  await authHandler({
    method: 'POST',
    socket: { encrypted: true },
    headers: {
      host: 'www.danielshort.me',
      origin: 'https://www.danielshort.me',
      authorization: 'Bearer exchange-token'
    }
  }, exchangeResponse, ['exchange']);
  assert.strictEqual(exchangeResponse.statusCode, 200);
  assert(exchangeResponse.headers['set-cookie'].includes('HttpOnly'));
  assert.strictEqual(exchangeResponse.headers['cache-control'], 'no-store');
  assert.strictEqual(exchangeResponse.headers.pragma, 'no-cache');
  assert(!exchangeResponse.body.includes('exchange-token'));
  assert(!exchangeResponse.body.includes(claims.idToken));

  const sessionResponse = createResponse();
  await authHandler({ method: 'GET', headers: {} }, sessionResponse, ['session']);
  assert.strictEqual(sessionResponse.statusCode, 200);
  assert.strictEqual(JSON.parse(sessionResponse.body).source, 'cookie');
  assert.strictEqual(sessionResponse.headers['cache-control'], 'no-store');
  assert.strictEqual(sessionResponse.headers.pragma, 'no-cache');

  const logoutResponse = createResponse();
  await authHandler({
    method: 'POST',
    socket: { encrypted: true },
    headers: { host: 'www.danielshort.me', origin: 'https://www.danielshort.me' }
  }, logoutResponse, ['logout']);
  assert.strictEqual(logoutResponse.statusCode, 200);
  assert(logoutResponse.headers['set-cookie'].includes('Max-Age=0'));
  assert.strictEqual(logoutResponse.headers['cache-control'], 'no-store');

  const methodResponse = createResponse();
  await authHandler({ method: 'GET', headers: {} }, methodResponse, ['exchange']);
  assert.strictEqual(methodResponse.statusCode, 405);
  assert.strictEqual(methodResponse.headers.allow, 'POST');

  const unknownResponse = createResponse();
  await authHandler({ method: 'GET', headers: {} }, unknownResponse, ['unknown']);
  assert.strictEqual(unknownResponse.statusCode, 404);

  const authEndpoint = fs.readFileSync('api/_lib/tools-endpoints/auth.js', 'utf8');
  const router = fs.readFileSync('api/tools/[...slug].js', 'utf8');
  const client = fs.readFileSync('js/accounts/tools-auth.js', 'utf8');
  const clientConfig = fs.readFileSync('js/accounts/tools-config.js', 'utf8');
  assert(authEndpoint.includes("action === 'exchange'") && authEndpoint.includes("action === 'session'") && authEndpoint.includes("action === 'logout'"));
  assert(router.includes("endpoint === 'auth'") && router.includes('handleAuth(req, res, rest)'));
  assert(client.includes('SESSION_MODES.has(configuredSessionMode)') && clientConfig.includes("sessionMode: 'dual'"));
  assert(client.includes('Cookie-only tools sessions require a same-origin API proxy'));

  const authStorage = new Map([[
    'toolsAuth',
    JSON.stringify({
      idToken: 'dual-mode-token',
      expiresAt: Date.now() + 60 * 60 * 1000,
      serverSession: true,
      claims: { sub: claims.sub, exp: now + 3600 }
    })
  ]]);
  const storage = {
    getItem: (key) => authStorage.has(key) ? authStorage.get(key) : null,
    setItem: (key, value) => authStorage.set(key, String(value)),
    removeItem: (key) => authStorage.delete(key)
  };
  const fetchCalls = [];
  const clientContext = {
    window: {
      location: {
        origin: 'https://www.danielshort.me',
        pathname: '/tools/dashboard',
        search: '',
        hash: ''
      },
      history: { replaceState() {} },
      TOOLS_AUTH_CONFIG: { sessionMode: 'dual' },
      addEventListener() {}
    },
    document: {
      body: { dataset: {} },
      documentElement: { dataset: {} },
      dispatchEvent() {},
      title: 'Tools'
    },
    localStorage: storage,
    sessionStorage: storage,
    fetch: async (url, options) => {
      fetchCalls.push({ url, options });
      return { ok: true, status: 200 };
    },
    Headers,
    URL,
    URLSearchParams,
    TextEncoder,
    atob,
    crypto: crypto.webcrypto,
    CustomEvent: class CustomEvent {},
    console,
    setTimeout,
    clearTimeout
  };
  vm.runInNewContext(client, clientContext, { filename: 'js/accounts/tools-auth.js' });
  await clientContext.window.ToolsAuth.fetchWithAuth('/api/tools/me');
  assert.strictEqual(fetchCalls.length, 1);
  assert.strictEqual(fetchCalls[0].options.headers.get('Authorization'), 'Bearer dual-mode-token');
  assert.strictEqual(fetchCalls[0].options.credentials, 'same-origin');

  const vercel = JSON.parse(fs.readFileSync('vercel.json', 'utf8'));
  const policies = vercel.headers
    .flatMap((entry) => entry.headers || [])
    .filter((header) => header.key === 'Content-Security-Policy')
    .map((header) => String(header.value || ''));
  assert(policies.length >= 1);
  for (const policy of policies) {
    for (const directive of [
      "base-uri 'self'",
      "object-src 'none'",
      "form-action 'self'",
      "frame-ancestors 'self'",
      "script-src-attr 'none'",
      "manifest-src 'self'"
    ]) {
      assert(policy.includes(directive), `CSP missing ${directive}`);
    }
    assert(/script-src [^;]*'unsafe-inline'/.test(policy), 'CSP migration should retain script-src unsafe-inline for now');
    assert(/style-src [^;]*'unsafe-inline'/.test(policy), 'CSP migration should retain style-src unsafe-inline for now');
    assert(!/frame-ancestors [^;]*https?:/i.test(policy), 'frame-ancestors should not allow external origins');
  }

  const staticResumePreviewSources = [
    'content/resumes/analytics.json',
    'content/resumes/data-science.json',
    'content/resumes/tourism.json'
  ];
  const staticManagedResumePreviews = [
    'pages/resume-analytics-pdf.html',
    'pages/resume-data-science-pdf.html',
    'pages/resume-tourism-pdf.html'
  ];
  listManagedHtmlFiles().forEach((file) => {
    const html = fs.readFileSync(file, 'utf8');
    assert(!/<(?:object|embed)\b/i.test(html), `${file} should not contain CSP-blocked object/embed markup`);
  });
  [
    'content/pages/resume-pdf-directory.json',
    'pages/resume-pdf.html'
  ].forEach((file) => {
    const html = fs.readFileSync(file, 'utf8');
    assert(html.includes('title=\\"Resume PDF preview\\"') || html.includes('title="Resume PDF preview"'),
      `${file} should include a titled PDF iframe`);
    assert(html.includes('resume-pdf-fallback'), `${file} should keep an always-visible PDF fallback link`);
  });
  [...staticResumePreviewSources, ...staticManagedResumePreviews].forEach((file) => {
    const html = fs.readFileSync(file, 'utf8');
    assert(!/<iframe\b/i.test(html), `${file} should use a static first-page preview instead of a PDF iframe`);
    assert(html.includes('resume-pdf-preview') && html.includes('Open PDF') && html.includes('Download PDF'),
      `${file} should include a linked static preview plus open and download actions`);
    assert(html.includes('resume-pdf-fallback'), `${file} should keep an always-visible PDF fallback link`);
  });
  const analyticsPreview = fs.readFileSync('pages/resume-analytics-pdf.html', 'utf8');
  assert(analyticsPreview.includes('src="img/resume-previews/resume-analytics-preview.png"'));

  const devServer = require('../build/dev');
  const localHeaders = {};
  devServer.applyResponseHeaders(
    '/',
    devServer.compileRoutes(vercel.headers),
    { headers: { host: '127.0.0.1:4177' } },
    { setHeader: (name, headerValue) => { localHeaders[String(name).toLowerCase()] = String(headerValue); } },
    new URL('http://127.0.0.1:4177/')
  );
  assert(localHeaders['content-security-policy']?.includes("frame-ancestors 'self'"));
  assert(localHeaders['x-content-type-options'] === 'nosniff');

  console.log('Tools auth session tests passed.');
}

run()
  .catch((err) => {
    console.error(err);
    process.exitCode = 1;
  })
  .finally(() => {
    Object.entries(originalEnv).forEach(([key, value]) => {
      if (typeof value === 'undefined') delete process.env[key];
      else process.env[key] = value;
    });
  });
