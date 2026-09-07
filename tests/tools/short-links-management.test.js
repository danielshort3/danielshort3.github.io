'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { createRequire } = require('module');
const management = require('../../api/_lib/short-links-management');
const helpers = require('../../api/_lib/short-links');
const sessions = require('../../api/_lib/tools-auth-session');
const store = require('../../api/_lib/short-links-store');
const redirect = require('../../api/go/[...slug]');

function loadModule(relative, overrides){
  const filename = path.resolve(__dirname, '../..', relative);
  const localRequire = createRequire(filename);
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), {
    module, exports: module.exports,
    require: name => Object.prototype.hasOwnProperty.call(overrides, name) ? overrides[name] : localRequire(name),
    Buffer, URL, URLSearchParams, Date, console, process, setTimeout, clearTimeout
  }, { filename });
  return module.exports;
}

function response(){
  return {
    headers: {}, statusCode: 0,
    setHeader(key, value){ this.headers[key] = value; },
    end(value){ this.raw = value || ''; try { this.body = JSON.parse(value); } catch {} }
  };
}

async function call(handler, method, url, body, headers = { 'x-admin-token': 'short-links-test-only' }){
  const res = response();
  await handler({ method, url, body, headers: { host: 'example.test', ...headers } }, res);
  return res;
}

async function run(){
  const priorEnv = {};
  for (const key of ['SHORTLINKS_ADMIN_TOKEN', 'TOOLS_SESSION_SECRET', 'TOOLS_SESSION_SECRETS', 'TOOLS_ADMIN_EMAILS', 'TOOLS_ADMIN_GROUPS', 'SHORTLINKS_DDB_TABLE', 'SHORTLINKS_DDB_CLICKS_TABLE', 'AWS_REGION']) {
    priorEnv[key] = process.env[key];
  }
  process.env.SHORTLINKS_ADMIN_TOKEN = 'short-links-test-only';
  process.env.TOOLS_SESSION_SECRET = 'a'.repeat(64);
  delete process.env.TOOLS_SESSION_SECRETS;
  delete process.env.TOOLS_ADMIN_EMAILS;
  delete process.env.TOOLS_ADMIN_GROUPS;
  process.env.SHORTLINKS_DDB_TABLE = 'short-links-test';
  process.env.AWS_REGION = 'us-east-2';
  delete process.env.SHORTLINKS_DDB_CLICKS_TABLE;
  try {
    const future = Math.floor(Date.now() / 1000) + 86400;
    const existing = {
      slug: 'Portfolio', destination: 'https://example.test/before', permanent: false,
      expiresAt: future, disabled: true, clicks: 27, label: 'Portfolio title', tags: ['work'],
      updatedAt: '2026-09-01T00:00:00.000Z', contextCompany: 'Example',
      qrDesign: { schemaVersion: 1, fg: '#3095AA' }
    };
    const patch = management.buildLinkPatch({ destination: 'https://example.test/after' }, existing);
    assert.deepStrictEqual(patch, { destination: 'https://example.test/after' });
    const command = store.buildLinkUpdate({ tableName: 'test-links', slug: existing.slug, patch, updatedAt: 'later', expectedUpdatedAt: existing.updatedAt });
    assert(!/clicks|disabled|expiresAt|contextCompany|qrDesign/.test(command.UpdateExpression), 'Editing a destination must not overwrite other link state');
    assert(command.ConditionExpression.includes('#updatedAt = :expectedUpdatedAt'), 'Concurrent edits must not silently overwrite each other');
    assert.throws(() => management.buildLinkPatch({ slug: 'renamed', label: 'New' }, existing), /cannot be renamed/);
    assert.deepStrictEqual(management.buildLinkPatch({ expiresAt: 0 }, existing), { expiresAt: 0 });
    assert.throws(() => management.buildLinkPatch({ expiresAt: future }, { ...existing, permanent: true, expiresAt: 0 }), /temporary redirect/);
    assert.deepStrictEqual(management.normalizeTags([' work ', 'work', 'launch']), ['work', 'launch']);
    assert.throws(() => management.normalizeTags('work'), /list/);

    const design = management.normalizeQrDesign({ schemaVersion: 1, fg: '#3095aa', ecc: 'H', payloadMode: 'wifi', wifiPassword: 'never-store', data: 'private', previewDataUrl: 'data:image/png;base64,YQ==' });
    assert.deepStrictEqual(Object.keys(design).sort(), ['schemaVersion', 'fg', 'ecc', 'previewDataUrl'].sort());
    assert.throws(() => management.normalizeQrDesign({ logoDataUrl: 'javascript:alert(1)' }), /logo/);
    assert.throws(() => management.normalizeQrDesign({ captionText: 'x'.repeat(128 * 1024) }), /128 KB/);
    assert.throws(() => management.normalizeQrDesign({ fg: 'red' }), /fg/);
    assert.equal(management.serializeLink(existing).qrDesign.fg, '#3095AA');
    assert.deepStrictEqual(management.serializeLink(existing).tags, ['work']);

    let storedRecord = existing;
    let lastTransaction;
    let collide = false;
    const awsDocument = require('@aws-sdk/lib-dynamodb');
    const isolatedStore = loadModule('api/_lib/short-links-store.js', {
      './aws-credentials': { resolveAwsCredentials: () => ({ cacheKey: 'test', credentials: { accessKeyId: 'test', secretAccessKey: 'test' } }) },
      '@aws-sdk/lib-dynamodb': {
        ...awsDocument,
        DynamoDBDocumentClient: { from: () => ({ send: async command => {
          if (command instanceof awsDocument.GetCommand) return { Item: storedRecord };
          if (command instanceof awsDocument.TransactWriteCommand) {
            lastTransaction = command.input;
            if (collide) {
              const error = new Error('ConditionalCheckFailed');
              error.name = 'TransactionCanceledException';
              error.CancellationReasons = [{ Code: 'ConditionalCheckFailed' }];
              throw error;
            }
            return {};
          }
          throw new Error(`Unexpected storage command ${command.constructor.name}`);
        } }) }
      }
    });
    await assert.rejects(isolatedStore.upsertLink({ slug: existing.slug, destination: 'https://example.test/new', createOnly: true }), error => error.code === 'SLUG_CONFLICT');
    assert.equal(lastTransaction, undefined, 'A create request must never update a preexisting record');
    storedRecord = null;
    await isolatedStore.upsertLink({ slug: 'new-link', destination: 'https://example.test/new', createOnly: true, updatedAt: 'now' });
    const atomicCreate = lastTransaction.TransactItems[1].Update;
    assert.equal(atomicCreate.ConditionExpression, 'attribute_not_exists(#slug)');
    assert(!atomicCreate.UpdateExpression.includes('REMOVE label'), 'Legacy metadata omitted from a request must be preserved');
    collide = true;
    await assert.rejects(isolatedStore.upsertLink({ slug: 'new-link', destination: 'https://example.test/new', createOnly: true, updatedAt: 'now' }), error => error.code === 'SLUG_CONFLICT');

    const target = redirect._internal.buildRedirectTarget({ url: '/go/Portfolio?__qr=1&utm_source=poster&slug=ignored' }, 'https://example.test/page?mode=read&__qr=old', 'https://dshort.me');
    assert.equal(target.channel, 'qr');
    assert.equal(target.finalUrl, 'https://example.test/page?mode=read&utm_source=poster');
    assert.equal(redirect._internal.buildRedirectTarget({ url: '/Portfolio?__qr=0' }, 'https://example.test', 'https://dshort.me').channel, 'link');
    assert.equal(store.buildClickEventItem({ slug: 'a', clickId: '1', clickedAt: 'now', channel: 'qr' }).channel, 'qr');
    assert.equal(store.buildClickEventItem({ slug: 'a', clickId: '1', clickedAt: 'now' }).channel, 'unknown');

    const cookie = claims => sessions.createSessionFromClaims({ sub: 'test-user', exp: Math.floor(Date.now() / 1000) + 3600, ...claims }).cookie.split(';')[0];
    const adminCookie = cookie({ 'cognito:groups': ['admin'] });
    assert.equal(await helpers.authorizeAdminRequest({ method: 'PATCH', headers: { host: 'example.test', origin: 'https://example.test', 'x-forwarded-proto': 'https', cookie: adminCookie } }, response()), true);
    const crossOrigin = response();
    assert.equal(await helpers.authorizeAdminRequest({ method: 'PATCH', headers: { host: 'example.test', origin: 'https://other.test', 'x-forwarded-proto': 'https', cookie: adminCookie } }, crossOrigin), false);
    assert.equal(crossOrigin.statusCode, 403);
    const nonAdmin = response();
    assert.equal(await helpers.authorizeAdminRequest({ method: 'GET', headers: { cookie: cookie({ email: 'reader@example.test', email_verified: true }) } }, nonAdmin), false);
    assert.equal(nonAdmin.statusCode, 403);
    assert.equal(helpers.isToolsAdminClaims({ email: 'daniel@danielshort.me', email_verified: false }), false);
    assert.equal(helpers.isToolsAdminClaims({ email: 'daniel@danielshort.me', email_verified: true }), true);
    assert.equal(await helpers.authorizeAdminRequest({ headers: { 'x-admin-token': 'short-links-test-only' } }, response()), true);
    assert.equal(await helpers.authorizeAdminRequest({ headers: {} }, response()), false);

    let writes = 0;
    let createdOptions;
    let updatedOptions;
    const memoryStore = {
      ...store,
      listLinks: async () => [existing],
      getLinkWithLegacyFallback: async slug => slug.toLowerCase() === 'portfolio' ? existing : null,
      upsertLink: async options => { writes += 1; createdOptions = options; return { ...options, ...options.metadata }; },
      updateLink: async options => { writes += 1; updatedOptions = options; return { ...existing, ...options.patch }; }
    };
    const index = loadModule('api/short-links/index.js', { '../_lib/short-links-store': memoryStore });
    const single = loadModule('api/short-links/[...slug].js', { '../_lib/short-links-store': memoryStore });
    let res = await call(index, 'POST', '/api/short-links', { intent: 'create', slug: 'Portfolio', destination: 'https://example.test/new' });
    assert.equal(res.statusCode, 409);
    assert.equal(writes, 0);
    res = await call(index, 'POST', '/api/short-links', { intent: 'create', slug: 'portfolio', destination: 'https://example.test/new' });
    assert.equal(res.statusCode, 409);
    assert.equal(writes, 0);
    res = await call(index, 'POST', '/api/short-links', { intent: 'create', destination: 'https://example.test/new', label: 'New campaign' });
    assert.equal(res.statusCode, 200);
    assert.equal(res.body.generated, true);
    assert.equal(createdOptions.createOnly, true);
    assert.equal(createdOptions.permanent, false);
    res = await call(index, 'POST', '/api/short-links', { slug: 'Portfolio', destination: 'https://example.test/new' });
    assert.equal(res.statusCode, 200, 'Legacy template and project upsert stays available');
    assert.equal(createdOptions.createOnly, false);
    res = await call(single, 'PATCH', '/api/short-links/portfolio', { destination: 'https://example.test/after' });
    assert.equal(res.statusCode, 200);
    assert.equal(updatedOptions.slug, 'Portfolio');
    assert.equal(res.body.link.expiresAt, existing.expiresAt);
    assert.equal(res.body.link.clicks, 27);
    assert.equal(res.body.link.disabled, true);
    assert.equal(res.body.link.contextCompany, 'Example');
    assert.equal(res.body.link.qrDesign.fg, '#3095AA');
    res = await call(single, 'PATCH', '/api/short-links/Portfolio', { qrDesign: null });
    assert.equal(res.body.link.qrDesign, null);
    const authorizedWrites = writes;
    res = await call(single, 'PATCH', '/api/short-links/Portfolio', { label: 'Denied' }, {});
    assert.equal(res.statusCode, 401);
    assert.equal(writes, authorizedWrites);

    const now = Date.parse('2026-09-05T12:00:00Z');
    const events = [
      { slug: 'Portfolio', clickId: 'baseline', entityType: 'clickBaseline' },
      { slug: 'Portfolio', clickId: '1', clickedAt: '2026-09-05T10:00:00Z', channel: 'qr' },
      { slug: 'Portfolio', clickId: '2', clickedAt: '2026-09-04T10:00:00Z', channel: 'link' },
      { slug: 'Portfolio', clickId: '3', clickedAt: '2026-09-04T11:00:00Z' },
      { slug: 'deleted', clickId: '4', clickedAt: '2026-09-04T11:00:00Z', channel: 'qr' }
    ];
    const report = management.buildAnalyticsReport({ links: [existing], items: events, days: '7', now });
    assert.deepStrictEqual(report.totals, { clicks: 3, qrScans: 1, linkClicks: 1, unknownClicks: 1, lifetimeClicks: 27 });
    assert.equal(report.daily.length, 7);
    assert.equal(report.completeness.complete, false);
    assert.equal(report.completeness.unattributedHistoricalClicks, 24);
    const all = management.buildAnalyticsReport({ links: [existing], items: events, days: 'all', now });
    assert.equal(all.totals.clicks, 27);
    assert.equal(all.totals.unknownClicks, 25);
    assert.equal(all.daily.reduce((sum, day) => sum + day.clicks, 0), 3, 'Undated activity must not be fabricated into the trend');
    assert.equal(management.buildAnalyticsReport({ links: [], items: [], truncated: true, now }).completeness.complete, false);

    let recorded;
    const redirectHandler = loadModule('api/go/[...slug].js', {
      '../_lib/short-links-store': {
        getLinkWithLegacyFallback: async () => ({ ...existing, disabled: false, permanent: true, expiresAt: 0 }),
        recordClick: async event => { recorded = event; }
      }
    });
    res = await call(redirectHandler, 'GET', '/go/Portfolio?__qr=1&utm_source=poster');
    assert.equal(res.statusCode, 302, 'QR redirects remain editable even for legacy permanent links');
    assert.equal(res.headers['Cache-Control'], 'no-store');
    assert.equal(recorded.channel, 'qr');
    assert(!res.headers.Location.includes('__qr'));
    recorded = null;
    await call(redirectHandler, 'HEAD', '/go/Portfolio?__qr=1');
    assert.equal(recorded, null, 'Health probes must not count as QR visits');
    process.stdout.write('Short links management tests passed\n');
  } finally {
    Object.entries(priorEnv).forEach(([key, value]) => { if (typeof value === 'undefined') delete process.env[key]; else process.env[key] = value; });
  }
}

run().catch(error => { process.stderr.write(`${error.stack || error}\n`); process.exitCode = 1; });
