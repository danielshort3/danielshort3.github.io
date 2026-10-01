'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const logs = require('../../api/_lib/chatbot-logs')._internal;

async function run() {
  const privateUrl = 'https://user:password@www.danielshort.me/tools/text-compare?input=private-value#private-fragment';
  assert.equal(logs.normalizeLogUrl(privateUrl), 'https://www.danielshort.me/tools/text-compare');
  assert.equal(logs.normalizeLogUrl('data:text/plain,private-value'), '');
  assert.equal(logs.normalizeLogUrl('not a valid \0 url'), '');
  assert.equal(logs.normalizePageContext({ url: privateUrl, title: 'Text Compare' }).url, 'https://www.danielshort.me/tools/text-compare');
  const metadata = logs.requestMetadata({ headers: {
    origin: 'https://www.danielshort.me', referer: 'https://referrer.example/page?email=private@example.com',
    'user-agent': 'unique-client', 'x-vercel-ip-country': 'us', 'x-vercel-ip-city': 'Denver', 'x-vercel-ip-country-region': 'CO'
  } });
  assert.deepEqual(metadata, { refererHost: 'referrer.example', country: 'US' });
  assert(!JSON.stringify(metadata).includes('private') && !JSON.stringify(metadata).includes('Denver'));
  const module = { exports: {} };
  const events = [];
  vm.runInNewContext(fs.readFileSync(require.resolve('../../api/go/[...slug]'), 'utf8'), {
    module, URL, URLSearchParams, console,
    require(name) {
      if (name === '../_lib/short-links-store') return {
        getLinkWithLegacyFallback: async () => ({ slug: 'test', destination: 'https://destination.example/page?fixed=public', clicks: 0 }),
        recordClick: async (event) => events.push(event)
      };
      if (name === '../_lib/short-links') return { normalizeSlug: (value) => value, getRequestBaseUrl: () => 'https://www.danielshort.me' };
      return require(name);
    }
  });
  const response = { setHeader() {}, end() {} };
  await module.exports({ method: 'GET', url: '/go/test?private=secret', query: { slug: 'test' }, headers: {
    host: 'www.danielshort.me', referer: 'https://referrer.example/page?private=secret',
    'user-agent': 'unique-client', 'x-vercel-ip-country': 'US', 'x-vercel-ip-city': 'Denver',
    'x-vercel-ip-country-region': 'CO', 'x-vercel-ip-timezone': 'America/Denver'
  } }, response);
  assert.equal(events.length, 1);
  assert.equal(events[0].destination, 'https://destination.example/page');
  assert.equal(events[0].refererHost, 'referrer.example');
  assert.equal(events[0].country, 'US');
  for (const field of ['userAgent', 'city', 'region', 'timezone', 'ip']) assert.equal(events[0][field], undefined);
  assert(!JSON.stringify(events).includes('secret'));
  console.log('Backend privacy metadata passed: URL credentials/query/fragment removed, host/country retained, detailed geography and user agent excluded. No network or stored click writes.');
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
