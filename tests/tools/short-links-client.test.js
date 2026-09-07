'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

async function run(){
  const storage = () => {
    const data = new Map();
    return { getItem: key => data.get(key) || null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
  };
  const requests = [];
  const events = [];
  const window = {
    location: { origin: 'https://www.danielshort.me' },
    sessionStorage: storage(), localStorage: storage(),
    dispatchEvent: event => events.push(event.type),
    ToolsAuth: { ensureFreshAuth: async () => ({ idToken: 'verified-test-id-token' }) }
  };
  const context = { window, document: { readyState: 'complete' }, URL, Headers,
    CustomEvent: class { constructor(type){ this.type = type; } },
    fetch: async (url, options) => {
      requests.push({ url, ...options });
      return { ok: true, json: async () => ({ ok: true, link: { slug: 'example' } }) };
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/tools/short-links-client.js'), 'utf8'), context);
  const client = window.ShortLinksClient;
  window.localStorage.setItem('shortlinks_admin_token', 'remembered');
  window.sessionStorage.setItem('shortlinks_admin_token', 'session');
  assert.strictEqual(client.getToken(), 'session', 'QR and shortener share session-first access');
  client.setToken('replacement', true);
  assert.strictEqual(window.sessionStorage.getItem('shortlinks_admin_token'), null);
  assert.strictEqual(client.getToken(), 'replacement');
  assert.strictEqual(events.pop(), 'shortlinks:access-changed');
  assert.strictEqual(client.publicUrl('p/portfolio'), 'https://dshort.me/p/portfolio');
  assert.strictEqual(client.publicUrl('p/portfolio', { qr: true }), 'https://dshort.me/p/portfolio?__qr=1');
  assert.strictEqual(client.qrEditorUrl('p/portfolio'), '/tools/qr-code-generator?link=p%2Fportfolio');
  await client.create({ destination: 'https://example.com', intent: 'overwrite' });
  assert.strictEqual(JSON.parse(requests.at(-1).body).intent, 'create', 'normal creation cannot fall back to upsert');
  assert.strictEqual(requests.at(-1).headers.get('Authorization'), 'Bearer replacement');
  client.clearToken();
  await client.update('p/portfolio', { label: 'Portfolio' });
  assert.strictEqual(requests.at(-1).url, '/api/short-links/p%2Fportfolio');
  assert.strictEqual(requests.at(-1).method, 'PATCH');
  assert.strictEqual(requests.at(-1).headers.get('Authorization'), 'Bearer verified-test-id-token');
  assert.strictEqual(requests.at(-1).credentials, 'same-origin');
  assert.deepStrictEqual(JSON.parse(requests.at(-1).body), { label: 'Portfolio' });
  await assert.rejects(client.request('https://other.example/api/short-links'), /workspace/);
  await assert.rejects(client.request('/api/unrelated'), /workspace/);
  context.fetch = async () => ({ ok: false, status: 403, json: async () => ({ ok: false, error: 'Unauthorized' }) });
  await assert.rejects(client.list(), error => error.status === 403 && /authorized account/.test(error.message));
  return 17;
}

module.exports = run;
if (require.main === module) run().then(count => console.log(`Short links client: ${count} checks passed.`)).catch(error => { console.error(error); process.exitCode = 1; });
