'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Module = require('node:module');
const { KNOWN_TOOL_IDS, normalizeKnownToolId } = require('../../api/_lib/tools-api');

const root = path.resolve(__dirname, '../..');
const accountSource = fs.readFileSync(path.join(root, 'js/accounts/tools-account-ui.js'), 'utf8');
const catalogMatch = /  const TOOL_CATALOG = (\{[\s\S]*?\n  \});/.exec(accountSource);
assert(catalogMatch, 'The shared account UI must expose its tool catalog');
const catalog = vm.runInNewContext(`(${catalogMatch[1]})`);
const toolIds = Object.keys(catalog);

function response() {
  return {
    statusCode: 200,
    headers: {},
    body: null,
    setHeader(name, value) { this.headers[String(name).toLowerCase()] = value; },
    end(value) { this.body = JSON.parse(value); }
  };
}

async function run() {
  assert(toolIds.includes('campaign-creative-tracker'), 'The browser regression must include Campaign Creative Tracker');
  for (const id of toolIds) {
    assert(KNOWN_TOOL_IDS.has(id), `Account UI tool ${id} must be accepted by activity and session APIs`);
    assert.equal(normalizeKnownToolId(id), id);
  }
  for (const file of fs.readdirSync(path.join(root, 'content/tools')).filter((name) => name.endsWith('.json'))) {
    const tool = JSON.parse(fs.readFileSync(path.join(root, 'content/tools', file), 'utf8'));
    assert(catalog[tool.slug], `Configured tool ${tool.slug} must have a shared account entry`);
    assert(KNOWN_TOOL_IDS.has(tool.slug), `Configured tool ${tool.slug} must be accepted by the API`);
  }

  const originalLoad = Module._load;
  const originalFetch = global.fetch;
  const activityPath = require.resolve('../../api/_lib/tools-endpoints/activity');
  const statePath = require.resolve('../../api/_lib/tools-endpoints/state');
  const authPath = require.resolve('../../api/_lib/tools-auth-session');
  const storePath = require.resolve('../../api/_lib/tools-store');
  const cachedHandlers = new Map([activityPath, statePath].map((file) => [file, require.cache[file]]));
  const calls = [];
  const store = {
    MAX_SNAPSHOT_BYTES: 512 * 1024,
    async logActivity(input) {
      calls.push({ kind: 'activity', ...input });
      return { ...input, id: 'fixture-event' };
    },
    async saveSession(input) {
      calls.push({ kind: 'save', ...input });
      return { ...input, sessionId: input.sessionId || 'fixture-session', version: 1 };
    },
    async updateSessionMeta(input) {
      calls.push({ kind: 'update', ...input });
      return { ...input, version: 2 };
    }
  };
  try {
    global.fetch = () => { throw new Error('Tool catalog tests must never access the network'); };
    Module._load = function (request, parent, isMain) {
      const resolved = Module._resolveFilename(request, parent, isMain);
      if (resolved === authPath) return { authenticateToolsRequest: async () => ({ claims: { sub: 'catalog-test-user' } }) };
      if (resolved === storePath) return store;
      return originalLoad.call(this, request, parent, isMain);
    };
    delete require.cache[activityPath];
    delete require.cache[statePath];
    const activity = require(activityPath);
    const state = require(statePath);

    for (const toolId of toolIds) {
      const result = response();
      await activity({ method: 'POST', headers: {}, body: JSON.stringify({ toolId, type: 'tool_open', summary: 'Opened tool' }) }, result);
      assert.equal(result.statusCode, 200, `${toolId} opening must not produce Invalid toolId`);
      assert.equal(result.body.event.toolId, toolId);
      assert.equal(result.body.event.sub, 'catalog-test-user');
      assert.equal(calls.at(-1).kind, 'activity');
      assert.equal(result.headers['cache-control'], 'no-store');
    }

    const snapshot = { toolId: 'campaign-creative-tracker', inputs: { campaignName: 'Spring campaign' }, output: { kind: 'campaign-creative-tracker', renditions: [] } };
    const saved = response();
    await state({ method: 'POST', headers: {}, body: { toolId: 'campaign-creative-tracker', snapshot, expectedVersion: 0, outputSummary: 'Spring campaign draft' } }, saved);
    assert.equal(saved.statusCode, 200, 'Campaign tracker manual save must use the same accepted tool ID');
    assert.equal(saved.body.session.toolId, 'campaign-creative-tracker');
    assert.deepEqual(calls.at(-1).snapshot, snapshot, 'The campaign snapshot must reach storage unchanged');
    assert.equal(calls.at(-1).expectedVersion, 0);

    const updated = response();
    await state({ method: 'PATCH', headers: {}, body: { toolId: 'campaign-creative-tracker', sessionId: 'fixture-session', expectedVersion: 1, title: 'Reviewed campaign' } }, updated);
    assert.equal(updated.statusCode, 200, 'Campaign tracker saved-session metadata must remain editable');
    assert.equal(calls.at(-1).kind, 'update');
    assert.equal(calls.at(-1).title, 'Reviewed campaign');

    for (const toolId of ['unknown-tool', '../campaign-creative-tracker', '']) {
      for (const handler of [activity, state]) {
        const count = calls.length;
        const rejected = response();
        await handler({ method: 'POST', headers: {}, body: { toolId, type: 'tool_open', snapshot: {}, expectedVersion: 0 } }, rejected);
        assert.equal(rejected.statusCode, 400, 'Unknown or malformed tool IDs must still fail validation');
        assert.equal(rejected.body.error, 'Invalid toolId');
        assert.equal(calls.length, count, 'Rejected tool IDs must not reach storage');
      }
    }
  } finally {
    Module._load = originalLoad;
    global.fetch = originalFetch;
    for (const [file, cached] of cachedHandlers) {
      if (cached) require.cache[file] = cached;
      else delete require.cache[file];
    }
  }
  console.log(`Tools API catalog: ${toolIds.length} account tools match backend validation; campaign activity, session save/edit and unknown-ID rejection passed without network or storage writes.`);
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
