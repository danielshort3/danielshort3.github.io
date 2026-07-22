import assert from 'node:assert/strict';
import test from 'node:test';
import { ToolbarActionLauncher } from '../src/background/action-launcher.js';
import { PRIVACY_NOTICE_VERSION } from '../src/shared/privacy-consent.js';

const currentConsent = {
  jobCopilotPrivacyConsentV1: {
    version: PRIVACY_NOTICE_VERSION,
    acceptedAt: '2026-07-18T18:00:00.000Z'
  }
};

test('toolbar action opens the tab panel immediately, then prepares a consented page', async () => {
  const events = [];
  let releaseStorage;
  const storageResult = new Promise(resolve => { releaseStorage = resolve; });
  const launcher = new ToolbarActionLauncher({
    sidePanel: {
      open(value) {
        events.push(['open', value]);
        return Promise.resolve();
      }
    },
    storageLocal: {
      get(keys) {
        events.push(['storage', keys]);
        return storageResult;
      }
    },
    scripting: {
      async executeScript(details) {
        events.push(['inject', details]);
      }
    }
  });

  const launchResult = launcher.launch({ id: 9, url: 'https://job-boards.greenhouse.io/example/jobs/123' });
  assert.deepEqual(events, [
    ['open', { tabId: 9 }],
    ['storage', ['jobCopilotPrivacyConsentV1']]
  ]);

  releaseStorage(currentConsent);
  assert.deepEqual(await launchResult, { panelOpened: true, injected: true, reason: 'ready' });
  assert.deepEqual(events.at(-1), ['inject', {
    target: { tabId: 9, frameIds: [0] },
    files: ['content/page-runtime.js']
  }]);
});

test('toolbar action opens without injecting before consent or on restricted pages', async () => {
  for (const scenario of [
    { url: 'http://jobs.example/apply', stored: {}, reason: 'privacy_consent_required' },
    { url: 'chrome://extensions', stored: currentConsent, reason: 'unsupported_page' },
    { url: 'file:///C:/private.html', stored: currentConsent, reason: 'unsupported_page' },
    { url: 'not a URL', stored: currentConsent, reason: 'unsupported_page' },
    { url: undefined, stored: currentConsent, reason: 'unsupported_page' }
  ]) {
    let openCount = 0;
    let injectionCount = 0;
    const launcher = new ToolbarActionLauncher({
      sidePanel: { async open() { openCount += 1; } },
      storageLocal: { async get() { return scenario.stored; } },
      scripting: { async executeScript() { injectionCount += 1; } }
    });
    assert.deepEqual(await launcher.launch({ id: 5, url: scenario.url }), {
      panelOpened: true,
      injected: false,
      reason: scenario.reason
    });
    assert.equal(openCount, 1);
    assert.equal(injectionCount, 0);
  }
});

test('toolbar action settles preparation failures without blocking the panel', async () => {
  const launcher = new ToolbarActionLauncher({
    sidePanel: { async open() {} },
    storageLocal: { async get() { return currentConsent; } },
    scripting: { async executeScript() { throw new Error('restricted page'); } }
  });
  assert.deepEqual(await launcher.launch({ id: 7, url: 'https://jobs.example/apply' }), {
    panelOpened: true,
    injected: false,
    reason: 'preparation_failed'
  });
});

test('toolbar action rejects an unsafe tab identity without side effects', async () => {
  let calls = 0;
  const launcher = new ToolbarActionLauncher({
    sidePanel: { async open() { calls += 1; } },
    storageLocal: { async get() { calls += 1; return currentConsent; } },
    scripting: { async executeScript() { calls += 1; } }
  });
  assert.deepEqual(await launcher.launch({ id: '7', url: 'https://jobs.example/apply' }), {
    panelOpened: false,
    injected: false,
    reason: 'missing_tab'
  });
  assert.equal(calls, 0);
});
