import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  isCurrentPrivacyConsent,
  ollamaHealthErrorPresentation,
  ollamaOriginCommand,
  PRIVACY_CONSENT_KEY,
  PRIVACY_NOTICE_VERSION,
  validateModelSettings
} from '../src/sidepanel/sidepanel.js';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('privacy consent is explicit, versioned, and timestamped', () => {
  assert.equal(PRIVACY_CONSENT_KEY, 'jobCopilotPrivacyConsentV1');
  assert.equal(PRIVACY_NOTICE_VERSION, 4);
  assert.equal(isCurrentPrivacyConsent(null), false);
  assert.equal(isCurrentPrivacyConsent({ version: 0, acceptedAt: new Date().toISOString() }), false);
  assert.equal(isCurrentPrivacyConsent({ version: 1, acceptedAt: 'not-a-date' }), false);
  assert.equal(isCurrentPrivacyConsent({ version: 1, acceptedAt: '2026-07-18T18:00:00.000Z' }), false);
  assert.equal(isCurrentPrivacyConsent({ version: 2, acceptedAt: '2026-07-19T18:00:00.000Z' }), false);
  assert.equal(isCurrentPrivacyConsent({ version: 3, acceptedAt: '2026-07-19T18:00:00.000Z' }), false);
  assert.equal(isCurrentPrivacyConsent({ version: 4, acceptedAt: '2026-07-20T18:00:00.000Z' }), true);
});

test('first-use disclosure blocks operational initialization until acceptance', async () => {
  const [html, source] = await Promise.all([
    readFile(path.join(packageDir, 'src/sidepanel/sidepanel.html'), 'utf8'),
    readFile(path.join(packageDir, 'src/sidepanel/sidepanel.js'), 'utf8')
  ]);
  assert.match(html, /data-privacy-onboarding[\s\S]*data-privacy-consent-form[\s\S]*name="acknowledged"[^>]*required/u);
  assert.match(html, /data-privacy-version>Version 4<\/span>/u);
  assert.match(html, /current page URL and form structure/u);
  assert.match(html, /custom thoughts \(including compensation preferences\)[\s\S]*Custom links, thoughts, and compensation preferences are never included/u);
  assert.match(html, /does not read values already entered/u);
  assert.match(html, /processed through Ollama at/u);
  assert.match(html, /Optional U\.S\. employment eligibility:[\s\S]*never sent to Ollama/u);
  assert.match(html, /Citizenship, immigration or visa status, I-9\/E-Verify, export-control/u);
  assert.match(html, /encrypted in this browser profile/u);
  assert.match(html, /explicit confirmation[\s\S]*sent over HTTPS/u);
  assert.match(html, /CAPTCHA pauses filling[\s\S]*Consent, signature, demographic, identity, security-code, employer file-upload, navigation, and submit controls are excluded/u);
  assert.match(html, /never submits an application[\s\S]*no advertising, analytics, or cloud AI/u);
  assert.match(html, /Original bytes for every supported import[\s\S]*Only retained PDF\/DOCX files/u);
  assert.match(html, /data-operational-shell hidden/u);

  const initializeStart = source.indexOf('const initialize = async');
  const previewBranch = source.indexOf('if (PREVIEW_MODE)', initializeStart);
  const storageAccess = source.indexOf('chrome.storage.local.setAccessLevel', initializeStart);
  assert.match(source, /privacyVersion\.textContent = `Version \$\{PRIVACY_NOTICE_VERSION\}`/u);
  const consentInstall = source.indexOf('installPrivacyActions();', initializeStart);
  const operationalStart = source.indexOf('if (state.privacyConsent) await startOperationalControllers();', initializeStart);
  assert.ok(initializeStart >= 0 && previewBranch > initializeStart);
  assert.ok(previewBranch < storageAccess, 'preview must return before any storage access');
  assert.ok(consentInstall < operationalStart, 'the consent handler must exist before operational startup is considered');
  assert.match(source, /const startOperationalControllers = async \(\) => \{\s*requirePrivacyConsent\(\);[\s\S]*new EncryptedIndexedDbVault\(\)[\s\S]*new OllamaClient\(\)[\s\S]*installVaultActions\(\)[\s\S]*installApplicationActions\(\)/u);
});

test('preview bypass is ephemeral and withdrawal retains encrypted vault data', async () => {
  const source = await readFile(path.join(packageDir, 'src/sidepanel/sidepanel.js'), 'utf8');
  const previewStart = source.indexOf('const initPreview = () =>');
  const previewEnd = source.indexOf('const startOperationalControllers', previewStart);
  const previewSource = source.slice(previewStart, previewEnd);
  assert.match(previewSource, /state\.privacyConsent = \{ version: PRIVACY_NOTICE_VERSION/u);
  assert.doesNotMatch(previewSource, /chrome\.storage\.(?:local|session)\.(?:set|remove)/u);

  const withdrawalStart = source.indexOf('const withdrawPrivacyConsent = async');
  const withdrawalEnd = source.indexOf('const installSettingsActions', withdrawalStart);
  const withdrawalSource = source.slice(withdrawalStart, withdrawalEnd);
  assert.match(withdrawalSource, /chrome\.storage\.local\.remove\(PRIVACY_CONSENT_KEY\)/u);
  assert.match(withdrawalSource, /state\.vault\?\.lock\(\)/u);
  assert.match(withdrawalSource, /clearTrustedSessionCapabilities\(\)/u);
  assert.doesNotMatch(withdrawalSource, /vault\.(?:reset|delete)/u);
});

test('runtime Ollama origin is derived from chrome.runtime.id and cloud names fail settings validation', () => {
  assert.equal(
    ollamaOriginCommand('abcdefghijklmnopabcdefghijklmnop'),
    'OLLAMA_ORIGINS=chrome-extension://abcdefghijklmnopabcdefghijklmnop'
  );
  assert.throws(() => validateModelSettings({
    generationModel: 'qwen3:cloud',
    fallbackGenerationModel: 'qwen3:8b',
    embeddingModel: 'nomic-embed-text'
  }), error => error.code === 'REMOTE_MODEL_NOT_ALLOWED');
});

test('HTTP 403 reports the exact installed extension origin without a wildcard', () => {
  const presentation = ollamaHealthErrorPresentation(
    { status: 403, code: 'HTTP_ERROR', message: 'Forbidden' },
    'jigajpmnbiofgmgcnmdeechgibpjlfop'
  );
  assert.deepEqual(presentation, {
    title: 'Ollama origin not allowed',
    copy: 'Ollama returned HTTP 403. Set OLLAMA_ORIGINS=chrome-extension://jigajpmnbiofgmgcnmdeechgibpjlfop, fully quit Ollama, and restart it.'
  });
  assert.doesNotMatch(presentation.copy, /chrome-extension:\/\/\*/u);
  assert.deepEqual(
    ollamaHealthErrorPresentation(
      { code: 'CONNECTION_FAILED', message: 'Failed to fetch' },
      'jigajpmnbiofgmgcnmdeechgibpjlfop'
    ),
    { title: 'Ollama unavailable', copy: 'Failed to fetch' }
  );
});
