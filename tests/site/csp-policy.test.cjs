'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { inlineScripts, hashesForBuild, updateConfig, ADMIN_SOURCE } = require('../../build/csp-hashes.cjs');
const config = require('../../vercel.json');
const policies = config.headers.flatMap(rule => rule.headers.filter(h => h.key === 'Content-Security-Policy').map(h => ({ source: rule.source, value: h.value })));
const directive = (policy, name) => policy.split(';').map(p => p.trim().split(/\s+/)).find(p => p[0] === name)?.slice(1);
test('public policies block arbitrary inline scripts and event handlers', () => {
  assert.ok(policies.length >= 11);
  for (const { source, value } of policies.filter(p => p.source !== ADMIN_SOURCE)) {
    for (const name of ['script-src', 'script-src-elem']) assert.ok(!directive(value, name)?.includes("'unsafe-inline'"), `${source}: ${name}`);
    assert.deepEqual(directive(value, 'script-src-attr'), ["'none'"], source);
  }
});
test('private tracker aliases retain same-origin script and framing restrictions', () => {
  const tracker = policies.filter(p => p.source.includes('job-application-tracker') && !p.source.startsWith('/:path('));
  assert.equal(tracker.length, 4);
  for (const { value } of tracker) {
    assert.deepEqual(directive(value, 'script-src'), ["'self'"]);
    assert.deepEqual(directive(value, 'frame-ancestors'), ["'none'"]);
  }
});
test('exact built demo hashes match the committed allowlist', () => {
  const report = hashesForBuild();
  assert.ok(report.count > 0 && report.hashes.length > 0);
  assert.deepEqual(updateConfig(config, report.hashes), config);
});
test('JSON data is inert and inline hashing preserves executable whitespace', () => {
  assert.deepEqual(inlineScripts('<script type="application/ld+json">{}</script><script src="/a.js"></script>'), []);
  assert.deepEqual(inlineScripts('<script>\r\n  run();\r\n</script>'), ['\n  run();\n']);
});
test('CMS compatibility exception cannot silently expand to public pages', () => {
  const exceptions = policies.filter(p => directive(p.value, 'script-src')?.includes("'unsafe-inline'"));
  assert.deepEqual(exceptions.map(p => p.source), [ADMIN_SOURCE]);
  assert.ok(fs.readFileSync('docs/RELEASE_SECURITY.md', 'utf8').includes('CMS preview'));
});
test('Vercel uses the locked install and exact-commit production guard', () => {
  assert.equal(config.installCommand, 'npm ci');
  assert.equal(config.buildCommand, 'node build/verify-production-release.cjs && npm run build');
});
