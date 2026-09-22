'use strict';
const { defineConfig } = require('@playwright/test');
const path = require('node:path');
const baseURL = `http://127.0.0.1:${process.env.RELEASE_TEST_PORT || 4219}`;
const reportPrefix = process.env.RELEASE_REPORT_PREFIX || 'tmp/release';
const reportPath = suffix => path.resolve(__dirname, '../..', `${reportPrefix}${suffix}`);
module.exports = defineConfig({
  testDir: __dirname,
  timeout: 90000,
  expect: { timeout: 15000, toHaveScreenshot: { animations: 'disabled', caret: 'hide', threshold: 0.2, maxDiffPixelRatio: 0.003 } },
  fullyParallel: false,
  workers: process.env.CI ? 2 : 1,
  retries: 0,
  forbidOnly: Boolean(process.env.CI),
  updateSnapshots: 'none',
  outputDir: reportPath('-results'),
  reporter: [['list'], ['html', { outputFolder: reportPath('-report'), open: 'never' }], ['json', { outputFile: reportPath('-results/report.json') }]],
  use: { baseURL, viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce', serviceWorkers: 'block', actionTimeout: 15000, navigationTimeout: 20000, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  webServer: { command: 'node tests/release/server.cjs', cwd: path.resolve(__dirname, '../..'), url: baseURL, reuseExistingServer: !process.env.CI, timeout: 30000 },
  projects: [
    ...['chromium', 'firefox', 'webkit'].map((browserName) => ({ name: browserName, testMatch: ['release.spec.cjs', 'library-links.spec.cjs'], use: { browserName } })),
    { name: 'visual', testMatch: 'visual.spec.cjs', use: { browserName: 'chromium' }, snapshotPathTemplate: '{testDir}/baselines/{arg}{ext}' }
  ]
});
