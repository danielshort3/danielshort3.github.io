'use strict';

const { spawnSync } = require('node:child_process');

const workflowFiles = new Set([
  '.github/status-report-public.pem',
  '.github/workflows/status-coverage-hourly.yml',
  '.github/workflows/status-report-publish.yml',
  'aws/aws-status/README.md',
  'build/status-report-transport.cjs'
]);

const previous = process.env.VERCEL_GIT_PREVIOUS_SHA;
const current = process.env.VERCEL_GIT_COMMIT_SHA;
const commitId = /^[a-f0-9]{40}$/;

// Vercel continues building on exit 1. Use its last successful deployment
// to include any site changes left over from a skipped or failed build.
if (!commitId.test(previous || '') || !commitId.test(current || '')) {
  console.log('Building: no previous and current Vercel Git commits to compare.');
  process.exit(1);
}

const diff = spawnSync('git', [
  'diff', '--name-only', '-z', '--no-renames', previous, current, '--'
], { encoding: 'utf8' });

if (diff.error || diff.status !== 0) {
  console.log('Building: Git could not compare the Vercel commits.');
  process.exit(1);
}

const changedFiles = diff.stdout.split('\0').filter(Boolean);
if (changedFiles.length && changedFiles.every(file => workflowFiles.has(file))) {
  console.log('Skipping Vercel build: only hourly status workflow files changed.');
  process.exit(0);
}

console.log('Building: comparison includes site changes or has an empty diff.');
process.exit(1);
