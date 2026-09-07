'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const {
  BROWSER_PROJECT_IDS, BROWSER_PROJECT_DOCUMENTS, BROWSER_PROJECT_DEMOS
} = require('../../build/lib/browser-project-assets');

const root = path.resolve(__dirname, '../..');
const read = (relative) => fs.readFileSync(path.join(root, relative), 'utf8');
const hash = (relative) => createHash('sha256').update(fs.readFileSync(path.join(root, relative))).digest('hex');
const awsUrl = /https?:\/\/[^\s"'<>]*(?:amazonaws\.com|\.on\.aws|amazoncognito\.com)/i;
let checked = 0;

function verifyPublishedAsset(relative) {
  assert(fs.statSync(path.join(root, relative)).size > 0, `${relative} must contain a real asset`);
  assert.equal(hash(`public/${relative}`), hash(relative), `${relative} must be shipped unchanged by the build`);
  checked += 1;
}

for (const id of BROWSER_PROJECT_IDS) {
  const source = read(`content/projects/${id}.json`);
  const project = JSON.parse(source);
  assert(!awsUrl.test(source), `${id} must not link its project content to AWS`);
  const published = read(`public/pages/portfolio/${id}.html`);
  for (const resource of Array.isArray(project.resources) ? project.resources : [project.resources]) {
    if (!resource?.url || !/^\/?documents\//.test(resource.url)) continue;
    const relative = resource.url.replace(/^\//, '');
    assert(BROWSER_PROJECT_DOCUMENTS.includes(relative), `${id} download must be explicitly allowlisted`);
    assert(published.includes(relative), `${id} published page must retain its self-hosted download`);
  }
}

for (const relative of BROWSER_PROJECT_DOCUMENTS) verifyPublishedAsset(relative);
for (const id of BROWSER_PROJECT_DEMOS) {
  const source = read(`demos/${id}-demo.html`);
  assert(!awsUrl.test(source), `${id} demo must not reference an AWS URL`);
  assert(!/DemoAws|js\/demos\/aws-client\.js|\/api\/demos\//.test(source), `${id} must work without an AWS demo API`);
  for (const match of source.matchAll(/<script\b[^>]*\bsrc="\/?(js\/demos\/[^"?#]+)"/g)) {
    verifyPublishedAsset(match[1]);
  }
  assert(read(`public/demos/${id}-demo.html`).includes('id="main"'), `${id} built demo must retain its main landmark`);
}

function checkDataDirectory(relative) {
  for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
    const child = `${relative}/${entry.name}`;
    if (entry.isDirectory()) checkDataDirectory(child);
    else if (entry.isFile()) verifyPublishedAsset(child);
  }
}
checkDataDirectory('demos/data');
console.log(`Browser project packaging passed: seven projects, five demos, ${checked} published assets.`);
