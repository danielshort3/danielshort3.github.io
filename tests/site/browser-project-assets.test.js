'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const {
  BROWSER_PROJECT_IDS, BROWSER_PROJECT_DOCUMENTS, BROWSER_PROJECT_DEMOS
} = require('../../build/lib/browser-project-assets');
const { resolveApprovedDocuments, copyApprovedDocuments } = require('../../build/copy-to-public');

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
const publishedDocuments = fs.readdirSync(path.join(root, 'public/documents'), { withFileTypes: true });
assert(publishedDocuments.every((entry) => entry.isFile()), 'public/documents must contain only approved files');
assert.deepEqual(
  publishedDocuments.map((entry) => entry.name).sort(),
  BROWSER_PROJECT_DOCUMENTS.map((relative) => path.basename(relative)).sort(),
  'public/documents must contain exactly the approved project downloads'
);

// A local document mentioned in repository text or an authored page must not
// become publishable. Keep this fixture outside the checkout and clean it up.
const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'site-documents-'));
try {
  const sourceRoot = path.join(fixtureRoot, 'source');
  const destinationRoot = path.join(fixtureRoot, 'public');
  fs.mkdirSync(path.join(sourceRoot, 'documents'), { recursive: true });
  fs.mkdirSync(path.join(sourceRoot, 'pages'), { recursive: true });
  BROWSER_PROJECT_DOCUMENTS.forEach((relative) => {
    fs.writeFileSync(path.join(sourceRoot, relative), `approved: ${relative}`);
  });
  fs.writeFileSync(path.join(sourceRoot, 'documents', 'unapproved.pdf'), 'unapproved');
  fs.writeFileSync(path.join(sourceRoot, 'README.md'), 'See documents/unapproved.pdf');
  fs.writeFileSync(path.join(sourceRoot, 'pages', 'extra.html'), '<a href="/documents/unapproved.pdf">Extra</a>');

  copyApprovedDocuments(resolveApprovedDocuments(sourceRoot), destinationRoot);
  assert.deepEqual(
    fs.readdirSync(path.join(destinationRoot, 'documents')).sort(),
    BROWSER_PROJECT_DOCUMENTS.map((relative) => path.basename(relative)).sort(),
    'references outside the allowlist must not expand published documents'
  );

  const missingDocument = BROWSER_PROJECT_DOCUMENTS[0];
  fs.rmSync(path.join(sourceRoot, missingDocument));
  assert.throws(
    () => resolveApprovedDocuments(sourceRoot),
    { message: `Missing browser-project asset: ${missingDocument}` },
    'a missing approved document must stop publication'
  );
} finally {
  assert.equal(path.dirname(path.resolve(fixtureRoot)), path.resolve(os.tmpdir()));
  fs.rmSync(fixtureRoot, { recursive: true, force: true });
}

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
