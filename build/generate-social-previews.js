#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { loadSocialPreviewRecords, renderSocialPreview } = require('./lib/social-previews');
const root = path.resolve(__dirname, '..');

async function main() {
  const records = loadSocialPreviewRecords(root);
  let updated = 0;
  for (const record of records) {
    const image = await renderSocialPreview(record, { root });
    const destination = path.join(root, record.file);
    if (fs.existsSync(destination) && fs.readFileSync(destination).equals(image)) continue;
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, image);
    updated++;
  }
  process.stdout.write(`[social-previews] ${records.length} tool/game cards; ${updated} updated.\n`);
}

if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
module.exports = { main };
