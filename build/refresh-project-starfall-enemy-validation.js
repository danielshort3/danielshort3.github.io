#!/usr/bin/env node
'use strict';

// Refresh file-integrity evidence after a reviewed import. This is not a motion
// approval and never writes artwork, source configs, inventory or runtime data.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_ROOT = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/enemies');
const REPORT_PATH = path.join(SOURCE_ROOT, 'validation-report.json');
const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const writeJson = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);

async function main() {
  const report = readJson(REPORT_PATH);
  const inventory = readJson(path.join(SOURCE_ROOT, 'inventory.json'));
  const registrations = readJson(path.join(SOURCE_ROOT, 'enemy-registration.json'));
  const verifiedAt = new Date().toISOString();
  const pendingEvidence = [];
  const importedHashes = new Map();
  let originalBackupsVerified = 0;
  for (const item of report.items) {
    const record = inventory.items.find(entry => entry.fileId === item.fileId);
    if (!record?.replacement) throw new Error(`${item.fileId}: missing inventory replacement.`);
    const canonicalId = record.aliasOf || record.replacement.aliasOf || item.fileId;
    const folder = path.join(SOURCE_ROOT, canonicalId);
    const config = readJson(path.join(folder, 'source.json'));
    const rawHash = hash(path.join(folder, config.source));
    if (rawHash !== item.sourceSha256 || rawHash !== config.sourceSha256 || rawHash !== record.replacement.sourceSha256) {
      throw new Error(`${item.fileId}: source hash changed; refresh cannot approve new artwork.`);
    }
    const productionHash = hash(path.join(ROOT, record.replacement.sheet));
    if (productionHash !== record.replacement.sheetSha256 || productionHash !== hash(path.join(folder, 'review-sheet.png'))) {
      throw new Error(`${item.fileId}: production does not match inventory and reviewed candidate.`);
    }
    if (JSON.stringify(registrations[item.fileId]) !== JSON.stringify(record.replacement.registration)) {
      throw new Error(`${item.fileId}: registration metadata mismatch.`);
    }
    for (const patch of Object.values(config.supplementalRows || {})) {
      const file = patch.repoPath ? path.join(ROOT, patch.repoPath) : path.join(folder, patch.file);
      if (patch.sha256 && hash(file) !== patch.sha256) throw new Error(`${item.fileId}: supplemental source hash changed.`);
    }
    for (const original of record.originals) {
      if (hash(path.join(ROOT, original.backupPath)) !== original.sha256) throw new Error(`${item.fileId}: original backup hash changed.`);
      originalBackupsVerified += 1;
    }
    const { data, info } = await sharp(path.join(ROOT, record.replacement.sheet)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    if (info.width !== 960 || info.height !== 1280) throw new Error(`${item.fileId}: wrong production sheet dimensions.`);
    let transparent = 0;
    let edgePixels = 0;
    for (let y = 0; y < info.height; y += 1) {
      for (let x = 0; x < info.width; x += 1) {
        const alpha = data[(y * info.width + x) * 4 + 3];
        if (!alpha) transparent += 1;
        if (alpha > 32 && (x % 160 === 0 || x % 160 === 159 || y % 160 === 0 || y % 160 === 159)) edgePixels += 1;
      }
    }
    if (edgePixels) throw new Error(`${item.fileId}: visible pixels touch cell boundaries.`);
    item.sheet = record.replacement.sheet;
    item.portrait = record.replacement.portrait;
    item.sheetSha256 = productionHash;
    item.portraitSha256 = hash(path.join(ROOT, item.portrait));
    item.transparentFraction = transparent / (info.width * info.height);
    item.edgePixels = edgePixels;
    item.registration = registrations[item.fileId];
    if (config.registrationReview) item.registrationReviewScope = config.registrationReview.scope;
    importedHashes.set(item.fileId, productionHash);
  }
  for (const source of report.protectedSourcesVerified) {
    if (hash(path.join(ROOT, source.path)) !== source.sha256) throw new Error(`${source.fileId}: protected study hash changed.`);
  }
  const aggregatePath = path.join(ROOT, 'output/enemy-idle-alignment/confirmed-idle-corrections.json');
  if (fs.existsSync(aggregatePath)) {
    const aggregate = readJson(aggregatePath);
    for (const evidence of aggregate) {
      const productionHash = importedHashes.get(evidence.id);
      if (productionHash !== evidence.afterReviewSheetHash) throw new Error(`${evidence.id}: evidence does not match production.`);
      evidence.productionImported = true;
      evidence.productionSheetSha256 = productionHash;
      evidence.productionVerifiedAt = verifiedAt;
      const evidencePath = path.join(ROOT, 'output/enemy-idle-alignment', evidence.id, 'alignment-evidence.json');
      if (fs.existsSync(evidencePath)) {
        const individual = readJson(evidencePath);
        if (individual.afterReviewSheetHash !== productionHash) throw new Error(`${evidence.id}: individual evidence does not match production.`);
        Object.assign(individual, { productionImported: true, productionSheetSha256: productionHash, productionVerifiedAt: verifiedAt });
        pendingEvidence.push([evidencePath, individual]);
      }
    }
    pendingEvidence.push([aggregatePath, aggregate]);
  }
  for (const id of ['clockwork-titan', 'rimewarden', 'dew-slime', 'brambleking', 'index-scribe']) {
    const folder = path.join(ROOT, 'output/enemy-idle-alignment', id);
    const evidencePath = path.join(folder, 'verification.json');
    if (!fs.existsSync(evidencePath)) continue;
    const productionHash = importedHashes.get(id);
    if (hash(path.join(folder, 'after-sheet.png')) !== productionHash) throw new Error(`${id}: verification candidate does not match production.`);
    const evidence = readJson(evidencePath);
    Object.assign(evidence, { productionImported: true, productionSheetSha256: productionHash, productionVerifiedAt: verifiedAt });
    pendingEvidence.push([evidencePath, evidence]);
  }
  const lavaPath = path.join(ROOT, 'output/lava-tick-alignment/evidence.json');
  if (fs.existsSync(lavaPath)) {
    const evidence = readJson(lavaPath);
    const productionHash = importedHashes.get('lava-tick');
    if (evidence.afterSheetSha256 !== productionHash) throw new Error('lava-tick: alignment evidence does not match production.');
    Object.assign(evidence, { productionImported: true, productionSheetSha256: productionHash, productionVerifiedAt: verifiedAt });
    pendingEvidence.push([lavaPath, evidence]);
  }
  report.outputIntegrityRefreshedAt = verifiedAt;
  report.outputIntegrityMethod = 'Actual production PNG SHA256 must equal inventory and reviewed candidate. Raw/supplemental/protected source hashes and original backup hashes must remain unchanged. Transparent fraction counts alpha=0; edgePixels counts alpha>32 on every 160px cell boundary. This refresh does not certify motion or anatomical alignment.';
  report.originalBackupsVerified = originalBackupsVerified;
  writeJson(REPORT_PATH, report);
  for (const [file, evidence] of pendingEvidence) writeJson(file, evidence);
  console.log(JSON.stringify({ productionSheetsVerified: report.items.length, originalBackupsVerified, protectedStudiesVerified: report.protectedSourcesVerified.length, scopedEvidenceFilesUpdated: pendingEvidence.length, edgePixels: 0 }));
}

main().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
