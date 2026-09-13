#!/usr/bin/env node
'use strict';

// Refresh from an official Tableau PNG, or a faithfully rasterized official SVG.
// Verify SVG fonts and full geometry first; see design/tableau/README.md.
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '../..');
const PROJECT_IDS = new Set(['pizzaDashboard', 'ufoDashboard']);
const USAGE = 'Usage: node build/tableau/update-dashboard-previews.js <pizzaDashboard|ufoDashboard> <published-export.png>';

function writeIfChanged(filePath, contents) {
  if (fs.existsSync(filePath) && fs.readFileSync(filePath).equals(contents)) return false;
  fs.writeFileSync(filePath, contents);
  return true;
}

function readProject(projectPath, projectId) {
  const project = JSON.parse(fs.readFileSync(projectPath, 'utf8'));
  if (project.id !== projectId || !project.mobilePreview || typeof project.mobilePreview !== 'object') {
    throw new Error(`Unexpected project metadata in ${projectPath}`);
  }
  return project;
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 1 && ['--help', '-h'].includes(args[0])) {
    process.stdout.write(`${USAGE}\n`);
    return;
  }
  const [projectId, inputPath] = args;
  if (args.length !== 2 || !PROJECT_IDS.has(projectId)) throw new Error(USAGE);

  const sourcePath = path.resolve(inputPath);
  if (!fs.statSync(sourcePath).isFile()) throw new Error('The PNG export must be a file.');
  const source = fs.readFileSync(sourcePath);
  const metadata = await sharp(source, { failOn: 'error' }).metadata();
  if (metadata.format !== 'png' || !metadata.width || !metadata.height || (metadata.pages || 1) !== 1) {
    throw new Error('Use a single-image PNG from the published Tableau export.');
  }

  const projectPath = path.join(ROOT, 'content/projects', `${projectId}.json`);
  readProject(projectPath, projectId);
  const base = path.join(ROOT, 'img/projects', projectId);
  const outputs = [{ filePath: `${base}.png`, data: source, width: metadata.width, height: metadata.height }];

  // Encode everything before replacing any existing preview assets.
  for (const width of [null, 640, 960]) {
    for (const format of ['webp', 'avif']) {
      let pipeline = sharp(source, { failOn: 'error' });
      if (width) pipeline = pipeline.resize({ width, withoutEnlargement: true });
      const encoded = format === 'webp'
        ? pipeline.webp({ quality: 92, effort: 6, smartSubsample: true })
        : pipeline.avif({ quality: 68, effort: 6, chromaSubsampling: '4:4:4' });
      const { data, info } = await encoded.toBuffer({ resolveWithObject: true });
      outputs.push({
        filePath: `${base}${width ? `-${width}` : ''}.${format}`,
        data,
        width: info.width,
        height: info.height
      });
    }
  }
  const preview = await sharp(source, { failOn: 'error' })
    .resize({ width: 1280, withoutEnlargement: true })
    .webp({ quality: 94, effort: 6, smartSubsample: true })
    .toBuffer({ resolveWithObject: true });
  outputs.push({
    filePath: `${base}-preview.webp`,
    data: preview.data,
    width: preview.info.width,
    height: preview.info.height
  });

  // Re-read after encoding so concurrent copy edits are retained.
  const project = readProject(projectPath, projectId);
  project.imageWidth = metadata.width;
  project.imageHeight = metadata.height;
  project.mobilePreview.width = preview.info.width;
  project.mobilePreview.height = preview.info.height;
  for (const output of outputs) {
    const changed = writeIfChanged(output.filePath, output.data);
    process.stdout.write(`[tableau-preview] ${path.relative(ROOT, output.filePath)} ${output.width}x${output.height}${changed ? '' : ' unchanged'}\n`);
  }
  writeIfChanged(projectPath, Buffer.from(`${JSON.stringify(project, null, 2)}\n`));
  process.stdout.write(`[tableau-preview] Updated dimensions in content/projects/${projectId}.json\n`);
}

main().catch((error) => {
  process.stderr.write(`[tableau-preview] ${error.message}\n`);
  process.exitCode = 1;
});
