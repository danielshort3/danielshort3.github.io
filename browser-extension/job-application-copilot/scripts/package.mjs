import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { zipSync } from 'fflate';
import { validateExtensionArchive } from './validate-package.mjs';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputDir = path.join(packageDir, 'dist');
const artifactsDir = path.join(packageDir, 'artifacts');
const targetArgument = process.argv.find(argument => argument.startsWith('--target='));
const target = targetArgument?.slice('--target='.length) || 'dev';
if (!['dev', 'store'].includes(target)) throw new Error('Unknown package target: ' + target);

const collectFiles = async (directory, relativeDirectory = '') => {
  const files = {};
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const relativePath = path.posix.join(relativeDirectory.replaceAll('\\', '/'), entry.name);
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      Object.assign(files, await collectFiles(absolutePath, relativePath));
    } else if (entry.isFile()) {
      files[relativePath] = new Uint8Array(await readFile(absolutePath));
    }
  }
  return files;
};

const manifestPath = path.join(outputDir, 'manifest.json');
if (!(await stat(manifestPath).catch(() => null))?.isFile()) {
  throw new Error('Run npm run build before packaging the extension.');
}

const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const assertBuiltAsset = async (relativePath) => {
  const target = path.resolve(outputDir, relativePath);
  const relative = path.relative(outputDir, target);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Manifest asset escapes the extension build: ${relativePath}`);
  }
  if (!(await stat(target).catch(() => null))?.isFile()) {
    throw new Error(`Manifest asset is missing from the extension build: ${relativePath}`);
  }
  return target;
};
const requiredPaths = [
  manifest.background?.service_worker,
  manifest.side_panel?.default_path,
  ...(manifest.content_scripts || []).flatMap(contentScript => contentScript.js || []),
  ...Object.values(manifest.icons || {}),
  ...Object.values(manifest.action?.default_icon || {})
].filter(Boolean);
for (const relativePath of requiredPaths) {
  await assertBuiltAsset(relativePath);
}
if (manifest.side_panel?.default_path) {
  const panelPath = manifest.side_panel.default_path;
  const panelHtml = await readFile(path.join(outputDir, panelPath), 'utf8');
  const localReferences = [...panelHtml.matchAll(/\b(?:href|src)="([^"]+)"/gu)]
    .map(match => match[1])
    .filter(reference => reference
      && !reference.startsWith('#')
      && !/^[A-Za-z][A-Za-z0-9+.-]*:/u.test(reference));
  for (const reference of localReferences) {
    const cleanReference = reference.split(/[?#]/u)[0];
    const relativePath = cleanReference.startsWith('/')
      ? cleanReference.slice(1)
      : path.posix.join(path.posix.dirname(panelPath), cleanReference);
    await assertBuiltAsset(relativePath);
  }
}
const files = await collectFiles(outputDir);
const packagedManifest = structuredClone(manifest);
if (target === 'store') delete packagedManifest.key;
files['manifest.json'] = new TextEncoder().encode(JSON.stringify(packagedManifest, null, 2) + '\n');
const archiveBytes = zipSync(files, { level: 9 });
validateExtensionArchive(archiveBytes, { target });

const archiveName = target === 'store'
  ? `job-application-copilot-${manifest.version}-chrome-web-store.zip`
  : `job-application-copilot-${manifest.version}.zip`;
await mkdir(artifactsDir, { recursive: true });
await writeFile(path.join(artifactsDir, archiveName), archiveBytes);
console.log(path.join(artifactsDir, archiveName));
