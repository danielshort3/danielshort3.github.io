import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { generateThirdPartyNotices } from './generate-third-party-notices.mjs';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourceDir = path.join(packageDir, 'src');
const outputDir = path.join(packageDir, 'dist');

const assertInsidePackage = (targetPath) => {
  const relative = path.relative(packageDir, path.resolve(targetPath));
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Refusing to modify path outside the extension package: ${targetPath}`);
  }
};

const fileExists = async (filePath) => {
  try {
    return (await stat(filePath)).isFile();
  } catch {
    return false;
  }
};

const addOptionalEntry = async (entries, relativeSourcePath) => {
  const sourcePath = path.join(sourceDir, relativeSourcePath);
  if (await fileExists(sourcePath)) entries.push(sourcePath);
};

const copySidepanelStatic = async () => {
  const sidepanelDir = path.join(sourceDir, 'sidepanel');
  try {
    const entries = await readdir(sidepanelDir, { withFileTypes: true });
    await mkdir(path.join(outputDir, 'sidepanel'), { recursive: true });
    for (const entry of entries) {
      if (!entry.isFile() || !/\.(?:css|html)$/.test(entry.name)) continue;
      await cp(path.join(sidepanelDir, entry.name), path.join(outputDir, 'sidepanel', entry.name));
    }
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
};

const validateManifest = (manifest) => {
  const expectedPermissions = ['activeTab', 'scripting', 'sidePanel', 'storage'];
  const expectedExtensionCsp = "default-src 'self'; base-uri 'none'; object-src 'none'; frame-src 'none'; script-src 'self'; style-src 'self'; worker-src 'self'; connect-src http://127.0.0.1:11434";
  const actualPermissions = [...(manifest.permissions || [])].sort();
  if (JSON.stringify(actualPermissions) !== JSON.stringify(expectedPermissions)) {
    throw new Error(`Unexpected extension permissions: ${actualPermissions.join(', ')}`);
  }
  if (JSON.stringify(manifest.host_permissions) !== JSON.stringify(['http://127.0.0.1:11434/*'])) {
    throw new Error('The extension must only request the loopback Ollama host permission.');
  }
  if (manifest.content_security_policy?.extension_pages !== expectedExtensionCsp) {
    throw new Error('The extension page CSP must remain self-only except for the loopback Ollama endpoint.');
  }
  if (!manifest.key) throw new Error('The manifest must retain its stable development key.');
  if (manifest.minimum_chrome_version !== '120') throw new Error('The extension must require Chrome 120 or newer.');
  if (manifest.homepage_url !== 'https://www.danielshort.me/job-application-copilot') {
    throw new Error('The manifest homepage must use the canonical product URL.');
  }
  const expectedIcons = {
    16: 'assets/icons/icon-16.png',
    32: 'assets/icons/icon-32.png',
    48: 'assets/icons/icon-48.png',
    128: 'assets/icons/icon-128.png'
  };
  if (JSON.stringify(manifest.icons) !== JSON.stringify(expectedIcons)
    || JSON.stringify(manifest.action?.default_icon) !== JSON.stringify(expectedIcons)) {
    throw new Error('The manifest and action must reference all required runtime icons.');
  }
};

assertInsidePackage(outputDir);
await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });

const moduleEntries = [
  path.join(sourceDir, 'background/service-worker.js'),
  path.join(sourceDir, 'workers/document-parser.worker.js')
];
await addOptionalEntry(moduleEntries, 'sidepanel/sidepanel.js');

const bundledInputs = new Set();
const moduleBuild = await build({
  entryPoints: moduleEntries,
  outdir: outputDir,
  outbase: sourceDir,
  entryNames: '[dir]/[name]',
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: ['chrome120'],
  sourcemap: false,
  legalComments: 'none',
  metafile: true,
  logLevel: 'info'
});
Object.keys(moduleBuild.metafile.inputs).forEach(input => bundledInputs.add(input));

const contentEntries = [];
await addOptionalEntry(contentEntries, 'content/tracker-bridge.js');
await addOptionalEntry(contentEntries, 'content/page-runtime.js');
if (contentEntries.length) {
  const contentBuild = await build({
    entryPoints: contentEntries,
    outdir: outputDir,
    outbase: sourceDir,
    entryNames: '[dir]/[name]',
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: ['chrome120'],
    sourcemap: false,
    legalComments: 'none',
    metafile: true,
    logLevel: 'info'
  });
  Object.keys(contentBuild.metafile.inputs).forEach(input => bundledInputs.add(input));
}

const manifestPath = path.join(packageDir, 'manifest.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
validateManifest(manifest);
await writeFile(path.join(outputDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
await copySidepanelStatic();
const noticePackageCount = await generateThirdPartyNotices({
  packageDir,
  inputPaths: [...bundledInputs],
  outputPath: path.join(outputDir, 'THIRD_PARTY_NOTICES.txt')
});
if (!noticePackageCount) throw new Error('No bundled third-party packages were detected for notices.');

const assetsDir = path.join(sourceDir, 'assets');
try {
  await cp(assetsDir, path.join(outputDir, 'assets'), { recursive: true });
} catch (error) {
  if (error?.code !== 'ENOENT') throw error;
}
