import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { unzipSync } from 'fflate';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const EXPECTED_PERMISSIONS = ['activeTab', 'scripting', 'sidePanel', 'storage'];
const EXPECTED_HOST_PERMISSIONS = ['http://127.0.0.1:11434/*'];
const EXPECTED_CSP = "default-src 'self'; base-uri 'none'; object-src 'none'; frame-src 'none'; script-src 'self'; style-src 'self'; worker-src 'self'; connect-src http://127.0.0.1:11434";
const REQUIRED_ICON_SIZES = ['16', '32', '48', '128'];
const decoder = new TextDecoder();

const fail = message => { throw new Error(`Invalid extension package: ${message}`); };

const readJson = (files, filename) => {
  const bytes = files[filename];
  if (!bytes) fail(`${filename} is missing`);
  try {
    return JSON.parse(decoder.decode(bytes));
  } catch {
    return fail(`${filename} is not valid JSON`);
  }
};

const pngDimensions = (bytes) => {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 24 || signature.some((value, index) => bytes[index] !== value)) fail('an icon is not a valid PNG');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16), height: view.getUint32(20) };
};

const assertReference = (files, relativePath) => {
  if (typeof relativePath !== 'string' || !relativePath || relativePath.startsWith('/') || relativePath.includes('..')) {
    fail(`unsafe manifest asset reference: ${relativePath}`);
  }
  if (!files[relativePath]) fail(`referenced asset is missing: ${relativePath}`);
};

export const validateExtensionArchive = (archiveBytes, { target = 'dev' } = {}) => {
  if (!['dev', 'store'].includes(target)) throw new Error(`Unknown package target: ${target}`);
  const files = unzipSync(archiveBytes);
  const paths = Object.keys(files).sort();
  if (!paths.length) fail('archive is empty');
  paths.forEach((filename) => {
    if (filename.startsWith('/') || filename.split('/').includes('..')) fail(`unsafe archive path: ${filename}`);
    if (/(?:^|\/)(?:node_modules|src|test|tests|store)(?:\/|$)/u.test(filename)) fail(`development-only path is packaged: ${filename}`);
    if (/\.(?:map|pem|p12|pfx|key)$/iu.test(filename)) fail(`prohibited file type is packaged: ${filename}`);
  });

  const manifest = readJson(files, 'manifest.json');
  if (target === 'store' && Object.hasOwn(manifest, 'key')) fail('Chrome Web Store manifest must not contain a key');
  if (target === 'dev' && !manifest.key) fail('development manifest must retain its stable key');
  if (manifest.minimum_chrome_version !== '120') fail('minimum_chrome_version must be 120');
  if (manifest.homepage_url !== 'https://www.danielshort.me/job-application-copilot') fail('homepage_url is incorrect');
  if (JSON.stringify([...(manifest.permissions || [])].sort()) !== JSON.stringify(EXPECTED_PERMISSIONS)) fail('permission surface changed');
  if (JSON.stringify(manifest.host_permissions || []) !== JSON.stringify(EXPECTED_HOST_PERMISSIONS)) fail('host permission surface changed');
  if (manifest.content_security_policy?.extension_pages !== EXPECTED_CSP) fail('extension page CSP changed');

  const referencedPaths = [
    manifest.background?.service_worker,
    manifest.side_panel?.default_path,
    ...(manifest.content_scripts || []).flatMap(contentScript => contentScript.js || [])
  ].filter(Boolean);
  referencedPaths.forEach(relativePath => assertReference(files, relativePath));

  for (const size of REQUIRED_ICON_SIZES) {
    const manifestIcon = manifest.icons?.[size];
    const actionIcon = manifest.action?.default_icon?.[size];
    if (!manifestIcon || actionIcon !== manifestIcon) fail(`manifest and action icon ${size} must use the same packaged PNG`);
    assertReference(files, manifestIcon);
    const dimensions = pngDimensions(files[manifestIcon]);
    if (dimensions.width !== Number(size) || dimensions.height !== Number(size)) {
      fail(`icon ${manifestIcon} must be ${size}x${size}`);
    }
  }

  const panelPath = manifest.side_panel?.default_path;
  if (panelPath) {
    const panelHtml = decoder.decode(files[panelPath]);
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
      assertReference(files, relativePath);
    }
  }

  const notices = files['THIRD_PARTY_NOTICES.txt'];
  if (!notices) fail('THIRD_PARTY_NOTICES.txt is missing');
  const noticeText = decoder.decode(notices);
  if (!/mammoth@/u.test(noticeText) || !/pdfjs-dist@/u.test(noticeText)) {
    fail('third-party notices must cover Mammoth and PDF.js');
  }
  return { files, manifest, paths };
};

const isMain = process.argv[1]
  && path.resolve(process.argv[1]).toLocaleLowerCase('en-US') === fileURLToPath(import.meta.url).toLocaleLowerCase('en-US');

if (isMain) {
  const targetArgument = process.argv.find(argument => argument.startsWith('--target='));
  const target = targetArgument?.slice('--target='.length) || 'dev';
  if (!['dev', 'store'].includes(target)) throw new Error(`Unknown package target: ${target}`);
  const packageMetadata = JSON.parse(await readFile(path.join(packageDir, 'package.json'), 'utf8'));
  const archiveName = target === 'store'
    ? `job-application-copilot-${packageMetadata.version}-chrome-web-store.zip`
    : `job-application-copilot-${packageMetadata.version}.zip`;
  const archivePath = path.join(packageDir, 'artifacts', archiveName);
  const result = validateExtensionArchive(new Uint8Array(await readFile(archivePath)), { target });
  process.stdout.write(`Validated ${archivePath} (${result.paths.length} files, ${target} target)\n`);
}
