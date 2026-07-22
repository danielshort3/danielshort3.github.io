'use strict';

const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const PACKAGE_ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIRECTORY = path.join(PACKAGE_ROOT, 'dist');
const OUTPUT_PATH = path.join(OUTPUT_DIRECTORY, 'job-application-tracker.zip');
const INSTALL_MANIFEST_FILES = Object.freeze(['package.json', 'package-lock.json']);
const RUNTIME_ROOT_FILES = Object.freeze(['index.js']);
const ARCHIVE_DATE = new Date('2000-01-01T00:00:00.000Z');
const NON_RUNTIME_DIRECTORY_NAMES = new Set([
  '.bin',
  '.github',
  'benchmark',
  'benchmarks',
  'doc',
  'docs',
  'example',
  'examples',
  'test',
  'tests'
]);

const normalizeArchivePath = value => String(value || '').replace(/\\/g, '/');

const isExcludedDependencyPath = relativePath => {
  const normalized = normalizeArchivePath(relativePath);
  const segments = normalized.split('/').filter(Boolean);
  const basename = (segments.at(-1) || '').toLowerCase();

  let index = segments[0]?.startsWith('@') ? 2 : 1;
  while (index < segments.length - 1) {
    if (segments[index].toLowerCase() === 'node_modules') {
      index += segments[index + 1]?.startsWith('@') ? 3 : 2;
      continue;
    }
    if (NON_RUNTIME_DIRECTORY_NAMES.has(segments[index].toLowerCase())) return true;
    index += 1;
  }
  if (/\.(?:zip|map|md|markdown)$/i.test(basename)) return true;
  if (/^(?:changelog|contributing|history|readme)(?:\.|$)/i.test(basename)) return true;
  if (/^(?:test|tests)\.(?:c?js|mjs|json)$/i.test(basename)) return true;
  if (/\.(?:spec|test)\.(?:c?js|mjs|json)$/i.test(basename)) return true;
  if (/^template\.ya?ml$/i.test(basename)) return true;
  return false;
};

const collectFiles = (directory, prefix = '') => {
  const entries = fs.readdirSync(directory, { withFileTypes: true })
    .sort((left, right) => left.name.localeCompare(right.name, 'en'));
  const files = [];

  for (const entry of entries) {
    const relativePath = normalizeArchivePath(path.join(prefix, entry.name));
    const absolutePath = path.join(directory, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`Refusing to package symbolic link: ${relativePath}`);
    }
    if (entry.isDirectory()) {
      files.push(...collectFiles(absolutePath, relativePath));
      continue;
    }
    if (!entry.isFile()) throw new Error(`Unsupported package entry: ${relativePath}`);
    files.push(relativePath);
  }

  return files;
};

const validateArchiveEntries = entries => {
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error('Lambda archive must contain runtime files.');
  }

  const normalizedEntries = entries.map(normalizeArchivePath);
  const uniqueEntries = new Set(normalizedEntries);
  if (uniqueEntries.size !== normalizedEntries.length) {
    throw new Error('Lambda archive contains duplicate paths.');
  }

  for (let entryIndex = 0; entryIndex < normalizedEntries.length; entryIndex += 1) {
    const entry = normalizedEntries[entryIndex];
    if (String(entries[entryIndex]) !== entry) {
      throw new Error(`Unsafe Lambda archive path separator: ${entries[entryIndex]}`);
    }
    const segments = entry.split('/');
    if (!entry || entry.startsWith('/') || segments.includes('..') || segments.includes('.')) {
      throw new Error(`Unsafe Lambda archive path: ${entry}`);
    }
    if (entry === 'index.js') continue;
    if (!entry.startsWith('node_modules/')) {
      throw new Error(`Lambda archive entry is outside the runtime allowlist: ${entry}`);
    }
    const dependencyPath = entry.slice('node_modules/'.length);
    if (!dependencyPath || isExcludedDependencyPath(dependencyPath)) {
      throw new Error(`Lambda archive contains a non-runtime dependency artifact: ${entry}`);
    }
  }

  if (!uniqueEntries.has('index.js')) throw new Error('Lambda archive is missing index.js.');
  if (!normalizedEntries.some(entry => entry.startsWith('node_modules/'))) {
    throw new Error('Lambda archive is missing production dependencies.');
  }

  return normalizedEntries.sort((left, right) => left.localeCompare(right, 'en'));
};

const pruneNonRuntimeDependencyFiles = nodeModulesDirectory => {
  const visit = directory => {
    const entries = fs.readdirSync(directory, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(directory, entry.name);
      const relativePath = path.relative(nodeModulesDirectory, absolutePath);
      if (isExcludedDependencyPath(relativePath)) {
        fs.rmSync(absolutePath, { recursive: true, force: true });
        continue;
      }
      if (entry.isDirectory()) visit(absolutePath);
    }
  };

  visit(nodeModulesDirectory);
};

const createDeterministicArchive = async (stagingDirectory, entries, outputPath) => {
  const { ZipArchive } = await import('archiver');
  if (typeof ZipArchive !== 'function') {
    throw new Error('Archiver did not provide the ZipArchive constructor.');
  }

  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(outputPath, { flags: 'wx' });
    const archive = new ZipArchive({ zlib: { level: 9 } });
    let settled = false;

    const fail = error => {
      if (settled) return;
      settled = true;
      reject(error);
    };

    output.on('close', () => {
      if (settled) return;
      settled = true;
      resolve();
    });
    output.on('error', fail);
    archive.on('error', fail);
    archive.pipe(output);

    for (const entry of entries) {
      const source = fs.readFileSync(path.join(stagingDirectory, ...entry.split('/')));
      archive.append(source, {
        date: ARCHIVE_DATE,
        mode: 0o644,
        name: entry
      });
    }
    archive.finalize().catch(fail);
  });
};

const findEndOfCentralDirectoryOffset = buffer => {
  const minimumOffset = Math.max(0, buffer.length - 65_557);
  for (let offset = buffer.length - 22; offset >= minimumOffset; offset -= 1) {
    if (buffer.readUInt32LE(offset) === 0x06054b50) return offset;
  }
  throw new Error('Lambda archive is missing a ZIP end-of-central-directory record.');
};

const readZipEntries = zipPath => {
  const buffer = fs.readFileSync(zipPath);
  const endOffset = findEndOfCentralDirectoryOffset(buffer);
  const entryCount = buffer.readUInt16LE(endOffset + 10);
  const centralDirectorySize = buffer.readUInt32LE(endOffset + 12);
  const centralDirectoryOffset = buffer.readUInt32LE(endOffset + 16);
  const centralDirectoryEnd = centralDirectoryOffset + centralDirectorySize;

  if (centralDirectoryEnd > endOffset || centralDirectoryEnd > buffer.length) {
    throw new Error('Lambda archive has an invalid ZIP central directory.');
  }

  const entries = [];
  let offset = centralDirectoryOffset;
  while (offset < centralDirectoryEnd) {
    if (buffer.readUInt32LE(offset) !== 0x02014b50) {
      throw new Error('Lambda archive has an invalid ZIP central-directory entry.');
    }
    const nameLength = buffer.readUInt16LE(offset + 28);
    const extraLength = buffer.readUInt16LE(offset + 30);
    const commentLength = buffer.readUInt16LE(offset + 32);
    const nameStart = offset + 46;
    entries.push(buffer.subarray(nameStart, nameStart + nameLength).toString('utf8'));
    offset = nameStart + nameLength + extraLength + commentLength;
  }

  if (entries.length !== entryCount) {
    throw new Error(`Lambda archive entry count mismatch: expected ${entryCount}, found ${entries.length}.`);
  }
  return entries;
};

const runNpmProductionInstall = stagingDirectory => {
  const npmArguments = ['ci', '--omit=dev', '--ignore-scripts', '--no-audit', '--no-fund'];
  const npmExecutable = process.platform === 'win32' ? (process.env.ComSpec || 'cmd.exe') : 'npm';
  const commandArguments = process.platform === 'win32'
    ? ['/d', '/s', '/c', 'npm.cmd', ...npmArguments]
    : npmArguments;
  const result = spawnSync(
    npmExecutable,
    commandArguments,
    {
      cwd: stagingDirectory,
      env: { ...process.env, NODE_ENV: 'production' },
      stdio: 'inherit'
    }
  );
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Production dependency install failed with exit code ${result.status}.`);
};

const smokeLoadRuntime = stagingDirectory => {
  const runtimeSmokeScript = [
    "process.env.APPLICATIONS_TABLE='package-smoke';",
    '(async () => {',
    "  const runtime = require('./index.js');",
    '  const archive = await runtime.__test.createZipArchive({ zlib: { level: 1 } });',
    "  if (archive.constructor.name !== 'ZipArchive') throw new Error('Runtime ZIP constructor unavailable.');",
    '  archive.abort();',
    '})().catch(error => {',
    '  console.error(error);',
    '  process.exitCode = 1;',
    '});'
  ].join('\n');
  const compatibilityFlags = process.allowedNodeEnvironmentFlags.has('--no-experimental-require-module')
    ? ['--no-experimental-require-module']
    : [];
  const result = spawnSync(
    process.execPath,
    [...compatibilityFlags, '-e', runtimeSmokeScript],
    { cwd: stagingDirectory, encoding: 'utf8' }
  );
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`Packaged Lambda failed to load: ${(result.stderr || result.stdout || '').trim()}`);
  }
};

const hashFile = filePath => createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');

const buildLambdaPackage = async () => {
  for (const file of [...INSTALL_MANIFEST_FILES, ...RUNTIME_ROOT_FILES]) {
    if (!fs.statSync(path.join(PACKAGE_ROOT, file), { throwIfNoEntry: false })?.isFile()) {
      throw new Error(`Required Lambda package input is missing: ${file}`);
    }
  }

  const stagingDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'job-application-tracker-lambda-'));
  const temporaryOutputPath = path.join(
    OUTPUT_DIRECTORY,
    `.job-application-tracker-${process.pid}-${Date.now()}.zip.tmp`
  );

  try {
    for (const file of INSTALL_MANIFEST_FILES) {
      fs.copyFileSync(path.join(PACKAGE_ROOT, file), path.join(stagingDirectory, file));
    }
    runNpmProductionInstall(stagingDirectory);
    for (const file of RUNTIME_ROOT_FILES) {
      fs.copyFileSync(path.join(PACKAGE_ROOT, file), path.join(stagingDirectory, file));
    }
    for (const file of INSTALL_MANIFEST_FILES) {
      fs.rmSync(path.join(stagingDirectory, file), { force: true });
    }

    const nodeModulesDirectory = path.join(stagingDirectory, 'node_modules');
    pruneNonRuntimeDependencyFiles(nodeModulesDirectory);
    smokeLoadRuntime(stagingDirectory);

    const entries = validateArchiveEntries(collectFiles(stagingDirectory));
    fs.mkdirSync(OUTPUT_DIRECTORY, { recursive: true });
    await createDeterministicArchive(stagingDirectory, entries, temporaryOutputPath);

    const packagedEntries = validateArchiveEntries(readZipEntries(temporaryOutputPath));
    if (JSON.stringify(packagedEntries) !== JSON.stringify(entries)) {
      throw new Error('Lambda archive contents do not match the staged runtime allowlist.');
    }

    fs.rmSync(OUTPUT_PATH, { force: true });
    fs.renameSync(temporaryOutputPath, OUTPUT_PATH);
    return {
      entryCount: entries.length,
      outputPath: OUTPUT_PATH,
      sha256: hashFile(OUTPUT_PATH),
      sizeBytes: fs.statSync(OUTPUT_PATH).size
    };
  } finally {
    fs.rmSync(temporaryOutputPath, { force: true });
    fs.rmSync(stagingDirectory, { recursive: true, force: true });
  }
};

if (require.main === module) {
  buildLambdaPackage()
    .then(result => {
      console.log(`Created ${result.outputPath}`);
      console.log(`Entries: ${result.entryCount}`);
      console.log(`Bytes: ${result.sizeBytes}`);
      console.log(`SHA-256: ${result.sha256}`);
    })
    .catch(error => {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
    });
}

module.exports = {
  OUTPUT_PATH,
  RUNTIME_ROOT_FILES,
  buildLambdaPackage,
  collectFiles,
  createDeterministicArchive,
  hashFile,
  isExcludedDependencyPath,
  readZipEntries,
  validateArchiveEntries
};
