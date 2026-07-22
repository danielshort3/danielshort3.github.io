import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const packageRootForInput = (packageDir, inputPath) => {
  const parts = path.resolve(packageDir, inputPath).split(path.sep);
  const nodeModulesIndex = parts.lastIndexOf('node_modules');
  if (nodeModulesIndex < 0 || nodeModulesIndex + 1 >= parts.length) return null;
  const packageEnd = parts[nodeModulesIndex + 1].startsWith('@')
    ? nodeModulesIndex + 3
    : nodeModulesIndex + 2;
  return parts.slice(0, packageEnd).join(path.sep);
};

const licenseFiles = async (packageRoot) => (await readdir(packageRoot, { withFileTypes: true }))
  .filter(entry => entry.isFile() && /^(?:license|licence|copying|notice)(?:\.|$)/iu.test(entry.name))
  .map(entry => entry.name)
  .sort((left, right) => left.localeCompare(right, 'en-US'));

export const generateThirdPartyNotices = async ({ packageDir, inputPaths, outputPath }) => {
  const packageRoots = [...new Set(inputPaths
    .map(inputPath => packageRootForInput(packageDir, inputPath))
    .filter(Boolean))]
    .sort((left, right) => left.localeCompare(right, 'en-US'));
  const sections = [];

  for (const packageRoot of packageRoots) {
    const metadata = JSON.parse(await readFile(path.join(packageRoot, 'package.json'), 'utf8'));
    const files = await licenseFiles(packageRoot);
    const contents = await Promise.all(files.map(async filename => ({
      filename,
      text: (await readFile(path.join(packageRoot, filename), 'utf8')).trim()
    })));
    const declaredLicense = typeof metadata.license === 'string'
      ? metadata.license
      : JSON.stringify(metadata.license || 'not declared');
    sections.push([
      `${metadata.name}@${metadata.version}`,
      `Declared license: ${declaredLicense}`,
      ...contents.flatMap(({ filename, text }) => [``, `--- ${filename} ---`, text]),
      ...(!contents.length ? ['', 'No standalone license file was included by this package; see its declared license above.'] : [])
    ].join('\n'));
  }

  const notice = [
    'Job Application Copilot - Third-Party Notices',
    '',
    'This file is generated from the third-party packages included in the browser bundles.',
    '',
    ...sections.flatMap((section, index) => [
      `${'='.repeat(72)}${index ? '\n' : ''}`,
      section,
      ''
    ])
  ].join('\n').trimEnd() + '\n';
  await writeFile(outputPath, notice, 'utf8');
  return packageRoots.length;
};
