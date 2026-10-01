'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const esbuild = require('esbuild');

async function main() {
  const root = path.resolve(__dirname, '..');
  const argument = process.argv.find((value) => value.startsWith('--output='));
  const output = path.resolve(argument ? argument.slice('--output='.length) : path.join(os.tmpdir(), 'website-contact-lambda.zip'));
  if (path.extname(output).toLowerCase() !== '.zip') throw new Error('Output must be an explicit .zip file.');
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const staging = fs.mkdtempSync(path.join(os.tmpdir(), 'website-contact-lambda-'));
  const entry = path.join(staging, 'index.js');
  await esbuild.build({
    entryPoints: [path.join(root, 'aws/contact-function/index.js')],
    outfile: entry,
    bundle: true,
    platform: 'node',
    target: 'node24',
    format: 'cjs',
    minify: false,
    logLevel: 'warning'
  });
  if (process.platform === 'win32') {
    const quote = (value) => `'${value.replace(/'/g, "''")}'`;
    execFileSync('powershell.exe', ['-NoProfile', '-Command', `Compress-Archive -LiteralPath ${quote(entry)} -DestinationPath ${quote(output)} -Force`], { stdio: 'inherit', windowsHide: true });
  } else {
    execFileSync('python3', ['-c', 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[2],"w",zipfile.ZIP_DEFLATED); z.write(sys.argv[1],"index.js"); z.close()', entry, output], { stdio: 'inherit' });
  }
  console.log(JSON.stringify({ output, bytes: fs.statSync(output).size, handler: 'index.handler', runtime: 'nodejs24.x' }));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
