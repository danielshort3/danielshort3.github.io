#!/usr/bin/env node
'use strict';

const childProcess = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

const TASKS = Object.freeze([
  Object.freeze({
    name: 'map backgrounds',
    generate: ['build/process-project-starfall-map-backgrounds.js'],
    validate: ['build/process-project-starfall-map-backgrounds.js', '--validate']
  }),
  Object.freeze({
    name: 'environment terrain and prop atlases',
    generate: ['build/process-project-starfall-environment-imagegen-assets.js'],
    validate: ['build/process-project-starfall-environment-imagegen-assets.js', '--validate']
  }),
  Object.freeze({
    name: 'playable class and equipment sheets',
    generate: ['build/process-project-starfall-player-ai-assets.js'],
    validate: ['build/process-project-starfall-player-ai-assets.js', '--validate']
  }),
  Object.freeze({
    name: 'item icons',
    generate: ['build/process-project-starfall-ai-item-icons.js']
  }),
  Object.freeze({
    name: 'skill icons',
    generate: ['build/process-project-starfall-skill-icons.js']
  })
]);

function runNodeTask(name, args) {
  if (!Array.isArray(args) || !args.length) return;
  const scriptPath = String(args[0] || '');
  if (scriptPath.toLowerCase().includes('enemy')) {
    throw new Error(`Refusing to run enemy-owned asset processor in non-enemy pass: ${scriptPath}`);
  }
  console.log(`\nProject Starfall ${name}`);
  childProcess.execFileSync(process.execPath, args, {
    cwd: ROOT,
    stdio: 'inherit'
  });
}

function main() {
  const validateOnly = process.argv.includes('--validate');
  TASKS.forEach((task) => {
    if (validateOnly) {
      if (task.validate) runNodeTask(`${task.name} validation`, task.validate);
      return;
    }
    runNodeTask(task.name, task.generate);
    if (task.validate) runNodeTask(`${task.name} validation`, task.validate);
  });
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(error && error.stack || error);
    process.exit(1);
  }
}
