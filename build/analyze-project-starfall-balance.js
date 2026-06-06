#!/usr/bin/env node
'use strict';

const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const {
  createBalanceReport,
  getScenarioResults
} = require('../tests/project-starfall-balance-harness.js');

function parseArgs(argv) {
  const options = {};
  argv.forEach((arg) => {
    if (arg === '--json') options.json = true;
    if (arg.startsWith('--level=')) options.level = Number(arg.slice('--level='.length));
    if (arg.startsWith('--rank=')) options.rank = Number(arg.slice('--rank='.length));
  });
  return options;
}

function formatResult(result, index) {
  const casts = Object.entries(result.casts || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([skillId, count]) => `${skillId} x${count}`)
    .join(', ');
  return `${String(index + 1).padStart(2, ' ')}. ${result.classId.padEnd(12)} ${String(result.dps).padStart(7)} DPS  ${String(result.damage).padStart(8)} dmg${casts ? `  [${casts}]` : ''}`;
}

function printTextReport(report) {
  process.stdout.write(`Project Starfall balance report (level ${report.level}, rank ${report.rank})\n`);
  report.assumptions.forEach((assumption) => {
    process.stdout.write(`- ${assumption}\n`);
  });
  Object.values(report.scenarios).forEach((scenario) => {
    process.stdout.write(`\n${scenario.label} (${scenario.duration}s)\n`);
    getScenarioResults(report, scenario.id).forEach((result, index) => {
      process.stdout.write(`${formatResult(result, index)}\n`);
    });
  });
}

const options = parseArgs(process.argv.slice(2));
const report = createBalanceReport(data, createProjectStarfallEngine, options);
if (options.json) {
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} else {
  printTextReport(report);
}
