'use strict';

// Compare separate invocations with --engine-root=<frozen project-starfall folder>.
// Each run drives actual combat with the training harness's fixed clock and seed.
const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const args = process.argv.slice(2);
const value = (name, fallback) => {
  const arg = args.find((entry) => entry.startsWith(`--${name}=`));
  return arg ? arg.slice(name.length + 3) : fallback;
};
const root = path.resolve(value('engine-root', path.join(__dirname, '../js/games/project-starfall')));
const runs = Math.max(1, Math.floor(Number(value('runs', 3))));
const measuredSeconds = Number(value('seconds', 90));
assert(Number.isFinite(runs) && runs <= 100 && Number.isFinite(measuredSeconds) && measuredSeconds > 0, 'valid benchmark duration and run count required');

function writeReport(report) {
  const output = value('output', '');
  if (output) fs.writeFileSync(path.resolve(output), `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

if (runs > 1) {
  // Runtime modules retain process-global caches. Isolate trials so replay and
  // timings cannot depend on the previous trial's accumulated runtime state.
  const childArgs = args.filter((arg) => !arg.startsWith('--runs=') && !arg.startsWith('--output='));
  let report = null;
  for (let repetition = 0; repetition < runs; repetition += 1) {
    const child = spawnSync(process.execPath, [__filename, ...childArgs, '--runs=1'], { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
    assert.strictEqual(child.status, 0, child.stderr || 'simulation benchmark worker failed');
    const trial = JSON.parse(child.stdout);
    if (!report) {
      report = trial;
      report.runs = runs;
    } else {
      trial.results.forEach((result, index) => {
        assert.strictEqual(result.checksum, report.results[index].checksum, 'isolated fixed input must replay identical gameplay outcomes');
        report.results[index].elapsedMs.push(...result.elapsedMs);
      });
    }
  }
  report.results.forEach((result) => {
    const sorted = result.elapsedMs.slice().sort((a, b) => a - b);
    result.medianMs = sorted[Math.floor(sorted.length / 2)];
  });
  writeReport(report);
} else {
  const data = require(path.join(root, 'project-starfall-data.js'));
  const { createProjectStarfallEngine } = require(path.join(root, 'project-starfall-engine.js'));
  const { runTrainingScenario } = require('../tests/project-starfall/project-starfall-training-harness.js');
  const fixtures = [
    { mapId: 'orebackQuarry', level: 28, classId: 'fighter', party: true },
    { mapId: 'cinderHollow', level: 28, classId: 'mage' },
    { mapId: 'greenrootMeadow', level: 6, classId: 'archer' }
  ];
  const results = fixtures.map((fixture) => ({ options: { ...fixture, seed: 137, warmupSeconds: 5, measuredSeconds, fps: 60 }, elapsedMs: [] }));
  // Discard a short JIT warmup, including the production simulation and bot.
  runTrainingScenario(data, createProjectStarfallEngine, { ...results[0].options, measuredSeconds: 15 });
  for (let repetition = 0; repetition < runs; repetition += 1) {
    for (const result of results) {
      const start = process.hrtime.bigint();
      const outcome = runTrainingScenario(data, createProjectStarfallEngine, result.options);
      result.elapsedMs.push(Number(process.hrtime.bigint() - start) / 1e6);
      const checksum = crypto.createHash('sha256').update(JSON.stringify(outcome)).digest('hex');
      if (result.checksum) assert.strictEqual(checksum, result.checksum, 'fixed input must replay identical gameplay outcomes');
      result.checksum = checksum;
      result.outcome = outcome;
    }
  }
  for (const result of results) {
    const sorted = result.elapsedMs.slice().sort((a, b) => a - b);
    result.medianMs = sorted[Math.floor(sorted.length / 2)];
  }
  const report = { engineRoot: root, node: process.version, runs, evidence: 'Node simulation and deterministic training controller; no rendering or FPS measurement', results };
  writeReport(report);
}
