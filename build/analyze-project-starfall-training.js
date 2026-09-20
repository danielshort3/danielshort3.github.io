#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const Module = require('module');
const args = Object.fromEntries(process.argv.slice(2).map((argument) => { const [key, ...value] = argument.replace(/^--/, '').split('='); return [key, value.length ? value.join('=') : true]; }));
// For a controlled before/after composition comparison, load a preserved
// authored publication module before the data facade and engine are required.
// Runtime stays current; this is explicitly not an old-game run.
for (const [option, source] of [['baseline-spawns', 'map-publication'], ['baseline-layouts', 'map-builders']]) {
  if (!args[option]) continue;
  const filename = require.resolve(`../js/games/project-starfall/data/${source}.js`);
  const baseline = new Module(filename, module);
  baseline.filename = filename;
  baseline.paths = Module._nodeModulePaths(path.dirname(filename));
  baseline._compile(fs.readFileSync(path.resolve(String(args[option])), 'utf8'), filename);
  baseline.loaded = true;
  require.cache[filename] = baseline;
}
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { TRAINING_PROTOCOL, getTrainingCohorts, getEligibleTrainingClasses, runTrainingScenario, summarizeTrainingRuns } = require('../tests/project-starfall/project-starfall-training-harness.js');
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const runtimeRoot = path.resolve(__dirname, '../js/games/project-starfall');
const comparisonOverrides = new Set(['map-publication.js', 'map-builders.js']);
const runtimeDependencies = Object.keys(require.cache).filter((filename) => filename.startsWith(`${runtimeRoot}${path.sep}`) && !comparisonOverrides.has(path.basename(filename)))
  .sort().map((filename) => [path.relative(runtimeRoot, filename).replace(/\\/g, '/'), hash(filename)]);
const sourceHashes = {
  runnerSourceHash: hash(__filename),
  runtimeSourceHash: hash(require.resolve('../js/games/project-starfall/project-starfall-engine.js')),
  runtimeDependencySourceHash: crypto.createHash('sha256').update(JSON.stringify(runtimeDependencies)).digest('hex'),
  spawnProfileSourceHash: hash(args['baseline-spawns'] ? path.resolve(String(args['baseline-spawns'])) : require.resolve('../js/games/project-starfall/data/map-publication.js')),
  layoutSourceHash: hash(args['baseline-layouts'] ? path.resolve(String(args['baseline-layouts'])) : require.resolve('../js/games/project-starfall/data/map-builders.js')),
  harnessSourceHash: hash(require.resolve('../tests/project-starfall/project-starfall-training-harness.js')),
  balanceHarnessSourceHash: hash(require.resolve('../tests/project-starfall/project-starfall-balance-harness.js')),
  routePlanSourceHash: args['route-plans'] ? hash(path.resolve(String(args['route-plans']))) : 'authored-map-trainingRoute'
};

if (args.worker) {
  process.once('message', (job) => {
    try { process.send({ index: job.index, run: runTrainingScenario(data, createProjectStarfallEngine, job.options), sourceHashes }, () => process.disconnect()); }
    catch (error) { process.send({ index: job.index, error: error.stack }, () => process.disconnect()); }
  });
} else main().catch((error) => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });

async function main() {

  const cohorts = getTrainingCohorts(data);
  const selectedMaps = args.maps ? args.maps.split(',') : null;
  const selectedClasses = args.classes ? args.classes.split(',') : args['route-gate'] ? ['fighter', 'mage', 'archer'] : null;
  const seeds = args.seeds ? args.seeds.split(',').map(Number) : args['route-gate'] ? [TRAINING_PROTOCOL.seeds[2]] : TRAINING_PROTOCOL.seeds;
  const levels = args.levels ? args.levels.split(',').map(Number) : args.level ? [Number(args.level)] : null;
  const routeScope = args.route || (args['route-gate'] ? 'full' : 'main');
  const routePlans = args['route-plans'] ? JSON.parse(fs.readFileSync(path.resolve(String(args['route-plans'])), 'utf8')) : null;
  const fields = data.MAPS.filter((map) => !map.safeZone && !map.adminOnly && !map.isDungeon && map.spawnGroups?.length && (!selectedMaps || selectedMaps.includes(map.id)));
  const selectedCohorts = args['route-gate'] ? fields.map((map) => ({ id: `route-${map.id}`, level: Math.floor((map.levelRange[0] + map.levelRange[1]) / 2), mapIds: [map.id] })) : levels ? levels.map((level) => ({ id: `level-${level}`, level, mapIds: fields.filter((map) => map.levelRange[0] <= level && map.levelRange[1] >= level).map((map) => map.id) })) : cohorts;
  const jobs = [];
  for (const cohort of selectedCohorts) {
    for (const mapId of cohort.mapIds.filter((id) => !selectedMaps || selectedMaps.includes(id))) {
      const classIds = getEligibleTrainingClasses(data, cohort.level).filter((id) => !selectedClasses || selectedClasses.includes(id));
      const modes = [...classIds.map((classId) => ({ classId, party: false })), ...(args['no-party'] || args['route-gate'] ? [] : [{ classId: 'fighter', party: true }])];
      for (const mode of modes) {
        for (const seed of seeds) {
          jobs.push({ mapId, level: cohort.level, ...mode, seed, routeScope, ...(routePlans ? { routePlan: routePlans[mapId] } : {}), ...(args.warmup ? { warmupSeconds: Number(args.warmup) } : {}), ...(args.seconds ? { measuredSeconds: Number(args.seconds) } : {}), ...(args.fps ? { fps: Number(args.fps) } : {}) });
        }
      }
    }
  }
  if (!jobs.length) throw new Error('No eligible training scenarios matched the requested maps, levels and classes.');
  const runs = new Array(jobs.length);
  const checkpoint = args.output ? `${path.resolve(String(args.output))}.runs.jsonl` : null;
  if (checkpoint) { fs.mkdirSync(path.dirname(checkpoint), { recursive: true }); fs.writeFileSync(checkpoint, ''); }
  const workers = Math.min(jobs.length, Math.max(1, Math.min(8, Number(args.workers || 1))));
  let cursor = 0;
  let completed = 0;
  let failed = false;
  const activeWorkers = new Set();
  // Engine module-level UID sequences affect deterministic movement staggering.
  // A fresh process per scenario prevents results depending on worker/job order.
  await Promise.all(Array.from({ length: workers }, async () => {
    while (cursor < jobs.length && !failed) {
      const index = cursor++;
      await new Promise((resolve, reject) => {
        const child = require('child_process').fork(__filename, ['--worker', ...['baseline-spawns', 'baseline-layouts', 'route-plans'].filter((option) => args[option]).map((option) => `--${option}=${args[option]}`)], { stdio: ['ignore', 'ignore', 'inherit', 'ipc'], windowsHide: true });
        activeWorkers.add(child);
        let received = false;
        const fail = (error) => {
          failed = true;
          activeWorkers.forEach((worker) => worker.kill());
          reject(error);
        };
        child.on('message', (message) => {
          if (message.error) { fail(new Error(message.error)); return; }
          if (JSON.stringify(message.sourceHashes) !== JSON.stringify(sourceHashes)) { fail(new Error('Runtime/source changed while starting workers. Rerun with a stable source snapshot.')); return; }
          received = true;
          runs[message.index] = message.run;
          if (checkpoint) fs.appendFileSync(checkpoint, `${JSON.stringify({ index: message.index, sourceHashes: message.sourceHashes, run: message.run })}\n`);
          completed += 1;
          const run = message.run;
          process.stderr.write(`${completed}/${jobs.length} ${run.mapId} L${run.level} ${run.party ? 'party' : run.classId}: ${run.leaderXpPerMinute} XP/min, route ${run.route.visitedPlatformIds.length}/${run.route.routePlatformCount}\n`);
        });
        child.on('error', fail);
        child.on('exit', (code) => {
          activeWorkers.delete(child);
          if (!received && !failed || code && !failed) fail(new Error(`Training worker exited ${code} before a complete scenario.`));
          else resolve();
        });
        child.send({ index, options: jobs[index] });
      });
    }
  }));
  const report = { evidenceKind: 'observed-engine', comparisonBasis: args['baseline-spawns'] ? `Reconstructed previous authored spawn profiles${args['baseline-layouts'] ? ' and layouts' : ''}, measured on current engine; not an old-build benchmark.` : 'Current authored spawn profiles, current engine and geometry.',
    ...sourceHashes, purpose: args['route-gate'] ? 'Controller preflight on map midpoints; diagnostic only, not an all-class balance matrix.' : 'Training comparison',
    protocol: TRAINING_PROTOCOL, sampledProtocol: { warmupSeconds: runs[0].warmupSeconds, measuredSeconds: runs[0].measuredSeconds, fps: runs[0].fps, seeds }, routeScope,
    environment: { nodeVersion: process.version, platform: process.platform, architecture: process.arch },
    isolation: 'Fresh Node process per scenario; no module-state carryover between maps, classes or seeds.', cohorts: selectedCohorts, summary: summarizeTrainingRuns(runs), runs };
  if (args.output) {
    const output = path.resolve(String(args.output));
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
    process.stdout.write(`Wrote ${runs.length} observed runs to ${output}\n`);
  }
  if (args.json) process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  else report.summary.forEach((row) => process.stdout.write(`${row.mapId} L${row.level} ${row.party ? 'party' : row.classId}: ${row.leaderXpPerMinute} XP/min, ${row.killsPerMinute} kills/min, travel ${row.travelPercent}%, wait ${row.respawnWaitingPercent}%, deaths ${row.deaths}, route ${row.visitedPlatformCount}/${row.routePlatformCount}\n`));
}
