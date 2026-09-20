#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { summarizeTrainingRuns, getEligibleTrainingClasses } = require('../tests/project-starfall/project-starfall-training-harness.js');

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return !sorted.length ? null : sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null;
}

function key(row) {
  return `${row.routeScope || 'full'}:${row.level}:${row.mapId}:${row.party ? 'party' : row.classId}`;
}

function compareTrainingReports(current, previous) {
  const currentRows = summarizeTrainingRuns(current.runs || []);
  const previousRows = new Map(summarizeTrainingRuns(previous && previous.runs || []).map((row) => [key(row), row]));
  const ordinaryFieldRoles = ['starterField', 'trainingField', 'deepField'];
  const layoutRole = (row) => data.MAPS.find((map) => map.id === row.mapId)?.layoutRole;
  const sourceHashKeys = ['runtimeSourceHash', 'runtimeDependencySourceHash', 'harnessSourceHash', 'runnerSourceHash', 'balanceHarnessSourceHash', 'routePlanSourceHash'];
  const referenceRun = current.runs && current.runs[0];
  const matchingSamples = (report) => !!referenceRun && Array.isArray(report && report.runs) && report.runs.length > 0 && report.runs.every((run) =>
    ['warmupSeconds', 'measuredSeconds', 'fps'].every((field) => Number.isFinite(run[field]) && run[field] === referenceRun[field]));
  // The CLI's protocol field describes defaults; use the actual run windows and
  // frame rate when judging a controlled before/after comparison.
  const matchingEnvironment = !current.environment && !previous?.environment || JSON.stringify(current.environment) === JSON.stringify(previous?.environment);
  const comparableProtocol = !!previous && matchingEnvironment && sourceHashKeys.every((field) => typeof current[field] === 'string' && current[field].length > 0 && current[field] === previous[field]) && matchingSamples(current) && matchingSamples(previous);
  const rows = currentRows.map((row) => {
    const map = data.MAPS.find((entry) => entry.id === row.mapId);
    const before = previousRows.get(key(row));
    const cohortPeers = currentRows.filter((peer) => peer.level === row.level && peer.classId === row.classId && peer.party === row.party && peer.routeScope === row.routeScope && ordinaryFieldRoles.includes(layoutRole(peer)));
    const ordinaryPeers = cohortPeers.filter((peer) => layoutRole(peer) !== 'deepField');
    const reference = median(ordinaryPeers.map((peer) => peer.leaderXpPerMinute));
    const dangerous = map.layoutRole === 'deepField';
    const referencePeers = ordinaryPeers.length ? ordinaryPeers : cohortPeers;
    const comparisonCohortEligible = referencePeers.every((peer) => peer.eligibleForAcceptance);
    const dangerAdvantagePercent = dangerous && reference > 0 ? round((row.leaderXpPerMinute / reference - 1) * 100) : null;
    const specialRoute = !ordinaryFieldRoles.includes(map.layoutRole);
    const cohortMedian = median(referencePeers.map((peer) => peer.leaderXpPerMinute));
    const deviation = cohortMedian > 0 ? round((row.leaderXpPerMinute / cohortMedian - 1) * 100) : null;
    const xpWithinTarget = specialRoute || cohortPeers.length <= 1 ? null : dangerAdvantagePercent !== null
      ? dangerAdvantagePercent >= 10 && dangerAdvantagePercent <= 20
      : Math.abs(deviation) <= 15;
    const runs = current.runs.filter((run) => key(run) === key(row));
    const previousRuns = previous && previous.runs.filter((run) => key(run) === key(row)) || [];
    const sameRoute = previousRuns.length > 0 && runs.every((run) => previousRuns.some((prior) => prior.seed === run.seed && JSON.stringify(prior.route.plannedPlatformIds || []) === JSON.stringify(run.route.plannedPlatformIds || [])));
    const matchedComparison = comparableProtocol && sameRoute && !!before && before.eligibleForAcceptance && row.eligibleForAcceptance;
    const perMinute = (name) => round(runs.reduce((sum, run) => sum + Number(run.totals[name] || 0) / (run.measuredSeconds / 60), 0) / Math.max(1, runs.length));
    const itemsPerMinute = (kind) => round(runs.reduce((sum, run) => sum + Object.entries(run.collectedItems || {}).reduce((quantity, [id, value]) => quantity + (id.startsWith(`${kind}:`) ? value : 0), 0) / (run.measuredSeconds / 60), 0) / Math.max(1, runs.length));
    const usefulLootPerMinute = (name) => round(runs.reduce((sum, run) => sum + Number(run.usefulLoot?.[name] || 0) / (run.measuredSeconds / 60), 0) / Math.max(1, runs.length));
    return {
      ...row, layoutRole: map.layoutRole, xpWithinTarget, dangerAdvantagePercent, comparisonCohortEligible,
      cohortMapCount: specialRoute ? 0 : cohortPeers.length, referenceMapCount: specialRoute ? 0 : referencePeers.length, cohortMedianXpPerMinute: specialRoute ? null : cohortMedian, deviationFromCohortMedianPercent: specialRoute ? null : deviation,
      minimumSeedXpPerMinute: Math.min(...runs.map((run) => run.leaderXpPerMinute)),
      maximumSeedXpPerMinute: Math.max(...runs.map((run) => run.leaderXpPerMinute)),
      currencyPerMinute: perMinute('currency'), pickedUpStacksPerMinute: perMinute('loot'), potionUsesPerMinute: perMinute('potion'),
      damageDealtPerMinute: perMinute('damage'), materialsCollectedPerMinute: itemsPerMinute('material'), consumablesCollectedPerMinute: itemsPerMinute('consumable'),
      equippableEquipmentPerMinute: usefulLootPerMinute('equippableEquipment'), potentialEquipmentResalePerMinute: usefulLootPerMinute('potentialEquipmentResale'),
      leaderDamageTakenPerMinute: row.damageTakenPerMinute,
      travelSeconds: round(runs.reduce((sum, run) => sum + Number(run.phases.travel || 0), 0) / Math.max(1, runs.length)),
      respawnWaitingSeconds: round(runs.reduce((sum, run) => sum + Number(run.phases.respawnWaiting || 0), 0) / Math.max(1, runs.length)),
      enemyHealingPerMinute: round(runs.reduce((sum, run) => sum + Number(run.enemyHealing || 0) / (run.measuredSeconds / 60), 0) / Math.max(1, runs.length)),
      rawMovingPercent: round(runs.reduce((sum, run) => sum + Number(run.rawMotion?.movingSeconds || 0) / run.measuredSeconds * 100, 0) / Math.max(1, runs.length)),
      engagedPursuitPercent: round(runs.reduce((sum, run) => sum + Number(run.rawMotion?.engagedPursuitSeconds || 0) / run.measuredSeconds * 100, 0) / Math.max(1, runs.length)),
      beforeXpPerMinute: before ? before.leaderXpPerMinute : null,
      pairedChangePercent: matchedComparison && before.leaderXpPerMinute > 0 ? round((row.leaderXpPerMinute / before.leaderXpPerMinute - 1) * 100) : null,
      diagnosticChangePercent: comparableProtocol && before && before.leaderXpPerMinute > 0 ? round((row.leaderXpPerMinute / before.leaderXpPerMinute - 1) * 100) : null,
      comparisonEligibleForAcceptance: matchedComparison,
      matchedResolvedRoute: sameRoute,
      status: !row.eligibleForAcceptance ? 'diagnostic-controller-or-coverage-limit' : specialRoute ? 'special-route-review' : !comparisonCohortEligible ? 'diagnostic-comparison-cohort' : xpWithinTarget === null ? 'standalone-review' : xpWithinTarget && row.pacingWithinTarget ? 'within-numerical-targets' : 'requires-tuning'
    };
  });
  const dominance = [];
  const scopedLevels = [...new Set(rows.map((row) => `${row.routeScope}:${row.level}`))];
  scopedLevels.forEach((scopeLevel) => {
    const [routeScope, levelText] = scopeLevel.split(':');
    const level = Number(levelText);
    const eligibleClasses = getEligibleTrainingClasses(data, level);
    const scopedRows = rows.filter((row) => row.level === level && row.routeScope === routeScope);
    const mapIds = [...new Set(scopedRows.filter((row) => ordinaryFieldRoles.includes(row.layoutRole)).map((row) => row.mapId))];
    mapIds.forEach((mapId) => mapIds.filter((other) => other !== mapId).forEach((other) => {
      const candidates = scopedRows.filter((row) => row.mapId === mapId && !row.party);
      const peers = new Map(scopedRows.filter((row) => row.mapId === other && !row.party).map((row) => [row.classId, row]));
      if (eligibleClasses.every((id) => candidates.some((row) => row.classId === id)) && candidates.every((row) => row.eligibleForAcceptance && peers.get(row.classId)?.eligibleForAcceptance && row.leaderXpPerMinute > peers.get(row.classId).leaderXpPerMinute * 1.25)) dominance.push({ routeScope, level, mapId, exceedsMapId: other, classCount: candidates.length, allEligibleClasses: true });
    }));
  });
  const sourceHashes = (report) => report ? Object.fromEntries(Object.entries(report).filter(([name]) => name.endsWith('SourceHash'))) : null;
  return { evidenceKind: 'observed-engine-comparison', comparableProtocol, comparisonBasis: previous && previous.comparisonBasis || 'No previous report supplied.', protocol: current.protocol, sampledProtocol: current.sampledProtocol, environment: current.environment, isolation: current.isolation, currentSourceHashes: sourceHashes(current), previousSourceHashes: sourceHashes(previous), currentRunCount: current.runs.length, previousRunCount: previous && previous.runs.length || 0, rows, dominance,
    caveats: [
      'Runtime movement, combat, enemy healing, respawns and rewards are measured; the deterministic input controller is not an optimal human player.',
      'A row needs all three fixed seeds, standard 60s/300s windows, at least 75% measured route coverage per seed and no more than 18s measured controller stalls before numerical acceptance.',
      'Leader XP and companion XP credits are distinct; a runtime companion party is not three identical human loadouts.',
      'Incoming damage is the leader damage event total including shield absorption, not party-wide HP loss. Companion deaths are not leader deaths.',
      'Companion down counts sum measured down-state transitions across seeds; down member-seconds average the sum of each ally recovery time per run. Two down allies contribute two member-seconds per second.',
      'Item counts retain actual identities and quantities. Equippable equipment meets the real class/level rules but need not be an upgrade; potential resale uses production shop appraisal without selling or crediting currency. Potion costs are real replacement prices, not net account profit.',
      'Pacing travel excludes pursuit with actual runtime aggro; raw moving time and engaged pursuit are also reported.',
      'Main-matrix danger premiums use authored deepField maps. Separately supplied optional circuits are compared against their own main circuits in optionalBranches.',
      'Ordinary field medians exclude deepField maps when ordinary references exist. Endless routes retain separate purpose review.',
      'Paired changes require both reports to pass the controller/coverage gate; diagnostic changes remain explicitly separate.',
      'Previous authored spawn/layout profiles run on the current engine; they do not reproduce a historical game build.'
    ] };
}

function toCsv(rows) {
  if (!rows.length) return '';
  const columns = Object.keys(rows[0]).filter((name) => rows[0][name] == null || typeof rows[0][name] !== 'object');
  const cell = (value) => value == null ? '' : `"${String(value).replace(/"/g, '""')}"`;
  return `${columns.map(cell).join(',')}\n${rows.map((row) => columns.map((column) => cell(row[column])).join(',')).join('\n')}\n`;
}

function compareTrainingRouteScopes(main, optional) {
  const sameRuntime = compareTrainingReports(main, optional).comparableProtocol && ['spawnProfileSourceHash', 'layoutSourceHash'].every((field) => main[field] && main[field] === optional[field]);
  const scenarioKey = (row) => `${row.level}:${row.mapId}:${row.party ? 'party' : row.classId}`;
  const mainRows = new Map(summarizeTrainingRuns(main.runs || []).filter((row) => row.routeScope === 'main').map((row) => [scenarioKey(row), row]));
  const rows = summarizeTrainingRuns(optional.runs || []).filter((row) => row.routeScope === 'optional').map((row) => {
    const normal = mainRows.get(scenarioKey(row));
    const eligible = sameRuntime && normal && normal.eligibleForAcceptance && row.eligibleForAcceptance;
    const advantage = eligible && normal.leaderXpPerMinute > 0 ? round((row.leaderXpPerMinute / normal.leaderXpPerMinute - 1) * 100) : null;
    const higherCompanionRecoveryBurden = eligible && row.party ? row.companionDowns > normal.companionDowns || row.companionDownMemberSeconds > normal.companionDownMemberSeconds * 1.1 : false;
    return { mapId: row.mapId, level: row.level, classId: row.classId, party: row.party, eligibleForComparison: !!eligible,
      mainXpPerMinute: normal?.leaderXpPerMinute ?? null, optionalXpPerMinute: row.leaderXpPerMinute, optionalXpAdvantagePercent: advantage,
      moderateRewardPremium: advantage === null ? null : advantage >= 10 && advantage <= 20,
      mainLeaderDamageTakenPerMinute: normal?.damageTakenPerMinute ?? null, optionalLeaderDamageTakenPerMinute: row.damageTakenPerMinute,
      mainConsumableCostPerMinute: normal?.consumableCostPerMinute ?? null, optionalConsumableCostPerMinute: row.consumableCostPerMinute,
      mainDeaths: normal?.deaths ?? null, optionalDeaths: row.deaths,
      mainCompanionDowns: normal?.companionDowns ?? null, optionalCompanionDowns: row.companionDowns,
      mainCompanionDownMemberSeconds: normal?.companionDownMemberSeconds ?? null, optionalCompanionDownMemberSeconds: row.companionDownMemberSeconds,
      mainTravelPercent: normal?.travelPercent ?? null, optionalTravelPercent: row.travelPercent,
      mainRespawnWaitingPercent: normal?.respawnWaitingPercent ?? null, optionalRespawnWaitingPercent: row.respawnWaitingPercent,
      mainStuckSeconds: normal?.stuckSeconds ?? null, optionalStuckSeconds: row.stuckSeconds,
      higherCompanionRecoveryBurden,
      higherMeasuredPressure: eligible ? higherCompanionRecoveryBurden || row.deaths > normal.deaths || row.damageTakenPerMinute > normal.damageTakenPerMinute * 1.1 || row.consumableCostPerMinute > normal.consumableCostPerMinute * 1.1 : null };
  });
  return { evidenceKind: 'observed-main-versus-optional', comparableProtocol: !!sameRuntime, rows, caveats: [
    'Main and optional circuits use the same current map profiles, engine, loadout rules, fixed seeds and sample windows; route scope is the intentional difference.',
    'Higher measured pressure means over 10% higher incoming leader damage or consumable replacement costs, more leader deaths, or greater companion recovery burden. Companion burden means more measured downs or over 10% more down member-seconds. These are diagnostic signals, not a complete difficulty rating.',
    'Every included class is reported; a branch name or enemy appearance does not establish greater danger. Incomplete controller samples cannot establish a paired conclusion.',
    'Companion party results use authored runtime allies; incoming damage and deaths cover the leader. Item quantities are not appraised loot value.'
  ] };
}

if (require.main === module) {
  const args = Object.fromEntries(process.argv.slice(2).map((argument) => { const [name, ...value] = argument.replace(/^--/, '').split('='); return [name, value.join('=')]; }));
  if (!args.current) throw new Error('Use --current=<report.json> [--before=<report.json>] [--output=<comparison.json>] [--csv=<summary.csv>]');
  const report = compareTrainingReports(JSON.parse(fs.readFileSync(path.resolve(args.current), 'utf8')), args.before ? JSON.parse(fs.readFileSync(path.resolve(args.before), 'utf8')) : null);
  if (args.optional) report.optionalBranches = compareTrainingRouteScopes(JSON.parse(fs.readFileSync(path.resolve(args.current), 'utf8')), JSON.parse(fs.readFileSync(path.resolve(args.optional), 'utf8')));
  if (args.output) { fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true }); fs.writeFileSync(path.resolve(args.output), `${JSON.stringify(report, null, 2)}\n`); }
  if (args.csv) { fs.mkdirSync(path.dirname(path.resolve(args.csv)), { recursive: true }); fs.writeFileSync(path.resolve(args.csv), toCsv(report.rows)); }
  if (args['optional-csv'] && report.optionalBranches) { fs.mkdirSync(path.dirname(path.resolve(args['optional-csv'])), { recursive: true }); fs.writeFileSync(path.resolve(args['optional-csv']), toCsv(report.optionalBranches.rows)); }
  const statuses = report.rows.reduce((counts, row) => { counts[row.status] = (counts[row.status] || 0) + 1; return counts; }, {});
  process.stdout.write(`${JSON.stringify({ currentRunCount: report.currentRunCount, previousRunCount: report.previousRunCount, comparableProtocol: report.comparableProtocol, statuses, dominance: report.dominance }, null, 2)}\n`);
}

module.exports = { compareTrainingReports, compareTrainingRouteScopes, toCsv };
