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
  if (report.skillSystem) {
    const skills = report.skillSystem;
    const coverage = skills.familyCoverage || {};
    const prereq = skills.prerequisiteHealth || {};
    const obsolete = skills.obsolescence || {};
    const leanOwners = (skills.owners || [])
      .slice()
      .sort((a, b) => a.classOwnedActiveCount - b.classOwnedActiveCount || a.classId.localeCompare(b.classId))
      .slice(0, 4)
      .map((entry) => `${entry.classId} ${entry.classOwnedActiveCount}/${entry.accessibleActiveCount}`)
      .join(', ');
    process.stdout.write('\nSkill system health\n');
    process.stdout.write(`Skills ${skills.skillCount} (${skills.activeSkillCount} active, ${skills.passiveSkillCount} passive), purpose coverage ${skills.purposeCoveragePercent}%, issues ${skills.issueCount}\n`);
    process.stdout.write(`Family coverage ${coverage.fullyCoveredOwnerCount}/${skills.ownerCount} accessible, ${coverage.classOwnedFullyCoveredOwnerCount}/${skills.ownerCount} class-owned, base-loop preservation floor ${coverage.minAdvancedBaseLoopPreservationPercent}%\n`);
    process.stdout.write(`Max active skills owned/accessed ${coverage.maxClassOwnedActiveCount}/${coverage.maxAccessibleActiveCount}, primary trainers ${coverage.primaryTrainingOwnerCount}/${skills.ownerCount}, lean owners ${leanOwners}\n`);
    process.stdout.write(`Prereqs missing/over-cap ${prereq.missingPrerequisiteCount || 0}/${prereq.rankOverCapCount || 0}, max depth ${prereq.maxPrerequisiteDepth || 0}, obsolete base skills ${obsolete.obsoleteBaseSkillCount || 0}\n`);
  }
  if (report.enemyEcosystem) {
    const ecosystem = report.enemyEcosystem;
    const archetypes = ecosystem.archetypeCoverage || {};
    const mapDistribution = ecosystem.mapDistribution || {};
    const counters = ecosystem.counterCoverage || {};
    const archetypeText = (archetypes.entries || [])
      .map((entry) => `${entry.id} ${entry.usedEnemyCount}`)
      .join(', ');
    const bracketText = (mapDistribution.bracketEntries || [])
      .filter((entry) => entry.mapCount)
      .map((entry) => `${entry.id} ${entry.uniqueEnemyCount}/${entry.archetypeCount}`)
      .join(', ');
    const counterText = (counters.entries || [])
      .map((entry) => `${entry.id} ${entry.enemyCount}`)
      .join(', ');
    process.stdout.write('\nEnemy ecosystem health\n');
    process.stdout.write(`Enemies ${ecosystem.usedEnemyCount}/${ecosystem.totalEnemyCount} used, behaviors ${ecosystem.behaviorCount}, families ${ecosystem.familyCount}, archetypes ${archetypes.coveredArchetypeCount}/${archetypes.requiredArchetypeCount}, issues ${ecosystem.issueCount}\n`);
    process.stdout.write(`Archetypes: ${archetypeText}\n`);
    process.stdout.write(`Map variety ${mapDistribution.mapsWithVarietyCount}/${mapDistribution.fieldDungeonMapCount} field/dungeon maps, boss arenas ${mapDistribution.bossArenaMapCount}, bracket enemy/archetype counts ${bracketText}\n`);
    process.stdout.write(`Counter coverage ${counters.coveredCounterCount}/${counters.counterCount}: ${counterText}\n`);
  }
  if (report.field && Array.isArray(report.field.maps)) {
    process.stdout.write('\nField map efficiency\n');
    report.field.maps.forEach((map) => {
      const top = (map.results || []).slice(0, 3).map((result) =>
        `${result.classId} ${result.efficiencyIndex}`).join(', ');
      const archetypes = map.profile && Array.isArray(map.profile.archetypes) ? map.profile.archetypes.join('/') : 'mixed';
      const reward = map.rewardCadence || {};
      process.stdout.write(`${map.name} L${map.level} [${archetypes}] target ${map.targetMinutes}m, TTK ${map.normalTtkSeconds}s, reward ${reward.smallVisibleMinutes || 0}/${reward.mediumProgressMinutes || 0}m: ${top}\n`);
    });
  }
  if (report.field && Array.isArray(report.field.progression)) {
    process.stdout.write('\nProgression pacing\n');
    report.field.progression.forEach((bracket) => {
      if (!bracket.mapCount) return;
      process.stdout.write(`${bracket.label}: median ${bracket.medianTimeToLevelMinutes}m, TTK ${bracket.medianNormalTtkSeconds}s, rewards ${bracket.medianSmallRewardMinutes}/${bracket.medianMediumProgressMinutes}m, range ${bracket.fastestMedianClassMinutes}-${bracket.slowestMedianClassMinutes}m across ${bracket.mapCount} map(s)\n`);
    });
  }
  if (report.field && report.field.rewardSource) {
    const reward = report.field.rewardSource;
    const tables = Object.entries(reward.tableMedians || {})
      .map(([tableId, dropsPerHour]) => `${tableId} ${dropsPerHour}/h`)
      .join(', ');
    process.stdout.write('\nReward source health\n');
    process.stdout.write(`Maps ${reward.mapCount}, optional ${reward.medianOptionalDropsPerHour}/h, deterministic ${reward.medianDeterministicProgressPerHour}/h, optional rare ${reward.medianOptionalRareOrBetterMinutes}m, max source share ${reward.maxOptionalTableShare}, issues ${reward.issueCount}\n`);
    process.stdout.write(`${tables}\n`);
  }
  if (report.field && report.field.survivability) {
    const survival = report.field.survivability;
    const classes = (survival.classSummaries || [])
      .slice()
      .sort((a, b) => b.medianPotionCostEarningsPercent - a.medianPotionCostEarningsPercent)
      .slice(0, 4)
      .map((entry) => `${entry.classId} ${entry.medianPotionCostEarningsPercent}%/${entry.medianDamageTakenPerHour} dmg`)
      .join(', ');
    process.stdout.write('\nField survivability and potion pressure\n');
    process.stdout.write(`Maps ${survival.mapCount}, median potion cost ${survival.medianPotionCostEarningsPercent}%, max potion cost ${survival.maxPotionCostEarningsPercent}%, max death risk ${survival.maxDeathRiskPerHour}/h, issues ${survival.issueCount}\n`);
    process.stdout.write(`${classes}\n`);
  }
  if (report.field && report.field.mapTuning) {
    const tuning = report.field.mapTuning;
    const warningMix = (tuning.warnings || []).reduce((counts, warning) => {
      const id = warning && warning.warningId || 'unknown';
      counts[id] = (counts[id] || 0) + 1;
      return counts;
    }, {});
    const warningText = Object.entries(warningMix)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([warningId, count]) => `${warningId} ${count}`)
      .join(', ');
    const retentionRisks = (tuning.maps || [])
      .slice()
      .sort((a, b) => Number(b.metrics && b.metrics.abandonmentRiskIndex || 0) - Number(a.metrics && a.metrics.abandonmentRiskIndex || 0))
      .slice(0, 3)
      .map((map) => `${map.mapId} ${map.metrics && map.metrics.abandonmentRiskIndex || 0}`)
      .join(', ');
    process.stdout.write('\nMap tuning and retention friction\n');
    process.stdout.write(`Combat maps ${tuning.mapCount} (${tuning.fieldMapCount} fields, ${tuning.dungeonMapCount} dungeons, ${tuning.bossArenaMapCount} bosses), median idle ${tuning.medianIdleTimePercent}%, travel ${tuning.medianTravelSharePercent}%, class spread ${tuning.medianClassPerformanceSpreadPercent}%, party overlap ${tuning.medianPartyOverlapPercent}%, warnings ${tuning.warningCount}\n`);
    const abuseControlPercent = Math.round(Number(tuning.highRiskFarmControlCoverage || 0) * 100);
    process.stdout.write(`High-risk farm maps ${tuning.highRiskFarmMapCount}, abuse-control coverage ${abuseControlPercent}%${warningText ? `, warnings: ${warningText}` : ''}\n`);
    if (retentionRisks) process.stdout.write(`Top abandonment risk: ${retentionRisks}\n`);
  }
  if (report.field && report.field.levelCurve) {
    const curve = report.field.levelCurve;
    process.stdout.write('\nLevel curve spike checks\n');
    process.stdout.write(`Checked L2-${curve.maxLevel}, max increase ${curve.maxIncreasePercent}%, unjustified spikes ${curve.unjustifiedSpikeCount}\n`);
    (curve.spikes || []).forEach((entry) => {
      const unlocks = (entry.unlocks || []).slice(0, 4).map((unlock) => unlock.category).join(', ') || 'none';
      process.stdout.write(`L${entry.level}: +${entry.increasePercent}% (${entry.previousMinutes}m -> ${entry.minutes}m), unlocks: ${unlocks}\n`);
    });
  }
  if (report.damageStatFormula) {
    const formula = report.damageStatFormula;
    const identity = formula.baseIdentity || {};
    const caps = formula.multiplierCaps && formula.multiplierCaps.maxByTier || {};
    const sources = formula.sourceCounts || {};
    const mitigationText = (formula.mitigation && formula.mitigation.samples || [])
      .map((sample) => `${sample.defense} def -> ${sample.damage}`)
      .join(', ');
    process.stdout.write('\nDamage/stat formula health\n');
    process.stdout.write(`Buckets ${formula.bucketCount} (${formula.positiveMultiplierBucketCount} positive multipliers), additive sources ${sources.additiveSourceCount || 0}, issues ${formula.issueCount}\n`);
    process.stdout.write(`Identity floors: Warrior eHP ${identity.fighterMinEffectiveHpIndex}, Mage MP ${identity.mageMinMpIndex}, Archer speed/range ${identity.archerMinSpeedIndex}/${identity.archerMinRangeIndex}, max base power spread ${identity.maxPowerSpreadPercent}%\n`);
    process.stdout.write(`Multiplier lines ${formula.multiplierCaps.lineCount}, rare/relic/celestial caps ${caps.rare}/${caps.relic}/${caps.celestial}, card/trait/passive sources ${sources.cardMultiplierBonuses || 0}/${sources.gearTraitMultiplierBonuses || 0}/${sources.passiveMultiplierSkills || 0}\n`);
    process.stdout.write(`Mitigation samples on 100 raw damage: ${mitigationText}\n`);
  }
  if (report.equipment) {
    const equipment = report.equipment;
    const unprotectedBands = equipment.upgradeBands && Array.isArray(equipment.upgradeBands.unprotected)
      ? equipment.upgradeBands.unprotected
      : [];
    const bandText = unprotectedBands
      .map((band) => `${band.label} ${Math.round((band.successChance || 0) * 100)}/${Math.round((band.failChance || 0) * 100)}/${Math.round((band.destroyChance || 0) * 100)}`)
      .join(', ');
    const materialSources = Object.entries(equipment.materialSources && equipment.materialSources.enemyDropSources || {})
      .map(([materialId, count]) => `${materialId} ${count}`)
      .join(', ');
    process.stdout.write('\nEquipment and upgrade health\n');
    process.stdout.write(`Items ${equipment.equipmentItemCount} (${equipment.shopItemCount} shop, ${equipment.randomDropItemCount} random, ${equipment.bossItemCount} boss), sets ${equipment.setCount}, advanced offhands ${equipment.advancedOffhandCoverage.coveredCount}/${equipment.advancedOffhandCoverage.classCount}, baseline safe +${equipment.upgradeBands.baselineSafeCeiling}, issues ${equipment.issueCount}\n`);
    process.stdout.write(`Upgrade bands S/F/D: ${bandText}; protection ${equipment.upgradeBands.protectionAideId || 'none'}, failure salvage ${equipment.upgradeBands.failureSalvageMode}\n`);
    process.stdout.write(`Upgrade material enemy sources: ${materialSources}; attunement rare/relic/celestial max ${equipment.attunementMultiplierCaps.maxByTier.rare}/${equipment.attunementMultiplierCaps.maxByTier.relic}/${equipment.attunementMultiplierCaps.maxByTier.celestial}\n`);
  }
  if (report.economy) {
    const economy = report.economy;
    const sinkTypes = economy.sinks && economy.sinks.byType || {};
    const sinkText = Object.entries(sinkTypes)
      .filter(([, entry]) => entry && entry.count > 0)
      .map(([typeId, entry]) => `${typeId} ${entry.count}`)
      .join(', ');
    const telemetryText = (economy.telemetryFields || []).join(', ');
    process.stdout.write('\nEconomy health\n');
    process.stdout.write(`Field coin faucet median ${economy.fieldFaucets.medianCurrencyPerHour}/h (${economy.fieldFaucets.minCurrencyPerHour}-${economy.fieldFaucets.maxCurrencyPerHour}/h), deterministic rewards ${economy.deterministicCurrencyRewards.currency} coins/${economy.deterministicCurrencyRewards.starTokens} Star Tokens, issues ${economy.issueCount}\n`);
    process.stdout.write(`Coin sinks ${economy.sinks.sinkCount} across ${economy.sinks.sinkTypeCount} types, repeatable ${economy.sinks.repeatableSinkCount}, median repeatable sink ${economy.medianRepeatableSinkMinutes}m, ${sinkText}\n`);
    process.stdout.write(`Item purpose coverage ${Math.round((economy.itemPurpose.purposeCoverage || 0) * 1000) / 10}% (${economy.itemPurpose.deadItemCount} dead), market listings ${economy.market.listingCount} (${economy.market.repeatableListingCount} repeatable/${economy.market.onceListingCount} once), cash-shop power ${economy.cashShopPowerItemCount}\n`);
    process.stdout.write(`Economy telemetry: ${telemetryText}\n`);
  }
  if (report.retention) {
    const retention = report.retention;
    const playerCoverage = retention.playerTypeCoverage || {};
    const cashShop = retention.cashShop || {};
    const directives = retention.fractureDirectives || {};
    const lanes = (retention.longTermLanes || [])
      .filter((lane) => lane.available)
      .map((lane) => lane.laneId)
      .slice(0, 6)
      .join(', ');
    const missingTypes = (playerCoverage.missingTypeIds || []).join(', ');
    const directiveIssueText = (directives.issueIds || []).join(', ');
    const directiveSeasonIds = (directives.referencedSeasonIds || []).join(', ') || 'none';
    const directiveSummaryLines = [
      'Fracture directives ' + (directives.weeklyChoiceCount || 0) +
        ' weekly choices, modeled ' + (directives.modeledDurationMinMinutes || 0) + '-' +
        (directives.modeledDurationMaxMinutes || 0) + 'm (median ' +
        (directives.modeledDurationMedianMinutes || 0) + 'm), reward parity ' +
        (directives.rewardParity ? 'yes' : 'no') + ', playstyles ' +
        (directives.uniquePlaystyleCount || 0) + '/' + (directives.weeklyChoiceCount || 0),
      'Directive season links ' + (directives.validSeasonReferenceDirectiveCount || 0) + '/' +
        (directives.weeklyChoiceCount || 0) + ' valid across ' +
        (directives.referencedSeasonCount || 0) + ' referenced season' +
        ((directives.referencedSeasonCount || 0) === 1 ? '' : 's') + ' (' + directiveSeasonIds +
        '), referenced rewards ' + (directives.referencedRewardMatchCount || 0) + '/' +
        (directives.weeklyChoiceCount || 0) + ' matched, missing ' +
        (directives.missingSeasonReferenceDirectiveIds || []).length + ', invalid ' +
        (directives.invalidSeasonReferenceDirectiveIds || []).length,
      'Directive references ' + (directives.validReferenceDirectiveCount || 0) + '/' +
        (directives.weeklyChoiceCount || 0) + ' valid, power rewards ' +
        (directives.powerRewardCount || 0) + ', stabilization visual-only ' +
        (directives.visualOnlyStabilizationCount || 0) + '/' + (directives.weeklyChoiceCount || 0) +
        ' and capped ' + (directives.cappedStabilizationCount || 0) + '/' +
        (directives.weeklyChoiceCount || 0) + ' (max ' +
        (directives.maxStabilizationSeals || 0) + ' seals), issues ' +
        (directives.issueCount || 0) + (directiveIssueText ? ' (' + directiveIssueText + ')' : '')
    ];
    process.stdout.write('\nRetention and chore health\n' + directiveSummaryLines.join('\n') + '\n');
    process.stdout.write(`Daily rewards ${retention.dailyLoginRewardCount}, milestones ${retention.dailyLoginMilestoneCount}, mandatory checklist ${retention.mandatoryDailyChecklistCount}/${retention.estimatedMandatoryDailyMinutes}m, active season goals ${retention.activeSeasonObjectiveCount}/${retention.estimatedSeasonGoalMinutes}m, issues ${retention.issueCount}\n`);
    process.stdout.write(`Long-term lanes ${retention.longTermLaneCount}, player coverage ${playerCoverage.coveredCount}/${playerCoverage.typeCount}${missingTypes ? ` missing ${missingTypes}` : ''}, examples ${lanes}\n`);
    process.stdout.write(`Accomplishments ${retention.accomplishmentCount} across ${retention.accomplishmentCategoryCount} categories/${retention.accomplishmentTierCount} tiers, monster guide ${retention.liveMonsterGuideCount}, cards ${retention.cardCount}, mastery tracks ${retention.classMasteryTrackCount}, roster ${retention.rosterTraitCount}/${retention.rosterSynergyCount}\n`);
    process.stdout.write(`Cash shop items ${cashShop.itemCount}, cosmetics ${cashShop.cosmeticItemCount}, power ${cashShop.powerItemCount}, earnable buffs ${cashShop.earnableBuffBundleCount}, max weekly limit ${cashShop.maxWeeklyLimit}, catch-up sources ${retention.catchUpSourceCount}\n`);
  }
  if (report.bossParty && report.bossParty.summary) {
    const summary = report.bossParty.summary;
    process.stdout.write('\nBoss and party checks\n');
    process.stdout.write(`Encounters ${summary.encounterCount}, min mechanic categories ${summary.minimumCategoryCount}, deterministic progress ${summary.deterministicProgressCount}, chase rewards ${summary.randomChaseCount}, issues ${summary.issueCount}\n`);
    if (report.bossParty.clearTime) {
      const clear = report.bossParty.clearTime;
      const target = Array.isArray(clear.targetMinutes) ? clear.targetMinutes.join('-') : '6-12';
      process.stdout.write(`Boss TTK target ${target}m, median solo ${clear.medianSoloClearMinutes}m, fastest ${clear.fastestSoloClearMinutes}m, slowest ${clear.slowestSoloClearMinutes}m, HP scale ${clear.minHpScale}-${clear.maxHpScale}x, issues ${clear.issueCount}\n`);
    }
    if (report.bossParty.dryStreak) {
      const dry = report.bossParty.dryStreak;
      const rarityParts = Object.entries(dry.byRarity || {})
        .map(([rarity, entry]) => `${rarity} p95 ${entry.maxP95Clears} clears`)
        .join(', ');
      process.stdout.write(`Boss chase dry streaks sources ${dry.sourceCount}, max p95 ${dry.maxP95Clears} clears/${dry.maxP95Hours}h, max p99 ${dry.maxP99Clears} clears, issues ${dry.issueCount}${rarityParts ? ` (${rarityParts})` : ''}\n`);
    }
    if (report.bossParty.supportContribution) {
      const support = report.bossParty.supportContribution;
      const top = (support.topContributors || [])
        .slice(0, 4)
        .map((entry) => `${entry.classId} ${entry.totalSupportIndex}`)
        .join(', ');
      process.stdout.write(`Support contribution classes ${support.classCount}, contributors ${support.supportContributorCount}, categories ${support.coveredCategoryCount}, max share ${support.maxSingleClassShare}, issues ${support.issueCount}${top ? ` (${top})` : ''}\n`);
    }
    (report.bossParty.categoryCoverage || []).forEach((category) => {
      process.stdout.write(`${category.label}: ${category.count} boss(es)\n`);
    });
    process.stdout.write('Party HP scaling: ');
    process.stdout.write((report.bossParty.partyScaling || []).map((entry) =>
      `${entry.players}p ${entry.hpMultiplier}x HP/${entry.throughputIndexVsSolo} throughput`).join(', '));
    process.stdout.write('\n');
  }
}

const options = parseArgs(process.argv.slice(2));
const report = createBalanceReport(data, createProjectStarfallEngine, options);
if (options.json) {
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} else {
  printTextReport(report);
}
