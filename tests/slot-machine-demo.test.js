const fs = require('fs');
const path = require('path');

module.exports = function runSlotMachineDemoTests({ assert, checkFileContains }) {
  const configPath = path.join('slot-config', 'classic.json');
  assert(fs.existsSync(configPath), 'slot-config/classic.json missing');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  assert(Array.isArray(config.symbols) && config.symbols.length >= 10, 'slot config should list at least 10 symbols');
  assert(Number.isFinite(config.baseRows) && Number.isFinite(config.maxRows), 'slot config missing base/max rows');
  assert(Number.isFinite(config.baseReels) && Number.isFinite(config.maxReels), 'slot config missing base/max reels');
  assert(config.upgradeCosts && Array.isArray(config.upgradeCosts.rows), 'slot config missing upgradeCosts.rows');
  assert(Number.isFinite(config.baseSymbolCount), 'slot config missing baseSymbolCount');

  const missingAssets = [];
  (config.symbols || []).forEach(symbol => {
    if (!symbol.asset) {
      missingAssets.push(symbol.key);
      return;
    }
    const assetPath = path.join(symbol.asset);
    if (!fs.existsSync(assetPath)) {
      missingAssets.push(`${symbol.key}:${symbol.asset}`);
    }
  });
  assert(missingAssets.length === 0, `slot symbol assets missing on disk: ${missingAssets.join(', ')}`);

  checkFileContains('demos/slot-machine-demo.html', 'data-machine-config="slot-config/classic.json"');
  checkFileContains('demos/slot-machine-demo.html', 'id="slot-grid"');
  checkFileContains('demos/slot-machine-demo.html', 'id="upgrade-grid"');
  checkFileContains('demos/slot-machine-demo.html', 'id="machine-list"');
  checkFileContains('demos/slot-machine-demo.html', 'id="gear-slots"');
  checkFileContains('demos/slot-machine-demo.html', 'id="card-slots"');
  checkFileContains('demos/slot-machine-demo.html', 'slot-config/upgrade-definitions.json');
  checkFileContains('demos/slot-machine-demo.html', 'id="debug-add-coins"');
  checkFileContains('demos/slot-machine-demo.html', 'id="export-state-btn"');
  checkFileContains('demos/slot-machine-demo.html', 'id="import-state-btn"');
  checkFileContains('demos/slot-machine-demo.html', 'id="daily-claim-btn"');
  checkFileContains('demos/slot-machine-demo.html', 'id="bonus-list"');
  checkFileContains('demos/slot-machine-demo.html', 'id="hud-vip"');

  const demoHtml = fs.readFileSync('demos/slot-machine-demo.html', 'utf8');
  assert(/slotDemoDebug/.test(demoHtml), 'slot demo debug hook missing');
  assert(demoHtml.includes('PLACEHOLDER_ASSET'), 'slot demo missing placeholder asset constant');
  assert(fs.existsSync('img/slot/placeholder.png'), 'slot placeholder asset missing');

  const upgradesPath = path.join('slot-config', 'upgrade-definitions.json');
  assert(fs.existsSync(upgradesPath), 'upgrade definitions missing');
  const upgradeDefs = JSON.parse(fs.readFileSync(upgradesPath, 'utf8'));
  assert(Array.isArray(upgradeDefs) && upgradeDefs.length >= 10, 'upgrade catalog should list multiple entries');
  ['idle', 'retrigger', 'rows', 'betMultiplier'].forEach(key => {
    assert(upgradeDefs.some(def => def.key === key), `upgrade catalog missing "${key}"`);
  });
  ['payoutBoostUnlock', 'payoutBoostEffect', 'payoutBoostDuration', 'wildBoostUnlock', 'wildBoostEffect', 'wildBoostDuration']
    .forEach(key => {
      assert(!upgradeDefs.some(def => def.key === key), `upgrade catalog should not expose "${key}"`);
    });
  const idleDef = upgradeDefs.find(def => def.key === 'idle') || {};
  assert(idleDef.defaultLevel === 1, 'idle upgrade should start at level 1 for all accounts');
  upgradeDefs.forEach(def => {
    assert(Number.isFinite(def.cost), `upgrade "${def.key}" missing numeric cost`);
    assert(typeof def.category === 'string' && def.category.length > 0, `upgrade "${def.key}" missing category`);
  });

  const machineIndexPath = path.join('slot-config', 'machines', 'index.json');
  assert(fs.existsSync(machineIndexPath), 'slot-config/machines/index.json missing');
  const machineIds = JSON.parse(fs.readFileSync(machineIndexPath, 'utf8'));
  ['classic','burst','charm','goldrush','cyber','pirate','dragon','cosmic'].forEach(id => {
    assert(machineIds.includes(id), `machine index missing ${id}`);
    const cfgPath = path.join('slot-config', 'machines', `${id}.json`);
    assert(fs.existsSync(cfgPath), `machine config missing for ${id}`);
    const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    assert(Array.isArray(cfg.symbols) && cfg.symbols.length >= 9, `${id} should include symbol definitions`);
    const iconPath = cfg.assets && cfg.assets.icon;
    assert(iconPath && fs.existsSync(iconPath), `${id} icon missing at ${iconPath}`);
    const symAsset = cfg.symbols[0] && cfg.symbols[0].asset;
    assert(symAsset && fs.existsSync(symAsset), `${id} first symbol asset missing`);
  });

  const dropTables = JSON.parse(fs.readFileSync(path.join('slot-config', 'drop-tables.json'), 'utf8'));
  const tableKeys = Object.keys(dropTables.tables || {});
  machineIds.forEach(id => {
    assert(tableKeys.includes(id), `drop table missing machine key ${id}`);
  });

  const gearDefsPath = path.join('slot-config', 'gear-definitions.json');
  assert(fs.existsSync(gearDefsPath), 'gear definitions missing');
  const gearDefs = JSON.parse(fs.readFileSync(gearDefsPath, 'utf8'));
  assert(Array.isArray(gearDefs.rarities) && gearDefs.rarities.length >= 3, 'gear definitions missing rarities');
  assert(gearDefs.bonuses && gearDefs.bonuses.Basic, 'gear bonuses missing Basic tier');
  assert(fs.existsSync('img/slot/gear/basic/gear_basic_accessory.png'), 'basic gear icon missing');

  const cardDefsPath = path.join('slot-config', 'card-definitions.json');
  assert(fs.existsSync(cardDefsPath), 'card definitions missing');
  const cardDefs = JSON.parse(fs.readFileSync(cardDefsPath, 'utf8'));
  assert(Array.isArray(cardDefs.definitions) && cardDefs.definitions.length > 10, 'card definitions too small');
  assert(cardDefs.effects && cardDefs.effects['Lucky Penny'], 'card effects missing Lucky Penny');
  assert(fs.existsSync('img/slot/cards/basic/vip_lucky_penny.png'), 'card icon missing');

  const lambda = require('../aws/slot-machine-function/index.js');
  assert(lambda && typeof lambda._getUpgradeDefinition === 'function', 'lambda upgrade helper missing');
  upgradeDefs.forEach(def => {
    assert(lambda._getUpgradeDefinition(def.key), `lambda missing handler for upgrade "${def.key}"`);
  });
  assert(typeof lambda._advanceDaily === 'function', 'daily advance helper missing');
  assert(typeof lambda._dailyRewardFor === 'function', 'daily reward helper missing');
  assert(typeof lambda._formatDailyPayload === 'function', 'daily payload formatter missing');
  const dayMs = 24 * 60 * 60 * 1000;
  const baseMs = Date.UTC(2024, 0, 10);
  const advanced = lambda._advanceDaily({ streak: 2, lastClaimMs: baseMs - dayMs, claimedToday: true }, baseMs + (2 * 60 * 60 * 1000));
  assert(advanced.streak === 3, 'daily streak should advance by one day continously');
  assert(advanced.claimedToday === false, 'daily claim flag should reset on new day');
  const reset = lambda._advanceDaily({ streak: 5, lastClaimMs: baseMs - (3 * dayMs), claimedToday: true }, baseMs);
  assert(reset.streak === 1, 'daily streak should reset after gaps');
  const dayOneReward = lambda._dailyRewardFor(1);
  assert(dayOneReward.vipMarks === 35, 'day 1 daily reward should start at 35 VIP marks');
  const dayThreeReward = lambda._dailyRewardFor(3);
  assert(dayThreeReward.vipMarks === 35 + (22 * 2), 'daily reward should step by 22 VIP marks per day');
  const daySevenReward = lambda._dailyRewardFor(7);
  assert(daySevenReward.drops.some(drop => drop.type === 'reelMod'), 'day 7 reward should include a mod drop');
  const dailyPayload = lambda._formatDailyPayload({ streak: 4, lastClaimMs: baseMs, claimedToday: false }, baseMs);
  assert(dailyPayload.todayReward && dailyPayload.todayReward.vipMarks, 'daily payload missing todayReward.vipMarks');
  assert(dailyPayload.ready === true, 'daily payload ready flag should be true when unclaimed');

  assert(typeof lambda._evaluateSkills === 'function', 'lambda skill evaluator missing');
  assert(typeof lambda._handleSync === 'function', 'lambda sync handler missing');
  const nowMs = 1_000_000;
  const activation = lambda._evaluateSkills({
    payload: { activeSkills: { dropRate: true } },
    upgrades: { dropBoostUnlock: 1, dropRateEffect: 2, dropRateDuration: 1 },
    skillState: {},
    nowMs
  });
  assert(activation.dropRateActive, 'drop skill should activate when unlocked');
  assert(activation.skillState.dropRate.activeUntil > nowMs, 'drop skill should set activeUntil');
  assert(activation.skillState.dropRate.cooldownUntil > activation.skillState.dropRate.activeUntil, 'drop skill cooldown should follow active window');
  const cooldown = lambda._evaluateSkills({
    payload: { activeSkills: { dropRate: true } },
    upgrades: { dropBoostUnlock: 1, dropRateEffect: 2, dropRateDuration: 1 },
    skillState: activation.skillState,
    nowMs: activation.skillState.dropRate.activeUntil + 1
  });
  assert(!cooldown.dropRateActive, 'drop skill should expire after active window');
  assert(cooldown.skillState.dropRate.cooldownUntil > cooldown.skillState.dropRate.activeUntil, 'cooldown should remain set after expiry');

  const demoHtmlUpgrade = fs.readFileSync('demos/slot-machine-demo.html', 'utf8');
  assert(demoHtmlUpgrade.includes('Account already exists. Sign in instead.'), 'register conflict hint missing');
  assert(demoHtmlUpgrade.includes("request('/session', { type }"), 'upgrade flow should use /session endpoint');
};
