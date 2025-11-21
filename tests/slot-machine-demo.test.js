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
  checkFileContains('demos/slot-machine-demo.html', 'slot-config/upgrade-definitions.json');
  checkFileContains('demos/slot-machine-demo.html', 'id="debug-add-coins"');

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

  const lambda = require('../aws/slot-machine-function/index.js');
  assert(lambda && typeof lambda._getUpgradeDefinition === 'function', 'lambda upgrade helper missing');
  upgradeDefs.forEach(def => {
    assert(lambda._getUpgradeDefinition(def.key), `lambda missing handler for upgrade "${def.key}"`);
  });

  assert(typeof lambda._evaluateSkills === 'function', 'lambda skill evaluator missing');
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
