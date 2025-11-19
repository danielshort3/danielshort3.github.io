const fs = require('fs');
const path = require('path');

module.exports = function runSlotMachineDemoTests({ assert, checkFileContains }) {
  const configPath = path.join('slot-config', 'classic.json');
  assert(fs.existsSync(configPath), 'slot-config/classic.json missing');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  assert(Array.isArray(config.symbols) && config.symbols.length >= 10, 'slot config should list at least 10 symbols');

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

  const demoHtml = fs.readFileSync('demos/slot-machine-demo.html', 'utf8');
  assert(/slotDemoDebug/.test(demoHtml), 'slot demo debug hook missing');
  assert(demoHtml.includes('PLACEHOLDER_ASSET'), 'slot demo missing placeholder asset constant');
  assert(fs.existsSync('img/slot/placeholder.png'), 'slot placeholder asset missing');
};
