'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const ROOT = path.resolve(__dirname, '..');
const OUTPUT = path.join(ROOT, 'demos', 'data');

function collectDatasets() {
  const files = [];
  const read = (source, destination) => {
    const bytes = fs.readFileSync(path.join(ROOT, source));
    const data = JSON.parse(bytes.toString('utf8'));
    files.push({ source, destination, bytes });
    return data;
  };
  const target = read('aws/target-empty-package/output/data.json', 'target-empty-package/data.json');
  if (target.rows.length !== target.meta.recordCount || !target.rows.length) throw new Error('Target record count mismatch.');
  for (const row of target.rows) {
    if (!/^Employee-\d+$/.test(row.employee) || !/^Location-\d+$/.test(row.location)) {
      throw new Error('Only anonymized Target employee and location identifiers may be published.');
    }
    if (Object.keys(row).some((key) => !['employee', 'datetime', 'value', 'location', 'condition', 'department', 'class', 'item'].includes(key))) {
      throw new Error('Unexpected Target record fields; review the publication schema.');
    }
  }
  const retail = read('aws/retail-loss-sales/output/data.json', 'retail-loss-sales/data.json');
  if (!/^Store_\d+$/.test(retail.meta.salesStore) || !retail.sales.weekly.length || !retail.incidents.stores.length) {
    throw new Error('Retail dataset is incomplete or has an unexpected store identifier.');
  }
  for (const row of [...retail.incidents.stores, ...retail.inventory.stores]) {
    if (!/^Store_\d+$/.test(row.store)) throw new Error('Only anonymized retail stores may be published.');
  }
  for (const row of retail.emptyPackages.employees) {
    if (!/^Employee(?:_?ID)?[_-]?\d+$/i.test(row.employee)) throw new Error('Only anonymized retail employees may be published.');
  }
  const covid = read('aws/covid-outbreak-drivers/output/meta.json', 'covid-outbreak/meta.json');
  if (!covid.dates.length || !covid.states.length || !covid.dates.includes(covid.latest)) throw new Error('COVID metadata is incomplete.');
  for (const date of covid.dates) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error('Unexpected COVID date.');
    const data = read(`aws/covid-outbreak-drivers/output/by-date/${date}.json`, `covid-outbreak/by-date/${date}.json`);
    if (data.date !== date || !Array.isArray(data.states) || !Array.isArray(data.hotspots)) throw new Error('COVID date data mismatch.');
  }
  for (const state of covid.states) {
    if (!/^[A-Z]{2}$/.test(state.id)) throw new Error('Unexpected COVID state.');
    const data = read(`aws/covid-outbreak-drivers/output/state/${state.id}.json`, `covid-outbreak/state/${state.id}.json`);
    if (data.state.id !== state.id || !Array.isArray(data.history)) throw new Error('COVID state data mismatch.');
  }
  return files;
}

function main() {
  // Validate the entire source set before updating any public asset. The ignored
  // preparation output is optional at runtime: the checked-in assets are complete.
  const files = collectDatasets();
  const manifest = {
    description: 'Existing public dashboard responses, preserved as static site data. Raw source workbooks and CSVs are not included.',
    sources: {
      'target-empty-package': 'aws/target-empty-package/precompute.py',
      'retail-loss-sales': 'aws/retail-loss-sales/precompute.py',
      'covid-outbreak': 'aws/covid-outbreak-drivers/precompute.py'
    },
    files: files.map(({ destination, bytes }) => ({
      path: destination,
      bytes: bytes.length,
      sha256: crypto.createHash('sha256').update(bytes).digest('hex')
    }))
  };
  for (const { destination, bytes } of files) {
    const output = path.join(OUTPUT, destination);
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, bytes);
  }
  fs.writeFileSync(path.join(OUTPUT, 'dashboard-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Prepared ${files.length} local dashboard datasets (${files.reduce((total, file) => total + file.bytes.length, 0)} bytes).`);
}

if (require.main === module) main();

module.exports = { collectDatasets };
