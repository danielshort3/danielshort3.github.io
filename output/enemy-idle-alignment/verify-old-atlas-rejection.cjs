const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '../..');
const { CASES, readIdleFrames, measureRegistration, assertRegistration } = require('../../tests/project-starfall/project-starfall-enemy-registration.test.js');

(async () => {
  const results = [];
  for (const [id, specification] of Object.entries(CASES)) {
    const file = id === 'lava-tick' ? path.join(root, 'output/lava-tick-alignment/before-sheet.png') : path.join(__dirname, id, 'before-sheet.png');
    const measurement = measureRegistration(await readIdleFrames(file), specification);
    let rejected = false;
    try { assertRegistration(measurement, id); } catch (error) { rejected = true; }
    if (!rejected) throw new Error(`${id}: original bad atlas was not rejected`);
    results.push({ id, horizontalSpan: measurement.horizontalSpan, horizontalLoopDelta: measurement.horizontalLoopDelta, rejected });
  }
  fs.writeFileSync(path.join(__dirname, 'old-atlas-rejection.json'), JSON.stringify(results, null, 2) + '\n');
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
