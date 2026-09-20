'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createMapRuntime } = require('../../js/games/project-starfall/engine/map-runtime.js');

const root = path.resolve(__dirname, '../..');
const sourceRoot = path.join(root, 'asset-sources/project-starfall/overhaul-v1/scenery');
const ledger = JSON.parse(fs.readFileSync(path.join(sourceRoot, 'ledger.json'), 'utf8'));
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

async function main() {
  const crossing = data.MAPS.find((map) => map.id === 'starfallCrossing');
  const runtime = createMapRuntime(crossing, null, { maps: data.MAPS });
  assert.strictEqual(runtime.townScene.structureTheme, 'crossingLandmarks', 'map publication and runtime must preserve the chosen landmark atlas');
  const atlas = data.ENVIRONMENT_STRUCTURE_ASSETS[runtime.townScene.structureTheme];
  assert(atlas && atlas.path !== data.ENVIRONMENT_STRUCTURE_ASSETS.townLandmarks.path,
    'Crossing architecture must not silently fall back to unrelated shared-town cells');
  const crossingRecord = ledger.assets.find((record) => record.id === 'crossing-landmarks');
  assert.strictEqual(data.ASSET_BACKUP_PATHS[atlas.path], crossingRecord.backupOutputPath,
    'the runtime fallback must be recreated by the authoritative scenery importer');
  assert.strictEqual(hash(path.join(root, crossingRecord.backupOutputPath)), crossingRecord.outputSha256,
    'Crossing fallback must preserve the accepted landmark identities');
  assert.strictEqual(crossingRecord.backupSha256, crossingRecord.outputSha256,
    'fallback provenance records the same accepted artwork');
  for (const map of data.MAPS.filter((map) => map.safeZone && map.townScene && map.id !== crossing.id)) {
    assert.strictEqual(map.townScene.structureTheme, 'townLandmarks', `${map.id}: retain its existing town architecture`);
  }
  for (const structure of runtime.townScene.rearStructures.concat(runtime.townScene.stationFacades)) {
    assert.strictEqual(structure.w, structure.h, `${structure.cell}: square source cells should not be stretched into wide buildings`);
    assert(Number.isInteger(data.ENVIRONMENT_STRUCTURE_CELLS[structure.cell]), `${structure.cell}: known landmark cell`);
  }
  const rearCells = new Set(runtime.townScene.rearStructures.map((structure) => structure.cell));
  const frontierGate = runtime.townScene.rearStructures.find((structure) => structure.cell === 'frontierGate');
  const greenrootPortal = crossing.portals.find((portal) => portal.id === 'crossing_greenroot');
  assert.strictEqual(frontierGate.x + frontierGate.w / 2, greenrootPortal.x + (greenrootPortal.w || 58) / 2,
    'the frontier gate opening must align with the actual Greenroot departure portal');
  assert(runtime.townScene.stationFacades.every((structure) => !rearCells.has(structure.cell)),
    'service kiosks should not be shrunken duplicates of major landmarks');
  const shops = data.MAPS.filter((map) => map.shopInterior);
  assert.strictEqual(shops.length, 24, 'all regional shop interiors participate in the scenery contract');
  assert.strictEqual(new Set(shops.map((map) => map.asset)).size, 4, 'each vendor type has a distinct illustrated room');
  for (const shop of shops) {
    assert.strictEqual(shop.asset, `img/project-starfall/maps/shop-${shop.shopVendorType}-interior.webp`, `${shop.id}: correct room identity`);
    assert.strictEqual(shop.townScene.rearStructures.length, 0, `${shop.id}: no outdoor houses inside rooms`);
    assert.strictEqual(shop.townScene.streetProps.length, 0, `${shop.id}: no scenery obscures the vendor or exit`);
    assert.strictEqual(shop.environment.terrain, 'shop-interior', shop.id + ': purposeful neutral indoor floor material');
    assert.strictEqual(shop.environment.density, 0, `${shop.id}: no procedural outdoor plants`);
    assert.deepStrictEqual(shop.environment.propKinds, [], `${shop.id}: keep the room's actor lane clear`);
    assert.strictEqual(shop.platforms.length, 1, `${shop.id}: retain the existing single-floor geometry`);
    assert.strictEqual(shop.questNpcs[0].x, 680, `${shop.id}: preserve vendor position`);
    assert.strictEqual(shop.portals[0].x, 110, `${shop.id}: preserve the return portal`);
  }

  for (const group of ['terrain', 'props', 'ramps']) {
    const ashglass = ledger.assets.find((record) => record.id === `ashglass-pass-${group}`);
    const quarry = ledger.assets.find((record) => record.id === `oreback-quarry-${group}`);
    assert(ashglass && quarry);
    assert.strictEqual(ashglass.kit, 'ashglass', `${group}: volcanic-glass source kit`);
    assert.notStrictEqual(ashglass.sourcePath, quarry.sourcePath, `${group}: source identity must match its biome`);
    assert.notStrictEqual(hash(path.join(root, ashglass.outputPath)), hash(path.join(root, quarry.outputPath)),
      `${group}: Ashglass cannot regress to a byte-identical Quarry atlas`);
  }
  const ashglassTerrain = ledger.assets.find((record) => record.id === 'ashglass-pass-terrain');
  const { data: terrainPixels } = await sharp(path.join(root, ashglassTerrain.outputPath)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  for (const cell of [1, 2, 5, 6]) {
    for (let x = 0; x < 64; x += 1) {
      assert(terrainPixels[(cell * 64 + x) * 4 + 3] >= 128,
        `Ashglass cell ${cell}: continuous level contact edge without per-tile alpha dips`);
    }
  }
  const terrainCell = async (cell) => sharp(path.join(root, ashglassTerrain.outputPath))
    .extract({ left: cell * 64, top: 0, width: 64, height: 64 }).raw().toBuffer();
  assert(!(await terrainCell(1)).equals(await terrainCell(2)),
    'Ashglass contact tiles retain different authored interior rock formations');

  const revisedIds = ['ashglass-pass-terrain', 'ashglass-pass-props', 'ashglass-pass-ramps', 'crossing-landmarks', 'cinder-hollow',
    'shop-weapon-interior', 'shop-armor-interior', 'shop-supply-interior', 'shop-special-interior', 'shop-interior-terrain'];
  for (const id of revisedIds) {
    const record = ledger.assets.find((candidate) => candidate.id === id);
    assert(record, `${id}: provenance record`);
    assert.strictEqual(hash(path.join(root, record.sourcePath)), record.sourceSha256, `${id}: source hash`);
    assert.strictEqual(hash(path.join(root, record.outputPath)), record.outputSha256, `${id}: output hash`);
    assert.strictEqual(hash(path.join(root, record.promptPath)), record.promptSha256, `${id}: exact generation prompt`);
    const metadata = await sharp(path.join(root, record.outputPath)).metadata();
    assert.deepStrictEqual([metadata.width, metadata.height], record.outputDimensions, `${id}: export contract`);
  }

  const { data: pixels, info } = await sharp(path.join(root, atlas.path)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  for (let index = 0; index < 8; index += 1) {
    let opaque = 0;
    let transparent = 0;
    const left = index % 4 * 256;
    const top = Math.floor(index / 4) * 256;
    for (let y = 0; y < 256; y += 1) for (let x = 0; x < 256; x += 1) {
      const alpha = pixels[((top + y) * info.width + left + x) * 4 + 3];
      if (alpha >= 16) opaque += 1;
      else transparent += 1;
      if (x === 0 || x === 255 || y === 0 || y === 255) assert(alpha < 16, `landmark ${index}: no clipped silhouette`);
    }
    assert(opaque > 1000 && transparent > 1000, `landmark ${index}: real object and genuine empty alpha`);
  }
  console.log('Project Starfall contextual scenery passed: dedicated biome sources, continuous contact edges, aligned Crossing landmarks, 24 shop interiors and ten provenance/export checks.');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
