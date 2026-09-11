'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const Data = require('../../js/games/project-starfall/project-starfall-data.js');
const EquipmentAttachments = require('../../js/games/project-starfall/engine/equipment-attachments.js');

const ROOT = path.resolve(__dirname, '..', '..');
const FRAME_SIZE = 160;
const SHEET_WIDTH = 960;
const FAMILY_IDS = Object.freeze(['fighter', 'mage', 'archer']);
const CLASS_FAMILIES = Object.freeze({
  fighter: 'fighter',
  guardian: 'fighter',
  berserker: 'fighter',
  duelist: 'fighter',
  mage: 'mage',
  fireMage: 'mage',
  runeMage: 'mage',
  stormMage: 'mage',
  archer: 'archer',
  sniper: 'archer',
  trapper: 'archer',
  beastArcher: 'archer'
});

function fullPath(repoPath) {
  return path.join(ROOT, repoPath);
}

function getFrameRaw(sheetRaw, rowId, frameIndex) {
  const rowIndex = Data.PLAYER_ANIMATION_ROWS.indexOf(rowId);
  const output = Buffer.alloc(FRAME_SIZE * FRAME_SIZE * 4);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const sourceStart = (((rowIndex * FRAME_SIZE + y) * SHEET_WIDTH) + frameIndex * FRAME_SIZE) * 4;
    sheetRaw.copy(output, y * FRAME_SIZE * 4, sourceStart, sourceStart + FRAME_SIZE * 4);
  }
  return output;
}

function getAlphaBounds(raw) {
  let minX = FRAME_SIZE;
  let minY = FRAME_SIZE;
  let maxX = -1;
  let maxY = -1;
  let count = 0;
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      if (raw[(y * FRAME_SIZE + x) * 4 + 3] <= 20) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      count += 1;
    }
  }
  return count ? { minX, minY, maxX, maxY, count } : null;
}

function getNearestVisiblePixelDistance(frameRaw, point) {
  let minimum = Number.POSITIVE_INFINITY;
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      if (frameRaw[(y * FRAME_SIZE + x) * 4 + 3] <= 20) continue;
      minimum = Math.min(minimum, Math.hypot(x - Number(point.x || 0), y - Number(point.y || 0)));
    }
  }
  return minimum;
}

function findEquipmentVisual(fileId) {
  return Object.values(Data.EQUIPMENT_VISUALS || {}).find((visual) => visual && visual.fileId === fileId);
}

async function validateEquipmentParts(sheetRaw, familyId) {
  const weaponFileIds = {
    fighter: 'training-sword',
    mage: 'training-wand',
    archer: 'training-bow'
  };
  const visualFileIds = ['stitched-vest', 'traveler-boots', 'fieldguard-helm', weaponFileIds[familyId]];
  const samples = [
    ['idle', 0],
    ['run', 2],
    ['basic', 2],
    ['skill', 3]
  ];
  const metadataBySheet = new Map();
  for (const [rowId, frameIndex] of samples) {
    const frameRaw = getFrameRaw(sheetRaw, rowId, frameIndex);
    for (const fileId of visualFileIds) {
      const visual = findEquipmentVisual(fileId);
      assert(visual, `${familyId} starter equipment should expose ${fileId}`);
      const parts = EquipmentAttachments.resolveEquipmentAtlasParts(visual, rowId, frameIndex);
      assert(parts.length, `${familyId} ${fileId} should resolve on ${rowId} frame ${frameIndex}`);
      for (const part of parts) {
        const sheetPath = fullPath(part.frame.sheet);
        assert(fs.existsSync(sheetPath), `${fileId} atlas should exist at ${part.frame.sheet}`);
        if (!metadataBySheet.has(part.frame.sheet)) {
          metadataBySheet.set(part.frame.sheet, await sharp(sheetPath).metadata());
        }
        const metadata = metadataBySheet.get(part.frame.sheet);
        assert((part.frame.frameIndex + 1) * part.frame.frameWidth <= metadata.width,
          `${fileId} frame column should fit its atlas`);
        assert((part.frame.row + 1) * part.frame.frameHeight <= metadata.height,
          `${fileId} frame row should fit its atlas`);
        const socketDistance = getNearestVisiblePixelDistance(frameRaw, part.socket);
        const socketLimit = fileId === 'traveler-boots' ? 33 : fileId === weaponFileIds[familyId] ? 24 : 8;
        assert(socketDistance <= socketLimit,
          `${familyId} ${fileId} socket should remain registered on ${rowId} frame ${frameIndex} (${socketDistance.toFixed(1)}px)`);
      }
    }
  }
}

async function main() {
  assert.strictEqual(Data.PLAYER_ART_VERSION, 'classic');
  assert.deepStrictEqual(Data.CLASS_FAMILY_IDS, FAMILY_IDS);
  assert.deepStrictEqual(Data.CLASS_BODY_FAMILIES, CLASS_FAMILIES);
  assert.strictEqual(Data.GENERIC_PLAYER_ASSET, 'img/project-starfall/characters/generic-player.png');
  assert.strictEqual(Data.GENERIC_PLAYER_ANIMATION_ASSET.sheet,
    'img/project-starfall/animations/players/generic-player-sheet.png');

  for (const [classId, familyId] of Object.entries(CLASS_FAMILIES)) {
    assert.strictEqual(Data.CLASS_ASSETS[classId], Data.GENERIC_PLAYER_ASSET,
      classId + ' should reuse the restored shared player portrait');
    assert.strictEqual(Data.PLAYER_ANIMATION_ASSETS[classId], Data.GENERIC_PLAYER_ANIMATION_ASSET,
      classId + ' should reuse the restored shared player animation object');
    assert.strictEqual(Data.getClassBodyFamilyId(classId), familyId,
      classId + ' should preserve its gameplay family');
  }

  const portrait = await sharp(fullPath(Data.GENERIC_PLAYER_ASSET)).metadata();
  assert.strictEqual(portrait.width, 320);
  assert.strictEqual(portrait.height, 320);
  assert(portrait.hasAlpha, 'the restored player portrait should retain transparency');
  const decoded = await sharp(fullPath(Data.GENERIC_PLAYER_ANIMATION_ASSET.sheet))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  assert.strictEqual(decoded.info.width, SHEET_WIDTH);
  assert.strictEqual(decoded.info.height, Data.PLAYER_ANIMATION_ROWS.length * FRAME_SIZE);

  for (const rowId of Data.PLAYER_ANIMATION_ROWS) {
    const state = Data.GENERIC_PLAYER_ANIMATION_ASSET.states[rowId];
    assert(state && state.frames > 0, rowId + ' should retain an animation definition');
    for (let frameIndex = 0; frameIndex < state.frames; frameIndex += 1) {
      const frameRaw = getFrameRaw(decoded.data, rowId, frameIndex);
      const bounds = getAlphaBounds(frameRaw);
      const registration = EquipmentAttachments.getPlayerSpriteRegistration(rowId, frameIndex);
      assert(bounds, rowId + ' frame ' + frameIndex + ' should contain visible art');
      assert(bounds.minX > 0 && bounds.maxX < FRAME_SIZE - 1 && bounds.minY > 0 && bounds.maxY < FRAME_SIZE - 1,
        rowId + ' frame ' + frameIndex + ' should fit inside its transparent cell');
      if (rowId === 'idle' || rowId === 'run') {
        assert(Math.abs(bounds.maxY - registration.groundY) <= 2,
          rowId + ' frame ' + frameIndex + ' should preserve registered ground contact');
        assert(registration.originX >= bounds.minX && registration.originX <= bounds.maxX,
          rowId + ' frame ' + frameIndex + ' should preserve its registered origin');
      }
    }
  }
  for (const familyId of FAMILY_IDS) await validateEquipmentParts(decoded.data, familyId);

  const engineSource = fs.readFileSync(fullPath('js/games/project-starfall/project-starfall-engine.js'), 'utf8');
  assert(engineSource.includes('getClassPlayerAsset(classId)'),
    'the runtime should retain class-aware portrait lookup with shared restored art');

  process.stdout.write('Project Starfall class identity tests passed.\n');
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exit(1);
});
