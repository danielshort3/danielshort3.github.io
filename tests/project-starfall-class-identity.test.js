'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const Data = require('../js/games/project-starfall/project-starfall-data.js');
const EquipmentAttachments = require('../js/games/project-starfall/engine/equipment-attachments.js');

const ROOT = path.resolve(__dirname, '..');
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

function fileHash(repoPath) {
  return crypto.createHash('sha256').update(fs.readFileSync(fullPath(repoPath))).digest('hex');
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

function countAlphaMaskDiff(first, second) {
  let count = 0;
  for (let pixel = 0; pixel < FRAME_SIZE * FRAME_SIZE; pixel += 1) {
    if ((first[pixel * 4 + 3] > 20) !== (second[pixel * 4 + 3] > 20)) count += 1;
  }
  return count;
}

function isVisiblePixelNearTransparency(raw, width, height, x, y) {
  for (let offsetY = -2; offsetY <= 2; offsetY += 1) {
    for (let offsetX = -2; offsetX <= 2; offsetX += 1) {
      if (!offsetX && !offsetY) continue;
      const neighborX = x + offsetX;
      const neighborY = y + offsetY;
      if (neighborX < 0 || neighborY < 0 || neighborX >= width || neighborY >= height) return true;
      if (raw[(neighborY * width + neighborX) * 4 + 3] <= 12) return true;
    }
  }
  return false;
}

function countVisibleGreenEdgeSpill(raw, width, height) {
  let count = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= 20) continue;
      const r = raw[offset];
      const g = raw[offset + 1];
      const b = raw[offset + 2];
      if (g <= 24 || g - Math.max(r, b) <= 8 || g - r <= 12 || g - b <= 12) continue;
      if (isVisiblePixelNearTransparency(raw, width, height, x, y)) count += 1;
    }
  }
  return count;
}

function countCyanHardwarePixels(raw, width, height) {
  let count = 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    const r = raw[offset];
    const g = raw[offset + 1];
    const b = raw[offset + 2];
    if (raw[offset + 3] > 20 && b > 70 && g > 55 && b - r > 25 && Math.abs(b - g) < 55) count += 1;
  }
  return count;
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
  assert.strictEqual(Data.PLAYER_ART_VERSION, 'v5');
  assert.deepStrictEqual(Data.CLASS_FAMILY_IDS, FAMILY_IDS);
  assert.deepStrictEqual(Data.CLASS_BODY_FAMILIES, CLASS_FAMILIES);

  const portraitPaths = FAMILY_IDS.map((familyId) => Data.CLASS_ASSETS[familyId]);
  const animationPaths = FAMILY_IDS.map((familyId) => Data.PLAYER_ANIMATION_ASSETS[familyId].sheet);
  assert.strictEqual(new Set(portraitPaths).size, FAMILY_IDS.length,
    'base classes should expose three cache-safe family portraits');
  assert.strictEqual(new Set(animationPaths).size, FAMILY_IDS.length,
    'base classes should expose three cache-safe family animation sheets');
  assert.strictEqual(new Set(portraitPaths.map(fileHash)).size, FAMILY_IDS.length,
    'family portraits should not be byte-identical aliases');
  assert.strictEqual(new Set(animationPaths.map(fileHash)).size, FAMILY_IDS.length,
    'family animation sheets should not be byte-identical aliases');

  for (const [classId, familyId] of Object.entries(CLASS_FAMILIES)) {
    assert.strictEqual(Data.CLASS_ASSETS[classId], Data.CLASS_ASSETS[familyId],
      `${classId} portrait should reuse the ${familyId} family`);
    assert.strictEqual(Data.PLAYER_ANIMATION_ASSETS[classId], Data.PLAYER_ANIMATION_ASSETS[familyId],
      `${classId} animation should reuse the ${familyId} family object`);
    assert.strictEqual(Data.getClassBodyFamilyId(classId), familyId);
  }
  assert(!Object.values(Data.CLASS_ASSETS).includes(Data.GENERIC_PLAYER_ASSET),
    'registered classes should not alias the generic recovery portrait');
  assert(!Object.values(Data.PLAYER_ANIMATION_ASSETS).includes(Data.GENERIC_PLAYER_ANIMATION_ASSET),
    'registered classes should not alias the generic recovery sheet');

  const familySheets = {};
  for (const familyId of FAMILY_IDS) {
    const portraitDecoded = await sharp(fullPath(Data.CLASS_ASSETS[familyId]))
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    assert.strictEqual(portraitDecoded.info.width, 320);
    assert.strictEqual(portraitDecoded.info.height, 320);
    assert.strictEqual(
      countVisibleGreenEdgeSpill(portraitDecoded.data, portraitDecoded.info.width, portraitDecoded.info.height),
      0,
      `${familyId} portrait should not retain green-dominant chroma fringe`
    );
    assert(countCyanHardwarePixels(portraitDecoded.data, portraitDecoded.info.width, portraitDecoded.info.height) >= 12,
      `${familyId} portrait despill should preserve cyan star-tech hardware`);
    const decoded = await sharp(fullPath(Data.PLAYER_ANIMATION_ASSETS[familyId].sheet))
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    assert.strictEqual(decoded.info.width, 960);
    assert.strictEqual(decoded.info.height, 1600);
    assert.strictEqual(
      countVisibleGreenEdgeSpill(decoded.data, decoded.info.width, decoded.info.height),
      0,
      `${familyId} animation sheet should not retain green-dominant chroma fringe`
    );
    familySheets[familyId] = decoded.data;

    for (const rowId of ['idle', 'run']) {
      for (let frameIndex = 0; frameIndex < 6; frameIndex += 1) {
        const frameRaw = getFrameRaw(decoded.data, rowId, frameIndex);
        const bounds = getAlphaBounds(frameRaw);
        const registration = EquipmentAttachments.getPlayerSpriteRegistration(rowId, frameIndex);
        assert(bounds, `${familyId} ${rowId} frame ${frameIndex} should contain visible art`);
        assert(Math.abs(bounds.maxY - registration.groundY) <= 2,
          `${familyId} ${rowId} frame ${frameIndex} should preserve registered ground contact`);
        assert(registration.originX >= bounds.minX && registration.originX <= bounds.maxX,
          `${familyId} ${rowId} frame ${frameIndex} should preserve its registered origin`);
      }
    }
    await validateEquipmentParts(decoded.data, familyId);
  }

  const silhouetteSamples = [
    ['idle', 0],
    ['run', 2],
    ['basic', 2],
    ['skill', 3]
  ];
  for (let firstIndex = 0; firstIndex < FAMILY_IDS.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < FAMILY_IDS.length; secondIndex += 1) {
      const firstId = FAMILY_IDS[firstIndex];
      const secondId = FAMILY_IDS[secondIndex];
      for (const [rowId, frameIndex] of silhouetteSamples) {
        const diff = countAlphaMaskDiff(
          getFrameRaw(familySheets[firstId], rowId, frameIndex),
          getFrameRaw(familySheets[secondId], rowId, frameIndex)
        );
        assert(diff >= 225,
          `${firstId} and ${secondId} should keep distinct ${rowId} silhouettes (${diff}px)`);
      }
    }
  }

  const engineSource = fs.readFileSync(fullPath('js/games/project-starfall/project-starfall-engine.js'), 'utf8');
  assert(engineSource.includes('getClassPlayerAsset(classId)'),
    'the runtime should resolve class-family portrait fallbacks explicitly');
  assert(!engineSource.includes("asset: Data.GENERIC_PLAYER_ASSET || ''"),
    'renderer snapshots should not publish the generic portrait for every class');

  process.stdout.write('Project Starfall class identity tests passed.\n');
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exit(1);
});
