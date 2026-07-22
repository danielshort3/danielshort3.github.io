#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const MANIFEST_PATH = path.join(ROOT, 'asset-sources/project-starfall/asset-generation-manifest.json');
const DATA_PATH = path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js');

function toPosix(filePath) {
  return String(filePath || '').replace(/\\/g, '/');
}

function fullPath(filePath) {
  return path.join(ROOT, filePath);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function assertExists(filePath, label) {
  assert(fs.existsSync(fullPath(filePath)), `${label || filePath} missing: ${filePath}`);
}

function assertDirectory(filePath) {
  assert(fs.existsSync(fullPath(filePath)) && fs.statSync(fullPath(filePath)).isDirectory(),
    `Required Project Starfall asset folder missing: ${filePath}`);
}

function assertArrayEquals(actual, expected, label) {
  assert(Array.isArray(actual), `${label} should be an array`);
  assert(Array.isArray(expected), `${label} expected value should be an array`);
  assert(actual.length === expected.length &&
    actual.every((value, index) => value === expected[index]),
    `${label} mismatch. Expected ${expected.join(', ')}, got ${actual.join(', ')}`);
}

function uniqueValues(values) {
  return Array.from(new Set(values));
}

function basenameWithoutExt(filePath) {
  return path.basename(String(filePath || ''), path.extname(String(filePath || '')));
}

async function assertImage(filePath, expected, label) {
  assertExists(filePath, label);
  const metadata = await sharp(fullPath(filePath)).metadata();
  assert(metadata.width === expected.width && metadata.height === expected.height,
    `${label || filePath} should be ${expected.width}x${expected.height}, got ${metadata.width}x${metadata.height}`);
  if (Object.prototype.hasOwnProperty.call(expected, 'alpha')) {
    assert(Boolean(metadata.hasAlpha) === Boolean(expected.alpha),
      `${label || filePath} alpha expectation mismatch`);
  }
  if (expected.format) {
    assert(String(metadata.format || '').toLowerCase() === String(expected.format).toLowerCase(),
      `${label || filePath} should be ${expected.format}, got ${metadata.format}`);
  }
  return metadata;
}

function cellVisibleAlpha(raw, sheetWidth, cellSize, row, column, threshold) {
  const x0 = column * cellSize;
  const y0 = row * cellSize;
  let visible = 0;
  for (let y = 0; y < cellSize; y += 1) {
    for (let x = 0; x < cellSize; x += 1) {
      if (raw[(((y0 + y) * sheetWidth) + x0 + x) * 4 + 3] > threshold) visible += 1;
    }
  }
  return visible;
}

function cellEdgeHasAlpha(raw, sheetWidth, cellSize, row, column, threshold) {
  const x0 = column * cellSize;
  const y0 = row * cellSize;
  for (let offset = 0; offset < cellSize; offset += 1) {
    if (raw[((y0 * sheetWidth) + x0 + offset) * 4 + 3] > threshold) return true;
    if (raw[(((y0 + cellSize - 1) * sheetWidth) + x0 + offset) * 4 + 3] > threshold) return true;
    if (raw[(((y0 + offset) * sheetWidth) + x0) * 4 + 3] > threshold) return true;
    if (raw[(((y0 + offset) * sheetWidth) + x0 + cellSize - 1) * 4 + 3] > threshold) return true;
  }
  return false;
}

function hashCellPixels(raw, sheetWidth, cellSize, row, column) {
  const hash = crypto.createHash('sha256');
  const x0 = column * cellSize;
  const y0 = row * cellSize;
  const rowBytes = cellSize * 4;
  for (let y = 0; y < cellSize; y += 1) {
    const offset = (((y0 + y) * sheetWidth) + x0) * 4;
    hash.update(raw.subarray(offset, offset + rowBytes));
  }
  return hash.digest('hex');
}

function isNearChromaKey(red, green, blue, target, tolerance) {
  const redDelta = red - target[0];
  const greenDelta = green - target[1];
  const blueDelta = blue - target[2];
  return redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta <= tolerance * tolerance;
}

function validateSkillFxPixels(id, raw, sheetWidth, contract, renderedHashes) {
  const frameSize = contract.frameSize;
  const rowCount = Array.isArray(contract.rows) ? contract.rows.length : Number(contract.rows || 0);
  const expectedFrameCount = Number(contract.expectedFrameCount || (contract.columns * rowCount));
  const quality = contract.quality || {};
  const alphaThreshold = Number(quality.alphaThreshold ?? 8);
  const minimumVisiblePixels = Number(quality.minimumVisiblePixelsPerCell ?? 1);
  const gutter = Number(quality.gutterPixels ?? 8);
  const chromaAlphaThreshold = Number(quality.chromaAlphaThreshold ?? 0);
  const chromaTolerance = Number(quality.chromaTolerance ?? 24);
  const chromaTargets = [
    { label: 'green', rgb: [0, 255, 0] },
    { label: 'magenta', rgb: [255, 0, 255] }
  ];

  assert(contract.columns * rowCount === expectedFrameCount,
    `Skill FX contract should describe ${expectedFrameCount} frames`);
  assert(expectedFrameCount === 24, `Skill FX sheets should contain exactly 24 frames, got ${expectedFrameCount}`);
  assert(gutter >= 8, `Skill FX quality gutter should be at least 8px, got ${gutter}`);

  let transparentResidue = null;
  let chromaResidue = null;
  for (let offset = 0; offset < raw.length; offset += 4) {
    const red = raw[offset];
    const green = raw[offset + 1];
    const blue = raw[offset + 2];
    const alpha = raw[offset + 3];
    if (!transparentResidue && alpha === 0 && (red !== 0 || green !== 0 || blue !== 0)) {
      const pixelIndex = offset / 4;
      transparentResidue = {
        x: pixelIndex % sheetWidth,
        y: Math.floor(pixelIndex / sheetWidth),
        rgba: `${red},${green},${blue},${alpha}`
      };
    }
    if (!chromaResidue && alpha > chromaAlphaThreshold) {
      const match = chromaTargets.find((target) =>
        isNearChromaKey(red, green, blue, target.rgb, chromaTolerance));
      if (match) {
        const pixelIndex = offset / 4;
        chromaResidue = {
          key: match.label,
          x: pixelIndex % sheetWidth,
          y: Math.floor(pixelIndex / sheetWidth),
          rgba: `${red},${green},${blue},${alpha}`
        };
      }
    }
    if (transparentResidue && chromaResidue) break;
  }
  assert(!transparentResidue,
    `${id} skill FX sheet has nonzero RGB hidden under transparent alpha at ` +
      `${transparentResidue && transparentResidue.x},${transparentResidue && transparentResidue.y}` +
      `${transparentResidue ? ` (${transparentResidue.rgba})` : ''}`);
  assert(!chromaResidue,
    `${id} skill FX sheet retains near-pure ${chromaResidue && chromaResidue.key} chroma-key pixels at ` +
      `${chromaResidue && chromaResidue.x},${chromaResidue && chromaResidue.y}` +
      `${chromaResidue ? ` (${chromaResidue.rgba})` : ''}`);

  const frameHashes = new Map();
  for (let row = 0; row < rowCount; row += 1) {
    for (let column = 0; column < contract.columns; column += 1) {
      const rowName = Array.isArray(contract.rows) ? contract.rows[row] : `row ${row + 1}`;
      const frameLabel = `${rowName} frame ${column + 1}`;
      const x0 = column * frameSize;
      const y0 = row * frameSize;
      let visible = 0;
      let gutterPixel = null;
      for (let y = 0; y < frameSize; y += 1) {
        for (let x = 0; x < frameSize; x += 1) {
          const alpha = raw[(((y0 + y) * sheetWidth) + x0 + x) * 4 + 3];
          if (alpha <= alphaThreshold) continue;
          visible += 1;
          if (!gutterPixel && (x < gutter || x >= frameSize - gutter || y < gutter || y >= frameSize - gutter)) {
            gutterPixel = { x, y, alpha };
          }
        }
      }
      assert(visible >= minimumVisiblePixels,
        `${id} skill FX ${frameLabel} should not be empty (found ${visible} visible pixels)`);
      assert(!gutterPixel,
        `${id} skill FX ${frameLabel} enters the required ${gutter}px gutter at ` +
          `${gutterPixel && gutterPixel.x},${gutterPixel && gutterPixel.y}`);

      const frameDigest = hashCellPixels(raw, sheetWidth, frameSize, row, column);
      assert(!frameHashes.has(frameDigest),
        `${id} skill FX ${frameLabel} exactly duplicates ${frameHashes.get(frameDigest) || 'another frame'}`);
      frameHashes.set(frameDigest, frameLabel);
    }
  }
  assert(frameHashes.size === expectedFrameCount,
    `${id} skill FX sheet should contain ${expectedFrameCount} distinct, nonempty frames`);

  const sheetDigest = crypto.createHash('sha256').update(raw).digest('hex');
  assert(!renderedHashes.has(sheetDigest),
    `${id} skill FX sheet duplicates ${renderedHashes.get(sheetDigest) || 'another skill'} at the rendered-pixel level`);
  renderedHashes.set(sheetDigest, id);
}

function itemIconFileName(itemId) {
  return `${String(itemId || '').replace(/_/g, '-')}.png`;
}

async function validatePlayers(manifest, data) {
  const contract = manifest.contracts.players;
  const fillPattern = (pattern, familyId) => String(pattern || '').replace('<family-id>', familyId);
  assertArrayEquals(data.PLAYER_ANIMATION_ROWS || [], contract.rows, 'Player animation rows');
  const expectedClassIds = contract.classes.slice().sort();
  const actualClassIds = Object.keys(data.CLASS_FILE_IDS || {}).sort();
  assertArrayEquals(actualClassIds, expectedClassIds, 'Player class IDs');
  assert(data.PLAYER_ART_VERSION === contract.artVersion,
    `Player art version should be ${contract.artVersion}`);
  assertArrayEquals((data.CLASS_FAMILY_IDS || []).slice().sort(), contract.families.slice().sort(), 'Player class families');
  assertArrayEquals(
    Object.keys(data.CLASS_BODY_FAMILIES || {}).sort(),
    Object.keys(contract.classFamilies || {}).sort(),
    'Player class-family mapping keys'
  );
  Object.entries(contract.classFamilies || {}).forEach(([classId, familyId]) => {
    assert(data.CLASS_BODY_FAMILIES[classId] === familyId,
      `${classId} data family should match the ${familyId} manifest family`);
  });
  assert(data.GENERIC_PLAYER_ASSET === contract.fallbackPortrait,
    `Generic player portrait should use ${contract.fallbackPortrait}`);
  assert(data.GENERIC_PLAYER_ANIMATION_ASSET && data.GENERIC_PLAYER_ANIMATION_ASSET.sheet === contract.fallbackAnimation,
    `Generic player animation should use ${contract.fallbackAnimation}`);

  await assertImage(data.GENERIC_PLAYER_ASSET, {
    width: contract.portraitWidth,
    height: contract.portraitHeight,
    alpha: true
  }, 'Generic player portrait');
  await assertImage(data.GENERIC_PLAYER_ANIMATION_ASSET.sheet, {
    width: contract.sheetWidth,
    height: contract.sheetHeight,
    alpha: true
  }, 'Generic player animation sheet');

  const expectedFamilyPortraits = new Set();
  const expectedFamilySheets = new Set();
  for (const familyId of contract.families) {
    const sourceFile = fillPattern(contract.sourcePattern, familyId);
    const generationSourceFile = fillPattern(contract.generationSourcePattern, familyId);
    const portraitPath = `${contract.portraitFolder}/${fillPattern(contract.portraitPattern, familyId)}`;
    const animationPath = `${contract.animationFolder}/${fillPattern(contract.animationPattern, familyId)}`;
    assertExists(`${contract.sourceFolder}/${generationSourceFile}`, `${familyId} player generation source`);
    assertExists(`${contract.sourceFolder}/${sourceFile}`, `${familyId} normalized player source sheet`);
    expectedFamilyPortraits.add(portraitPath);
    expectedFamilySheets.add(animationPath);
    await assertImage(portraitPath, {
      width: contract.portraitWidth,
      height: contract.portraitHeight,
      alpha: true
    }, `${familyId} family portrait`);
    await assertImage(animationPath, {
      width: contract.sheetWidth,
      height: contract.sheetHeight,
      alpha: true
    }, `${familyId} family animation sheet`);
  }
  assert(expectedFamilyPortraits.size === contract.families.length,
    'Each player family should have one cache-safe portrait');
  assert(expectedFamilySheets.size === contract.families.length,
    'Each player family should have one cache-safe animation sheet');

  for (const classId of Object.keys(data.CLASS_FILE_IDS || {})) {
    assert(contract.classes.includes(classId), `Manifest missing player class ${classId}`);
    const familyId = contract.classFamilies[classId];
    assert(contract.families.includes(familyId), `${classId} should map to a declared player family`);
    const expectedPortrait = `${contract.portraitFolder}/${fillPattern(contract.portraitPattern, familyId)}`;
    const expectedAnimation = `${contract.animationFolder}/${fillPattern(contract.animationPattern, familyId)}`;
    assert(data.CLASS_ASSETS[classId] === expectedPortrait,
      `${classId} portrait should resolve through the ${familyId} family`);
    const animation = data.PLAYER_ANIMATION_ASSETS[classId];
    assert(animation && animation.frameWidth === contract.frameSize && animation.frameHeight === contract.frameSize,
      `${classId} player animation should use ${contract.frameSize}px frames`);
    assert(animation.sheet === expectedAnimation,
      `${classId} animation should resolve through the ${familyId} family`);
    assert(animation === data.PLAYER_FAMILY_ANIMATION_ASSETS[familyId],
      `${classId} should reuse the ${familyId} animation object for cache-safe decoding`);
  }
}

async function validateEquipmentAtlases(manifest, data) {
  const contract = manifest.contracts.equipmentAtlases;
  const attachments = require(path.join(ROOT, 'js/games/project-starfall/engine/equipment-attachments.js'));
  const visuals = Object.values(data.EQUIPMENT_VISUALS || {});
  assert(visuals.length === contract.visualCount,
    `Project Starfall should expose ${contract.visualCount} equipment visuals, got ${visuals.length}`);

  const expectedFiles = new Set();
  for (const visual of visuals) {
    const atlas = visual && visual.atlas;
    const label = visual && visual.id || 'equipment visual';
    assert(visual && visual.renderMode === 'atlas', `${label} should use atlas rendering`);
    assert(atlas && String(atlas.sheet || '').startsWith(`${contract.folder}/`),
      `${label} should use the equipment-atlas folder`);
    assert(atlas.frameWidth === contract.cellSize && atlas.frameHeight === contract.cellSize,
      `${label} atlas cells should be ${contract.cellSize}x${contract.cellSize}`);
    assert(atlas.pivotX === contract.cellSize / 2 && atlas.pivotY === contract.cellSize / 2,
      `${label} atlas should use the centered ${contract.cellSize / 2}px pivot`);

    const kind = String(atlas.kind || visual.kind || '').toLowerCase();
    assert(kind && kind === String(visual.kind || '').toLowerCase(),
      `${label} atlas kind should match its visual kind`);
    const expectedAngles = attachments.getEquipmentAngleSet(kind);
    assert(expectedAngles.length === contract.angleCount,
      `${label} shared angle set should contain ${contract.angleCount} angles`);
    assertArrayEquals(atlas.angles || [], expectedAngles, `${label} atlas angles`);

    const expectedVariants = kind === 'bow' ? contract.bowVariants : contract.defaultVariants;
    assertArrayEquals(atlas.variants || [], expectedVariants, `${label} atlas variants`);
    const expectedWidth = contract.cellSize * contract.columns;
    const expectedHeight = contract.cellSize * expectedVariants.length;
    assert(expectedWidth === contract.sheetWidth, 'Equipment atlas manifest sheet width is internally inconsistent');
    assert(expectedHeight === (kind === 'bow' ? contract.bowSheetHeight : contract.defaultSheetHeight),
      `${label} equipment atlas manifest height is internally inconsistent`);

    const expectedFileName = String(contract.pattern || '<equipment-file-id>-atlas-v2.png')
      .replace('<equipment-file-id>', visual.fileId);
    const expectedPath = `${contract.folder}/${expectedFileName}`;
    assert(atlas.sheet === expectedPath, `${label} atlas should use ${expectedPath}`);
    expectedFiles.add(path.basename(expectedPath));
    await assertImage(atlas.sheet, {
      width: expectedWidth,
      height: expectedHeight,
      alpha: true
    }, `${label} equipment atlas`);

    const decoded = await sharp(fullPath(atlas.sheet)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    for (let row = 0; row < expectedVariants.length; row += 1) {
      for (let column = 0; column < contract.columns; column += 1) {
        const cellLabel = `${label} ${expectedVariants[row]} angle ${atlas.angles[column]}`;
        assert(cellVisibleAlpha(decoded.data, decoded.info.width, contract.cellSize, row, column, contract.alphaThreshold) >= contract.minimumVisiblePixelsPerCell,
          `${cellLabel} should not be empty`);
        if (contract.transparentCellEdges) {
          assert(!cellEdgeHasAlpha(decoded.data, decoded.info.width, contract.cellSize, row, column, contract.alphaThreshold),
            `${cellLabel} should not touch a cell edge`);
        }
      }
    }
  }

  assert(expectedFiles.size === contract.visualCount,
    `Project Starfall equipment visuals should map to ${contract.visualCount} unique atlas files`);
  const actualFiles = fs.readdirSync(fullPath(contract.folder))
    .filter((file) => /-atlas(?:-v\d+)?\.png$/i.test(file))
    .sort();
  const missingFiles = Array.from(expectedFiles).filter((file) => !actualFiles.includes(file));
  const unexpectedFiles = actualFiles.filter((file) => !expectedFiles.has(file));
  assert(!missingFiles.length && !unexpectedFiles.length && actualFiles.length === contract.visualCount,
    `Equipment atlas folder should contain exactly ${contract.visualCount} registered files` +
      `${missingFiles.length ? `; missing ${missingFiles.join(', ')}` : ''}` +
      `${unexpectedFiles.length ? `; unexpected ${unexpectedFiles.join(', ')}` : ''}`);
}

async function validateEnemies(manifest, data) {
  const contract = manifest.contracts.enemies;
  assertArrayEquals(data.ENEMY_ANIMATION_ROWS || [], contract.rows, 'Enemy animation rows');
  assert((data.ENEMIES || []).length > 0, 'Project Starfall enemies should be populated');

  for (const enemy of data.ENEMIES || []) {
    const fileId = basenameWithoutExt(enemy.asset);
    assertExists(`${contract.sourceFolder}/${fileId}-compact-source.png`, `${enemy.id} compact enemy source sheet`);
    await assertImage(enemy.asset, { width: 320, height: 320, alpha: true }, `${enemy.id} enemy portrait`);
    assert(enemy.animation && enemy.animation.frameWidth === contract.frameSize && enemy.animation.frameHeight === contract.frameSize,
      `${enemy.id} compact enemy animation should use ${contract.frameSize}px frames`);
    assert(toPosix(enemy.animation.sheet).endsWith(`/${fileId}-compact-sheet.png`),
      `${enemy.id} compact enemy animation filename should match portrait file ID`);
    await assertImage(enemy.animation.sheet, {
      width: contract.sheetWidth,
      height: contract.sheetHeight,
      alpha: true
    }, `${enemy.id} compact enemy animation sheet`);
  }
}

async function validateFx(manifest, data) {
  const globalContract = manifest.contracts.globalFx;
  for (const [id, animation] of Object.entries(data.FX_ANIMATION_ASSETS || {})) {
    await assertImage(animation.sheet, {
      width: globalContract.sheetWidth,
      height: globalContract.sheetHeight,
      alpha: true
    }, `${id} global FX sheet`);
  }
  assertExists(`${globalContract.sourceFolder}/ai-global-fx-sheet.png`, 'Global FX source sheet');

  const skillContract = manifest.contracts.skillFx;
  assertArrayEquals(data.SKILL_FX_ANIMATION_ROWS || [], skillContract.rows, 'Skill FX animation rows');
  const skillFxHashes = new Map();
  for (const [id, animation] of Object.entries(data.SKILL_FX_ANIMATION_ASSETS || {})) {
    await assertImage(animation.sheet, {
      width: skillContract.sheetWidth,
      height: skillContract.sheetHeight,
      alpha: true
    }, `${id} skill FX sheet`);
    assert(animation.frameWidth === skillContract.frameSize && animation.frameHeight === skillContract.frameSize,
      `${id} skill FX animation should use ${skillContract.frameSize}px frames`);
    const decoded = await sharp(fullPath(animation.sheet)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    validateSkillFxPixels(id, decoded.data, decoded.info.width, skillContract, skillFxHashes);
  }
  assert(skillFxHashes.size === Object.keys(data.SKILL_FX_ANIMATION_ASSETS || {}).length,
    'Every skill FX sheet should be unique at the rendered-pixel level');

  const basicContract = manifest.contracts.basicAttackFx;
  assertArrayEquals(data.BASIC_ATTACK_FX_ANIMATION_ROWS || [], basicContract.rows, 'Basic attack FX animation rows');
  for (const [id, animation] of Object.entries(data.BASIC_ATTACK_FX_ANIMATION_ASSETS || {})) {
    await assertImage(animation.sheet, {
      width: basicContract.sheetWidth,
      height: basicContract.sheetHeight,
      alpha: true
    }, `${id} basic attack FX sheet`);
  }

  const enemyFxContract = manifest.contracts.enemyCombatFx;
  assertArrayEquals(data.ENEMY_COMBAT_FX_ANIMATION_ROWS || [], enemyFxContract.rows, 'Enemy combat FX animation rows');
  const enemyFxHashes = new Map();
  for (const [id, animation] of Object.entries(data.ENEMY_COMBAT_FX_ANIMATION_ASSETS || {})) {
    await assertImage(animation.sheet, {
      width: enemyFxContract.sheetWidth,
      height: enemyFxContract.sheetHeight,
      alpha: true
    }, `${id} enemy combat FX sheet`);
    const pixels = await sharp(fullPath(animation.sheet)).ensureAlpha().raw().toBuffer();
    const digest = crypto.createHash('sha256').update(pixels).digest('hex');
    assert(!enemyFxHashes.has(digest),
      `${id} enemy combat FX sheet duplicates ${enemyFxHashes.get(digest) || 'another enemy'} at the rendered-pixel level`);
    enemyFxHashes.set(digest, id);
  }
  assert(enemyFxHashes.size === Object.keys(data.ENEMY_COMBAT_FX_ANIMATION_ASSETS || {}).length,
    'Every enemy combat FX sheet should have a unique file digest');

  const projectileContract = manifest.contracts.enemyProjectiles;
  for (const [id, animation] of Object.entries(data.ENEMY_PROJECTILE_ANIMATION_ASSETS || {})) {
    await assertImage(animation.sheet, {
      width: projectileContract.sheetWidth,
      height: projectileContract.sheetHeight,
      alpha: true
    }, `${id} enemy projectile sheet`);
  }

  const portalContract = manifest.contracts.portals;
  for (const variant of portalContract.variants) {
    const animation = data.PORTAL_ANIMATION_ASSETS && data.PORTAL_ANIMATION_ASSETS[variant];
    assert(animation, `Missing portal animation variant ${variant}`);
    await assertImage(animation.sheet, {
      width: portalContract.sheetWidth,
      height: portalContract.sheetHeight,
      alpha: true
    }, `${variant} portal animation sheet`);
  }
  assertExists(`${portalContract.sourceFolder}/ai-portals-sheet.png`, 'Portal source sheet');

  const petContract = manifest.contracts.pet;
  assertArrayEquals(data.PET_ANIMATION_ROWS || [], petContract.rows, 'Pet animation rows');
  await assertImage(data.PET_ANIMATION_ASSET.sheet, {
    width: petContract.sheetWidth,
    height: petContract.sheetHeight,
    alpha: true
  }, 'Pet animation sheet');
}

async function validateItemsAndCards(manifest, data) {
  const itemContract = manifest.contracts.items;
  for (const sheet of itemContract.sourceSheets || []) {
    assertExists(`${itemContract.sourceFolder}/${sheet.file}`, `${sheet.file} item source sheet`);
    assertExists(`${itemContract.sheetFolder}/${sheet.outputFile}`, `${sheet.outputFile} processed item sheet`);
  }
  for (const sheet of itemContract.externalSheets || []) {
    assertExists(`${itemContract.sheetFolder}/${sheet.file}`, `${sheet.file} external processed item sheet`);
    for (const id of sheet.ids || []) {
      assert(data.ITEM_ASSETS && data.ITEM_ASSETS[id], `External item sheet ID missing ITEM_ASSETS mapping: ${id}`);
    }
  }

  const itemAssets = data.ITEM_ASSETS || {};
  assert(Object.keys(itemAssets).length > 0, 'Project Starfall ITEM_ASSETS should be populated');
  for (const [itemId, assetPath] of Object.entries(itemAssets)) {
    assert(toPosix(assetPath).startsWith(`${itemContract.iconFolder}/`),
      `${itemId} should use standalone item icon folder`);
    assert(path.basename(assetPath) === itemIconFileName(itemId),
      `${itemId} item icon filename should be kebab-case`);
    await assertImage(assetPath, {
      width: itemContract.iconWidth,
      height: itemContract.iconHeight,
      alpha: true
    }, `${itemId} item icon`);
  }

  const cardContract = manifest.contracts.cards;
  assertExists(cardContract.source, 'Card icon source sheet');
  assert((data.CARD_DEFINITIONS || []).length > 0, 'Project Starfall CARD_DEFINITIONS should be populated');
  for (const card of data.CARD_DEFINITIONS || []) {
    const assetPath = data.CARD_ASSETS && data.CARD_ASSETS[card.id];
    assert(assetPath === `${cardContract.iconFolder}/${card.id}.png`,
      `${card.id} card asset path should match card icon folder`);
    await assertImage(assetPath, {
      width: cardContract.iconWidth,
      height: cardContract.iconHeight,
      alpha: true
    }, `${card.id} card icon`);
  }
}

async function validateSkills(manifest, data) {
  const contract = manifest.contracts.skills;
  for (const sourceSheet of contract.sourceSheets || []) {
    assertExists(`${contract.sourceFolder}/${sourceSheet}`, `${sourceSheet} skill source sheet`);
  }
  assert((data.SKILLS || []).length > 0, 'Project Starfall skills should be populated');
  for (const skill of data.SKILLS || []) {
    if (!skill.iconAsset) continue;
    await assertImage(skill.iconAsset, {
      width: contract.iconWidth,
      height: contract.iconHeight,
      alpha: true
    }, `${skill.id} skill icon`);
    assert(toPosix(skill.iconAsset).startsWith(`${contract.baseFolder}/`) ||
      toPosix(skill.iconAsset).startsWith(`${contract.advancedFolder}/`),
      `${skill.id} skill icon should live in a skill icon folder`);
  }
}

async function validateMapsAndEnvironment(manifest, data) {
  const mapContract = manifest.contracts.maps;
  for (const map of data.MAPS || []) {
    if (!map.asset) continue;
    await assertImage(map.asset, {
      width: map.backgroundMode === 'panorama' ? mapContract.width * 2 : mapContract.width,
      height: mapContract.height,
      format: mapContract.format
    }, `${map.id} map background`);
  }
  for (const assetPath of Object.values(data.CLASS_TRIAL_ASSETS || {})) {
    await assertImage(assetPath, {
      width: mapContract.width,
      height: mapContract.height,
      format: mapContract.format
    }, `${assetPath} class trial background`);
  }
  await assertImage(manifest.contracts.worldMap.path, {
    width: manifest.contracts.worldMap.width,
    height: manifest.contracts.worldMap.height,
    format: manifest.contracts.worldMap.format
  }, 'Project Starfall world map atlas');

  const environmentContract = manifest.contracts.environment;
  for (const [themeId, asset] of Object.entries(data.ENVIRONMENT_ASSETS && data.ENVIRONMENT_ASSETS.terrain || {})) {
    assert(asset.cellSize === environmentContract.terrain.cellSize &&
      asset.columns === environmentContract.terrain.columns &&
      asset.schema === environmentContract.terrain.schema,
      `${themeId} terrain metadata should match terrain contract`);
    await assertImage(asset.path, {
      width: environmentContract.terrain.width,
      height: environmentContract.terrain.height,
      alpha: true
    }, `${themeId} terrain atlas`);
  }
  for (const [themeId, asset] of Object.entries(data.ENVIRONMENT_ASSETS && data.ENVIRONMENT_ASSETS.props || {})) {
    assert(asset.cellSize === environmentContract.props.cellSize &&
      asset.columns === environmentContract.props.columns,
      `${themeId} prop metadata should match prop contract`);
    await assertImage(asset.path, {
      width: environmentContract.props.width,
      height: environmentContract.props.height,
      alpha: true
    }, `${themeId} prop atlas`);
  }
  for (const [themeId, asset] of Object.entries(data.ENVIRONMENT_ASSETS && data.ENVIRONMENT_ASSETS.ramps || {})) {
    assert(asset.cellSize === environmentContract.ramps.cellSize &&
      asset.columns === environmentContract.ramps.columns &&
      asset.schema === environmentContract.ramps.schema,
      `${themeId} ramp metadata should match ramp contract`);
    await assertImage(asset.path, {
      width: environmentContract.ramps.width,
      height: environmentContract.ramps.height,
      alpha: true
    }, `${themeId} ramp atlas`);
  }
  const townLandmarks = environmentContract.structures.townLandmarks;
  assertExists(townLandmarks.source, 'Town landmark source sheet');
  await assertImage(townLandmarks.path, {
    width: townLandmarks.width,
    height: townLandmarks.height,
    alpha: true
  }, 'Town landmark structure atlas');
}

async function validateUiAndStations(manifest, data) {
  const uiContract = manifest.contracts.ui;
  assertExists(uiContract.source, 'Menu icon source sheet');
  for (const screen of uiContract.screens || []) {
    await assertImage(screen.path, {
      width: screen.width,
      height: screen.height,
      alpha: screen.alpha
    }, screen.path);
  }

  const expectedMenuIcons = uiContract.menuIcons || {};
  assertArrayEquals(Object.keys(data.MENU_ICON_ASSETS || {}).sort(), Object.keys(expectedMenuIcons).sort(), 'Menu icon IDs');
  for (const [id, filename] of Object.entries(expectedMenuIcons)) {
    const assetPath = data.MENU_ICON_ASSETS[id];
    assert(assetPath === `${uiContract.menuIconFolder}/${filename}`,
      `${id} menu icon should use ${filename}`);
    await assertImage(assetPath, {
      width: uiContract.menuIconWidth,
      height: uiContract.menuIconHeight,
      alpha: true
    }, `${id} menu icon`);
  }

  const stationContract = manifest.contracts.stations;
  for (const id of stationContract.ids || []) {
    const assetPath = data.STATION_ASSETS && data.STATION_ASSETS[id];
    assert(assetPath === `${stationContract.folder}/${id}.png`, `${id} station should use station folder`);
    await assertImage(assetPath, {
      width: stationContract.width,
      height: stationContract.height,
      alpha: true
    }, `${id} station`);
  }
  assert(data.STATION_ASSETS && data.STATION_ASSETS.plinko === data.STATION_ASSETS.slots,
    'Plinko station should intentionally reuse slots station art');
}

async function validateManifest() {
  assertExists('ASSET_GENERATION_GUIDE.md', 'Asset generation guide');
  assertExists('img/project-starfall/asset-prompts.md', 'Asset prompt provenance');
  assert(fs.existsSync(MANIFEST_PATH), 'Project Starfall asset-generation manifest missing');
  assert(fs.existsSync(DATA_PATH), 'Project Starfall runtime data missing');

  const manifest = readJson(MANIFEST_PATH);
  const data = require(DATA_PATH);
  assert(manifest.version === 1, 'Project Starfall asset-generation manifest should be version 1');
  assert(manifest.guide === 'ASSET_GENERATION_GUIDE.md', 'Manifest should point to ASSET_GENERATION_GUIDE.md');
  assert(manifest.promptTemplates === 'asset-sources/project-starfall/prompts/README.md',
    'Manifest should point to source prompt templates');
  assertExists(manifest.promptTemplates, 'Prompt template README');

  for (const folder of manifest.requiredFolders || []) assertDirectory(folder);
  for (const script of manifest.buildScripts || []) assertExists(script, `${script} build script`);

  const requiredContractKeys = [
    'players',
    'equipmentAtlases',
    'enemies',
    'enemyProjectiles',
    'globalFx',
    'skillFx',
    'basicAttackFx',
    'enemyCombatFx',
    'portals',
    'pet',
    'items',
    'cards',
    'skills',
    'maps',
    'worldMap',
    'environment',
    'ui',
    'stations'
  ];
  const contractKeys = Object.keys(manifest.contracts || {});
  requiredContractKeys.forEach((key) => {
    assert(contractKeys.includes(key), `Manifest missing ${key} contract`);
  });

  await validatePlayers(manifest, data);
  await validateEquipmentAtlases(manifest, data);
  await validateEnemies(manifest, data);
  await validateFx(manifest, data);
  await validateItemsAndCards(manifest, data);
  await validateSkills(manifest, data);
  await validateMapsAndEnvironment(manifest, data);
  await validateUiAndStations(manifest, data);

  const runtimeAssetRoots = uniqueValues([
    ...Object.values(data.CLASS_ASSETS || {}),
    ...Object.values(data.ENEMY_ASSETS || {}),
    ...Object.values(data.ITEM_ASSETS || {}),
    ...Object.values(data.CARD_ASSETS || {}),
    ...Object.values(data.MAP_ASSETS || {}),
    ...Object.values(data.MENU_ICON_ASSETS || {}),
    ...Object.values(data.STATION_ASSETS || {})
  ].map((assetPath) => toPosix(assetPath).split('/').slice(0, 2).join('/')));
  assert(runtimeAssetRoots.every((root) => root === 'img/project-starfall'),
    `All Starfall runtime assets should stay under img/project-starfall, got ${runtimeAssetRoots.join(', ')}`);

  console.log(`Validated Project Starfall asset generation manifest: ${manifest.requiredFolders.length} folders, ${requiredContractKeys.length} contracts, ${Object.keys(data.ITEM_ASSETS || {}).length} item icons, ${(data.ENEMIES || []).length} enemies.`);
}

validateManifest().catch((error) => {
  console.error(error && error.stack || error);
  process.exitCode = 1;
});
