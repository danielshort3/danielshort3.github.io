#!/usr/bin/env node
'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const Data = require('../js/games/project-starfall/project-starfall-data.js');

const FRAME_SIZE = 160;
const COLUMNS = 6;
const ROWS = 4;
const ALPHA_THRESHOLD = 8;
const MIN_VISIBLE_PIXELS = 48;
const MIN_MARGIN = 8;

function digest(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function extractFrame(raw, sheetWidth, row, column) {
  const frame = Buffer.alloc(FRAME_SIZE * FRAME_SIZE * 4);
  const sourceX = column * FRAME_SIZE;
  const sourceY = row * FRAME_SIZE;
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const sourceOffset = (((sourceY + y) * sheetWidth) + sourceX) * 4;
    raw.copy(frame, y * FRAME_SIZE * 4, sourceOffset, sourceOffset + FRAME_SIZE * 4);
  }
  return frame;
}

function isVisibleKeyColor(r, g, b) {
  const green = g >= 220 && r <= 95 && b <= 95 && g - Math.max(r, b) >= 125;
  const magenta = r >= 205 && b >= 205 && g <= 80 && Math.min(r, b) - g >= 140;
  return green || magenta;
}

function inspectFrame(frame, label) {
  let visible = 0;
  let hiddenColor = 0;
  let keyColor = 0;
  let minX = FRAME_SIZE;
  let minY = FRAME_SIZE;
  let maxX = -1;
  let maxY = -1;

  for (let pixel = 0; pixel < FRAME_SIZE * FRAME_SIZE; pixel += 1) {
    const offset = pixel * 4;
    const r = frame[offset];
    const g = frame[offset + 1];
    const b = frame[offset + 2];
    const alpha = frame[offset + 3];
    if (!alpha && (r || g || b)) hiddenColor += 1;
    if (alpha > ALPHA_THRESHOLD && isVisibleKeyColor(r, g, b)) keyColor += 1;
    if (alpha <= ALPHA_THRESHOLD) continue;
    const x = pixel % FRAME_SIZE;
    const y = Math.floor(pixel / FRAME_SIZE);
    visible += 1;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  assert(visible >= MIN_VISIBLE_PIXELS, `${label} should contain visible effect art`);
  assert.strictEqual(hiddenColor, 0, `${label} should zero RGB beneath fully transparent pixels`);
  assert.strictEqual(keyColor, 0, `${label} should not retain visible green or magenta key pixels`);
  assert(minX >= MIN_MARGIN && minY >= MIN_MARGIN && maxX <= FRAME_SIZE - MIN_MARGIN - 1 && maxY <= FRAME_SIZE - MIN_MARGIN - 1,
    `${label} should preserve a ${MIN_MARGIN}px safety gutter; bounds were ${minX},${minY}-${maxX},${maxY}`);

  return Object.freeze({
    visible,
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1
  });
}

async function main() {
  const entries = Object.entries(Data.SKILL_FX_ANIMATION_ASSETS || {});
  const activeSkills = (Data.SKILLS || []).filter((skill) => skill && skill.type !== 'Passive' && Data.SKILL_FX_ANIMATION_ASSETS[skill.id]);
  assert.strictEqual(entries.length, 70, 'Project Starfall should expose 70 active-skill FX sheets');
  assert.strictEqual(activeSkills.length, entries.length, 'Every active mapped skill should have exactly one FX sheet');

  const sheetDigests = new Map();
  const frameDigests = new Map();
  let totalBytes = 0;
  let totalFrames = 0;

  for (const [skillId, animation] of entries) {
    const filePath = path.join(ROOT, animation.sheet);
    assert(fs.existsSync(filePath), `${skillId} runtime sheet should exist`);
    const stats = fs.statSync(filePath);
    totalBytes += stats.size;

    const decoded = await sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    assert.strictEqual(decoded.info.width, FRAME_SIZE * COLUMNS, `${skillId} sheet width`);
    assert.strictEqual(decoded.info.height, FRAME_SIZE * ROWS, `${skillId} sheet height`);
    assert.strictEqual(decoded.info.channels, 4, `${skillId} should decode as RGBA`);

    const sheetHash = digest(decoded.data);
    assert(!sheetDigests.has(sheetHash), `${skillId} duplicates ${sheetDigests.get(sheetHash)} at rendered-pixel level`);
    sheetDigests.set(sheetHash, skillId);

    const localFrameDigests = new Set();
    for (let row = 0; row < ROWS; row += 1) {
      for (let column = 0; column < COLUMNS; column += 1) {
        const label = `${skillId} ${Data.SKILL_FX_ANIMATION_ROWS[row]} frame ${column + 1}`;
        const frame = extractFrame(decoded.data, decoded.info.width, row, column);
        inspectFrame(frame, label);
        const frameHash = digest(frame);
        assert(!localFrameDigests.has(frameHash), `${label} duplicates another frame in the same sheet`);
        assert(!frameDigests.has(frameHash), `${label} duplicates ${frameDigests.get(frameHash)} across skills`);
        localFrameDigests.add(frameHash);
        frameDigests.set(frameHash, label);
        totalFrames += 1;
      }
    }

    const cast = animation.states && animation.states.cast;
    const impact = animation.states && animation.states.impact;
    assert(cast && cast.loop === false, `${skillId} cast should be a one-shot animation`);
    assert(impact && impact.loop === false, `${skillId} impact should be a one-shot animation`);
  }

  const backupPaths = Data.ASSET_BACKUP_PATHS || {};
  entries.forEach(([skillId, animation]) => {
    assert(!backupPaths[animation.sheet], `${skillId} should not advertise the excluded legacy skill backup sheet`);
  });

  assert.strictEqual(sheetDigests.size, entries.length, 'Every active skill sheet should have unique rendered pixels');
  assert.strictEqual(frameDigests.size, entries.length * ROWS * COLUMNS, 'Every skill frame should be unique across the catalog');
  console.log(`Project Starfall skill FX quality passed: ${entries.length} sheets, ${totalFrames} unique frames, ${(totalBytes / 1024 / 1024).toFixed(2)} MiB.`);
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exitCode = 1;
});
