(function initProjectStarfallEnemyHurtboxes(global) {
  'use strict';

  const Data = (typeof require === 'function' ? require('../data/enemy-hurtboxes.js') : null) || global.ProjectStarfallEnemyHurtboxesData;
  const Visuals = (typeof require === 'function' ? require('./visuals.js') : null) || global.ProjectStarfallEngineModules && global.ProjectStarfallEngineModules.visuals;
  const decodedFrames = new Map();

  function decodeFrame(encoded) {
    const bytes = typeof Buffer === 'function'
      ? Buffer.from(encoded, 'base64')
      : Uint8Array.from(global.atob(encoded), character => character.charCodeAt(0));
    let nibbleIndex = 4;
    const nibble = () => {
      const value = bytes[nibbleIndex >> 1];
      return (nibbleIndex++ & 1) ? value & 15 : value >> 4;
    };
    const byte = () => nibble() * 16 + nibble();
    const rectangles = [];
    const startY = bytes[0];
    const endY = startY + bytes[1];
    let previous = [];
    let previousRectangles = new Map();
    let minX = Infinity;
    let maxX = -Infinity;
    let pixelCount = 0;
    let sumX = 0;
    let sumY = 0;
    for (let y = startY; y < endY;) {
      let count = nibble();
      const repeat = nibble() + 1;
      if (count === 15) count = byte();
      const row = [];
      for (let index = 0; index < count * 2; index += 1) {
        const value = nibble();
        row.push(value === 15 ? byte() : (previous[index] || 0) + ((value & 1) ? -(value + 1) / 2 : value / 2));
      }
      const currentRectangles = new Map();
      for (let index = 0; index < row.length; index += 2) {
        const x = row[index];
        const width = row[index + 1] - x;
        const key = `${x}:${width}`;
        let rect = previousRectangles.get(key);
        if (rect && rect.y + rect.h === y) rect.h += repeat;
        else {
          rect = { x, y, w: width, h: repeat };
          rectangles.push(rect);
        }
        currentRectangles.set(key, rect);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x + width);
        const area = width * repeat;
        pixelCount += area;
        sumX += (x + width / 2) * area;
        sumY += (y + repeat / 2) * area;
      }
      previousRectangles = currentRectangles;
      previous = row;
      y += repeat;
    }
    if (!pixelCount) return null;
    const centroidX = sumX / pixelCount;
    const centroidY = sumY / pixelCount;
    let aim = null;
    let nearest = Infinity;
    rectangles.forEach(rect => {
      // Choose the center of a solid source pixel, including for hollow actors.
      const x = Math.max(rect.x + 0.5, Math.min(rect.x + rect.w - 0.5, Math.floor(centroidX) + 0.5));
      const y = Math.max(rect.y + 0.5, Math.min(rect.y + rect.h - 0.5, Math.floor(centroidY) + 0.5));
      const distance = (x - centroidX) ** 2 + (y - centroidY) ** 2;
      if (distance < nearest) {
        nearest = distance;
        aim = { x, y };
      }
    });
    return { rectangles, bounds: { x: minX, y: startY, w: maxX - minX, h: endY - startY }, aim, pixelCount };
  }

  function createEnemyHurtbox(animation, frame, renderBox, facing) {
    if (!Data || !Visuals || !animation || !frame || !renderBox) return null;
    const sheet = Data.sheets[animation.sheet];
    if (!sheet || Number(frame.frameWidth || animation.frameWidth) !== sheet.frameWidth || Number(frame.frameHeight || animation.frameHeight) !== sheet.frameHeight) return null;
    const row = Number(frame.row || 0);
    const column = Number(frame.frameIndex || 0);
    if (!Number.isInteger(row) || !Number.isInteger(column) || row < 0 || row >= sheet.rows || column < 0 || column >= sheet.columns) return null;
    const index = row * sheet.columns + column;
    const key = `${sheet.maskIndex}:${index}`;
    if (!decodedFrames.has(key)) decodedFrames.set(key, decodeFrame(Data.masks[sheet.maskIndex][index]));
    const mask = decodedFrames.get(key);
    if (!mask) return null;
    const draw = Visuals.createAnimationFrameDrawState(frame, renderBox.x, renderBox.y, renderBox.w, renderBox.h, facing, { registration: animation.registration });
    if (!draw) return null;
    const scaleX = draw.drawWidth / sheet.frameWidth * draw.scaleX;
    const scaleY = draw.drawHeight / sheet.frameHeight * draw.scaleY;
    const originX = draw.translateX + draw.drawX * draw.scaleX;
    const originY = draw.translateY + draw.drawY * draw.scaleY;
    const bounds = {
      x: Math.min(originX + mask.bounds.x * scaleX, originX + (mask.bounds.x + mask.bounds.w) * scaleX),
      y: Math.min(originY + mask.bounds.y * scaleY, originY + (mask.bounds.y + mask.bounds.h) * scaleY),
      w: mask.bounds.w * Math.abs(scaleX),
      h: mask.bounds.h * Math.abs(scaleY)
    };
    return { bounds, mask, originX, originY, scaleX, scaleY, sheet: animation.sheet, row, frameIndex: column };
  }

  function getBounds(hurtbox) {
    return hurtbox ? hurtbox.bounds : null;
  }

  function getAimPoint(hurtbox) {
    return hurtbox ? {
      x: hurtbox.originX + hurtbox.mask.aim.x * hurtbox.scaleX,
      y: hurtbox.originY + hurtbox.mask.aim.y * hurtbox.scaleY
    } : null;
  }

  function forEachRect(hurtbox, callback) {
    if (!hurtbox || typeof callback !== 'function') return;
    hurtbox.mask.rectangles.forEach(rect => {
      callback({
        x: Math.min(hurtbox.originX + rect.x * hurtbox.scaleX, hurtbox.originX + (rect.x + rect.w) * hurtbox.scaleX),
        y: Math.min(hurtbox.originY + rect.y * hurtbox.scaleY, hurtbox.originY + (rect.y + rect.h) * hurtbox.scaleY),
        w: rect.w * Math.abs(hurtbox.scaleX),
        h: rect.h * Math.abs(hurtbox.scaleY)
      });
    });
  }

  function overlaps(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }

  function intersectsRect(hurtbox, rect) {
    if (!hurtbox || !rect || !(rect.w > 0) || !(rect.h > 0) || !overlaps(hurtbox.bounds, rect)) return false;
    const x1 = (rect.x - hurtbox.originX) / hurtbox.scaleX;
    const x2 = (rect.x + rect.w - hurtbox.originX) / hurtbox.scaleX;
    const y1 = (rect.y - hurtbox.originY) / hurtbox.scaleY;
    const y2 = (rect.y + rect.h - hurtbox.originY) / hurtbox.scaleY;
    const source = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
    return hurtbox.mask.rectangles.some(candidate => overlaps(candidate, source));
  }

  function intersectsCircle(hurtbox, x, y, radius) {
    if (!hurtbox || !Number.isFinite(x) || !Number.isFinite(y) || !(radius >= 0)) return false;
    const bounds = hurtbox.bounds;
    const nearestX = Math.max(bounds.x, Math.min(x, bounds.x + bounds.w));
    const nearestY = Math.max(bounds.y, Math.min(y, bounds.y + bounds.h));
    const radiusSquared = radius * radius;
    if ((nearestX - x) ** 2 + (nearestY - y) ** 2 > radiusSquared) return false;
    return hurtbox.mask.rectangles.some(rect => {
      const x1 = hurtbox.originX + rect.x * hurtbox.scaleX;
      const x2 = hurtbox.originX + (rect.x + rect.w) * hurtbox.scaleX;
      const y1 = hurtbox.originY + rect.y * hurtbox.scaleY;
      const y2 = hurtbox.originY + (rect.y + rect.h) * hurtbox.scaleY;
      const closestX = Math.max(Math.min(x1, x2), Math.min(x, Math.max(x1, x2)));
      const closestY = Math.max(Math.min(y1, y2), Math.min(y, Math.max(y1, y2)));
      return (closestX - x) ** 2 + (closestY - y) ** 2 <= radiusSquared;
    });
  }

  const api = { createEnemyHurtbox, intersectsRect, intersectsCircle, getBounds, getAimPoint, forEachRect };
  const modules = global.ProjectStarfallEngineModules || {};
  modules.enemyHurtboxes = Object.assign({}, modules.enemyHurtboxes || {}, api);
  global.ProjectStarfallEngineModules = modules;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
