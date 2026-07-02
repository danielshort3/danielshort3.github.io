(function initProjectStarfallCoreMath(global) {
  'use strict';

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function colorWithAlpha(color, alpha) {
    const value = String(color || '#ffffff');
    if (!/^#[0-9a-f]{6}$/i.test(value)) return value;
    const channel = clamp(Math.round((Number(alpha) || 0) * 255), 0, 255);
    return `${value}${channel.toString(16).padStart(2, '0')}`;
  }

  function hashString(value) {
    const text = String(value || '');
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function seededUnit(seed, salt) {
    let value = hashString(`${seed}:${salt}`);
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return ((value >>> 0) % 10000) / 10000;
  }

  function seededPick(items, seed, salt) {
    const options = (items || []).filter(Boolean);
    if (!options.length) return '';
    return options[Math.floor(seededUnit(seed, salt) * options.length) % options.length];
  }

  function rectsOverlap(a, b) {
    return !!(a && b &&
      a.x < b.x + b.w &&
      a.x + a.w > b.x &&
      a.y < b.y + b.h &&
      a.y + a.h > b.y);
  }

  function rectOverlapsBox(rect, x, y, w, h) {
    return !!rect && x < rect.x + rect.w && x + w > rect.x && y < rect.y + rect.h && y + h > rect.y;
  }

  function positiveModulo(value, divisor) {
    const base = Math.max(1, Number(divisor || 1));
    return ((Number(value || 0) % base) + base) % base;
  }

  function randItem(items) {
    return items[Math.floor(Math.random() * items.length)];
  }

  function weightedItem(items) {
    const options = (items || []).filter(Boolean);
    if (!options.length) return null;
    const total = options.reduce((sum, item) => sum + Math.max(1, Number(item.weight || 1)), 0);
    let roll = Math.random() * total;
    for (const item of options) {
      roll -= Math.max(1, Number(item.weight || 1));
      if (roll <= 0) return item;
    }
    return options[options.length - 1];
  }

  function randomInt(min, max) {
    const low = Math.ceil(Number(min) || 0);
    const high = Math.floor(Number(max) || low);
    if (high <= low) return low;
    return low + Math.floor(Math.random() * (high - low + 1));
  }

  function rollNormalStatVariance() {
    const u1 = Math.max(Number.MIN_VALUE, Math.random());
    const u2 = Math.random();
    return clamp(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2), -2.35, 2.35);
  }

  const api = {
    clamp,
    colorWithAlpha,
    hashString,
    seededUnit,
    seededPick,
    rectsOverlap,
    rectOverlapsBox,
    positiveModulo,
    randItem,
    weightedItem,
    randomInt,
    rollNormalStatVariance
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
