'use strict';

// Read-only audit of shipped sprite pixels and the renderer's actual transform.
// Run from any directory: node output/starfall-hitbox-review/audit-enemy-alpha.cjs
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const ROOT = path.resolve(__dirname, '../..');
const Data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
const Visuals = require(path.join(ROOT, 'js/games/project-starfall/engine/visuals.js'));
const Feedback = require(path.join(ROOT, 'js/games/project-starfall/engine/combat-feedback.js'));
const THRESHOLD = Number(process.env.STARFALL_ALPHA_THRESHOLD || 64);
const OUT = __dirname;
const round = value => Math.round(value * 100) / 100;
const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

function bodyFor(enemy) {
  return enemy.behavior === 'boss' ? enemy.id === 'stormbreakRoc' ? { w: 124, h: 96 } : enemy.id === 'astralArchivist' ? { w: 92, h: 112 } : { w: 110, h: 124 } : enemy.id === 'crackedMimic' ? { w: 64, h: 58 } : enemy.behavior === 'flyer' ? { w: 42, h: 42 } : { w: 46, h: 46 };
}

function transformFor(enemy, state, column, facing, reaction = false) {
  const actor = { id: enemy.id, data: enemy, ...bodyFor(enemy), x: 0, y: 0, facing };
  let box = Visuals.createEnemySpriteRenderBox(actor);
  if (reaction) box = Feedback.applyEnemyHitReactionToBox(box, Feedback.getEnemyHitReactionState(Feedback.createEnemyHitReaction({ startedAtMs: 1000, direction: facing, critical: true }), 1000));
  const animation = enemy.animation;
  const frame = { frameIndex: column, row: animation.states[state].row, frameWidth: animation.frameWidth, frameHeight: animation.frameHeight };
  const draw = Visuals.createAnimationFrameDrawState(frame, box.x, box.y, box.w, box.h, facing, { registration: animation.registration });
  return { actor, draw, box, scale: draw.drawWidth / frame.frameWidth };
}

// Exact separable squared Euclidean distance to opaque pixel centers.
function edt1d(f) {
  const n = f.length, v = new Int32Array(n), z = new Float64Array(n + 1), d = new Float64Array(n);
  let k = 0;
  z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s;
    do {
      const p = v[k];
      s = ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p);
      if (s <= z[k]) k--;
      else break;
    } while (k >= 0);
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) ** 2 + f[v[k]];
  }
  return d;
}

function inspectPixels(data, width, height) {
  const alpha = new Uint8Array(width * height), opaque = [];
  let left = width, right = 0, top = height, bottom = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (data[(y * width + x) * 4 + 3] >= THRESHOLD) {
    alpha[y * width + x] = 1; opaque.push([x, y]);
    left = Math.min(left, x); right = Math.max(right, x + 1); top = Math.min(top, y); bottom = Math.max(bottom, y + 1);
  }
  const temp = new Float64Array(width * height), distance = new Float64Array(width * height);
  for (let y = 0; y < height; y++) temp.set(edt1d(Array.from({ length: width }, (_, x) => alpha[y * width + x] ? 0 : 1e9)), y * width);
  for (let x = 0; x < width; x++) {
    const column = edt1d(Array.from({ length: height }, (_, y) => temp[y * width + x]));
    for (let y = 0; y < height; y++) distance[y * width + x] = Math.sqrt(column[y]);
  }
  return { alpha, opaque, bounds: { left, right, top, bottom }, distance, width, height };
}

function worldPoint(draw, scale, x, y) {
  return { x: draw.translateX + draw.scaleX * (draw.drawX + x * scale), y: draw.translateY + draw.drawY + y * scale };
}

function measure(pixels, transform) {
  const { actor, draw, scale } = transform;
  let samples = 0, empty = 0, worstEstimate = -1, worstPoint = null;
  for (let y = 0.5; y < actor.h; y++) for (let x = 0.5; x < actor.w; x++) {
    const sx = ((x - draw.translateX) * draw.scaleX - draw.drawX) / scale;
    const sy = (y - draw.translateY - draw.drawY) / scale;
    const ix = Math.floor(sx), iy = Math.floor(sy);
    const inside = ix >= 0 && iy >= 0 && ix < pixels.width && iy < pixels.height;
    const occupied = inside && pixels.alpha[iy * pixels.width + ix];
    samples++; if (!occupied) empty++;
    const clampedX = Math.min(pixels.width - 1, Math.max(0, ix)), clampedY = Math.min(pixels.height - 1, Math.max(0, iy));
    const estimate = pixels.distance[clampedY * pixels.width + clampedX] * scale + (!inside ? Math.hypot(clampedX - ix, clampedY - iy) * scale : 0);
    if (estimate > worstEstimate) { worstEstimate = estimate; worstPoint = { x, y, sx, sy }; }
  }
  // Refine the selected candidate against opaque pixel rectangles, not centers.
  // This is a proven lower bound on the largest empty radius in the body rectangle.
  let exactDistance = Infinity;
  for (const [x, y] of pixels.opaque) {
    const dx = Math.max(x - worstPoint.sx, 0, worstPoint.sx - x - 1), dy = Math.max(y - worstPoint.sy, 0, worstPoint.sy - y - 1);
    exactDistance = Math.min(exactDistance, Math.hypot(dx, dy) * scale);
  }
  const a = worldPoint(draw, scale, pixels.bounds.left, pixels.bounds.top), b = worldPoint(draw, scale, pixels.bounds.right, pixels.bounds.bottom);
  const bounds = { x: Math.min(a.x, b.x), y: a.y, w: Math.abs(b.x - a.x), h: b.y - a.y };
  const margins = { left: Math.max(0, bounds.x), right: Math.max(0, actor.w - bounds.x - bounds.w), top: Math.max(0, bounds.y), bottom: Math.max(0, actor.h - bounds.y - bounds.h) };
  let visibleOutsideBody = 0;
  for (const [x, y] of pixels.opaque) {
    const p = worldPoint(draw, scale, x + 0.5, y + 0.5);
    if (p.x < 0 || p.y < 0 || p.x >= actor.w || p.y >= actor.h) visibleOutsideBody++;
  }
  return {
    oldBody: { x: 0, y: 0, w: actor.w, h: actor.h },
    visibleBounds: Object.fromEntries(Object.entries(bounds).map(([key, value]) => [key, round(value)])),
    whollyEmptyExternalMarginsPx: Object.fromEntries(Object.entries(margins).map(([key, value]) => [key, round(value)])),
    emptyAreaPercent: round(empty / samples * 100),
    maxEmptyRadiusLowerBoundPx: round(exactDistance),
    emptyWitness: { x: worstPoint.x, y: worstPoint.y },
    visiblePixelCentersOutsideOldBodyPercent: round(visibleOutsideBody / pixels.opaque.length * 100),
    spriteScale: round(scale)
  };
}

async function overlay(entry, width = 400, height = 320) {
  const enemy = Data.ENEMIES.find(value => value.id === entry.enemyId);
  const transform = transformFor(enemy, entry.action, entry.column, entry.facing, entry.reaction);
  const source = await sharp(path.join(ROOT, enemy.animation.sheet)).extract({ left: entry.column * enemy.animation.frameWidth, top: enemy.animation.states[entry.action].row * enemy.animation.frameHeight, width: enemy.animation.frameWidth, height: enemy.animation.frameHeight }).png().toBuffer();
  const { draw, scale, actor } = transform;
  const zoom = Math.min(2, (width - 30) / draw.drawWidth, (height - 65) / draw.drawHeight);
  const ox = width / 2 - actor.w / 2 * zoom, oy = height - 45 - actor.h * zoom;
  const imageX = draw.translateX + (draw.scaleX < 0 ? -draw.drawX - draw.drawWidth : draw.drawX), imageY = draw.translateY + draw.drawY;
  const image = entry.facing < 0 ? await sharp(source).flop().png().toBuffer() : source;
  const witness = entry.emptyWitness;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="#172027"/><text x="14" y="22" fill="#fff" font-family="Arial" font-size="16">${escape(enemy.name)} · ${entry.action} ${entry.column + 1}${entry.reaction ? ' · recoil' : ''}</text><image href="data:image/png;base64,${image.toString('base64')}" x="${ox + imageX * zoom}" y="${oy + imageY * zoom}" width="${draw.drawWidth * zoom}" height="${draw.drawHeight * zoom}"/><rect x="${ox}" y="${oy}" width="${actor.w * zoom}" height="${actor.h * zoom}" fill="#f06a6026" stroke="#f06a60" stroke-width="2"/><circle cx="${ox + witness.x * zoom}" cy="${oy + witness.y * zoom}" r="${entry.maxEmptyRadiusLowerBoundPx * zoom}" fill="none" stroke="#ffd477" stroke-width="1.5"/><circle cx="${ox + witness.x * zoom}" cy="${oy + witness.y * zoom}" r="2" fill="#ffd477"/><text x="14" y="${height - 14}" fill="#f9d188" font-family="Arial" font-size="14">${entry.maxEmptyRadiusLowerBoundPx}px empty reach · ${entry.emptyAreaPercent}% rectangle transparent</text></svg>`;
  return sharp(Buffer.from(svg)).png().toBuffer();
}

async function main() {
  const frames = [], sheets = new Map();
  for (const enemy of Data.ENEMIES) {
    if (!enemy.animation?.sheet) continue;
    const animation = enemy.animation;
    if (!sheets.has(animation.sheet)) sheets.set(animation.sheet, await sharp(path.join(ROOT, animation.sheet)).ensureAlpha().raw().toBuffer({ resolveWithObject: true }));
    const sheet = sheets.get(animation.sheet);
    for (const [action, definition] of Object.entries(animation.states)) {
      const columns = [...new Set(definition.sequence || Array.from({ length: definition.frames }, (_, i) => i))];
      for (const column of columns) {
        const rgba = Buffer.alloc(animation.frameWidth * animation.frameHeight * 4);
        for (let y = 0; y < animation.frameHeight; y++) {
          const start = ((definition.row * animation.frameHeight + y) * sheet.info.width + column * animation.frameWidth) * 4;
          sheet.data.copy(rgba, y * animation.frameWidth * 4, start, start + animation.frameWidth * 4);
        }
        const pixels = inspectPixels(rgba, animation.frameWidth, animation.frameHeight);
        for (const facing of [-1, 1]) for (const reaction of (action === 'hit' ? [false, true] : [false])) {
          frames.push({ enemyId: enemy.id, name: enemy.name, guideVisibility: enemy.guide.visibility, behavior: enemy.behavior, action, column, facing, reaction, damageableAction: action !== 'defeat', ...measure(pixels, transformFor(enemy, action, column, facing, reaction)) });
        }
      }
    }
  }
  const active = frames.filter(value => value.damageableAction && !value.reaction), byEnemy = Data.ENEMIES.map(enemy => {
    const rows = active.filter(value => value.enemyId === enemy.id), worst = rows.reduce((a, b) => a.maxEmptyRadiusLowerBoundPx > b.maxEmptyRadiusLowerBoundPx ? a : b);
    const idle = rows.filter(value => value.action === 'idle').reduce((a, b) => a.maxEmptyRadiusLowerBoundPx > b.maxEmptyRadiusLowerBoundPx ? a : b);
    return { enemyId: enemy.id, name: enemy.name, visibility: enemy.guide.visibility, behavior: enemy.behavior, body: bodyFor(enemy), worst, idle };
  }).sort((a, b) => b.worst.maxEmptyRadiusLowerBoundPx - a.worst.maxEmptyRadiusLowerBoundPx);
  const report = { generatedAt: new Date().toISOString(), alphaThreshold: THRESHOLD, enemyIds: byEnemy.length, uniqueSheets: sheets.size, frameFacingReactionCases: frames.length, activeNeutralCases: active.length, method: 'Current runtime visual transform; all registered frames both facings. Old rectangle sampled at world-pixel centers. Worst candidate selected using exact Euclidean distance transform then refined against alpha pixel rectangles. Reported empty radius is a proven lower bound on the maximum, not a continuous optimizer. Defeat frames are evidence only; neutral and peak critical-hit reaction separated. No production artwork or runtime files edited.', recommendation: 'Keep platform/physics body. Derive combat hurt geometry from each current sprite frame alpha as row spans or exact occupancy, using the exact renderer registration, scale, facing, and recoil box; query narrow phase after broad bounds. Do not replace the generic body with one inflated AABB.', byEnemy, frames };
  fs.writeFileSync(path.join(OUT, 'enemy-alpha-audit.json'), JSON.stringify(report, null, 2) + '\n');
  const examples = ['lavaTick', 'clockworkTitan', 'brambleking', 'rimewarden', 'stormbreakRoc', 'astralArchivist', 'emberWisp', 'voidMote', 'eclipseSovereign', 'dewSlime', 'banditCutter', 'briarStag'].map(id => byEnemy.find(value => value.enemyId === id)).filter(Boolean).map(value => value.worst);
  const images = await Promise.all(examples.map((entry, i) => overlay(entry).then(input => ({ input, left: (i % 4) * 400, top: Math.floor(i / 4) * 320 }))));
  await sharp({ create: { width: 1600, height: Math.ceil(images.length / 4) * 320, channels: 3, background: '#172027' } }).composite(images).png().toFile(path.join(OUT, 'old-body-overlays.png'));
  const lines = ['# Enemy sprite / combat-body pixel audit', '', `Audited ${byEnemy.length} enemy IDs / ${sheets.size} unique sheets; ${frames.length} frame/facing/reaction cases. Alpha threshold ${THRESHOLD}/255.`, '', 'Distances are at actual game scale, not the 160px atlas scale. Red overlay is the current generic body. Gold circles show a proven empty-space witness; pixels are original shipped sprites.', '', '| Enemy | Visibility | Worst active pose | Empty radius at least | Idle at least | Transparent old body |', '| --- | --- | --- | ---: | ---: | ---: |', ...byEnemy.map(value => `| ${value.name} | ${value.visibility} | ${value.worst.action} ${value.worst.column + 1} | ${value.worst.maxEmptyRadiusLowerBoundPx}px | ${value.idle.maxEmptyRadiusLowerBoundPx}px | ${value.worst.emptyAreaPercent}% |`), '', report.method, '', report.recommendation];
  fs.writeFileSync(path.join(OUT, 'enemy-alpha-audit.md'), lines.join('\n') + '\n');
  console.log(JSON.stringify({ enemyIds: byEnemy.length, uniqueSheets: sheets.size, cases: frames.length, worst: byEnemy.slice(0, 8).map(value => ({ id: value.enemyId, action: value.worst.action, emptyRadius: value.worst.maxEmptyRadiusLowerBoundPx })), lavaTick: byEnemy.find(value => value.enemyId === 'lavaTick') }, null, 2));
}

if (require.main === module) main().catch(error => { console.error(error); process.exit(1); });
module.exports = { bodyFor, transformFor, inspectPixels, measure, worldPoint, overlay };
