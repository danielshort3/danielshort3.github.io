#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const Masks = require('../js/games/project-starfall/data/enemy-hurtboxes.js');
const Visuals = require('../js/games/project-starfall/engine/visuals.js');
const Hurtboxes = require('../js/games/project-starfall/engine/enemy-hurtboxes.js');
const Feedback = require('../js/games/project-starfall/engine/combat-feedback.js');
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'output/starfall-hitbox-review');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT, file))).digest('hex').slice(0, 12);
const url = file => `${path.relative(OUT, path.join(ROOT, file)).replace(/\\/g, '/')}?v=${hash(file)}`;

function terrainBody(enemy) {
  return enemy.behavior === 'boss' ? enemy.id === 'stormbreakRoc' ? { w: 124, h: 96 } : enemy.id === 'astralArchivist' ? { w: 92, h: 112 } : { w: 110, h: 124 } : enemy.id === 'crackedMimic' ? { w: 64, h: 58 } : enemy.behavior === 'flyer' ? { w: 42, h: 42 } : { w: 46, h: 46 };
}

const enemies = Data.ENEMIES.map(enemy => {
  const animation = enemy.animation;
  const mask = Masks.sheets[animation.sheet];
  if (!mask || !mask.sha256.startsWith(hash(animation.sheet))) throw new Error(`${enemy.id}: mask does not match current sprite pixels`);
  const actions = Object.fromEntries(Object.entries(animation.states).map(([id, state]) => {
    const sequence = state.sequence || Array.from({ length: state.frames }, (_, i) => i);
    const holds = sequence.map(frame => 1000 * Math.max(1, state.holds?.[frame] || 1) / Math.max(1, state.fps));
    if (state.loop && state.loopDelay) holds[holds.length - 1] += state.loopDelay * 1000;
    return [id, { sequence, holds, loop: state.loop }];
  }));
  const body = terrainBody(enemy);
  const viewBounds = { left: 0, top: 0, right: body.w, bottom: body.h };
  // A fixed camera union across every pose/facing/recoil prevents preview-only
  // camera jumps and clipping without changing the actor's runtime geometry.
  for (const definition of Object.values(animation.states)) for (let column = 0; column < definition.frames; column++) for (const facing of [-1, 1]) for (const recoil of [false, true]) {
    let box = Visuals.createEnemySpriteRenderBox({ x: 0, y: 0, ...body, id: enemy.id, data: enemy });
    if (recoil) box = Feedback.applyEnemyHitReactionToBox(box, Feedback.getEnemyHitReactionState(Feedback.createEnemyHitReaction({ startedAtMs: 1000, critical: true, direction: facing }), 1000));
    const bounds = Hurtboxes.getBounds(Hurtboxes.createEnemyHurtbox(animation, { row: definition.row, frameIndex: column, frameWidth: animation.frameWidth, frameHeight: animation.frameHeight }, box, facing));
    viewBounds.left = Math.min(viewBounds.left, bounds.x); viewBounds.top = Math.min(viewBounds.top, bounds.y);
    viewBounds.right = Math.max(viewBounds.right, bounds.x + bounds.w); viewBounds.bottom = Math.max(viewBounds.bottom, bounds.y + bounds.h);
  }
  return { id: enemy.id, name: enemy.name, behavior: enemy.behavior, visibility: enemy.guide.visibility, body, viewBounds, image: url(animation.sheet), animation, actions };
});
const sourceModules = ['engine/visuals.js', 'engine/combat-feedback.js', 'data/enemy-hurtboxes.js', 'engine/enemy-hurtboxes.js'];
const data = { enemies, alphaThreshold: Masks.alphaThreshold, generatedAt: new Date().toISOString() };
fs.mkdirSync(OUT, { recursive: true });
const template = fs.readFileSync(path.join(ROOT, 'build/templates/starfall-hitbox-review.template.html'), 'utf8');
const scripts = sourceModules.map(file => `<script src="${url(`js/games/project-starfall/${file}`)}"></script>`).join('\n');
fs.writeFileSync(path.join(OUT, 'review-data.json'), JSON.stringify(data, null, 2) + '\n');
fs.writeFileSync(path.join(OUT, 'index.html'), template.replace('__MODULES__', scripts).replace('__REVIEW_DATA__', JSON.stringify(data).replace(/</g, '\\u003c')));
console.log(`Generated enemy collision review: ${enemies.length} identities, current production pixels and shared runtime collision helper.`);
