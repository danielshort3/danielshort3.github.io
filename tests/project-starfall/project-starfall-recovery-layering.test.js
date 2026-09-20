'use strict';

const assert = require('assert');
const Data = require('../../js/games/project-starfall/project-starfall-data');
const Visuals = require('../../js/games/project-starfall/engine/visuals');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine');
const { createRenderer } = require('../../js/games/project-starfall/project-starfall-renderer-pixi');

// Exercise the real Canvas frame order with an opaque-actor marker between paint calls.
const engine = createProjectStarfallEngine(null, Data);
const canvasCalls = [];
let start, end, orb;
engine.ctx = {
  clearRect() {}, save() {}, restore() {}, beginPath() {}, rect() {}, clip() {}, scale() {}, translate() {},
  fillRect(x, y, w, h) { assert.strictEqual(y, engine.getPlayfieldHeight() + engine.getViewportMetrics().solidPlatformHeight, 'HUD backing must not cover gameplay'); assert.strictEqual(y + h, engine.canvas.height); },
  moveTo(x, y) { start = [x, y]; },
  lineTo(x, y) { end = [x, y]; },
  stroke() { canvasCalls.push({ type: 'line', points: [...start, ...end], width: this.lineWidth }); },
  arc(x, y, radius) { orb = [x, y, radius]; },
  fill() { canvasCalls.push({ type: 'orb', points: orb }); }
};
engine.canvas = { width: 1280, height: 806 };
engine.state.player.classId = 'fighter';
engine.shouldUseRendererBackend = () => false;
engine.profilePerformancePhase = (section, name, draw) => draw();
engine.getCombatFeedbackCameraOffset = () => ({ x: 0, y: 0 });
engine.getVisibleProjectilesForDraw = () => [];
engine.getVisibleEnemiesForDraw = () => [{}];
engine.getQuestGuidanceSnapshot = () => ({});
engine.currentMapLootDrops = () => [];
for (const name of ['drawBackground', 'drawWorldBaseBand', 'drawMap', 'drawPet', 'drawPartyMembers', 'drawQuestNavigationArrow']) engine[name] = () => {};
engine.drawEnemy = () => canvasCalls.push({ type: 'actor' });
engine.drawPlayer = () => canvasCalls.push({ type: 'actor' });

// Use the actual Pixi scene graph and effect method without requiring WebGL.
class Container {
  constructor() { this.children = []; }
  addChild(...children) { this.children.push(...children); }
}
const renderer = createRenderer({ PIXI: { Container, Graphics: Container }, data: Data });
renderer.app = { stage: new Container() };
renderer.setupScene();
const layers = renderer.worldLayer.children;
assert(layers.indexOf(renderer.vfxSprites) < layers.indexOf(renderer.entitySprites), 'Recovery ground contours remain behind sprites.');
assert(layers.indexOf(renderer.damageSprites) > layers.indexOf(renderer.entitySprites), 'Recovery symbols remain above sprite actors.');
assert(layers.indexOf(renderer.damageSprites) > layers.indexOf(renderer.entityGraphics), 'Recovery symbols remain above procedural actor fallbacks.');
const pixiCalls = [];
renderer.drawLine = (pool, x1, y1, x2, y2, width) => pixiCalls.push({ pool, type: 'line', points: [x1, y1, x2, y2], width });
renderer.drawShape = (pool, shape, x, y, w) => pixiCalls.push({ pool, type: 'orb', points: [x, y, w / 2] });

for (const ownership of ['player', 'enemy']) {
  for (const phase of ['prepare', 'release']) {
    for (const recoveryKind of ['heal', 'resource']) {
      for (const ttl of [0.5, 0.25, 0.01]) {
        const effect = { type: 'recoveryPulse', phase, ownership, recoveryKind, x: 250, y: 430, r: 32, ttl, duration: 0.5 };
        const state = Visuals.createSemanticRecoveryDrawState(effect);
        const ground = state.lines.filter((line) => line.layer === 'ground');
        const symbols = state.lines.filter((line) => line.layer === 'symbol');
        assert.strictEqual(ground.length, ownership === 'enemy' ? 36 : 48, 'One ground contour preserves the ownership shape.');
        assert(state.alpha > 0, 'The semantic cue stays visible throughout its active lifetime.');
        assert(state.orbs.every((entry) => entry.layer === 'symbol'));
        if (phase === 'release' && recoveryKind === 'heal') {
          assert.strictEqual(symbols.length, 6, 'Three complete healing plus marks remain visible.');
          assert.strictEqual(state.orbs.length, 0, 'Healing does not borrow the resource droplet silhouette.');
        } else {
          assert.strictEqual(state.orbs.length, 5, 'Gathering and resource droplets survive foreground routing.');
        }
        engine.prepareVisualDrawLists = () => ({ worldEffects: [effect], damageSplats: [], levelUpBursts: [] });
        canvasCalls.length = 0;
        engine.draw();
        const firstActor = canvasCalls.findIndex((call) => call.type === 'actor');
        const lastActor = canvasCalls.map((call) => call.type).lastIndexOf('actor');
        assert.strictEqual(firstActor, ground.length, 'Canvas paints only the single ground ring before actors.');
        assert.strictEqual(canvasCalls.length - lastActor - 1, symbols.length + state.orbs.length, 'Canvas paints every semantic mark after all actors.');
        const canvasGround = canvasCalls.slice(0, firstActor);
        const canvasSymbols = canvasCalls.slice(lastActor + 1);
        for (const visualQuality of [{ level: 'normal' }, { level: 'reduced', reduceEffects: true, combatPressure: true }]) {
          pixiCalls.length = 0;
          renderer.renderWorldEffects({ worldEffects: [engine.getWorldEffectRenderSnapshot(effect)], visualQuality });
          const normalize = ({ pool, ...call }) => call;
          assert.deepStrictEqual(pixiCalls.filter((call) => call.pool === 'vfx').map(normalize), canvasGround, 'Canvas and Pixi share identical ground geometry without duplicate rings.');
          assert.deepStrictEqual(pixiCalls.filter((call) => call.pool === 'damage').map(normalize), canvasSymbols, 'Both renderers preserve identical foreground symbols, including reduced effects.');
        }
      }
    }
  }
}

const delayed = { type: 'recoveryPulse', recoveryKind: 'heal', activationDelay: 0.01, ttl: 0.5, duration: 0.5 };
canvasCalls.length = 0;
engine.drawEffect(engine.ctx, delayed, 'ground');
engine.drawEffect(engine.ctx, delayed, 'symbol');
assert.strictEqual(canvasCalls.length, 0, 'Separating the layers cannot reveal a recovery cue before activation.');
assert.strictEqual(engine.getWorldEffectRenderSnapshot(delayed), null, 'Pixi also receives no cue before activation.');
console.log('Project Starfall recovery ground/symbol layering and Canvas/Pixi parity passed.');
