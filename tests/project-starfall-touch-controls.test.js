'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const input = require(path.join(ROOT, 'js/games/project-starfall/ui/input.js'));
require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
const { ProjectStarfallUi } = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'));

function createClassList() {
  const values = new Set();
  return {
    toggle(name, active) {
      if (active) values.add(name);
      else values.delete(name);
    },
    contains(name) {
      return values.has(name);
    }
  };
}

function createTarget(actionId) {
  const attributes = new Map([
    ['data-starfall-touch-action', actionId],
    ['aria-pressed', 'false']
  ]);
  return {
    actionId,
    disabled: false,
    classList: createClassList(),
    captured: [],
    released: [],
    closest(selector) {
      return String(selector || '').includes('[data-starfall-touch-action]') ? this : null;
    },
    getAttribute(name) {
      return attributes.get(name) || '';
    },
    setAttribute(name, value) {
      attributes.set(name, String(value));
    },
    setPointerCapture(pointerId) {
      this.captured.push(pointerId);
    },
    releasePointerCapture(pointerId) {
      this.released.push(pointerId);
    }
  };
}

function createPointerEvent(target, pointerId, options) {
  const settings = options || {};
  return {
    target,
    pointerId,
    pointerType: settings.pointerType || 'touch',
    button: settings.button || 0,
    prevented: false,
    preventDefault() {
      this.prevented = true;
    }
  };
}

const leftTarget = createTarget('moveLeft');
const root = { contains: (target) => target === leftTarget };
const metadata = input.getTouchControlPointerAction(createPointerEvent(leftTarget, 7), { root });
assert.strictEqual(metadata.handled, true, 'a touch control inside the game root should resolve through the touch pointer contract');
assert.strictEqual(metadata.pointerKey, 'touch:7', 'each pointer should receive a stable virtual key identity');
assert.strictEqual(metadata.shouldSetPointerCapture, true, 'touch controls should capture their owning pointer through release');
assert.strictEqual(input.getTouchControlPointerAction(createPointerEvent(leftTarget, 8, { pointerType: 'mouse', button: 2 }), { root }).handled, false,
  'secondary mouse buttons should not masquerade as touch controls');

const engineEvents = [];
const actions = {
  moveLeft: { id: 'moveLeft', label: 'Move Left', type: 'hold', input: 'left', onboardingEvent: 'move' },
  moveRight: { id: 'moveRight', label: 'Move Right', type: 'hold', input: 'right', onboardingEvent: 'move' },
  jump: { id: 'jump', label: 'Jump', type: 'hold', input: 'jump', onboardingEvent: 'jump' },
  attack: { id: 'attack', label: 'Attack', type: 'action', action: 'attack', onboardingEvent: 'attack' },
  'skill:alpha': { id: 'skill:alpha', label: 'Alpha Strike', type: 'skill', skillId: 'alpha' },
  'skill:beta': { id: 'skill:beta', label: 'Beta Field', type: 'skill', skillId: 'beta' },
  'skill:gamma': { id: 'skill:gamma', label: 'Gamma Rush', type: 'skill', skillId: 'gamma' },
  'skill:delta': { id: 'skill:delta', label: 'Delta Guard', type: 'skill', skillId: 'delta' }
};
const targets = new Set();
const ui = Object.create(ProjectStarfallUi.prototype);
Object.assign(ui, {
  root: { contains: (target) => targets.has(target) },
  engine: {
    setInput(name, value) {
      engineEvents.push(['input', name, value]);
    },
    basicAttack() {
      engineEvents.push(['attack']);
      return true;
    },
    setHeldSkill(skillId, value) {
      engineEvents.push(['heldSkill', skillId, value]);
    },
    useSkill(skillId) {
      engineEvents.push(['skill', skillId]);
      return true;
    },
    clearHeldSkills() {
      engineEvents.push(['clearHeldSkills']);
    }
  },
  heldAttackKeys: new Set(),
  heldSkillKeys: new Map(),
  touchControlPointers: new Map(),
  touchControlActionCounts: new Map(),
  isStartScreenOpen: false,
  isCharacterSelectOpen: false,
  getBindableAction: (actionId) => actions[actionId] || null,
  recordControlOnboardingEvent: (action) => engineEvents.push(['onboarding', action.id]),
  focusCanvas: () => engineEvents.push(['focus']),
  startPortalTransition: () => false,
  commitAdminRatePreviewFromEvent: () => false,
  stopPlinkoDropHold: () => false,
  stopPotentialPromptDomDrag: () => false
});

const firstLeft = createTarget('moveLeft');
const secondLeft = createTarget('moveLeft');
const attackTarget = createTarget('attack');
const skillTarget = createTarget('skill:alpha');
[firstLeft, secondLeft, attackTarget, skillTarget].forEach((target) => targets.add(target));

const firstLeftDown = createPointerEvent(firstLeft, 1);
const secondLeftDown = createPointerEvent(secondLeft, 2);
const attackDown = createPointerEvent(attackTarget, 3);
assert.strictEqual(ui.handleTouchControlPointerDown(firstLeftDown), true, 'left movement should start through the bindable hold action');
assert.strictEqual(ui.handleTouchControlPointerDown(secondLeftDown), true, 'a second finger may own the same held direction');
assert.strictEqual(ui.handleTouchControlPointerDown(attackDown), true, 'attack should run concurrently with movement');
assert.strictEqual(engineEvents.filter((entry) => entry[0] === 'input' && entry[1] === 'left' && entry[2] === true).length, 1,
  'same-action multitouch should not duplicate the engine hold transition');
assert.strictEqual(engineEvents.filter((entry) => entry[0] === 'attack').length, 1,
  'attack touch should reuse the existing one-shot basic-attack path');
assert.strictEqual(firstLeftDown.prevented && secondLeftDown.prevented && attackDown.prevented, true,
  'touch controls should suppress browser gestures only after claiming a game control');

ui.handlePointerCancel(createPointerEvent(firstLeft, 1));
assert.strictEqual(engineEvents.some((entry) => entry[0] === 'input' && entry[1] === 'left' && entry[2] === false), false,
  'cancelling one of two movement pointers should preserve the remaining hold');
assert.strictEqual(ui.hasActiveTouchControlAction('attack'), true, 'movement cancellation should not release a simultaneous attack pointer');
ui.handlePointerCancel(createPointerEvent(secondLeft, 2));
assert.strictEqual(engineEvents.filter((entry) => entry[0] === 'input' && entry[1] === 'left' && entry[2] === false).length, 1,
  'the last movement pointer should release its engine hold on pointercancel');
ui.handlePointerUp(createPointerEvent(attackTarget, 3));
assert.strictEqual(engineEvents.some((entry) => entry[0] === 'input' && entry[1] === 'attack' && entry[2] === false), true,
  'the attack pointer should release the shared attack input on pointerup');

ui.handleTouchControlPointerDown(createPointerEvent(skillTarget, 4));
ui.handlePointerCancel(createPointerEvent(skillTarget, 4));
assert.deepStrictEqual(engineEvents.filter((entry) => entry[0] === 'skill'), [['skill', 'alpha']],
  'skill touch should use the existing skill activation path exactly once');
assert(engineEvents.some((entry) => entry[0] === 'heldSkill' && entry[1] === 'alpha' && entry[2] === true),
  'skill touch should enter the existing held-skill path');
assert(engineEvents.some((entry) => entry[0] === 'heldSkill' && entry[1] === 'alpha' && entry[2] === false),
  'pointercancel should cleanly release the held skill');

const outsideTarget = { closest: () => null };
const outsideEvent = createPointerEvent(outsideTarget, 9);
assert.strictEqual(ui.handleTouchControlPointerDown(outsideEvent), false, 'touches outside the game controls should remain unclaimed');
assert.strictEqual(outsideEvent.prevented, false, 'unclaimed page touches should remain available for scrolling');

const controlsElement = {
  hidden: true,
  innerHTML: '',
  attributes: new Map(),
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  },
  querySelectorAll() {
    return [];
  }
};
ui.elements = { touchControls: controlsElement };
ui.snapshot = {
  state: { player: { classId: 'vanguard', advancedClassId: '' } },
  activeCooldowns: []
};
ui.getHudRenderSnapshot = () => ui.snapshot;
ui.getSkillBindActions = () => [actions['skill:alpha'], actions['skill:beta'], actions['skill:gamma'], actions['skill:delta']];
ui.touchControlsRenderKey = '';
assert.strictEqual(ui.renderTouchControls(ui.snapshot), true, 'an active character should render the persistent touch control deck');
assert.strictEqual((controlsElement.innerHTML.match(/data-starfall-touch-action=/g) || []).length, 8,
  'the deck should expose left, jump, right, attack, and four skill actions');
assert(controlsElement.innerHTML.includes('Skill 4: Delta Guard'), 'the fourth learned skill should occupy the fourth touch slot');

const hudCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/hud.css'), 'utf8');
const responsiveCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/responsive.css'), 'utf8');
const page = fs.readFileSync(path.join(ROOT, 'pages/games/project-starfall.html'), 'utf8');
const uiCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');
assert(page.includes('data-starfall-touch-controls hidden'), 'the page should provide a persistent control mount outside the frequently replaced HUD');
assert(hudCss.includes('touch-action: pan-y') && hudCss.includes('.project-starfall-touch-control') && hudCss.includes('touch-action: none'),
  'only actual control buttons should suppress touch gestures while the surrounding deck preserves vertical scrolling');
assert(responsiveCss.includes('(pointer: coarse)') && responsiveCss.includes('min-height: 52px'),
  'controls should appear for coarse pointers and narrow screens with accessible targets larger than 44px');
assert(!uiCode.includes('Touch movement controls are not available yet'), 'the obsolete keyboard-required mobile warning should be removed');

console.log('Project Starfall touch controls tests passed.');
