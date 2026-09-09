'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../js/tools/ocean-wave-camera.js'), 'utf8');

class Element {
  constructor(tagName = 'div') {
    this.tagName = tagName;
    this.dataset = {};
    this.hidden = false;
    this.clientHeight = 600;
    this.attributes = new Map();
    this.listeners = new Map();
    this.captures = new Set();
    const classes = new Set();
    this.classList = {
      add: name => classes.add(name), remove: name => classes.delete(name), contains: name => classes.has(name),
      toggle: (name, force) => { if (force) classes.add(name); else classes.delete(name); },
    };
  }
  addEventListener(name, callback) {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name).add(callback);
  }
  removeEventListener(name, callback) { this.listeners.get(name)?.delete(callback); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  getAttribute(name) { return this.attributes.get(name); }
  closest(selector) { return selector.split(',').some(part => part.trim() === this.tagName) ? this : null; }
  querySelectorAll() { return this.children || []; }
  setPointerCapture(id) { this.captures.add(id); }
  hasPointerCapture(id) { return this.captures.has(id); }
  releasePointerCapture(id) { this.captures.delete(id); }
  fire(type, data = {}) {
    const event = { type, target: this, preventDefault() { this.defaultPrevented = true; }, ...data };
    for (const callback of this.listeners.get(type) || []) callback(event);
    return event;
  }
}

const harness = (pose = {}, options = {}) => {
  const document = new Element('document');
  document.hidden = false;
  const window = new Element('window');
  const stage = new Element();
  stage.focus = () => { document.activeElement = stage; };
  const toggleButton = new Element('button');
  const relaxButton = new Element('button');
  const resetButton = new Element('button');
  const pad = new Element();
  pad.children = ['forward', 'back', 'left', 'right', 'up', 'down'].map(action => {
    const button = new Element('button');
    button.dataset.oceanMove = action;
    return button;
  });
  const hint = new Element();
  const camera = { x: 0, z: 0, height: 3.5, yaw: 0, pitch: -.1, minPitch: -1.2, maxPitch: .5, ...pose };
  const queue = new Map();
  let sequence = 0;
  let timestamp = 0;
  let changes = 0;
  let commits = 0;
  let resets = 0;
  window.requestAnimationFrame = callback => { const id = ++sequence; queue.set(id, callback); return id; };
  window.cancelAnimationFrame = id => queue.delete(id);
  vm.runInNewContext(source, { window, document, Math, Map, Number, Object, Error });
  const controller = window.OceanWaveCamera.create({
    stage, camera, toggleButton, relaxButton, resetButton, pad, hint,
    onChange: () => { changes++; }, onCommit: () => { commits++; }, onReset: () => { resets++; },
    ...options,
  });
  return {
    window, document, stage, camera, toggleButton, relaxButton, resetButton, pad, hint, controller, queue,
    get changes() { return changes; }, get commits() { return commits; }, get resets() { return resets; },
    key: (code, target = stage) => stage.fire('keydown', { code, key: code.startsWith('Key') ? code.slice(3).toLowerCase() : code, target }),
    release: (code, target = stage) => window.fire('keyup', { code, target }),
    step: (frames = 60) => {
      for (let i = 0; i < frames; i++) {
        timestamp += 1000 / 60;
        const callbacks = [...queue.values()];
        queue.clear();
        callbacks.forEach(callback => callback(timestamp));
      }
    },
    pointer: (type, point = {}) => stage.fire(type, {
      pointerId: 1, pointerType: 'mouse', button: 0, clientX: 100, clientY: 100, ...point,
    }),
  };
};

{
  const app = harness();
  assert.equal(app.controller.isEnabled(), false, 'Exploration should require an explicit action.');
  assert.equal(app.pad.hidden, true);
  assert.equal(app.hint.hidden, true);
  assert.equal(app.resetButton.hidden, true);
  assert.equal(app.relaxButton.getAttribute('aria-pressed'), 'true');
  assert.equal(app.queue.size, 0, 'A stationary camera must not add an animation loop.');
  assert.equal(app.key('KeyW').defaultPrevented, undefined, 'W must remain available to the page outside Explore mode.');
  app.step();
  assert.equal(app.camera.z, 0);
  app.toggleButton.fire('click');
  assert.equal(app.controller.isEnabled(), true);
  assert.equal(app.pad.hidden, false);
  assert.equal(app.resetButton.hidden, false);
  assert.equal(app.relaxButton.getAttribute('aria-pressed'), 'false');
  assert.equal(app.toggleButton.getAttribute('aria-pressed'), 'true');
  assert.equal(app.stage.classList.contains('is-exploring'), true);
  assert.equal(app.key('KeyW').defaultPrevented, true);
  app.step(1);
  assert.ok(app.camera.z > 0 && app.camera.z < .04, 'Movement should ease in instead of jumping to full speed.');
  app.step(59);
  assert.ok(app.camera.z > 1.5 && app.camera.z < 2.5, 'Holding forward must move slowly through the ocean.');
  app.release('KeyW', new Element('input'));
  app.step(180);
  const stoppedAt = app.camera.z;
  app.step(60);
  assert.equal(app.camera.z, stoppedAt, 'Releasing a movement key over another control must stop travel.');
  assert.equal(app.queue.size, 0, 'The demand animation loop must end after easing finishes.');
  assert.ok(app.changes > 1 && app.commits > 0);
  app.controller.dispose();
}

{
  const straight = harness({ yaw: Math.PI / 2 });
  straight.controller.setEnabled(true);
  straight.key('KeyW');
  straight.step();
  assert.ok(straight.camera.x > 1.5 && Math.abs(straight.camera.z) < .001, 'Forward must follow the direction the camera faces.');
  const diagonal = harness();
  diagonal.controller.setEnabled(true);
  diagonal.key('KeyW');
  diagonal.key('KeyD');
  diagonal.step();
  assert.ok(Math.abs(Math.hypot(diagonal.camera.x, diagonal.camera.z) - straight.camera.x) < .01,
    'Diagonal movement must not be faster than straight movement.');
  straight.controller.dispose();
  diagonal.controller.dispose();
}

for (const stop of ['blur', 'hidden', 'focusout', 'offscreen', 'Escape', 'disabled', 'dispose']) {
  const app = harness();
  app.controller.setEnabled(true);
  app.key('KeyW');
  app.step(20);
  if (stop === 'blur') app.window.fire('blur');
  if (stop === 'hidden') { app.document.hidden = true; app.document.fire('visibilitychange'); }
  if (stop === 'focusout') app.stage.fire('focusout', { relatedTarget: new Element('input') });
  if (stop === 'offscreen') app.stage.fire('ocean:visibility', { detail: { visible: false } });
  if (stop === 'Escape') app.key('Escape');
  if (stop === 'disabled') app.controller.setEnabled(false);
  if (stop === 'dispose') app.controller.dispose();
  const pose = JSON.stringify(app.camera);
  app.step(120);
  assert.equal(JSON.stringify(app.camera), pose, `${stop} must immediately stop all camera movement.`);
  assert.equal(app.queue.size, 0, `${stop} must release the camera animation loop.`);
  assert.equal(app.controller.isMoving(), false, `${stop} must clear held input.`);
  app.controller.dispose();
}

{
  const app = harness();
  app.controller.setEnabled(true);
  for (const tag of ['input', 'select', 'textarea', 'button', 'a']) {
    assert.equal(app.key('KeyW', new Element(tag)).defaultPrevented, undefined, `Camera keys must not hijack ${tag} controls.`);
  }
  assert.equal(app.stage.fire('keydown', { code: 'KeyW', key: 'w', ctrlKey: true }).defaultPrevented, undefined);
  app.step();
  assert.equal(app.camera.z, 0);
  app.key('ArrowUp');
  app.step(120);
  app.release('ArrowUp');
  app.step(120);
  assert.ok(app.camera.pitch > .45 && app.camera.pitch <= .5, 'Arrow up must look upward while respecting pitch bounds.');
  app.key('KeyQ');
  app.step(600);
  app.release('KeyQ');
  app.step(120);
  assert.ok(app.camera.height >= 1.8 && app.camera.height < 1.81, 'Lowering the camera must keep it above the sea.');
  app.key('Home');
  assert.equal(app.camera.height, 3.5);
  assert.equal(app.camera.pitch, -.1);
  assert.equal(app.camera.x, 0);
  assert.equal(app.camera.z, 0);
  assert.equal(app.resets, 1);
  app.controller.dispose();
}

{
  const app = harness();
  const restingPose = JSON.stringify(app.camera);
  app.pointer('pointerdown');
  app.pointer('pointermove', { clientX: 250, clientY: 120 });
  app.pointer('pointerup');
  app.step(120);
  app.key('ArrowUp');
  app.step(120);
  assert.equal(JSON.stringify(app.camera), restingPose, 'Relax mode must keep the camera steady during accidental drags and arrow keys.');
  assert.equal(app.stage.captures.size, 0);
  assert.equal(app.queue.size, 0, 'Relax mode must not start a camera animation loop.');
  app.controller.setEnabled(true);
  app.pointer('pointerdown');
  app.pointer('pointermove', { clientX: 250, clientY: 120 });
  app.pointer('pointerup');
  app.step(120);
  assert.ok(app.camera.yaw > .4 && app.camera.yaw < .5, 'Explore must allow dragging to look around.');
  app.relaxButton.fire('click');
  assert.equal(app.controller.isEnabled(), false);
  assert.equal(app.resetButton.hidden, true);
  assert.equal(app.relaxButton.getAttribute('aria-pressed'), 'true');
  app.controller.reset();
  app.pointer('pointerdown', { pointerType: 'touch' });
  const scroll = app.pointer('pointermove', { pointerType: 'touch', clientX: 102, clientY: 160 });
  app.step();
  assert.equal(scroll.defaultPrevented, undefined, 'Vertical touch gestures outside Explore/fullscreen must preserve page scrolling.');
  assert.equal(app.camera.pitch, -.1);
  assert.equal(app.stage.captures.size, 0);
  const wheel = app.stage.fire('wheel', { deltaY: 100 });
  assert.equal(wheel.defaultPrevented, undefined, 'The page must continue scrolling when Explore is disabled.');
  app.controller.setEnabled(true);
  assert.equal(app.stage.fire('wheel', { deltaY: 100, target: new Element('input') }).defaultPrevented, undefined);
  assert.equal(app.stage.fire('wheel', { deltaY: 100, ctrlKey: true }).defaultPrevented, undefined, 'Browser zoom must remain available.');
  assert.equal(app.stage.fire('wheel', { deltaY: 100 }).defaultPrevented, true);
  app.step(120);
  assert.ok(app.camera.height > 4.3 && app.camera.height < 4.5);
  app.controller.dispose();
}

{
  const app = harness({ x: 14, z: 9, height: 8, yaw: 1, pitch: .3 });
  app.controller.setHomePose({ x: 0, z: 0, height: 2.2, yaw: 0, pitch: -.1 });
  assert.equal(app.camera.x, 14, 'Updating the composed home view must preserve a shared URL camera until recentering.');
  app.controller.reset();
  assert.equal(app.camera.x, 0);
  assert.equal(app.camera.z, 0);
  assert.equal(app.camera.height, 2.2);
  assert.equal(app.camera.yaw, 0);
  assert.equal(app.camera.pitch, -.1);
  app.controller.dispose();
}

{
  const app = harness();
  app.controller.setEnabled(true);
  app.pointer('pointerdown', { pointerType: 'touch', pointerId: 1, clientX: 100, clientY: 100 });
  app.pointer('pointerdown', { pointerType: 'touch', pointerId: 2, clientX: 200, clientY: 100 });
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 1, clientX: 100, clientY: 150 });
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 2, clientX: 200, clientY: 150 });
  app.step(120);
  assert.ok(app.camera.z > 1, 'Two-finger dragging should travel through the ocean.');
  assert.equal(app.camera.yaw, 0, 'Two-finger travel must not unexpectedly rotate the horizon.');
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 2, clientX: 250, clientY: 150 });
  app.step(120);
  assert.ok(app.camera.height < 2.4 && app.camera.height >= 1.8, 'Pinching out should bring the camera closer to the water.');
  app.key('Escape');
  assert.equal(app.stage.captures.size, 0, 'Escape must release all touch captures.');
  app.controller.dispose();
}

{
  const app = harness();
  app.controller.setEnabled(true);
  const forward = app.pad.children[0];
  forward.fire('pointerdown', { pointerId: 9, pointerType: 'touch', button: 0 });
  app.step(60);
  assert.ok(app.camera.z > 1.5, 'Touch movement buttons must move while held.');
  forward.fire('pointerup', { pointerId: 9 });
  app.step(180);
  assert.equal(app.queue.size, 0);
  const beforeClick = app.camera.z;
  forward.fire('click', { detail: 0 });
  app.step(120);
  assert.ok(Math.abs(app.camera.z - beforeClick - .8) < .001, 'Keyboard or assistive activation must offer a predictable movement step.');
  app.camera.x = 40;
  app.camera.yaw = -3.1;
  app.controller.sync();
  app.step(120);
  assert.equal(app.camera.x, 40, 'Externally restored positions must not be overwritten by old camera targets.');
  app.controller.dispose();
  assert.equal([...app.window.listeners.values()].every(listeners => listeners.size === 0), true, 'Cleanup must remove global listeners.');
  assert.equal([...app.stage.listeners.values()].every(listeners => listeners.size === 0), true, 'Cleanup must remove stage listeners.');
  assert.equal(app.pad.hidden, true);
  app.key('KeyW');
  app.step(60);
  assert.equal(app.camera.x, 40, 'A disposed controller must not react to input.');
}

{
  const app = harness({}, { constrainPose: pose => { pose.x = Math.min(2, Math.max(-2, pose.x)); } });
  app.controller.setEnabled(true);
  app.key('KeyD');
  app.step(240);
  assert.equal(app.camera.x, 2, 'Exploration must stop at the scene boundary.');
  app.release('KeyD');
  app.step(180);
  assert.equal(app.queue.size, 0, 'A constrained camera must settle without leaving an unreachable movement target.');
  app.camera.x = -40;
  app.controller.sync();
  assert.equal(app.camera.x, -2, 'Externally restored cameras must respect the scene boundary.');
  app.controller.dispose();
}

console.log('Ocean camera movement, touch, accessibility, lifecycle, and idle rendering checks passed.');
