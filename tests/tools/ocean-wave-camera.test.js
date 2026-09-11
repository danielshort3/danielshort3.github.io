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
  closest(selector) {
    const matches = selector.split(',').some(part => {
      const token = part.trim();
      return token.startsWith('.') ? this.classList.contains(token.slice(1)) : token === this.tagName;
    });
    return matches ? this : this.parentElement?.closest(selector) || null;
  }
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
  const motionPreference = new Element('media-query');
  motionPreference.matches = Boolean(options.reducedMotion);
  window.matchMedia = () => motionPreference;
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
  const floatChanges = [];
  window.requestAnimationFrame = callback => { const id = ++sequence; queue.set(id, callback); return id; };
  window.cancelAnimationFrame = id => queue.delete(id);
  vm.runInNewContext(source, { window, document, Math, Map, Number, Object, Error });
  const controller = window.OceanWaveCamera.create({
    stage, camera, toggleButton, relaxButton, resetButton, pad, hint,
    onChange: () => { changes++; }, onCommit: () => { commits++; }, onReset: () => { resets++; },
    onFloatChange: enabled => floatChanges.push(enabled),
    ...options,
  });
  return {
    window, document, stage, camera, toggleButton, relaxButton, resetButton, pad, hint, controller, queue, motionPreference, floatChanges,
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
  assert.ok(app.camera.yaw < -.4 && app.camera.yaw > -.5, 'Dragging right in Explore must turn the camera left.');
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

{
  const app = harness();
  const canonicalPose = JSON.stringify(app.camera);
  const conditions = { waveHeight: 5, period: 8, direction: Math.PI / 2 };
  assert.equal(app.controller.isFloatEnabled(), false, 'Floating must require an explicit opt in.');
  for (let frame = 0; frame <= 60; frame++) {
    assert.equal(JSON.stringify(app.controller.getRenderPose(frame / 60, conditions)), canonicalPose,
      'Floating must not change the rendered pose while disabled.');
  }
  app.controller.setFloatEnabled(true);
  let maximumRise = 0;
  let maximumPitch = 0;
  let lastPose = { ...app.camera };
  for (let frame = 61; frame <= 1800; frame++) {
    const pose = { ...app.controller.getRenderPose(frame / 60, conditions) };
    maximumRise = Math.max(maximumRise, Math.abs(pose.height - app.camera.height));
    maximumPitch = Math.max(maximumPitch, Math.abs(pose.pitch - app.camera.pitch));
    assert.ok(Math.abs(pose.height - lastPose.height) < .006, 'Floating must ease in and stay gentle between frames.');
    assert.ok(Math.abs(pose.pitch - lastPose.pitch) < .0002, 'Pitch changes must remain subtle.');
    assert.equal(pose.yaw, app.camera.yaw, 'Floating must not turn or roll the horizon.');
    assert.equal(pose.x, app.camera.x);
    assert.equal(pose.z, app.camera.z);
    lastPose = pose;
  }
  assert.ok(maximumRise > .08 && maximumRise <= .22, 'Floating must produce visible, bounded rise and fall even in a rough sea.');
  assert.ok(maximumPitch > .001 && maximumPitch <= .0055, 'Floating pitch must remain below a third of a degree.');
  assert.equal(JSON.stringify(app.camera), canonicalPose, 'Render offsets must never contaminate the saved or navigable camera pose.');
  assert.equal(app.changes, 0);
  assert.equal(app.commits, 0, 'Floating must not generate URL or session updates each frame.');
  assert.equal(app.queue.size, 0, 'Floating must use the sea renderer clock without starting its own animation loop.');
  const paused = JSON.stringify(app.controller.getRenderPose(30, conditions));
  for (let frame = 0; frame < 100; frame++) {
    assert.equal(JSON.stringify(app.controller.getRenderPose(30, conditions)), paused, 'Pausing the sea clock must freeze floating.');
  }
  app.controller.setFloatEnabled(false);
  assert.equal(JSON.stringify(app.controller.getRenderPose(30, conditions)), paused, 'Switching floating off should ease back from the current view.');
  for (let frame = 1801; frame <= 3000; frame++) app.controller.getRenderPose(frame / 60, conditions);
  assert.equal(JSON.stringify(app.controller.getRenderPose(50, conditions)), canonicalPose, 'Disabling floating must settle back onto the original camera.');
  assert.deepEqual(app.floatChanges, [true, false]);
  app.controller.dispose();
}

{
  const app = harness({}, { reducedMotion: true });
  assert.equal(app.controller.setFloatEnabled(true), false, 'Reduced motion must override a restored floating preference.');
  assert.equal(app.controller.isFloatEnabled(), false);
  assert.deepEqual(app.floatChanges, [false], 'The UI must receive the effective disabled preference.');
  const canonicalPose = JSON.stringify(app.camera);
  app.controller.getRenderPose(0, { waveHeight: 3 });
  assert.equal(JSON.stringify(app.controller.getRenderPose(10, { waveHeight: 3 })), canonicalPose);
  app.motionPreference.matches = false;
  app.motionPreference.fire('change');
  assert.equal(app.controller.isFloatEnabled(), false, 'Ending reduced motion must not enable unsolicited camera movement.');
  app.controller.setFloatEnabled(true);
  for (let frame = 0; frame < 240; frame++) app.controller.getRenderPose(frame / 60, { waveHeight: 2 });
  assert.notEqual(JSON.stringify(app.controller.getRenderPose(4, { waveHeight: 2 })), canonicalPose);
  app.motionPreference.matches = true;
  app.motionPreference.fire('change');
  assert.equal(app.controller.isFloatEnabled(), false, 'A live reduced motion change must disable floating.');
  assert.equal(JSON.stringify(app.controller.getRenderPose(4, { waveHeight: 2 })), canonicalPose,
    'Reduced motion must remove offsets immediately, including while the sea is paused.');
  app.controller.dispose();
  assert.equal([...app.motionPreference.listeners.values()].every(listeners => listeners.size === 0), true,
    'Disposal must remove the reduced motion listener.');
}

{
  const sample = (fps) => {
    const app = harness({ yaw: .8 });
    app.controller.setFloatEnabled(true);
    let result;
    for (let frame = 0; frame <= fps * 30; frame++) {
      result = { ...app.controller.getRenderPose(frame / fps, { waveHeight: 1.6, period: 11, direction: .4 }) };
    }
    app.controller.dispose();
    return result;
  };
  const fast = sample(60);
  const slow = sample(15);
  assert.ok(Math.abs(fast.height - slow.height) < .003, 'Floating should follow the same sea clock at 15 and 60 FPS.');
  assert.ok(Math.abs(fast.pitch - slow.pitch) < .0001, 'Pitch damping should be independent of render frame rate.');
}

for (const camera of [{ height: 1.8, pitch: -1.2 }, { height: 24, pitch: .5 }]) {
  const app = harness(camera);
  app.controller.setFloatEnabled(true);
  for (let frame = 0; frame < 1800; frame++) {
    const pose = app.controller.getRenderPose(frame / 60, { waveHeight: 8, period: .5, direction: Math.PI / 2 });
    assert.ok(pose.height >= 1.8 && pose.height <= 24, 'Floating must respect the camera height limits.');
    assert.ok(pose.pitch >= -1.2 && pose.pitch <= .5, 'Floating must respect the pitch limits.');
  }
  for (let frame = 1800; frame <= 3600; frame++) app.controller.getRenderPose(frame / 60, { waveHeight: 0 });
  const calm = app.controller.getRenderPose(60, { waveHeight: 0 });
  assert.ok(Math.abs(calm.height - camera.height) < .00001 && Math.abs(calm.pitch - camera.pitch) < .00001,
    'A flat sea must settle to a stationary view even with floating enabled.');
  app.controller.dispose();
}

{
  const floated = harness();
  const steady = harness();
  floated.controller.setFloatEnabled(true);
  for (const app of [floated, steady]) { app.controller.setEnabled(true); app.key('KeyW'); app.key('ArrowRight'); }
  for (let frame = 0; frame < 240; frame++) {
    floated.step(1);
    steady.step(1);
    const pose = floated.controller.getRenderPose(frame / 60, { waveHeight: 2, direction: .2 });
    assert.equal(pose.x, floated.camera.x);
    assert.equal(pose.z, floated.camera.z);
    assert.equal(JSON.stringify(floated.camera), JSON.stringify(steady.camera), 'Explore movement must be identical with floating enabled.');
  }
  const beforeHidden = JSON.stringify(floated.controller.getRenderPose(4, { waveHeight: 2 }));
  floated.document.hidden = true;
  assert.equal(JSON.stringify(floated.controller.getRenderPose(60, { waveHeight: 2 })), beforeHidden,
    'Hidden stages must not advance floating.');
  floated.document.hidden = false;
  assert.equal(JSON.stringify(floated.controller.getRenderPose(60, { waveHeight: 2 })), beforeHidden,
    'Returning to a stage must not catch up with a sudden camera jump.');
  floated.controller.reset();
  assert.equal(JSON.stringify(floated.controller.getRenderPose(60, { waveHeight: 2 })), JSON.stringify(floated.camera),
    'Recenter must clear floating offsets together with navigation.');
  floated.controller.dispose();
  steady.controller.dispose();
}

{
  const app = harness();
  const canvas = new Element('canvas');
  canvas.parentElement = app.stage;
  app.controller.setEnabled(true);
  const down = app.pointer('pointerdown', { pointerType: 'touch', target: canvas });
  const move = app.pointer('pointermove', { pointerType: 'touch', clientX: 180, clientY: 130 });
  app.pointer('pointerup', { pointerType: 'touch' });
  app.window.fire('pointerup', { pointerId: 1, pointerType: 'touch' });
  app.step(120);
  assert.equal(down.defaultPrevented, true);
  assert.equal(move.defaultPrevented, true);
  assert.ok(Math.abs(app.camera.yaw + .24) < .0001, 'A one-finger drag right must turn the camera left in Explore.');
  assert.ok(Math.abs(app.camera.pitch + .025) < .0001, 'A one-finger canvas drag must change pitch.');
  assert.equal(app.camera.x, 0, 'One-finger looking must preserve camera position.');
  assert.equal(app.camera.z, 0);
  assert.equal(app.stage.captures.size, 0);
  assert.equal(app.stage.classList.contains('ocean-wave-dragging'), false);
  app.pointer('pointerdown', { pointerType: 'touch', pointerId: 7, clientX: 140, clientY: 140 });
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 7, clientX: 100, clientY: 160 });
  app.pointer('pointerup', { pointerType: 'touch', pointerId: 7 });
  app.step(120);
  assert.ok(Math.abs(app.camera.yaw + .12) < .0001, 'A new touch drag left must turn right from the previous heading.');
  assert.ok(Math.abs(app.camera.pitch - .025) < .0001, 'A new touch drag must use its own gesture origin.');
  assert.equal(app.queue.size, 0);
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
  const translated = { ...app.camera };
  assert.ok(translated.z > 1);
  app.pointer('pointerup', { pointerType: 'touch', pointerId: 1 });
  app.step(120);
  assert.equal(app.camera.yaw, translated.yaw, 'Lifting one finger after translation must not rotate the view.');
  assert.equal(app.camera.pitch, translated.pitch);
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 2, clientX: 280, clientY: 170 });
  app.step(120);
  assert.ok(Math.abs(app.camera.yaw - translated.yaw + .24) < .0001, 'The remaining finger must continue as a fresh look gesture.');
  assert.ok(Math.abs(app.camera.pitch - translated.pitch - .05) < .0001);
  assert.equal(app.camera.x, translated.x);
  assert.equal(app.camera.z, translated.z, 'Switching from two fingers to one must not continue translation.');
  app.pointer('pointerup', { pointerType: 'touch', pointerId: 2 });
  app.step(120);
  assert.equal(app.stage.captures.size, 0);
  app.controller.dispose();
}

{
  const app = harness();
  app.controller.setEnabled(true);
  app.pointer('pointerdown', { pointerType: 'touch', pointerId: 1, clientX: 100, clientY: 100 });
  app.pointer('pointerdown', { pointerType: 'touch', pointerId: 2, clientX: 200, clientY: 100 });
  app.pointer('pointerdown', { pointerType: 'touch', pointerId: 3, clientX: 300, clientY: 100 });
  assert.equal(app.stage.captures.size, 2, 'Extra fingers must not join a camera gesture.');
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 3, clientX: 340, clientY: 180 });
  app.step(120);
  assert.equal(app.camera.z, 0);
  assert.equal(app.camera.yaw, 0);
  app.pointer('pointerup', { pointerType: 'touch', pointerId: 1 });
  app.pointer('pointermove', { pointerType: 'touch', pointerId: 2, clientX: 300, clientY: 100 });
  app.step(120);
  assert.ok(Math.abs(app.camera.yaw + .3) < .0001, 'An ignored third finger must not replace the released gesture finger.');
  assert.equal(app.camera.x, 0);
  assert.equal(app.camera.z, 0);
  app.pointer('pointerup', { pointerType: 'touch', pointerId: 3 });
  app.pointer('pointerup', { pointerType: 'touch', pointerId: 2 });
  app.controller.dispose();
}

for (const ending of ['pointercancel', 'lostpointercapture']) {
  const app = harness();
  app.controller.setEnabled(true);
  app.pointer('pointerdown', { pointerType: 'touch' });
  app.pointer('pointermove', { pointerType: 'touch', clientX: 240, clientY: 130 });
  app.step(6);
  app.pointer('pointercancel', { pointerType: 'touch', pointerId: 99 });
  assert.equal(app.stage.classList.contains('ocean-wave-dragging'), true, 'Cancellation for an unrelated finger must not stop a gesture.');
  if (ending === 'lostpointercapture') app.stage.releasePointerCapture(1);
  app.pointer(ending, { pointerType: 'touch' });
  const endedPose = JSON.stringify(app.camera);
  app.step(120);
  if (ending === 'pointercancel') assert.equal(JSON.stringify(app.camera), endedPose, 'Touch cancellation must stop camera movement immediately.');
  assert.equal(app.stage.captures.size, 0);
  assert.equal(app.stage.classList.contains('ocean-wave-dragging'), false);
  assert.equal(app.queue.size, 0);
  const settledPose = JSON.stringify(app.camera);
  app.pointer('pointermove', { pointerType: 'touch', clientX: 600, clientY: 250 });
  app.step(120);
  assert.equal(JSON.stringify(app.camera), settledPose, 'Canceled or uncaptured touches must not keep steering the camera.');
  app.controller.dispose();
}

for (const ending of ['pointerup', 'pointercancel']) {
  const app = harness();
  app.controller.setEnabled(true);
  app.stage.setPointerCapture = () => { throw Error('Pointer capture unavailable'); };
  app.pointer('pointerdown', { pointerType: 'touch' });
  app.pointer('pointermove', { pointerType: 'touch', clientX: 200, clientY: 140 });
  app.step(6);
  app.window.fire(ending, { pointerId: 1, pointerType: 'touch' });
  app.step(120);
  assert.equal(app.stage.classList.contains('ocean-wave-dragging'), false,
    `Window ${ending} must release a gesture when pointer capture failed.`);
  assert.equal(app.queue.size, 0);
  const pose = JSON.stringify(app.camera);
  app.pointer('pointermove', { pointerType: 'touch', clientX: 400, clientY: 200 });
  app.window.fire(ending, { pointerId: 1, pointerType: 'touch' });
  app.step(120);
  assert.equal(JSON.stringify(app.camera), pose, 'Outside-stage release must be final and safe to receive twice.');
  app.controller.dispose();
}

for (const ending of ['pointerup', 'pointercancel']) {
  const app = harness();
  app.controller.setEnabled(true);
  const forward = app.pad.children[0];
  forward.setPointerCapture = () => { throw Error('Pointer capture unavailable'); };
  forward.fire('pointerdown', { pointerId: 9, pointerType: 'touch', button: 0 });
  app.step(30);
  assert.ok(app.camera.z > .5);
  app.window.fire(ending, { pointerId: 9, pointerType: 'touch' });
  app.step(180);
  const stopped = app.camera.z;
  app.step(120);
  assert.equal(app.camera.z, stopped, `Window ${ending} must stop a held movement button when pointer capture failed.`);
  assert.equal(app.controller.isMoving(), false);
  assert.equal(app.queue.size, 0);
  forward.fire(ending, { pointerId: 9, pointerType: 'touch' });
  app.window.fire(ending, { pointerId: 9, pointerType: 'touch' });
  app.step(120);
  assert.equal(app.camera.z, stopped, 'Duplicate button and window release must not restart movement.');
  app.controller.dispose();
}

for (const className of ['ocean-wave-hud', 'ocean-wave-camera-pad', 'ocean-wave-settings']) {
  const app = harness();
  app.controller.setEnabled(true);
  const panel = new Element();
  panel.classList.add(className);
  const child = new Element('span');
  child.parentElement = panel;
  const original = JSON.stringify(app.camera);
  const down = app.pointer('pointerdown', { pointerType: 'touch', target: child });
  app.pointer('pointermove', { pointerType: 'touch', clientX: 200, clientY: 160 });
  app.pointer('pointerup', { pointerType: 'touch' });
  app.step(120);
  assert.equal(down.defaultPrevented, undefined, 'Touching controls or their blank areas must remain available to that panel.');
  assert.equal(app.stage.captures.size, 0);
  assert.equal(JSON.stringify(app.camera), original, 'Drags beginning inside controls must not move the view.');
  app.controller.dispose();
}

console.log('Ocean camera movement, floating, reduced motion, touch, accessibility, lifecycle, and idle rendering checks passed.');
