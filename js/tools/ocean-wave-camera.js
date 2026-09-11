(() => {
  'use strict';

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const wrap = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));
  const KEYS = {
    KeyW: 'forward', KeyS: 'back', KeyA: 'left', KeyD: 'right',
    KeyQ: 'down', KeyE: 'up', ArrowLeft: 'lookLeft', ArrowRight: 'lookRight',
    ArrowUp: 'lookUp', ArrowDown: 'lookDown',
  };
  const INTERACTIVE = 'button, input, select, textarea, a, label, [contenteditable], .ocean-wave-settings, .ocean-wave-hud, .ocean-wave-camera-pad, .ocean-wave-place, .ocean-wave-rest-screen';

  const create = ({ stage, camera, toggleButton, relaxButton, resetButton, pad, hint, onChange = () => {}, onCommit = () => {}, onFloatChange = () => {},
    onReset = () => {}, constrainPose = () => {}, minHeight = 1.8, maxHeight = 24, speed = 2.4 } = {}) => {
    if (!stage || !camera) throw new Error('An ocean stage and camera are required.');
    const minimumHeight = () => typeof minHeight === 'function' ? minHeight() : minHeight;
    const initial = {
      x: Number(camera.x) || 0, z: Number(camera.z) || 0,
      height: Number(camera.height) || 3.5, yaw: Number(camera.yaw) || 0, pitch: Number(camera.pitch) || 0,
    };
    camera.x = initial.x;
    camera.z = initial.z;
    const target = { ...initial };
    const velocity = { x: 0, z: 0, height: 0 };
    const pressed = new Map();
    const pointers = new Map();
    const padPointers = new Map();
    const listeners = [];
    const motionPreference = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    const floatOffset = { height: 0, pitch: 0 };
    const renderPose = { ...camera };
    let floatEnabled = false;
    let lastFloatTime = null;
    let enabled = false;
    let disposed = false;
    let rafId = 0;
    let lastFrame = null;
    let changed = false;
    let gesture = null;
    const on = (node, name, callback, options) => {
      if (!node) return;
      node.addEventListener(name, callback, options);
      listeners.push(() => node.removeEventListener(name, callback, options));
    };
    const angleDifference = (a, b) => wrap(a - b);
    const pitchBounds = () => [camera.minPitch ?? -Math.PI * .4, camera.maxPitch ?? Math.PI * .3];
    const resetFloat = () => {
      floatOffset.height = 0;
      floatOffset.pitch = 0;
      lastFloatTime = null;
    };
    const setFloatEnabled = (value) => {
      if (disposed) return false;
      const next = Boolean(value) && !motionPreference?.matches;
      const notify = next !== floatEnabled || (Boolean(value) && !next)
        || (motionPreference?.matches && (floatOffset.height !== 0 || floatOffset.pitch !== 0));
      floatEnabled = next;
      if (motionPreference?.matches) resetFloat();
      if (notify) onFloatChange(floatEnabled);
      return floatEnabled;
    };
    // The renderer owns this clock, so pausing the sea also freezes floating.
    // Offsets never feed back into the user pose or its navigation targets.
    const getRenderPose = (time, { waveHeight = 0, period = 10, direction = 0 } = {}) => {
      Object.assign(renderPose, camera);
      if (disposed || motionPreference?.matches) {
        resetFloat();
        return renderPose;
      }
      const now = Number.isFinite(time) ? time : 0;
      const dt = lastFloatTime === null ? 0 : clamp(now - lastFloatTime, 0, .1);
      lastFloatTime = now;
      if (document.hidden || stage.dataset.oceanVisible === 'false') {
        lastFloatTime = null;
      } else if (dt > 0) {
        const seconds = clamp(Number(period) || 10, 7, 18);
        const heading = Number.isFinite(direction) ? direction : 0;
        const wavelength = 9.81 * seconds * seconds / (Math.PI * 2);
        const position = camera.x * Math.cos(heading) + camera.z * Math.sin(heading);
        const phase = now * Math.PI * 2 / seconds - position * Math.PI * 2 / wavelength;
        const amplitude = floatEnabled ? clamp(Number(waveHeight) || 0, 0, 2.2) * .1 : 0;
        const height = amplitude * (.88 * Math.sin(phase) + .12 * Math.sin(phase * .63 + .8));
        // Pitch follows the broad swell only; there is deliberately no roll.
        const alongView = Math.sin(camera.yaw) * Math.cos(heading) + Math.cos(camera.yaw) * Math.sin(heading);
        const pitch = -amplitude * .025 * Math.cos(phase) * alongView;
        const blend = 1 - Math.exp(-dt / 1.25);
        floatOffset.height += (height - floatOffset.height) * blend;
        floatOffset.pitch += (pitch - floatOffset.pitch) * blend;
        if (!floatEnabled && Math.abs(floatOffset.height) + Math.abs(floatOffset.pitch) < .00001) resetFloat();
      }
      renderPose.height = clamp(camera.height + floatOffset.height, minimumHeight(), maxHeight);
      renderPose.pitch = clamp(camera.pitch + floatOffset.pitch, ...pitchBounds());
      return renderPose;
    };
    const hasAction = (action) => [...pressed.values()].includes(action);
    const commit = () => {
      if (!changed) return;
      changed = false;
      onCommit(camera);
    };
    const isMoving = () => pressed.size > 0 || Math.abs(velocity.x) + Math.abs(velocity.z)
      + Math.abs(velocity.height) > .001 || Math.abs(angleDifference(target.yaw, camera.yaw)) > .00001
      || Math.abs(target.pitch - camera.pitch) > .00001 || Math.abs(target.height - camera.height) > .0001
      || Math.abs(target.x - camera.x) + Math.abs(target.z - camera.z) > .0001;
    const update = (elapsed) => {
      if (disposed || document.hidden || stage.dataset.oceanVisible === 'false') return false;
      const dt = clamp(Number(elapsed) || 0, 0, .05);
      if (!dt) return false;
      const before = [camera.x, camera.z, camera.height, camera.yaw, camera.pitch];
      const blend = 1 - Math.exp(-10 * dt);
      const vertical = Number(hasAction('up')) - Number(hasAction('down'));
      const forward = Number(hasAction('forward')) - Number(hasAction('back'));
      const right = Number(hasAction('right')) - Number(hasAction('left'));
      const length = Math.max(1, Math.hypot(forward, right));
      const travel = enabled ? speed / length : 0;
      const desiredX = (Math.sin(camera.yaw) * forward + Math.cos(camera.yaw) * right) * travel;
      const desiredZ = (Math.cos(camera.yaw) * forward - Math.sin(camera.yaw) * right) * travel;
      velocity.x += (desiredX - velocity.x) * blend;
      velocity.z += (desiredZ - velocity.z) * blend;
      velocity.height += ((enabled ? vertical * .8 : 0) - velocity.height) * blend;
      for (const axis of ['x', 'z', 'height']) {
        if (Math.abs(velocity[axis]) < .001) velocity[axis] = 0;
        target[axis] += velocity[axis] * dt;
      }
      target.height = clamp(target.height, minimumHeight(), maxHeight);
      constrainPose(target);
      target.yaw += (Number(hasAction('lookRight')) - Number(hasAction('lookLeft'))) * .48 * dt;
      const [minPitch, maxPitch] = pitchBounds();
      target.pitch = clamp(target.pitch + (Number(hasAction('lookUp')) - Number(hasAction('lookDown'))) * .35 * dt, minPitch, maxPitch);
      for (const axis of ['x', 'z', 'height', 'pitch']) {
        camera[axis] += (target[axis] - camera[axis]) * blend;
        if (Math.abs(target[axis] - camera[axis]) < .0001) camera[axis] = target[axis];
      }
      camera.yaw = wrap(camera.yaw + angleDifference(target.yaw, camera.yaw) * blend);
      if (Math.abs(angleDifference(target.yaw, camera.yaw)) < .00001) camera.yaw = wrap(target.yaw);
      constrainPose(camera);
      if (before.some((value, index) => value !== [camera.x, camera.z, camera.height, camera.yaw, camera.pitch][index])) {
        changed = true;
        onChange(camera);
        return true;
      }
      return false;
    };
    const tick = (timestamp) => {
      rafId = 0;
      if (disposed || document.hidden || stage.dataset.oceanVisible === 'false') { clear(); return; }
      update(lastFrame === null ? 1 / 60 : (timestamp - lastFrame) / 1000);
      lastFrame = timestamp;
      if (isMoving()) rafId = window.requestAnimationFrame(tick);
      else { lastFrame = null; commit(); }
    };
    const wake = () => {
      if (!rafId && !disposed && !document.hidden && stage.dataset.oceanVisible !== 'false') {
        lastFrame = null;
        rafId = window.requestAnimationFrame(tick);
      }
    };
    const releaseCapture = (node, id) => {
      try { if (node.hasPointerCapture?.(id)) node.releasePointerCapture(id); } catch {}
    };
    const clear = () => {
      if (rafId) window.cancelAnimationFrame(rafId);
      rafId = 0;
      lastFrame = null;
      pressed.clear();
      gesture = null;
      const captured = [...pointers.keys()];
      pointers.clear();
      captured.forEach(id => releaseCapture(stage, id));
      const padCaptured = [...padPointers];
      padPointers.clear();
      padCaptured.forEach(([id, button]) => releaseCapture(button, id));
      velocity.x = 0;
      velocity.z = 0;
      velocity.height = 0;
      for (const axis of ['x', 'z', 'height', 'yaw', 'pitch']) target[axis] = camera[axis];
      stage.classList.remove('ocean-wave-dragging');
      commit();
    };
    const sync = () => { constrainPose(camera); clear(); };
    const setHomePose = (pose = {}) => {
      for (const axis of ['x', 'z', 'height', 'yaw', 'pitch']) {
        if (Number.isFinite(pose[axis])) initial[axis] = pose[axis];
      }
    };
    const reset = () => {
      if (disposed) return;
      clear();
      resetFloat();
      Object.assign(camera, initial);
      camera.height = clamp(camera.height, minimumHeight(), maxHeight);
      constrainPose(camera);
      Object.assign(target, camera);
      onReset(camera);
      onChange(camera);
      onCommit(camera);
    };
    const setEnabled = (value) => {
      if (disposed) return;
      clear();
      enabled = Boolean(value);
      stage.classList.toggle('is-exploring', enabled);
      stage.dataset.oceanCameraMode = enabled ? 'explore' : 'relax';
      toggleButton?.setAttribute('aria-pressed', String(enabled));
      toggleButton?.setAttribute('aria-label', 'Explore');
      relaxButton?.setAttribute('aria-pressed', String(!enabled));
      if (resetButton) resetButton.hidden = !enabled;
      if (pad) pad.hidden = !enabled;
      if (hint) hint.hidden = !enabled;
    };

    on(toggleButton, 'click', () => { setEnabled(true); stage.focus({ preventScroll: true }); });
    on(relaxButton, 'click', () => { setEnabled(false); });
    on(stage, 'keydown', (event) => {
      if (event.key === 'Escape') { clear(); return; }
      if (event.target !== stage || event.ctrlKey || event.altKey || event.metaKey) return;
      if (event.key === 'Home') { event.preventDefault(); if (!event.repeat) reset(); return; }
      const code = event.code || event.key;
      const action = KEYS[code];
      if (!action || !enabled) return;
      event.preventDefault();
      pressed.set(code, action);
      wake();
    });
    on(window, 'keyup', (event) => {
      const code = event.code || event.key;
      if (pressed.delete(code)) wake();
    });
    on(window, 'blur', clear);
    on(stage, 'focusout', (event) => { if (event.relatedTarget !== stage) clear(); });
    on(document, 'visibilitychange', () => { if (document.hidden) clear(); });
    on(stage, 'ocean:visibility', (event) => { if (!event.detail?.visible) clear(); });
    on(window, 'pagehide', clear);
    const onMotionPreference = () => {
      if (motionPreference.matches) setFloatEnabled(false);
    };
    if (motionPreference?.addEventListener) on(motionPreference, 'change', onMotionPreference);
    else if (motionPreference?.addListener) {
      motionPreference.addListener(onMotionPreference);
      listeners.push(() => motionPreference.removeListener(onMotionPreference));
    }

    const measureGesture = () => {
      const points = [...pointers.values()];
      const a = points[0];
      if (!a) { gesture = null; return; }
      const b = points[1];
      gesture = {
        x: b ? (a.x + b.x) / 2 : a.x, y: b ? (a.y + b.y) / 2 : a.y,
        distance: b ? Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)) : 0,
        yaw: target.yaw, pitch: target.pitch, height: target.height,
        worldX: target.x, worldZ: target.z,
      };
    };
    on(stage, 'pointerdown', (event) => {
      if (!enabled) return;
      if (event.target?.closest?.(INTERACTIVE) || (event.pointerType === 'mouse' && event.button !== 0)) return;
      // Extra fingers must not reset an in-progress pan or pinch.
      if (pointers.size >= 2) return;
      stage.focus({ preventScroll: true });
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      measureGesture();
      stage.classList.add('ocean-wave-dragging');
      try { stage.setPointerCapture(event.pointerId); } catch {}
      event.preventDefault();
    });
    on(stage, 'pointermove', (event) => {
      const point = pointers.get(event.pointerId);
      if (!point || !gesture) return;
      point.x = event.clientX;
      point.y = event.clientY;
      const points = [...pointers.values()];
      const a = points[0];
      const b = points[1];
      const dx = (b ? (a.x + b.x) / 2 : a.x) - gesture.x;
      const dy = (b ? (a.y + b.y) / 2 : a.y) - gesture.y;
      if (b) {
        const panScale = .022;
        target.x = gesture.worldX + (Math.cos(gesture.yaw) * -dx + Math.sin(gesture.yaw) * dy) * panScale;
        target.z = gesture.worldZ + (-Math.sin(gesture.yaw) * -dx + Math.cos(gesture.yaw) * dy) * panScale;
        const distance = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
        target.height = clamp(gesture.height * gesture.distance / distance, minimumHeight(), maxHeight);
      } else {
        target.yaw = gesture.yaw - dx * .003;
        target.pitch = clamp(gesture.pitch + dy * .0025, ...pitchBounds());
      }
      event.preventDefault();
      wake();
    });
    const endGesture = (event) => {
      if (!pointers.delete(event.pointerId)) return;
      releaseCapture(stage, event.pointerId);
      measureGesture();
      if (!pointers.size) stage.classList.remove('ocean-wave-dragging');
      wake();
    };
    const cancelGesture = (event) => { if (pointers.has(event.pointerId)) clear(); };
    on(stage, 'pointerup', endGesture);
    on(stage, 'pointercancel', cancelGesture);
    on(stage, 'lostpointercapture', endGesture);
    on(stage, 'wheel', (event) => {
      if (!enabled || event.ctrlKey || event.metaKey || event.target?.closest?.(INTERACTIVE)) return;
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? stage.clientHeight : 1;
      target.height = clamp(target.height + clamp(event.deltaY * unit, -120, 120) * .009, minimumHeight(), maxHeight);
      wake();
    }, { passive: false });

    const releasePad = (event) => {
      const button = padPointers.get(event.pointerId);
      if (!button) return;
      padPointers.delete(event.pointerId);
      pressed.delete(`pad-${event.pointerId}`);
      releaseCapture(button, event.pointerId);
      wake();
    };
    for (const button of pad?.querySelectorAll('[data-ocean-move]') || []) {
      const action = button.dataset.oceanMove;
      if (!['forward', 'back', 'left', 'right', 'up', 'down'].includes(action)) continue;
      on(button, 'pointerdown', (event) => {
        if (!enabled || (event.pointerType === 'mouse' && event.button !== 0)) return;
        event.preventDefault();
        pressed.set(`pad-${event.pointerId}`, action);
        padPointers.set(event.pointerId, button);
        try { button.setPointerCapture(event.pointerId); } catch {}
        wake();
      });
      on(button, 'pointerup', releasePad);
      on(button, 'pointercancel', releasePad);
      on(button, 'lostpointercapture', releasePad);
      on(button, 'click', (event) => {
        // Keyboard and assistive-technology activation is a small, predictable step.
        if (!enabled || event.detail) return;
        const forward = action === 'forward' ? 1 : action === 'back' ? -1 : 0;
        const right = action === 'right' ? 1 : action === 'left' ? -1 : 0;
        target.x += (Math.sin(camera.yaw) * forward + Math.cos(camera.yaw) * right) * .8;
        target.z += (Math.cos(camera.yaw) * forward - Math.sin(camera.yaw) * right) * .8;
        target.height = clamp(target.height + (action === 'up' ? .4 : action === 'down' ? -.4 : 0), minimumHeight(), maxHeight);
        wake();
      });
    }
    // Also release outside the stage when a browser cannot capture the pointer.
    on(window, 'pointerup', (event) => { endGesture(event); releasePad(event); }, { capture: true });
    on(window, 'pointercancel', (event) => { cancelGesture(event); releasePad(event); }, { capture: true });
    const dispose = () => {
      if (disposed) return;
      clear();
      setEnabled(false);
      resetFloat();
      disposed = true;
      listeners.splice(0).forEach(remove => remove());
    };
    setEnabled(false);
    return {
      update, reset, sync, setHomePose, setEnabled, isEnabled: () => enabled, isMoving,
      setFloatEnabled, isFloatEnabled: () => floatEnabled, getRenderPose, dispose,
    };
  };

  window.OceanWaveCamera = { create };
})();
