'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/common/motion.js'), 'utf8');

// Model browser scheduling without wall-clock sleeps. Promises remain native so
// cancelled animations and queued completion callbacks follow real microtasks.
function createHarness({ reduced = false } = {}) {
  let now = 0;
  let timerId = 0;
  const timers = new Map();
  const observers = [];
  const windowListeners = new Map();
  const mediaListeners = new Set();
  const media = {
    matches: reduced,
    addEventListener: (name, callback) => mediaListeners.add(callback)
  };
  const setTimeout = (callback, delay) => {
    const id = ++timerId;
    timers.set(id, { callback, due: now + delay });
    return id;
  };
  const clearTimeout = (id) => timers.delete(id);
  const flush = async () => {
    for (let index = 0; index < 6; index += 1) await Promise.resolve();
  };

  const animate = (element, frames, options, transitionProperty) => {
    let resolve;
    let reject;
    let settled = false;
    const finished = new Promise((done, fail) => { resolve = done; reject = fail; });
    finished.catch(() => {});
    const animation = {
      effect: { target: element },
      frames,
      options,
      transitionProperty,
      finished,
      progress: 0,
      playState: 'running',
      complete() {
        if (settled) return;
        settled = true;
        animation.progress = 1;
        animation.playState = 'finished';
        resolve();
      },
      cancel() {
        animation.playState = 'idle';
        if (settled) return;
        settled = true;
        reject(new Error('Animation cancelled'));
      }
    };
    element.animations.push(animation);
    return animation;
  };

  const element = ({ hidden = true, naturalHeight = 240, cssTransitions = true, tokens = {} } = {}) => {
    const classes = new Set();
    const properties = {
      '--motion-fast': '160ms',
      '--motion-base': '220ms',
      '--motion-slow': '.32s',
      '--easing-standard': 'ease-out',
      ...tokens
    };
    const node = {
      hidden,
      naturalHeight,
      isConnected: true,
      dataset: {},
      animations: [],
      reads: [],
      transitionDuration: null,
      transitionDelay: '0s',
      paddingTop: '0px',
      paddingBottom: '0px',
      borderTopWidth: '0px',
      borderBottomWidth: '0px',
      style: {
        height: '',
        overflow: 'visible',
        maxHeight: '440px',
        boxSizing: 'content-box',
        setProperty: (name, value) => { properties[name] = value; }
      },
      classList: {
        contains: (name) => classes.has(name),
        toggle(name, value) {
          const changed = classes.has(name) !== value;
          if (value) classes.add(name);
          else classes.delete(name);
          if (!changed || !cssTransitions || media.matches) return;
          node.animations.filter((animation) => animation.transitionProperty).forEach((animation) => animation.cancel());
          animate(node, [], {}, 'opacity');
          animate(node, [], {}, 'transform');
        }
      },
      getBoundingClientRect() {
        node.reads.push({ hidden: node.hidden, open: classes.has('is-open') });
        if (node.hidden) return { height: 0 };
        const active = node.animations.findLast((animation) => animation.playState !== 'idle' && animation.frames[0]?.height);
        if (active) {
          const start = parseFloat(active.frames[0].height);
          const end = parseFloat(active.frames.at(-1).height);
          return { height: start + (end - start) * active.progress };
        }
        return { height: node.style.height && node.style.height !== 'auto' ? parseFloat(node.style.height) : node.naturalHeight };
      },
      getAnimations: () => node.animations.filter((animation) => animation.playState !== 'idle'),
      animate: (frames, options) => animate(node, frames, options)
    };
    node.computed = {
      getPropertyValue: (name) => properties[name] || '',
      get transitionDuration() { return node.transitionDuration || properties['--motion-duration'] || '0s'; },
      get transitionDelay() { return node.transitionDelay; },
      get boxSizing() { return node.style.boxSizing; },
      get paddingTop() { return node.paddingTop; },
      get paddingBottom() { return node.paddingBottom; },
      get borderTopWidth() { return node.borderTopWidth; },
      get borderBottomWidth() { return node.borderBottomWidth; }
    };
    return node;
  };

  const window = {
    matchMedia: () => media,
    setTimeout,
    addEventListener(name, callback) {
      if (!windowListeners.has(name)) windowListeners.set(name, new Set());
      windowListeners.get(name).add(callback);
    }
  };
  const context = vm.createContext({
    window,
    document: { documentElement: {} },
    getComputedStyle: (node) => node.computed,
    clearTimeout,
    MutationObserver: class {
      constructor(callback) { observers.push(callback); }
      observe() {}
    }
  });
  vm.runInContext(source, context, { filename: 'js/common/motion.js' });

  return {
    motion: window.SiteMotion,
    element,
    animate,
    timers,
    flush,
    async advance(delay) {
      const end = now + delay;
      while (true) {
        const next = [...timers].filter(([, timer]) => timer.due <= end).sort((a, b) => a[1].due - b[1].due)[0];
        if (!next) break;
        now = next[1].due;
        timers.delete(next[0]);
        next[1].callback();
        await flush();
      }
      now = end;
      await flush();
    },
    reduce(value) {
      media.matches = value;
      mediaListeners.forEach((callback) => callback({ matches: value }));
    },
    disconnect(node) {
      node.isConnected = false;
      observers.forEach((callback) => callback());
    },
    resize() { windowListeners.get('resize')?.forEach((callback) => callback()); },
    complete(node) { node.getAnimations().forEach((animation) => animation.complete()); }
  };
}

module.exports = async function runMotionTests({ assert }) {
  {
    const { motion, element, reduce } = createHarness();
    const node = element();
    assert(motion.duration(node, '--motion-slow') === 320, 'motion duration should parse seconds as milliseconds');
    assert(motion.duration(node, '--motion-fast') === 160, 'motion duration should preserve millisecond tokens');
    assert(motion.duration(node, '--missing', 75) === 75, 'missing tokens should use the caller fallback');
    assert(motion.duration(node, 88) === 88, 'explicit durations should be supported');
    reduce(true);
    assert(motion.duration(node, '--motion-slow') === 0, 'reduced motion should override a CSS duration');
  }

  {
    const h = createHarness();
    const node = h.element();
    let completed = 0;
    const opening = h.motion.presence(node, true, { onFinish: () => { completed += 1; } });
    assert(node.reads[0].hidden === false && node.reads[0].open === false,
      'opening should render an unhidden starting frame before applying the open class');
    assert(node.classList.contains('is-open') && !node.hidden && completed === 0,
      'opening should remain visible while waiting for its own transitions');
    node.getAnimations()[0].complete();
    await h.flush();
    assert(completed === 0, 'one finished property should not finish a surface with another running transition');
    node.getAnimations()[1].complete();
    assert(await opening === true && completed === 1 && node.dataset.motionState === 'open',
      'all surface transitions should finish the opening once');

    const closing = h.motion.presence(node, false);
    assert(!node.hidden && !node.classList.contains('is-open'), 'closing must keep content rendered during the exit');
    h.complete(node);
    assert(await closing === true && node.hidden, 'closing should hide the surface after its transition completes');
    await h.advance(1000);
    assert(completed === 1 && h.timers.size === 0, 'completed motion should remove fallback timers and avoid duplicate callbacks');
  }

  {
    const h = createHarness();
    const node = h.element();
    const counts = [0, 0, 0];
    const first = h.motion.presence(node, true, { onFinish: () => { counts[0] += 1; } });
    const second = h.motion.presence(node, false, { onFinish: () => { counts[1] += 1; } });
    const third = h.motion.presence(node, true, { onFinish: () => { counts[2] += 1; } });
    assert(await first === false && await second === false, 'rapid reversal should resolve superseded operations as cancelled');
    h.complete(node);
    assert(await third === true, 'the most recent reversed operation should finish normally');
    await h.advance(1000);
    assert(counts.join(',') === '0,0,1' && !node.hidden && node.classList.contains('is-open'),
      'stale closing callbacks must never hide a reopened menu');
  }

  {
    const h = createHarness();
    const node = h.element({ hidden: false });
    let completed = 0;
    const opening = h.motion.presence(node, true);
    h.motion.finish(node);
    assert(await opening === true && node.dataset.motionState === 'open', 'finish should synchronously settle the requested endpoint');
    const closing = h.motion.presence(node, false, { onFinish: () => { completed += 1; } });
    h.motion.cancel(node);
    h.complete(node);
    await h.advance(1000);
    assert(await closing === false && !node.hidden && completed === 0,
      'cancel should discard completion cleanup without hiding the current surface');
    const final = h.motion.presence(node, false, { onFinish: () => { completed += 1; } });
    h.motion.finish(node);
    h.motion.finish(node);
    assert(await final === true && node.hidden && completed === 1, 'finish should be idempotent and apply delayed hiding exactly once');
  }

  {
    const h = createHarness();
    const node = h.element({ naturalHeight: 200 });
    const original = { ...node.style };
    let staleCompletion = 0;
    const opening = h.motion.height(node, true, { onFinish: () => { staleCompletion += 1; } });
    const first = node.animations.at(-1);
    first.progress = .4;
    const closing = h.motion.height(node, false);
    const reverse = node.animations.at(-1);
    assert(first.frames[0].height === '0px' && first.frames[1].height === '200px', 'height opening should use collapsed and measured natural heights');
    assert(reverse.frames[0].height === '80px', 'height reversal must start from the current visible height');
    assert(await opening === false && staleCompletion === 0, 'interrupted height opening must not call its completion handler');
    reverse.complete();
    assert(await closing === true && node.hidden, 'reversed height closing should settle to hidden');
    assert(['height', 'overflow', 'maxHeight', 'boxSizing'].every((key) => node.style[key] === original[key]),
      'height completion should restore the original inline sizing and overflow');

    const cancelled = h.motion.height(node, true);
    h.motion.cancel(node);
    assert(await cancelled === false && !node.hidden, 'height cancellation should leave the content available for the next operation');
    assert(['height', 'overflow', 'maxHeight', 'boxSizing'].every((key) => node.style[key] === original[key]),
      'height cancellation must also restore inline styles');
  }

  {
    const h = createHarness();
    const node = h.element({ hidden: false, naturalHeight: 200 });
    node.paddingTop = node.paddingBottom = '10px';
    node.borderTopWidth = node.borderBottomWidth = '2px';
    let updates = 0;
    const changed = h.motion.swap(node, () => { node.naturalHeight = 320; updates += 1; });
    const animation = node.animations.at(-1);
    assert(updates === 1, 'content updates must happen synchronously so selection and accessibility state remain current');
    assert(animation.frames[0].height === '176px' && animation.frames[1].height === '296px',
      'content swaps should account for padding and borders while preserving measured outer heights');
    animation.complete();
    assert(await changed === true && node.style.overflow === 'visible', 'content swap completion should release temporary clipping');
  }

  {
    const h = createHarness({ reduced: true });
    const menu = h.element();
    const panel = h.element();
    const opening = h.motion.presence(menu, true);
    const expansion = h.motion.height(panel, true);
    assert(!menu.hidden && !panel.hidden && menu.dataset.motionState === 'open' && panel.dataset.motionState === 'open',
      'reduced-motion operations should settle immediately without waiting for animation events');
    assert(await opening === true && await expansion === true && h.timers.size === 0 && panel.animations.length === 0,
      'reduced motion must not schedule animation fallback timers or height animation');
    const closing = h.motion.presence(menu, false);
    assert(menu.hidden && await closing === true, 'reduced-motion close should hide immediately');
  }

  {
    const h = createHarness();
    const menu = h.element({ hidden: false });
    const panel = h.element();
    const closing = h.motion.presence(menu, false);
    const expansion = h.motion.height(panel, true);
    h.reduce(true);
    assert(await closing === true && menu.hidden && await expansion === true && !panel.hidden,
      'enabling reduced motion during active operations should finish both their endpoints');
    assert(panel.style.height === '' && panel.style.overflow === 'visible' && h.timers.size === 0,
      'reduced-motion changes should remove all temporary layout styles and timers');
  }

  {
    const h = createHarness();
    const menu = h.element();
    const panel = h.element();
    let completed = 0;
    const opening = h.motion.presence(menu, true, { onFinish: () => { completed += 1; } });
    const expansion = h.motion.height(panel, true, { onFinish: () => { completed += 1; } });
    h.disconnect(menu);
    h.disconnect(panel);
    h.complete(menu);
    await h.advance(1000);
    assert(await opening === false && await expansion === false && completed === 0,
      'removed route content should cancel operations without firing stale completion callbacks');
    assert(panel.style.height === '' && panel.style.overflow === 'visible' && h.timers.size === 0,
      'disconnection should restore sizing and release animation timers');
  }

  {
    const h = createHarness();
    const panel = h.element();
    const expansion = h.motion.height(panel, true);
    h.resize();
    assert(await expansion === true && !panel.hidden && panel.style.height === '',
      'viewport resizing should finish measured-height motion at a natural, responsive height');
  }

  {
    const h = createHarness();
    const node = h.element({ hidden: false, cssTransitions: false });
    node.transitionDuration = '100ms, .2s';
    node.transitionDelay = '20ms';
    let completed = 0;
    const closing = h.motion.presence(node, false, { onFinish: () => { completed += 1; } });
    await h.advance(269);
    assert(!node.hidden && completed === 0, 'fallback completion should respect the longest transition plus its repeated delay');
    await h.advance(1);
    assert(await closing === true && node.hidden && completed === 1,
      'surfaces without observable animation events should still finish using the computed fallback');
  }

  {
    const h = createHarness();
    const node = h.element();
    const loop = h.animate(node, [], { iterations: Infinity });
    const opening = h.motion.presence(node, true);
    node.getAnimations().filter((animation) => animation.transitionProperty).forEach((animation) => animation.complete());
    assert(await opening === true && loop.playState === 'running',
      'a looping keyframe animation must not prevent surface transition completion');
    assert(h.timers.size === 0, 'unrelated animation work should not retain the surface fallback timer');
  }

  {
    const h = createHarness();
    const node = h.element();
    const other = h.element();
    const opening = h.motion.presence(node, true);
    const unrelated = h.animate(other, [], {}, 'opacity');
    unrelated.complete();
    await h.flush();
    assert(node.dataset.motionState === 'opening', 'another surface completing must not settle this surface');
    h.motion.finish(node);
    await opening;
    let updates = 0;
    assert(await h.motion.presence(null, true) === false && await h.motion.height(null, true) === false,
      'missing surfaces should resolve safely as inactive operations');
    assert(await h.motion.swap(null, () => { updates += 1; }) === true && updates === 1,
      'missing animation containers must not suppress the underlying content update');
  }
};

if (require.main === module) {
  let assertions = 0;
  module.exports({
    assert(condition, message) {
      if (!condition) throw new Error(message);
      assertions += 1;
    }
  }).then(() => {
    console.log(`Motion behavior tests passed (${assertions} assertions).`);
  }, (error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
