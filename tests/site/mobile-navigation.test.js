'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..', '..');
const read = (relative) => fs.readFileSync(path.join(root, relative), 'utf8');
const extractFunction = (source, name) => {
  const start = source.indexOf(`function ${name}(`);
  const body = source.indexOf('{', start);
  let depth = 0;
  for (let index = body; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}' && --depth === 0) return source.slice(start, index + 1);
  }
  throw new Error(`Missing function ${name}`);
};

module.exports = function runMobileNavigationTests({ assert }) {
  const frames = [];
  const scrolls = [];
  let headers = [{ height: 62, bottom: 62 }, { height: 0, bottom: 0 }];
  const trigger = { getBoundingClientRect: () => ({ top: 134 }) };
  const context = vm.createContext({
    document: { querySelectorAll: () => headers.map((rect) => ({ getBoundingClientRect: () => rect })) },
    window: { scrollY: 900, scrollTo: (options) => scrolls.push(options), requestAnimationFrame: (callback) => frames.push(callback) },
    railLayoutQuery: { matches: false },
    reducedMotionQuery: { matches: false },
    isLibraryMode: false,
    activeId: 'tools',
    panelMotions: new Map(),
    triggerById: new Map([['tools', trigger]]),
    Promise: { resolve: (value) => value || { then: (callback) => callback() } }
  });
  const accordion = read('js/home/category-accordion.js');
  vm.runInContext(`${extractFunction(accordion, 'getVisibleHeaderBottom')}\n${extractFunction(accordion, 'revealPanelTrigger')}`, context);
  vm.runInContext("revealPanelTrigger('tools');", context);
  frames.shift()();
  assert(scrolls[0].top === 972 && scrolls[0].behavior === 'smooth',
    'category reveal should subtract the actual 62px masthead once, independently of CSS scroll padding and margins');
  headers = [{ height: 0, bottom: 0 }, { height: 60, bottom: 60 }];
  assert(vm.runInContext('getVisibleHeaderBottom()', context) === 60,
    'landscape category reveal should measure the visible desktop header instead of using a portrait masthead fallback');
  let resolvePanel;
  context.panelMotions.set('tools', { then: (callback) => { resolvePanel = callback; } });
  vm.runInContext("revealPanelTrigger('tools');", context);
  context.activeId = 'games';
  resolvePanel();
  assert(frames.length === 0 && scrolls.length === 1,
    'a superseded panel animation must not scroll an earlier category into view');
};

if (require.main === module) {
  const assert = require('assert');
  module.exports({ assert });
  process.stdout.write('Mobile navigation tests passed.\n');
}
