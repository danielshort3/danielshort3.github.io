'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = path.join(__dirname, '../..');
const voidTags = new Set(['INPUT', 'BR', 'HR', 'IMG', 'SOURCE', 'WBR']);
const decode = (text) => String(text).replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');

class TestEvent {
  constructor(type, options = {}) { Object.assign(this, { type, bubbles: false, defaultPrevented: false }, options); }
  preventDefault() { this.defaultPrevented = true; }
}

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.listeners = new Map();
    this.style = { setProperty(name, value) { this[name] = value; } };
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this.value = '';
    this.defaultValue = '';
    this.id = '';
    this.className = '';
    this._text = '';
    this.classList = {
      contains: (value) => this.className.split(/\s+/).includes(value),
      add: (...values) => { this.className = [...new Set([...this.className.split(/\s+/), ...values])].join(' '); },
      remove: (...values) => { this.className = this.className.split(/\s+/).filter((value) => !values.includes(value)).join(' '); },
      toggle: (value, enabled) => { if (enabled) this.classList.add(value); else this.classList.remove(value); }
    };
  }

  append(...children) { children.forEach((child) => this.appendChild(child)); }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
  set textContent(value) { this._text = String(value); this.children = []; }
  set innerHTML(value) { this._text = ''; this.children = []; parse(String(value), this); }
  get innerHTML() { return this.children.length ? this.children.map((child) => `<${child.tagName.toLowerCase()}>${child.textContent}</${child.tagName.toLowerCase()}>`).join('') : this._text; }
  get options() { return this.children.filter((child) => child.tagName === 'OPTION'); }
  get selectedOptions() { return this.options.filter((option) => option.value === this.value); }
  get parentElement() { return this.parentNode; }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === 'class') this.className = String(value);
    else if (['id', 'name', 'type'].includes(name)) this[name] = String(value);
    else if (name === 'value') this.value = this.defaultValue = decode(value);
    else if (['hidden', 'checked', 'disabled', 'selected'].includes(name)) this[name] = true;
    else if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
  }
  getAttribute(name) {
    if (name === 'class') return this.className;
    if (name.startsWith('data-')) return this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] ?? null;
    return this.attributes[name] ?? null;
  }
  matches(selector) {
    if (selector === '*') return true;
    if (selector.includes(',')) return selector.split(',').some((part) => this.matches(part.trim()));
    const tag = /^[a-z][\w-]*/i.exec(selector)?.[0];
    if (tag && tag.toUpperCase() !== this.tagName) return false;
    const id = /#([\w-]+)/.exec(selector)?.[1];
    if (id && this.id !== id) return false;
    for (const match of selector.matchAll(/\.([\w-]+)/g)) if (!this.classList.contains(match[1])) return false;
    for (const match of selector.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
      const actual = this.getAttribute(match[1]);
      if (actual === null || (match[2] !== undefined && actual !== match[2])) return false;
    }
    return !selector.includes(':checked') || this.checked;
  }
  closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector) || null; }
  querySelectorAll(selector) { return this.children.flatMap((child) => [child, ...child.querySelectorAll('*')]).filter((child) => child.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  addEventListener(type, listener) { const entries = this.listeners.get(type) || []; entries.push(listener); this.listeners.set(type, entries); }
  dispatchEvent(event) {
    if (!event.target) event.target = this;
    (this.listeners.get(event.type) || []).forEach((listener) => listener(event));
    if (event.bubbles && this.parentNode) this.parentNode.dispatchEvent(event);
    return !event.defaultPrevented;
  }
  focus() { this.focused = true; }
  scrollIntoView() { this.scrolled = true; }
}

function parse(html, target) {
  const stack = [target];
  for (const match of html.matchAll(/<!--[\s\S]*?-->|<\s*(\/?)([\w-]+)\b([^>]*?)>|([^<]+)/g)) {
    const parent = stack[stack.length - 1];
    if (match[4]) { parent._text += decode(match[4]); continue; }
    if (!match[2]) continue;
    if (match[1]) {
      const closed = stack.pop();
      assert(closed && closed.tagName === match[2].toUpperCase(), `Balanced markup for ${match[2]}`);
      if (closed.tagName === 'TEXTAREA') closed.value = closed.defaultValue = closed.textContent;
      if (closed.tagName === 'SELECT') closed.value = (closed.options.find((option) => option.selected) || closed.options[0])?.value || '';
      continue;
    }
    const child = new Element(match[2]);
    for (const attribute of match[3].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) child.setAttribute(attribute[1], attribute[2] || '');
    parent.appendChild(child);
    if (!voidTags.has(child.tagName) && !match[3].endsWith('/')) stack.push(child);
  }
}

function harness(slug) {
  const html = fs.readFileSync(path.join(root, 'pages', `${slug}.html`), 'utf8');
  const main = html.slice(html.indexOf('<main id="main">'), html.indexOf('</main>') + 7);
  const document = new Element('document');
  document.readyState = 'complete';
  document.body = document.appendChild(new Element('body'));
  document.createElement = (tag) => new Element(tag);
  document.getElementById = (id) => document.querySelector(`#${id}`);
  parse(main, document.body);
  const ids = document.querySelectorAll('[id]').map((element) => element.id);
  assert.strictEqual(ids.length, new Set(ids).size, `${slug} must keep IDs unique`);
  const storage = new Map();
  const window = new Element('window');
  Object.assign(window, {
    location: new URL(`https://example.test/tools/${slug}`),
    localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key) },
    matchMedia: () => ({ matches: false }),
    TextCompareCore: require('../../js/tools/text-compare-core.js')
  });
  const context = vm.createContext({ window, document, Element, Event: TestEvent, CustomEvent: TestEvent, URL, URLSearchParams, Blob, navigator: {}, requestAnimationFrame: (callback) => callback(), setTimeout: () => 1, clearTimeout() {}, console });
  vm.runInContext(fs.readFileSync(path.join(root, 'js/tools/tool-workspace.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(root, 'js/tools', `${slug}.js`), 'utf8'), context);
  const get = (id) => document.getElementById(id);
  const fire = (element, type, options = {}) => element.dispatchEvent(new TestEvent(type, { bubbles: true, ...options }));
  const fill = (id, value) => { get(id).value = value; fire(get(id), 'input'); };
  return { get, fill, fire, window, document, html };
}

async function run() {
  const compare = harness('text-compare');
  assert.strictEqual(compare.document.body.style['--textcompare-ins-bg'], '#DDF2EC', 'fresh comparisons use the soft insertion highlight');
  assert.strictEqual(compare.document.body.style['--textcompare-del-bg'], '#FBE4E8', 'fresh comparisons use the soft deletion highlight');
  compare.fill('textcompare-ins-bg', '#C0FFEE');
  assert.strictEqual(compare.document.body.style['--textcompare-ins-bg'], '#C0FFEE', 'custom picker colors remain authoritative');
  compare.fill('textcompare-original', 'Ship the draft on Monday.');
  compare.fill('textcompare-revised', 'Ship the final draft on Tuesday.');
  compare.get('textcompare-original').scrollTop = 42;
  compare.fire(compare.get('textcompare-form'), 'submit');
  await Promise.resolve();
  await Promise.resolve();
  assert.strictEqual(compare.get('textcompare-view-comparison').hidden, false, 'Compare opens the result');
  assert(compare.get('textcompare-summary').textContent.includes('word'), 'real comparison still returns a summary');
  compare.window.ToolWorkspace.selectTab('textcompare-view-drafts');
  assert.strictEqual(compare.get('textcompare-original').value, 'Ship the draft on Monday.');
  assert.strictEqual(compare.get('textcompare-original').scrollTop, 42, 'changing views preserves editor scrolling');
  compare.fire(compare.get('textcompare-clear'), 'click');
  assert.strictEqual(compare.get('textcompare-view-drafts').hidden, false);
  assert.strictEqual(compare.get('textcompare-original').value, '');
  compare.fire(compare.document, 'tools:session-applied', { detail: { toolId: 'text-compare', snapshot: { output: { kind: 'html', html: '<p>Saved comparison</p>', summary: 'Saved draft changes' } } } });
  assert.strictEqual(compare.get('textcompare-view-comparison').hidden, false, 'restoring a saved comparison opens its result');
  assert.strictEqual(compare.get('textcompare-summary').textContent, 'Saved draft changes');
  assert.strictEqual(compare.get('textcompare-ins-bg').value, '#C0FFEE', 'restored output preserves the selected formatting');

  const frequency = harness('word-frequency');
  frequency.fill('wordfreq-text', 'Handoff handoff handoff. Support support.');
  frequency.fire(frequency.get('wordfreq-form'), 'submit');
  const term = frequency.get('wordfreq-results').querySelector('.wordfreq-term-btn');
  assert.strictEqual(term.textContent, 'handoff');
  assert.strictEqual(frequency.get('wordfreq-results').querySelector('.wordfreq-count').textContent, '3');
  assert.strictEqual(frequency.get('wordfreq-results').querySelector('.wordfreq-score').hidden, true, 'frequency does not repeat the same count');
  frequency.fire(term, 'click');
  assert.strictEqual(frequency.get('wordfreq-results-occurrences').hidden, false);
  assert(frequency.get('wordfreq-occurrence-summary').textContent.includes('3 matches'));
  frequency.window.ToolWorkspace.selectTab('wordfreq-results-fulltext');
  assert(frequency.get('wordfreq-fulltext').querySelector('.is-selected'), 'selected term remains highlighted in full text');
  frequency.window.ToolWorkspace.selectTab('wordfreq-results-terms');
  assert(frequency.get('wordfreq-results').querySelector('.is-selected'), 'selected result survives switching views');
  frequency.get('wordfreq-score').value = 'share';
  frequency.fire(frequency.get('wordfreq-score'), 'change');
  assert.strictEqual(frequency.get('wordfreq-results').querySelector('.wordfreq-score').hidden, false, 'non-count scores remain visible');
  frequency.fire(frequency.get('wordfreq-clear'), 'click');
  frequency.window.ToolWorkspace.selectTab('wordfreq-results-occurrences');
  assert.strictEqual(frequency.get('wordfreq-occurrence-panel').hidden, false, 'empty tab retains explanatory content');
  assert(frequency.get('wordfreq-occurrence-summary').textContent.includes('Select a term'));
  frequency.get('wordfreq-text').value = 'Review review review.';
  frequency.fire(frequency.document, 'tools:session-applied', { detail: { toolId: 'word-frequency' } });
  assert.strictEqual(frequency.get('wordfreq-results').querySelector('.wordfreq-count').textContent, '3', 'restored inputs regenerate frequency results');

  const pov = harness('point-of-view-checker');
  assert.strictEqual(pov.get('povcheck-stats').hidden, true, 'empty counts are not presented as analyzed results');
  pov.fill('povcheck-text', 'I reviewed the draft. You can share it. They will review the final version.');
  pov.fire(pov.get('povcheck-form'), 'submit');
  assert.strictEqual(pov.get('povcheck-stats').hidden, false);
  assert.strictEqual(pov.get('povcheck-first-count').textContent, '1');
  assert.strictEqual(pov.get('povcheck-second-count').textContent, '1');
  pov.window.ToolWorkspace.selectTab('povcheck-results-drift');
  const sentence = pov.get('povcheck-drift-list').querySelector('button');
  pov.fire(sentence, 'click');
  assert.strictEqual(pov.get('povcheck-results-highlights').hidden, false, 'sentence navigation reveals its highlighted target');
  pov.window.ToolWorkspace.selectTab('povcheck-results-tokens');
  pov.fire(pov.get('povcheck-first-list').querySelector('button'), 'click');
  assert.strictEqual(pov.get('povcheck-results-highlights').hidden, false, 'token navigation reveals highlighted text');
  pov.window.ToolWorkspace.selectTab('povcheck-setup-rules');
  pov.get('povcheck-mode-basic').checked = false;
  pov.get('povcheck-mode-advanced').checked = true;
  pov.fire(pov.get('povcheck-mode-advanced'), 'change');
  assert.strictEqual(pov.get('povcheck-advanced-panel').hidden, false);
  assert(pov.get('povcheck-settings-summary').textContent.includes('Advanced rules'));
  pov.fire(pov.get('povcheck-clear'), 'click');
  assert.strictEqual(pov.get('povcheck-setup-text').hidden, false);
  assert.strictEqual(pov.get('povcheck-stats').hidden, true);
  pov.get('povcheck-text').value = 'We reviewed the draft.';
  pov.fire(pov.document, 'tools:session-applied', { detail: { toolId: 'point-of-view-checker' } });
  assert.strictEqual(pov.get('povcheck-first-count').textContent, '1', 'restored POV text rebuilds the count strip');
  assert(pov.get('povcheck-settings-summary').textContent.includes('4 words'), 'restored text updates the setup summary');

  for (const current of [compare, frequency, pov]) {
    assert.strictEqual(current.document.querySelectorAll('[data-tool-share-link]').length, 1, 'one deliberate share location');
    assert(current.html.indexOf('js/tools/tool-workspace.js') < current.html.lastIndexOf('js/tools/'), 'shared tabs initialize before the entry script');
    current.document.querySelectorAll('[data-workspace-tab]').forEach((tab) => {
      const panel = current.get(tab.getAttribute('aria-controls'));
      assert(panel && panel.getAttribute('aria-labelledby') === tab.id, 'tab and panel have matching accessible labels');
    });
  }
  console.log('Text tool workspace tests passed.');
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
