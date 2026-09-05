'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/portfolio/modal-helpers.js'), 'utf8');
const origin = 'https://example.test';

function createFrame({ src = '/demos/baby-names-demo.html', fit = 'content', inaccessible = false, workspace = true } = {}) {
  let bottom = 760;
  let height = '';
  let heightWrites = 0;
  const document = {
    body: { scrollHeight: 2200 },
    documentElement: { scrollHeight: 2200, clientHeight: 2200 },
    querySelector: () => workspace ? { getBoundingClientRect: () => ({ bottom }) } : null
  };
  const contentWindow = { scrollY: 0, getComputedStyle: () => ({ paddingBottom: '12px' }) };
  const frame = {
    dataset: { src, projectDemoFit: fit },
    getAttribute: (name) => name === 'src' ? src : null,
    contentWindow,
    style: {
      get height() { return height; },
      set height(value) { height = value; heightWrites += 1; }
    }
  };
  Object.defineProperty(frame, 'contentDocument', { get() {
    if (inaccessible) throw new Error('Cross-origin document');
    return document;
  } });
  return { frame, document, setBottom: (value) => { bottom = value; }, heightWrites: () => heightWrites };
}

function createHarness(frames) {
  let messageListener;
  const window = { addEventListener(type, listener) { if (type === 'message') messageListener = listener; } };
  vm.runInNewContext(source.replace(/\}\)\(\);\s*$/, 'window.testResizeIframe = resizeIframeToContent;\n})();'), {
    window,
    document: { addEventListener() {}, querySelectorAll: () => frames.map((item) => item.frame) },
    location: { origin, href: `${origin}/portfolio` },
    URL, console
  });
  return {
    resize: (item) => window.testResizeIframe(item.frame),
    send: (item, overrides = {}) => messageListener({
      source: item.frame.contentWindow, origin,
      data: { type: 'baby-names-demo-resize', height: 2200 }, ...overrides
    })
  };
}

const content = createFrame();
const missingWorkspace = createFrame({ workspace: false });
const inaccessible = createFrame({ inaccessible: true });
const tableau = createFrame({ src: 'https://public.tableau.com/views/example', inaccessible: true });
const chat = createFrame({ src: '/demos/chatbot-demo.html', fit: 'viewport' });
const harness = createHarness([content, missingWorkspace, inaccessible, tableau, chat]);

harness.resize(content);
assert.strictEqual(content.frame.style.height, '772px', 'Initial sizing should include workspace bottom and body padding');
content.setBottom(1400);
harness.send(content);
assert.strictEqual(content.frame.style.height, '1412px', 'Opening a disclosure should grow its iframe');
content.setBottom(700);
harness.send(content);
assert.strictEqual(content.frame.style.height, '712px', 'Closing a disclosure must shrink despite the old reported document height');
const writesBeforeRepeat = content.heightWrites();
harness.send(content);
assert.strictEqual(content.heightWrites(), writesBeforeRepeat, 'Repeated unchanged measurements must not rewrite height');
content.setBottom(300);
harness.send(content);
assert.strictEqual(content.frame.style.height, '560px', 'Short demos retain the canonical usable minimum');

content.setBottom(900);
harness.send(content, { origin: 'https://untrusted.test' });
harness.send(content, { source: {} });
harness.send(content, { data: { type: 'prefix-baby-names-demo-resize', height: 9999 } });
assert.strictEqual(content.frame.style.height, '560px', 'Wrong origin, source, or event type must not resize an embed');
harness.send(content, { data: { type: 'baby-names-demo-resize', height: Infinity } });
assert.strictEqual(content.frame.style.height, '912px', 'Untrusted height values should be replaced by a real content measurement');

harness.resize(missingWorkspace);
assert.strictEqual(missingWorkspace.frame.style.height, '2200px', 'A same-origin document without a workspace can use its body fallback');
inaccessible.frame.style.height = '640px';
harness.send(inaccessible);
assert.strictEqual(inaccessible.frame.style.height, '640px', 'An inaccessible document must retain its existing fallback');
tableau.frame.style.height = '70vh';
harness.resize(tableau);
harness.send(tableau, { origin: 'https://public.tableau.com' });
assert.strictEqual(tableau.frame.style.height, '70vh', 'Cross-origin Tableau sizing must remain untouched');

chat.frame.style.height = '2200px';
harness.resize(chat);
assert.strictEqual(chat.frame.style.height, 'clamp(560px, 75vh, 900px)', 'A chat view must release any previously inherited content height');
const chatWrites = chat.heightWrites();
chat.setBottom(4500);
harness.send(chat, { data: { type: 'chatbot-demo-resize', height: 4500 } });
assert.strictEqual(chat.heightWrites(), chatWrites, 'Chat transcript growth must not change its bounded viewport');

console.log('Portfolio demo sizing: grow, shrink, duplicate events, trusted source, document fallback, Tableau, and chat viewport passed.');
