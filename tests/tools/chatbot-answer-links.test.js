'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '../../demos/chatbot-demo.html'), 'utf8');
const start = html.indexOf('    function sourceLabel(');
const end = html.indexOf('    function answerText(', start);
assert(start >= 0 && end > start, 'the real chatbot answer renderer must be available');

// The fallback renderer only needs append/text nodes here. Full citation placement
// is verified against the browser DOM; these checks protect URL/token integrity.
class TextNode {
  constructor(value) { this.textContent = String(value); }
}

class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
  }

  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.append(node); return node; }
  set textContent(value) { this.children = [new TextNode(value)]; }
  get textContent() { return this.children.map(node => node.textContent).join(''); }
}

const env = {
  URL,
  document: {
    createElement: tag => new Element(tag),
    createTextNode: value => new TextNode(value)
  }
};
vm.runInNewContext(html.slice(start, end), env);
const allLinks = node => node.children?.flatMap(child => [
  ...(child.tagName === 'A' ? [child] : []), ...allLinks(child)
]) || [];
const render = text => {
  const node = new Element('div');
  env.renderMarkdownFallback(node, text, env.linkState());
  return node;
};

const balancedUrl = 'https://example.com/Grand_Mesa_(Colorado)';
let answer = render(`Explore [Grand Mesa](${balancedUrl}).`);
assert.equal(answer.textContent, 'Explore Grand Mesa.');
assert.equal(allLinks(answer)[0].href, balancedUrl);

answer = render('Try [a scenic route](https://example.com/route?name=Grand%20Mesa&mode=easy!)!');
assert.equal(answer.textContent, 'Try a scenic route!');
assert.equal(allLinks(answer)[0].href, 'https://example.com/route?name=Grand%20Mesa&mode=easy!');

answer = render('Read [the guide](<https://example.com/guide?view=all> "Visitor guide").');
assert.equal(answer.textContent, 'Read the guide.');
assert.equal(allLinks(answer)[0].href, 'https://example.com/guide?view=all');

answer = render('Read [the guide](https://example.com/route\\(easy\\)).');
assert.equal(answer.textContent, 'Read the guide.');
assert.equal(allLinks(answer)[0].href, 'https://example.com/route(easy)');

answer = render(`Explore (${balancedUrl}).`);
assert.equal(answer.textContent, `Explore (${balancedUrl}).`, 'bare URL linking must not discard sentence punctuation');
assert.equal(allLinks(answer)[0].href, balancedUrl);

answer = render('Read https://example.com/guide, then head outside.');
assert.equal(answer.textContent, 'Read https://example.com/guide, then head outside.');
assert.equal(allLinks(answer)[0].href, 'https://example.com/guide');

answer = render('Read (https://example.com/route?lang=en).');
assert.equal(answer.textContent, 'Read (https://example.com/route?lang=en).');
assert.equal(allLinks(answer)[0].href, 'https://example.com/route?lang=en');

answer = render('Read <https://example.com/route?value=literal!>.');
assert.equal(answer.textContent, 'Read https://example.com/route?value=literal!.');
assert.equal(allLinks(answer)[0].href, 'https://example.com/route?value=literal!');

answer = render('[Grand Mesa](https://example.com/mesa) and [the same guide](https://example.com/mesa).');
assert.equal(allLinks(answer).length, 2, 'explicit repeat links must behave like the full Markdown renderer');

answer = render('Visit [Grand Mesa][mesa].\n\n[mesa]: https://example.com/mesa "Grand Mesa guide"');
assert.equal(answer.textContent, 'Visit Grand Mesa.');
assert.equal(allLinks(answer)[0].href, 'https://example.com/mesa');

answer = render('Visit [Grand Mesa][] and [Grand Mesa].\n\n[Grand Mesa]: <https://example.com/mesa>');
assert.equal(answer.textContent, 'Visit Grand Mesa and Grand Mesa.');
assert.equal(allLinks(answer).length, 2);

answer = render('Keep `[Grand Mesa][mesa]` literal.\n\n[mesa]: https://example.com/mesa');
assert.equal(answer.textContent, 'Keep [Grand Mesa][mesa] literal.');
assert.equal(allLinks(answer).length, 0);

answer = render('Keep `https://example.com/guide` and ``[example](https://example.com/guide)`` as code.');
assert.equal(allLinks(answer).length, 0);
assert.equal(answer.textContent, 'Keep https://example.com/guide and [example](https://example.com/guide) as code.');

answer = render('[unsafe](javascript:alert(1)) and [data](data:text/html,hello).');
assert.equal(allLinks(answer).length, 0);
assert.equal(env.safeAnswerUrl('javascript:alert(1)'), '');
assert.equal(env.safeAnswerUrl('data:text/html,hello'), '');
assert.equal(env.safeAnswerUrl('/relative'), '');
assert.equal(env.safeAnswerUrl('https://example.com/Case?Token=AbC'), 'https://example.com/Case?Token=AbC');
assert.notEqual(env.sourceKey('https://example.com/Case'), env.sourceKey('https://example.com/case'));
assert.equal(env.sourceKey('https://EXAMPLE.com/guide/#more'), env.sourceKey('https://example.com/guide'));
assert.equal(env.sourceLabel('https://example.com/grand%20mesa.html', 0), 'Grand Mesa');

const sources = env.sourceItems({
  source_details: [
    { url: 'javascript:alert(1)', label: 'Unsafe' },
    { url: 'https://example.com/mesa', label: 'Grand Mesa' }
  ],
  source_urls: ['https://example.com/mesa', 'data:text/html,hello']
});
assert.equal(sources.length, 1);
assert.equal(sources[0].label, 'Grand Mesa');

for (let index = 0; index <= 100; index += 1) {
  const partial = `Explore [Grand Mesa](${balancedUrl}).`.slice(0, index);
  assert.doesNotThrow(() => render(partial), 'partial streamed Markdown must remain renderable');
}

console.log('Chatbot answer link regression checks passed.');
