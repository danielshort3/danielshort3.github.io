'use strict';

const assert = require('node:assert/strict');
const { buildKnowledge, hasIndexableContent } = require('../../build/generate-chatbot-knowledge');

assert.equal(hasIndexableContent(
  'Ocean Wave Simulation',
  'Adjust wave, light, and wind parameters in a real-time canvas simulation sandbox.',
  'Your browser does not support the HTML canvas element. Calm dawn'
), true, 'A concise canvas experience must remain discoverable from its real title and description.');
assert.equal(hasIndexableContent('Coming soon', '', 'Coming soon.'), false, 'Placeholder pages must remain excluded.');
assert.equal(hasIndexableContent('A descriptive title '.repeat(20), '', ''), false, 'A long title cannot make an empty page eligible.');
assert.equal(hasIndexableContent('', '', ''), false);

const knowledge = buildKnowledge();
const ocean = knowledge.pages.find(page => page.url === '/games/ocean-wave-simulation');
assert(ocean, 'The actual concise simulator page must survive the generator pipeline.');
assert.equal(ocean.title, 'Ocean Wave Simulation');
assert(knowledge.chunks.some(chunk => chunk.url === ocean.url && chunk.text.includes(ocean.description)),
  'The simulator must retain a citeable chunk grounded in its page description.');
assert(!knowledge.pages.some(page => ['/privacy', '/games/project-starfall', '/tools/dashboard'].includes(page.url)),
  'Metadata eligibility must not bypass excluded or noindex routes.');

console.log('Chatbot knowledge eligibility passed: concise canvas metadata, blank pages, citation content, and private-route exclusions.');
