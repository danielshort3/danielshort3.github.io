'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/accounts/tools-account-ui.js'), 'utf8');
const start = source.indexOf('  const enhanceAccountDisclosures = ');
const end = source.indexOf('\n  const isPersonalAccordionToolsPage', start);

module.exports = ({ assert }) => {
  const document = { activeElement: null };
  const element = () => ({
    childNodes: [],
    dataset: {},
    attributes: {},
    listeners: {},
    open: false,
    hidden: false,
    appendChild(node) {
      if (node.parent) node.parent.childNodes = node.parent.childNodes.filter((child) => child !== node);
      node.parent = this;
      this.childNodes.push(node);
    },
    querySelector() { return this.childNodes.find((node) => node.isSummary); },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    contains(node) { return this.childNodes.includes(node); },
    focus() { document.activeElement = this; }
  });
  document.createElement = element;
  const details = element();
  const summary = element();
  summary.isSummary = true;
  const input = element();
  details.appendChild(summary);
  details.appendChild(input);
  const root = { querySelectorAll: () => [details] };
  const pending = new Map();
  const motion = {
    height(node, expanded, options) {
      node.hidden = false;
      node.dataset.motionState = expanded ? 'opening' : 'closing';
      pending.set(node, () => {
        pending.delete(node);
        node.hidden = !expanded;
        node.dataset.motionState = expanded ? 'open' : 'closed';
        options.onFinish();
      });
    },
    cancel(node) { pending.delete(node); }
  };
  const context = vm.createContext({ window: { SiteMotion: motion }, document });
  vm.runInContext(`${source.slice(start, end)}\nglobalThis.enhance = enhanceAccountDisclosures;`, context);
  context.enhance(root);
  const content = details.childNodes[1];
  assert(content.hidden && content.inert, 'closed session details must hide their fields and keep them out of keyboard navigation');
  assert(content.childNodes[0] === input, 'enhancement must preserve existing field nodes and their values');
  context.enhance(root);
  assert(details.childNodes.length === 2, 'rerendering must not nest duplicate disclosure wrappers');
  let prevented = false;
  const click = () => summary.listeners.click({ preventDefault() { prevented = true; } });
  click();
  assert(prevented && details.open && !content.hidden && !content.inert, 'opening must expose the native details before animating its content');
  assert(summary.attributes['aria-expanded'] === 'true', 'opening must communicate expanded state');
  pending.get(content)();
  document.activeElement = input;
  click();
  assert(details.open && !content.hidden && content.inert, 'closing must retain visible content until the height animation finishes');
  assert(document.activeElement === summary, 'closing focused content must return focus to its summary');
  click();
  pending.get(content)();
  assert(details.open && !content.hidden && !content.inert, 'rapid reopening must finish open without an older close hiding the fields');
  click();
  pending.get(content)();
  assert(!details.open && content.hidden && summary.attributes['aria-expanded'] === 'false', 'completed close must restore native collapsed semantics');
  details.open = true;
  details.listeners.toggle();
  assert(!content.hidden && !content.inert, 'native find-in-page opening must reveal enhanced fields');
  details.open = false;
  details.listeners.toggle();
  assert(content.hidden && content.inert, 'native programmatic closing must keep enhanced state synchronized');
  context.window.SiteMotion = null;
  const nativeDetails = element();
  context.enhance({ querySelectorAll: () => [nativeDetails] });
  assert(!nativeDetails.dataset.motionDisclosure, 'the native disclosure must remain available if motion support is absent');
};

if (require.main === module) {
  let checks = 0;
  module.exports({ assert(pass, message) { if (!pass) throw new Error(message); checks += 1; } });
  console.log(`Account disclosure tests passed (${checks} assertions).`);
}
