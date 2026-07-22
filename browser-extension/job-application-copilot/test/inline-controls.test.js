import assert from 'node:assert/strict';
import test from 'node:test';

import {
  RISK_CLASSES,
  RUNTIME_CHANNEL,
  RUNTIME_VERSION,
  createPageRuntime,
  overlayActionPresentation
} from '../src/content/page-runtime.js';
import { FieldSelectionRouter, validateFieldSelectedMessage } from '../src/background/field-selection-router.js';

class FakeField {
  constructor({ label, name, type }, doc, index) {
    this.ownerDocument = doc;
    this.tagName = type === 'textarea' ? 'TEXTAREA' : 'INPUT';
    this.type = type === 'textarea' ? '' : type;
    this.name = name;
    this.id = `inline-field-${index}`;
    this.labels = [{
      textContent: label,
      hidden: false,
      isConnected: true,
      getAttribute: () => '',
      getBoundingClientRect: () => ({ width: 160, height: 20, top: 0, right: 160, bottom: 20, left: 0 })
    }];
    this.required = true;
    this.disabled = false;
    this.readOnly = false;
    this.hidden = false;
    this.isConnected = true;
    this.isContentEditable = false;
    this.multiple = false;
    this.maxLength = -1;
    this.parentElement = doc.body;
    this.form = null;
    this.options = [];
    this.attributes = new Map([
      ['id', this.id],
      ['name', this.name],
      ['type', type]
    ]);
  }

  getAttribute(name) {
    if (['aria-hidden', 'aria-required', 'aria-label', 'aria-labelledby', 'aria-describedby', 'title'].includes(name)) return '';
    return this.attributes.get(name) || '';
  }

  hasAttribute(name) {
    return Boolean(this.getAttribute(name));
  }

  closest() {
    return null;
  }

  matches(selector) {
    const tagName = this.tagName.toLowerCase();
    return String(selector || '').split(',').some(part => part.trim().startsWith(tagName));
  }

  getBoundingClientRect() {
    return { width: 300, height: 36, top: 40, right: 340, bottom: 76, left: 40 };
  }

  getClientRects() {
    return [this.getBoundingClientRect()];
  }
}

const createFakeDocument = () => {
  const doc = {
    location: { href: 'https://jobs.example.test/apply' },
    defaultView: {
      Event,
      getComputedStyle: () => ({ display: 'block', visibility: 'visible' })
    },
    body: {
      tagName: 'BODY',
      children: [],
      parentElement: null,
      getAttribute: () => ''
    },
    querySelector: () => null,
    querySelectorAll(selector) {
      return /input|textarea|select|contenteditable/u.test(String(selector)) ? this.fields : [];
    },
    getElementById: () => null
  };
  doc.documentElement = doc.body;
  doc.fields = [
    new FakeField({ label: 'Email address', name: 'email', type: 'email' }, doc, 0),
    new FakeField({ label: 'Salary expectation', name: 'salary', type: 'textarea' }, doc, 1)
  ];
  doc.body.children = doc.fields;
  return doc;
};

test('inline controls expose compact accessible review and regenerate actions', () => {
  const generated = overlayActionPresentation({
    field: { label: 'Salary expectation', riskClass: RISK_CLASSES.F2_REVIEW },
    proposal: { action: 'fill', confidence: 'review' }
  });
  assert.equal(generated.review.title, 'Review in sidebar');
  assert.match(generated.review.ariaLabel, /Open the answer for Salary expectation/u);
  assert.equal(generated.regenerate.title, 'Regenerate from saved sources');
  assert.match(generated.regenerate.ariaLabel, /Regenerate the answer for Salary expectation/u);
  assert.equal(generated.regenerate.disabled, false);

  const deterministic = overlayActionPresentation({
    field: { label: 'Email address', riskClass: RISK_CLASSES.F1_VERIFIED },
    proposal: { action: 'fill', confidence: 'high' }
  });
  assert.equal(deterministic.regenerate.disabled, true);
  assert.equal(deterministic.regenerate.title, 'Saved profile facts are deterministic');
});

test('page runtime emits review and regenerate intents from the overlay controller', async () => {
  const doc = createFakeDocument();
  const sent = [];
  let overlayCallbacks;
  const overlayFactory = (callbacks) => {
    overlayCallbacks = callbacks;
    return {
      render() {},
      setCopyOnly() {},
      clearCopyOnly() {},
      resetCopyOnly() {},
      destroy() {}
    };
  };
  const runtime = createPageRuntime({
    doc,
    view: doc.defaultView,
    overlayFactory,
    runtimeApi: {
      id: 'extension-id',
      async sendMessage(message) { sent.push(message); },
      onMessage: { addListener() {}, removeListener() {} }
    }
  });
  const scan = runtime.scan();
  const email = scan.result.fields.find(field => field.label === 'Email address');
  const salary = scan.result.fields.find(field => field.label === 'Salary expectation');
  overlayCallbacks.onSelect(email.fieldId);
  overlayCallbacks.onRegenerate(salary.fieldId);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(sent.map(message => message.payload.intent), ['review', 'regenerate']);
  assert.deepEqual(sent.map(message => message.payload.field.fieldId), [email.fieldId, salary.fieldId]);
  runtime.destroy();
});

test('field-selection router preserves regenerate intent while opening the sidebar', async () => {
  const values = {};
  const opened = [];
  const router = new FieldSelectionRouter({
    storageSession: { async set(update) { Object.assign(values, update); } },
    sidePanel: { async open(value) { opened.push(value); } },
    extensionId: 'extension-id',
    now: () => new Date('2026-07-19T12:00:00.000Z')
  });
  const message = {
    channel: RUNTIME_CHANNEL,
    version: RUNTIME_VERSION,
    type: 'FIELD_SELECTED',
    requestId: 'overlay-regenerate-1',
    payload: {
      intent: 'regenerate',
      field: {
        fieldId: 'field-0123456789abcdef',
        fingerprint: '0123456789abcdef',
        label: 'Salary expectation',
        type: 'textarea',
        options: [],
        nearbyText: '',
        required: true,
        riskClass: RISK_CLASSES.F2_REVIEW
      }
    }
  };
  assert.equal(validateFieldSelectedMessage(message), message);
  await router.handle(message, {
    id: 'extension-id',
    frameId: 0,
    url: 'https://jobs.example.test/apply',
    tab: { id: 12 }
  });
  assert.deepEqual(opened, [{ tabId: 12 }]);
  assert.equal(values.jobCopilotSelectedField.intent, 'regenerate');
  assert.equal(values.jobCopilotSelectedField.fieldId, 'field-0123456789abcdef');
});
