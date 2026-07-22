import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  RISK_CLASSES,
  RUNTIME_CHANNEL,
  RUNTIME_VERSION,
  classifyFieldMetadata,
  createFieldFingerprint,
  createPageRuntime,
  createSanitizedExport,
  detectCaptchaPresence,
  fieldRecordHasValue,
  fillFieldRecord,
  sanitizeLabelText,
  scanDocument,
  validateRuntimeMessage
} from '../src/content/page-runtime.js';
import { validatePageBatchRequest } from '../src/shared/validators.js';

const fixture = JSON.parse(await readFile(new URL('./fixtures/applicantpro-structure.json', import.meta.url), 'utf8'));
const liveApplicantProFixture = JSON.parse(await readFile(new URL('./fixtures/applicantpro-live-structure.json', import.meta.url), 'utf8'));

class FakeLabel {
  constructor(text) {
    this.textContent = text;
    this.isConnected = true;
    this.hidden = false;
  }

  getAttribute() {
    return '';
  }

  getBoundingClientRect() {
    return { width: 120, height: 20, top: 0, right: 120, bottom: 20, left: 0 };
  }
}

class FakeField {
  constructor(spec, doc, index) {
    this.spec = spec;
    this.ownerDocument = doc;
    this.tagName = spec.tagName || 'INPUT';
    this.type = spec.type === 'textarea' ? '' : spec.type;
    this.name = spec.name ?? `field_${index}`;
    this.id = spec.id ?? `fixture-field-${index}`;
    this.labels = [new FakeLabel(spec.label ?? '')];
    this.required = Boolean(spec.required);
    this.disabled = Boolean(spec.disabled);
    this.readOnly = Boolean(spec.readOnly);
    this.hidden = Boolean(spec.hidden);
    this.isConnected = true;
    this.isContentEditable = spec.type === 'contenteditable';
    this.multiple = spec.type === 'select-multiple';
    this.maxLength = spec.maxLength ?? -1;
    this.parentElement = spec.parentElement === undefined ? doc.body : spec.parentElement;
    this.form = null;
    this.events = [];
    this.style = spec.style || { display: 'block', visibility: 'visible', opacity: '1' };
    this._width = spec.width ?? 300;
    this._height = spec.height ?? 36;
    this._root = spec.root || doc;
    this._value = '';
    this._checked = false;
    this._textContent = '';
    this.resetOnInput = Boolean(spec.resetOnInput);
    this.resetInputAttempts = Math.max(0, Number(spec.resetInputAttempts) || 0);
    this.attributes = new Map([
      ['id', this.id],
      ['name', this.name],
      ['type', spec.type || 'text'],
      ['autocomplete', spec.autocomplete || ''],
      ['placeholder', spec.placeholder || ''],
      ['role', spec.role || '']
    ]);
    [
      ['aria-autocomplete', spec.ariaAutocomplete],
      ['aria-hidden', spec.ariaHidden],
      ['aria-label', spec.ariaLabel],
      ['aria-labelledby', spec.ariaLabelledBy],
      ['aria-haspopup', spec.ariaHasPopup]
    ].forEach(([name, value]) => {
      if (value !== undefined) this.attributes.set(name, String(value));
    });
    Object.entries(spec.attributes || {}).forEach(([name, value]) => this.attributes.set(name, String(value)));
    this.options = Array.from(spec.options || []).map((label) => ({
      label,
      textContent: label,
      selected: false
    }));
    this.selectedIndex = this.options.length ? 0 : -1;
    if (spec.throwOnValueRead) {
      Object.defineProperty(this, 'value', {
        configurable: true,
        get() {
          throw new Error('Scanner read a current field value.');
        },
        set(value) {
          this._value = value;
        }
      });
    }
  }

  get value() {
    return this._value;
  }

  set value(value) {
    this._value = String(value);
  }

  get checked() {
    return this._checked;
  }

  set checked(value) {
    this._checked = Boolean(value);
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value);
  }

  getAttribute(name) {
    if (name === 'contenteditable') return this.isContentEditable ? 'true' : '';
    return this.attributes.get(name) || '';
  }

  hasAttribute(name) {
    return this.attributes.has(name);
  }

  getRootNode() {
    return this._root;
  }

  closest() {
    return null;
  }

  matches(selector) {
    const tagName = this.tagName.toLowerCase();
    return String(selector || '').split(',').some((part) => {
      const candidate = part.trim();
      if (candidate.startsWith(tagName)) return true;
      if (candidate.startsWith('[contenteditable')) return this.isContentEditable;
      const roleMatch = /^\[role="([^"]+)"/u.exec(candidate);
      return roleMatch ? this.getAttribute('role') === roleMatch[1] : false;
    });
  }

  getBoundingClientRect() {
    return {
      width: this._width,
      height: this._height,
      top: 40,
      right: 40 + this._width,
      bottom: 40 + this._height,
      left: 40
    };
  }

  getClientRects() {
    return [this.getBoundingClientRect()];
  }

  dispatchEvent(event) {
    this.events.push(event.type);
    if (event.type === 'input' && (this.resetOnInput || this.resetInputAttempts > 0)) {
      this._value = '';
      if (this.resetInputAttempts > 0) this.resetInputAttempts -= 1;
    }
    return true;
  }
}

const createFakeCaptchaNode = ({
  tagName = 'IFRAME',
  src = '',
  name = '',
  id = '',
  type = '',
  title = '',
  className = '',
  ariaLabel = '',
  dataSize = '',
  dataAppearance = '',
  dataExecution = '',
  insideBadge = false,
  visible = true,
  width = 300,
  height = 70
} = {}) => ({
  tagName,
  hidden: false,
  isConnected: true,
  getAttribute(attribute) {
    return {
      src,
      name,
      id,
      type,
      title,
      class: className,
      'aria-label': ariaLabel,
      'data-size': dataSize,
      'data-appearance': dataAppearance,
      'data-execution': dataExecution,
      'aria-hidden': ''
    }[attribute] || '';
  },
  closest(selector) {
    return insideBadge && selector === '.grecaptcha-badge' ? {} : null;
  },
  getBoundingClientRect() {
    return visible
      ? { width, height, top: 40, right: 40 + width, bottom: 40 + height, left: 40 }
      : { width: 0, height: 0, top: 0, right: 0, bottom: 0, left: 0 };
  },
  getClientRects() {
    return visible ? [this.getBoundingClientRect()] : [];
  }
});

const isCaptchaSelector = selector => /recaptcha|hcaptcha|turnstile|aria-label\*="captcha"/iu.test(String(selector || ''));

const createFakeDocument = (fieldSpecs, { captcha = false, url = fixture.url, application = false } = {}) => {
  const doc = {
    id: application ? 'application-form' : '',
    location: { href: url },
    defaultView: {
      Event,
      getComputedStyle(node) {
        return node?.style || { display: 'block', visibility: 'visible' };
      }
    },
    body: {
      tagName: 'BODY',
      children: [],
      parentElement: null,
      getAttribute() {
        return '';
      }
    },
    querySelector(selector) {
      if (isCaptchaSelector(selector)) return this.captchaNodes[0] || null;
      if (selector === 'iframe') return this.iframes[0] || null;
      return null;
    },
    querySelectorAll(selector) {
      if (isCaptchaSelector(selector)) return this.captchaNodes;
      if (selector === 'iframe') return this.iframes;
      if (selector === '*') return [...this.fields, ...this.hosts, ...this.iframes];
      return this.fields;
    },
    getElementById(id) {
      return [...this.fields, ...this.hosts, ...this.iframes].find((node) => node.id === id) || null;
    }
  };
  doc.hosts = [];
  doc.iframes = [];
  doc.documentElement = doc.body;
  doc.captchaNodes = captcha ? [createFakeCaptchaNode({
    src: 'https://www.google.com/recaptcha/api2/bframe?k=test-key',
    visible: true
  })] : [];
  doc.fields = fieldSpecs.map((spec, index) => new FakeField(spec, doc, index));
  doc.body.children = [...doc.fields, ...doc.iframes];
  return doc;
};

const appendFakeShadowField = (root, doc, spec = {}) => {
  const field = new FakeField({ ...spec, parentElement: null, root }, doc, root.elements.length);
  if (spec.useAriaLabel) {
    const labelId = spec.ariaLabelledBy || `shadow-label-${root.elements.length}`;
    field.labels = [];
    field.attributes.set('aria-labelledby', labelId);
    root.labelNodes.set(labelId, new FakeLabel(spec.label || ''));
  }
  root.elements.push(field);
  return field;
};

const attachOpenShadowHost = (doc, {
  id,
  parentRoot = null,
  fieldSpecs = [],
  overlay = false
} = {}) => {
  const attributes = new Map([
    ['id', id || 'shadow-host'],
    ['name', ''],
    ['data-job-application-copilot-overlay', overlay ? 'true' : '']
  ]);
  const host = {
    tagName: 'DIV',
    id: id || 'shadow-host',
    name: '',
    parentElement: parentRoot ? null : doc.body,
    children: [],
    getAttribute(name) {
      return attributes.get(name) || '';
    },
    getRootNode() {
      return parentRoot || doc;
    },
    closest() {
      return null;
    },
    matches() {
      return false;
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    }
  };
  const shadowRoot = {
    mode: 'open',
    host,
    elements: [],
    captchaNodes: [],
    labelNodes: new Map(),
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    },
    querySelectorAll(selector) {
      if (isCaptchaSelector(selector)) return this.captchaNodes;
      if (selector === '*') return this.elements;
      return this.elements.filter((element) => element instanceof FakeField && element.matches(selector));
    },
    getElementById(labelId) {
      return this.labelNodes.get(labelId)
        || this.elements.find((element) => element.id === labelId)
        || null;
    }
  };
  host.shadowRoot = shadowRoot;
  if (parentRoot) parentRoot.elements.push(host);
  else {
    doc.hosts.push(host);
    doc.body.children = [...doc.fields, ...doc.hosts];
  }
  fieldSpecs.forEach((spec) => appendFakeShadowField(shadowRoot, doc, spec));
  return { host, shadowRoot, fields: shadowRoot.elements.filter((element) => element instanceof FakeField) };
};

const fixtureAdapter = {
  id: 'applicantpro',
  collectCandidates: (doc) => doc.fields,
  getSupplementalLabel: () => '',
  getNearbyText: (element) => element.spec?.nearbyText || '',
  extractJobMetadata: () => ({
    ...fixture.job,
    jobUrl: 'https://acme.applicantpro.com/jobs/12345?jobId=12345'
  })
};

const envelope = (type, requestId, payload = {}) => ({
  channel: RUNTIME_CHANNEL,
  version: RUNTIME_VERSION,
  type,
  requestId,
  payload
});

test('F0-F4 classifier follows the approved runtime taxonomy', () => {
  const cases = [
    [{ type: 'password', label: 'Password' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'text', label: 'Verification OTP code' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'email', label: 'Email address' }, RISK_CLASSES.F1_VERIFIED],
    [{ type: 'date', label: 'Interview date' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'textarea', label: 'Why do you want this role?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Desired salary' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'radio', label: 'Will you require sponsorship?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'radio', label: 'Will you require U.S. visa sponsorship?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'What is your citizenship?' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'What is your current visa status, and will you require sponsorship?' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'What type of visa sponsorship do you need?' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'Do you have a green card and authorization to work in the U.S.?' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'Enter your I-9 or E-Verify information' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'Are you a U.S. person under ITAR export-control rules?' }, RISK_CLASSES.F3_SENSITIVE],
    [{
      type: 'select-one',
      label: 'What is your preferred method of communication?',
      name: 'text_approval',
      options: ['Email', 'Text Message']
    }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Are you legally eligible to work within the country you are applying to?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Have you previously worked for this company?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Current company', name: 'org' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Current location', name: 'location' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Do you identify as transgender?' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'date', label: 'Date of birth' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'Birthday', autocomplete: 'bday' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'select-one', label: 'Veteran status' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'text', label: 'University attended' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'text', label: 'Professional certifications' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'email', label: 'Supervisor email' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'email', label: "Manager's email" }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'tel', label: 'Emergency contact phone' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'url', label: 'Reference website' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'email', label: 'Email of a reference' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'email', label: 'School contact email' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'email', label: 'Company email' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'url', label: 'Employer website' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'text', label: 'Website' }, RISK_CLASSES.F1_VERIFIED],
    [{ type: 'url', label: 'Personal website' }, RISK_CLASSES.F1_VERIFIED],
    [{ type: 'url', label: 'Tableau Public profile' }, RISK_CLASSES.F1_VERIFIED],
    [{ type: 'text', label: 'Tableau Public profile' }, RISK_CLASSES.F1_VERIFIED],
    [{ type: 'text', label: 'Writing samples link' }, RISK_CLASSES.F1_VERIFIED],
    [{ type: 'text', label: 'Reference profile' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'url', label: 'Unlabeled field' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'text', label: 'Account reference' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'textarea', label: 'Anything else?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'textarea', label: 'What project are you most proud of?' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'textarea', label: 'If yes, please elaborate.' }, RISK_CLASSES.F2_REVIEW],
    [{ type: 'textarea', label: 'What is your gender?' }, RISK_CLASSES.F3_SENSITIVE],
    [{ type: 'textarea', label: 'What is your password?' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'textarea', label: 'Search jobs', name: 'job_search', role: 'searchbox' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'text', label: 'Account reference', placeholder: 'Email address' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'text', label: 'Login', autocomplete: 'current-password' }, RISK_CLASSES.F0_EXCLUDED],
    [{ type: 'checkbox', label: 'I certify these answers are true' }, RISK_CLASSES.F4_CONSENT],
    [{ type: 'submit', label: 'Continue' }, RISK_CLASSES.F4_CONSENT]
  ];
  cases.forEach(([metadata, expected]) => assert.equal(classifyFieldMetadata(metadata), expected));
});

test('live-derived ApplicantPro fields retain their expected safety classes', () => {
  liveApplicantProFixture.applicationForm.fields.forEach((field) => {
    assert.equal(
      classifyFieldMetadata(field),
      field.riskClass,
      `${field.name} should remain ${field.riskClass}`
    );
  });
  assert.equal(liveApplicantProFixture.captchaPresent, true);
});

test('scanner emits stable actionable descriptors without reading values', () => {
  const specs = fixture.fields.map((field) => ({ ...field, throwOnValueRead: true }));
  specs[0].maxLength = 80;
  specs[2].nearbyText = 'Current answer daniel@example.com or 303-555-0123';
  const doc = createFakeDocument(specs, { captcha: true });
  doc.fields.forEach((field, index) => {
    field.spec = specs[index];
  });
  const first = scanDocument({ doc, url: fixture.url, adapter: fixtureAdapter });
  const second = scanDocument({ doc, url: fixture.url, adapter: fixtureAdapter });
  assert.deepEqual(first.result.fields.map((field) => field.fieldId), second.result.fields.map((field) => field.fieldId));
  assert.match(first.result.pageId, /^page-[a-f0-9]{16}$/);
  assert.match(first.result.urlHash, /^[a-f0-9]{16}$/);
  assert.equal(first.result.domRevision, 0);
  assert.equal(first.result.fields[0].maxLength, 80);
  assert.equal(first.result.fields[0].nearbyText, '');
  assert.equal(first.result.fields[1].nearbyText, '');
  assert.equal(first.result.fields[2].nearbyText, 'Current answer [redacted] or [redacted]');
  assert.equal(first.result.fields[3].nearbyText, '');
  assert.deepEqual(first.result.fields.map((field) => field.riskClass), [
    RISK_CLASSES.F1_VERIFIED,
    RISK_CLASSES.F1_VERIFIED,
    RISK_CLASSES.F2_REVIEW,
    RISK_CLASSES.F2_REVIEW
  ]);
  assert.equal(first.result.captchaPresent, true);
  assert.equal(first.result.exclusionCounts.F0_EXCLUDED, 1);
  assert.equal(first.result.exclusionCounts.F3_SENSITIVE, 1);
  assert.equal(first.result.exclusionCounts.F4_CONSENT, 1);
  assert.deepEqual(first.result.job, {
    company: fixture.job.company,
    title: fixture.job.title,
    jobUrl: 'https://acme.applicantpro.com/jobs/12345?jobId=12345',
    location: fixture.job.location,
    source: fixture.job.source,
    description: ''
  });
  assert.doesNotThrow(() => validatePageBatchRequest({
    pageId: first.result.pageId,
    urlHash: first.result.urlHash,
    domRevision: first.result.domRevision,
    fields: first.result.fields
  }));
  const serialized = JSON.stringify(first.result);
  assert.equal(serialized.includes('Password'), false);
  assert.equal(serialized.includes('Gender'), false);
  assert.equal(serialized.includes('certify'), false);
  assert.equal(serialized.includes('current-secret-value'), false);
  assert.equal(serialized.includes('daniel@example.com'), false);
  assert.equal(serialized.includes('303-555-0123'), false);
});

test('label sanitization removes PII-like dynamic tokens before descriptors are retained', () => {
  const raw = 'Email address daniel@example.com | phone +1 (303) 555-0123 | applicant 550e8400-e29b-41d4-a716-446655440000';
  const sanitized = sanitizeLabelText(raw);
  assert.match(sanitized, /Email address \[redacted\]/u);
  assert.doesNotMatch(sanitized, /daniel@example\.com|303|550e8400/u);

  const doc = createFakeDocument([{
    tagName: 'INPUT',
    type: 'email',
    name: 'email',
    label: raw
  }]);
  const result = scanDocument({ doc, adapter: fixtureAdapter }).result;
  assert.equal(result.fields.length, 1);
  assert.doesNotMatch(JSON.stringify(result.fields), /daniel@example\.com|303|550e8400/u);
});

test('scanner prefers a bounded structural label over nested ATS status text', () => {
  const doc = createFakeDocument([{
    tagName: 'INPUT',
    type: 'text',
    name: 'location',
    label: 'Current location No location found. Try entering a different location Loading'
  }]);
  const adapter = {
    ...fixtureAdapter,
    id: 'generic',
    getSupplementalLabel: () => 'Current location'
  };
  const result = scanDocument({ doc, adapter }).result;
  assert.equal(result.fields.length, 1);
  assert.equal(result.fields[0].label, 'Current location');
  assert.equal(result.fields[0].riskClass, RISK_CLASSES.F2_REVIEW);
});

test('unknown fields fail closed and never enter the actionable model batch', () => {
  const doc = createFakeDocument([
    { tagName: 'TEXTAREA', type: 'textarea', name: 'misc', label: 'Miscellaneous entry' },
    { tagName: 'INPUT', type: 'number', name: 'reference', label: 'Account reference' }
  ]);
  const result = scanDocument({ doc, adapter: fixtureAdapter }).result;
  assert.deepEqual(result.fields, []);
  assert.equal(result.exclusionCounts.F0_EXCLUDED, 2);
});

test('passive invisible CAPTCHA integration does not pause filling, but rendered interaction does', () => {
  const passiveNodes = [
    createFakeCaptchaNode({ tagName: 'SCRIPT', src: 'https://www.google.com/recaptcha/api.js?render=test-key' }),
    createFakeCaptchaNode({
      src: 'https://www.recaptcha.net/recaptcha/enterprise/anchor?k=test-key&size=invisible',
      insideBadge: true
    }),
    createFakeCaptchaNode({ tagName: 'TEXTAREA', name: 'g-recaptcha-response', visible: false }),
    createFakeCaptchaNode({ tagName: 'DIV', dataSize: 'invisible' }),
    createFakeCaptchaNode({ tagName: 'DIV', dataAppearance: 'interaction-only' })
  ];
  const fakeDocument = nodes => ({
    defaultView: { getComputedStyle: () => ({ display: 'block', visibility: 'visible' }) },
    querySelectorAll: () => nodes
  });
  assert.equal(detectCaptchaPresence(fakeDocument(passiveNodes)), false);
  assert.equal(detectCaptchaPresence(fakeDocument([
    createFakeCaptchaNode({
      src: 'https://www.google.com/recaptcha/api2/bframe?k=test-key',
      visible: false
    })
  ])), false);
  assert.equal(detectCaptchaPresence(fakeDocument([
    createFakeCaptchaNode({
      src: 'https://www.google.com/recaptcha/api2/anchor?k=test-key&size=normal'
    })
  ])), true);
  assert.equal(detectCaptchaPresence(fakeDocument([
    createFakeCaptchaNode({
      src: 'https://www.google.com/recaptcha/api2/bframe?k=test-key'
    })
  ])), true);
  assert.equal(detectCaptchaPresence(fakeDocument([])), false);
});

test('scanner caps actionable fields at the shared 50-field limit', () => {
  const specs = Array.from({ length: 70 }, (_, index) => ({
    tagName: 'INPUT',
    type: 'text',
    name: `first_name_${index}`,
    label: `First name ${index}`
  }));
  const doc = createFakeDocument(specs);
  const result = scanDocument({ doc, adapter: fixtureAdapter }).result;
  assert.equal(result.fields.length, 50);
  assert.equal(result.truncated, true);
});

test('sanitized export contains structural actionable data only', () => {
  const doc = createFakeDocument(fixture.fields);
  doc.fields.forEach((field, index) => {
    field.spec = fixture.fields[index];
  });
  const { result } = scanDocument({ doc, url: fixture.url, adapter: fixtureAdapter });
  const exported = createSanitizedExport(result);
  exported.fields.forEach((field) => {
    assert.deepEqual(Object.keys(field).sort(), ['fieldId', 'label', 'options', 'riskClass', 'type']);
    assert.equal(Object.hasOwn(field, 'nearbyText'), false);
    assert.equal(Object.hasOwn(field, 'value'), false);
    assert.ok([RISK_CLASSES.F1_VERIFIED, RISK_CLASSES.F2_REVIEW].includes(field.riskClass));
  });
  assert.equal(JSON.stringify(exported).includes('Gender'), false);
});

test('semantic field fingerprints survive unrelated positional DOM changes', () => {
  const specs = [
    { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' },
    { tagName: 'INPUT', type: 'url', name: 'linkedin', label: 'LinkedIn Profile' }
  ];
  const doc = createFakeDocument(specs);
  doc.fields.forEach((field, index) => {
    field.spec = specs[index];
  });
  const before = scanDocument({ doc, url: fixture.url, adapter: fixtureAdapter });
  const validationNode = new FakeField({
    tagName: 'INPUT',
    type: 'hidden',
    name: 'validation_helper',
    label: 'Validation helper'
  }, doc, 99);
  doc.body.children = [validationNode, ...doc.fields];
  const after = scanDocument({ doc, url: fixture.url, adapter: fixtureAdapter });
  assert.deepEqual(
    after.result.fields.map(field => field.fieldId),
    before.result.fields.map(field => field.fieldId)
  );
});

test('fingerprints are deterministic and structural', () => {
  assert.equal(
    createFieldFingerprint(['applicantpro', 'email', 'email', 'Email address']),
    createFieldFingerprint(['applicantpro', 'email', 'email', 'Email address'])
  );
  assert.notEqual(
    createFieldFingerprint(['applicantpro', 'email', 'email', 'Email address']),
    createFieldFingerprint(['applicantpro', 'tel', 'phone', 'Phone number'])
  );
});

test('runtime message validation rejects unbounded and malformed requests', () => {
  assert.equal(validateRuntimeMessage(envelope('PAGE_SCAN_REQUEST', 'scan-1')).ok, true);
  assert.equal(validateRuntimeMessage(envelope('PAGE_SCAN_REQUEST', 'x'.repeat(129))).error, 'invalid_request_id');
  const validFill = validateRuntimeMessage(envelope('FIELD_FILL_REQUEST', 'fill-1', {
    fieldId: 'field-0123456789abcdef',
    value: 'safe answer',
    confirmed: true,
    skipIfPopulated: true
  }));
  assert.equal(validFill.ok, true);
  assert.equal(validFill.value.payload.skipIfPopulated, true);
  assert.equal(validateRuntimeMessage(envelope('FIELD_FILL_REQUEST', 'fill-invalid-skip', {
    fieldId: 'field-0123456789abcdef',
    value: 'safe answer',
    skipIfPopulated: 'true'
  })).error, 'invalid_skip_if_populated');
  assert.equal(validateRuntimeMessage(envelope('FIELD_FILL_REQUEST', 'fill-2', {
    fieldId: 'not-a-field',
    value: 'safe answer'
  })).error, 'invalid_field_id');
  assert.equal(validateRuntimeMessage(envelope('PAGE_PROPOSALS_UPDATE', 'proposal-1', {
    proposals: [{
      fieldId: 'field-0123456789abcdef',
      value: 'x'.repeat(12_001),
      confidence: 'review',
      risk_class: RISK_CLASSES.F2_REVIEW
    }]
  })).error, 'invalid_answer');
  assert.equal(validateRuntimeMessage(envelope('PAGE_PROPOSALS_UPDATE', 'proposal-2', {
    proposals: [{
      fieldId: 'field-0123456789abcdef',
      value: 'bounded answer',
      confidence: 'certain',
      risk_class: RISK_CLASSES.F2_REVIEW
    }]
  })).error, 'invalid_proposal_metadata');
  assert.equal(validateRuntimeMessage(envelope('PAGE_PROPOSALS_UPDATE', 'proposal-cap', {
    proposals: Array.from({ length: 51 }, () => ({
      fieldId: 'field-0123456789abcdef',
      value: 'bounded answer',
      confidence: 'review',
      risk_class: RISK_CLASSES.F2_REVIEW
    }))
  })).error, 'invalid_proposals');
  const directModelProposal = validateRuntimeMessage(envelope('PAGE_PROPOSALS_UPDATE', 'proposal-3', {
    proposals: [{
      field_id: 'field-0123456789abcdef',
      action: 'fill',
      value_type: 'selected_values',
      value: '',
      selected_values: ['Yes'],
      checked: false,
      confidence: 'review',
      risk_class: RISK_CLASSES.F2_REVIEW,
      citation_ids: ['source-1', 'source-2']
    }]
  }));
  assert.equal(directModelProposal.ok, true);
  assert.deepEqual(directModelProposal.value.payload.proposals[0].value, ['Yes']);
  assert.equal(directModelProposal.value.payload.proposals[0].citationCount, 2);
});

test('active CAPTCHA challenge blocks fill before setters or events', async () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', name: 'first_name', label: 'First name' }
  ], { captcha: true });
  const runtimeApi = {
    id: 'extension-id',
    sendMessage() {
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'captcha-scan'));
  const field = scanResponse.payload.fields[0];
  const fillResponse = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'captcha-fill', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: 'Daniel',
    confirmed: true
  }));
  assert.equal(fillResponse.payload.status, 'copy_only');
  assert.equal(fillResponse.payload.reason, 'captcha_present');
  assert.equal(doc.fields[0]._value, '');
  assert.deepEqual(doc.fields[0].events, []);
  runtime.destroy();
});

test('passive invisible CAPTCHA badge allows an approved ordinary field fill', async () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', name: 'first_name', label: 'First name' }
  ]);
  doc.captchaNodes = [
    createFakeCaptchaNode({ tagName: 'SCRIPT', src: 'https://www.google.com/recaptcha/api.js?render=test-key' }),
    createFakeCaptchaNode({
      src: 'https://www.recaptcha.net/recaptcha/enterprise/anchor?k=test-key&size=invisible',
      insideBadge: true
    }),
    createFakeCaptchaNode({ tagName: 'TEXTAREA', name: 'g-recaptcha-response', visible: false })
  ];
  const runtimeApi = {
    id: 'extension-id',
    sendMessage() {
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'passive-captcha-scan'));
  assert.equal(scanResponse.payload.captchaPresent, false);
  const field = scanResponse.payload.fields[0];
  const fillResponse = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'passive-captcha-fill', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: 'Daniel',
    confirmed: true
  }));
  assert.equal(fillResponse.payload.status, 'filled');
  assert.equal(fillResponse.payload.verified, true);
  assert.equal(doc.fields[0]._value, 'Daniel');
  assert.deepEqual(doc.fields[0].events, ['input', 'change', 'blur']);
  runtime.destroy();
});

test('native fill uses input, change, and blur without click or submit', async () => {
  const doc = createFakeDocument([{ tagName: 'INPUT', type: 'text', label: 'First name' }]);
  const element = doc.fields[0];
  const record = {
    descriptor: { riskClass: RISK_CLASSES.F1_VERIFIED },
    elements: [element],
    optionRecords: [],
    type: 'text'
  };
  const result = await fillFieldRecord(record, 'Daniel');
  assert.equal(result.attempted, true);
  assert.equal(element.value, 'Daniel');
  assert.deepEqual(element.events, ['input', 'change', 'blur']);
  assert.equal(element.events.includes('click'), false);
  assert.equal(element.events.includes('submit'), false);
});

test('native fill bypasses a React-style own value tracker setter', async () => {
  const doc = createFakeDocument([{ tagName: 'INPUT', type: 'url', label: 'LinkedIn Profile' }]);
  const element = doc.fields[0];
  let ownSetterCalls = 0;
  Object.defineProperty(element, 'value', {
    configurable: true,
    get() {
      return this._value;
    },
    set(value) {
      ownSetterCalls += 1;
      this._value = `tracked:${value}`;
    }
  });
  const result = await fillFieldRecord({
    descriptor: { riskClass: RISK_CLASSES.F1_VERIFIED },
    elements: [element],
    optionRecords: [],
    type: 'url'
  }, 'https://www.linkedin.com/in/example');
  assert.equal(result.attempted, true);
  assert.equal(ownSetterCalls, 0);
  assert.equal(element.value, 'https://www.linkedin.com/in/example');
  assert.deepEqual(element.events, ['input', 'change', 'blur']);

});
test('one fill request retries a transient controlled-field reset exactly once', async () => {
  const doc = createFakeDocument([{
    tagName: 'INPUT',
    type: 'url',
    name: 'linkedin',
    label: 'LinkedIn Profile',
    resetInputAttempts: 1
  }]);
  const runtimeApi = {
    id: 'extension-id',
    sendMessage() {
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'transient-reset-scan'));
  const field = scanResponse.payload.fields[0];
  const fillResponse = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'transient-reset-fill', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: 'https://www.linkedin.com/in/example'
  }));
  assert.equal(fillResponse.payload.status, 'filled');
  assert.equal(fillResponse.payload.verified, true);
  assert.equal(fillResponse.payload.reason, 'verified_after_retry');
  assert.equal(doc.fields[0].value, 'https://www.linkedin.com/in/example');
  assert.deepEqual(doc.fields[0].events, [
    'input', 'change', 'blur',
    'input', 'change', 'blur'
  ]);
  assert.equal(doc.fields[0].events.includes('click'), false);
  assert.equal(doc.fields[0].events.includes('submit'), false);
  runtime.destroy();
});

test('fieldRecordHasValue detects entered text, meaningful selections, and checked controls', () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', label: 'First name' },
    { tagName: 'DIV', type: 'contenteditable', label: 'Summary' },
    { tagName: 'SELECT', type: 'select-one', label: 'State', options: ['Select', 'Colorado'] },
    { tagName: 'SELECT', type: 'select-multiple', label: 'Skills', options: ['SQL', 'Python'] },
    { tagName: 'INPUT', type: 'checkbox', label: 'Remote' }
  ]);
  const [textField, contenteditable, stateSelect, skillsSelect, checkbox] = doc.fields;
  const record = (element, type, optionRecords = []) => ({
    descriptor: { riskClass: RISK_CLASSES.F1_VERIFIED },
    elements: [element],
    optionRecords,
    type
  });

  assert.equal(fieldRecordHasValue(record(textField, 'text')), false);
  textField.value = 'Daniel';
  assert.equal(fieldRecordHasValue(record(textField, 'text')), true);

  assert.equal(fieldRecordHasValue(record(contenteditable, 'contenteditable')), false);
  contenteditable.textContent = 'Evidence-backed summary';
  assert.equal(fieldRecordHasValue(record(contenteditable, 'contenteditable')), true);

  const stateOptions = stateSelect.options.map((node) => ({ label: node.label, node }));
  assert.equal(fieldRecordHasValue(record(stateSelect, 'select-one', stateOptions)), false);
  stateSelect.selectedIndex = 1;
  assert.equal(fieldRecordHasValue(record(stateSelect, 'select-one', stateOptions)), true);

  const skillOptions = skillsSelect.options.map((node) => ({ label: node.label, node }));
  assert.equal(fieldRecordHasValue(record(skillsSelect, 'select-multiple', skillOptions)), false);
  skillOptions[0].node.selected = true;
  assert.equal(fieldRecordHasValue(record(skillsSelect, 'select-multiple', skillOptions)), true);

  assert.equal(fieldRecordHasValue(record(checkbox, 'checkbox')), false);
  checkbox.checked = true;
  assert.equal(fieldRecordHasValue(record(checkbox, 'checkbox')), true);
});

test('batch-safe fill skips an occupied field without mutation while ordinary fill can replace it', async () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', name: 'first_name', label: 'First name' }
  ]);
  doc.fields[0].value = 'Existing answer';
  const runtimeApi = {
    id: 'extension-id',
    sendMessage() {
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'occupied-scan'));
  const field = scanResponse.payload.fields[0];
  const skipped = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'occupied-skip', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: 'Replacement answer',
    skipIfPopulated: true
  }));
  assert.equal(skipped.payload.status, 'skipped');
  assert.equal(skipped.payload.reason, 'field_already_has_value');
  assert.equal(skipped.payload.verified, false);
  assert.equal(skipped.payload.copyOnly, false);
  assert.equal(doc.fields[0].value, 'Existing answer');
  assert.deepEqual(doc.fields[0].events, []);

  const replaced = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'occupied-replace', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: 'Replacement answer'
  }));
  assert.equal(replaced.payload.status, 'filled');
  assert.equal(replaced.payload.verified, true);
  assert.equal(doc.fields[0].value, 'Replacement answer');
  assert.deepEqual(doc.fields[0].events, ['input', 'change', 'blur']);
  runtime.destroy();
});

test('page runtime re-scans, requires F2 confirmation, and falls back to copy-only', async () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', name: 'first_name', label: 'First name', resetOnInput: true },
    { tagName: 'TEXTAREA', type: 'textarea', name: 'interest', label: 'Why are you interested?' }
  ]);
  let scanCount = 0;
  const originalQuerySelectorAll = doc.querySelectorAll.bind(doc);
  doc.querySelectorAll = (...args) => {
    scanCount += 1;
    return originalQuerySelectorAll(...args);
  };
  const sent = [];
  const runtimeApi = {
    id: 'extension-id',
    sendMessage(message) {
      sent.push(message);
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'scan-runtime'));
  const firstName = scanResponse.payload.fields.find((field) => field.label === 'First name');
  const prose = scanResponse.payload.fields.find((field) => field.label.includes('interested'));
  const rejectedProposal = await runtime.handleMessage(envelope('PAGE_PROPOSALS_UPDATE', 'proposal-risk-mismatch', {
    proposals: [{
      fieldId: firstName.fieldId,
      value: 'Daniel',
      confidence: 'high',
      risk_class: RISK_CLASSES.F2_REVIEW,
      confirmed: true
    }]
  }));
  assert.equal(rejectedProposal.payload.acceptedCount, 0);
  const acceptedProposal = await runtime.handleMessage(envelope('PAGE_PROPOSALS_UPDATE', 'proposal-risk-match', {
    proposals: [{
      fieldId: firstName.fieldId,
      value: 'Daniel',
      confidence: 'high',
      risk_class: RISK_CLASSES.F1_VERIFIED,
      confirmed: true
    }]
  }));
  assert.equal(acceptedProposal.payload.acceptedCount, 1);
  const failedControlled = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'fill-controlled', {
    fieldId: firstName.fieldId,
    fingerprint: firstName.fingerprint,
    value: 'Daniel'
  }));
  assert.equal(failedControlled.payload.status, 'copy_only');
  assert.equal(failedControlled.payload.verified, false);
  assert.deepEqual(doc.fields[0].events, [
    'input', 'change', 'blur',
    'input', 'change', 'blur'
  ]);
  const unconfirmed = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'fill-review', {
    fieldId: prose.fieldId,
    fingerprint: prose.fingerprint,
    value: 'Evidence-backed response',
    confirmed: false
  }));
  assert.equal(unconfirmed.payload.reason, 'review_confirmation_required');
  assert.ok(scanCount >= 3, 'each fill request should re-scan before acting');
  assert.equal(doc.fields.some((field) => field.events.includes('submit')), false);
  runtime.destroy();
});

test('cosmetic validation mutations preserve proposals while real form changes increment domRevision', async () => {
  class FakeMutationObserver {
    static latest = null;

    constructor(callback) {
      this.callback = callback;
      FakeMutationObserver.latest = this;
    }

    observe() {}

    disconnect() {}
  }

  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' }
  ]);
  doc.defaultView.MutationObserver = FakeMutationObserver;
  const sent = [];
  const runtimeApi = {
    id: 'extension-id',
    sendMessage(message) {
      sent.push(message);
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const initial = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'revision-initial'));
  assert.equal(initial.payload.domRevision, 0);
  FakeMutationObserver.latest.callback([{
    type: 'attributes',
    target: doc.fields[0],
    addedNodes: [],
    removedNodes: []
  }]);
  await new Promise(resolve => setTimeout(resolve, 100));
  const cosmetic = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'revision-cosmetic'));
  assert.equal(cosmetic.payload.domRevision, 0);
  assert.equal(sent.length, 0);
  const added = new FakeField({ tagName: 'INPUT', type: 'url', name: 'linkedin', label: 'LinkedIn Profile' }, doc, 1);
  doc.fields.push(added);
  doc.body.children = doc.fields;
  FakeMutationObserver.latest.callback([{
    type: 'childList',
    target: doc.body,
    addedNodes: [added],
    removedNodes: []
  }]);
  await new Promise(resolve => setTimeout(resolve, 100));
  const changed = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'revision-structural'));
  assert.equal(changed.payload.domRevision, 1);
  assert.equal(changed.payload.fields.length, 2);
  assert.equal(sent.some(message => message.type === 'PAGE_SCAN_RESULT' && message.payload.stale), true);
  runtime.destroy();
});
test('free-format safety excludes honeypots and near-zero fields', () => {
  assert.equal(classifyFieldMetadata({
    type: 'text',
    label: 'Website',
    nearbyText: 'Leave this field blank. Robots only.'
  }), RISK_CLASSES.F0_EXCLUDED);
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', name: 'first_name', label: 'First name', width: 1, height: 1 },
    { tagName: 'INPUT', type: 'url', name: 'robots_only', label: 'Website - robots only' },
    { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' }
  ], { url: 'https://custom.example/apply' });
  const result = scanDocument({ doc, adapter: fixtureAdapter }).result;
  assert.deepEqual(result.fields.map((field) => field.label), ['Email address']);
  assert.equal(result.exclusionCounts.F0_EXCLUDED, 2);
});

test('zero-height passive hCaptcha containers do not pause while visible challenges do', () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'text', name: 'first_name', label: 'First name' }
  ], { url: 'https://custom.example/apply', application: true });
  doc.captchaNodes = [createFakeCaptchaNode({
    src: 'https://newassets.hcaptcha.com/captcha/v1/anchor.html',
    width: 300,
    height: 0
  })];
  assert.equal(detectCaptchaPresence(doc), false);
  const passiveScan = scanDocument({ doc, url: doc.location.href }).result;
  assert.equal(passiveScan.captchaPresent, false);
  assert.equal(passiveScan.discovery.mode, 'free_format');
  assert.equal(passiveScan.discovery.unsupportedCount, 0);
  doc.captchaNodes = [createFakeCaptchaNode({
    src: 'https://js.hcaptcha.com/1/api.js?frame=challenge',
    width: 300,
    height: 70
  })];
  assert.equal(detectCaptchaPresence(doc), true);
});

test('whole-document discovery reports a likely cross-origin application frame without a form root', () => {
  const doc = createFakeDocument([], { url: 'https://careers.example.test/openings/1' });
  const applicationFrame = createFakeCaptchaNode({
    src: 'https://apply.vendor.example/application/123',
    title: 'Job application',
    ariaLabel: 'Candidate application'
  });
  const unrelatedFrame = createFakeCaptchaNode({
    src: 'https://video.example/embed/123',
    title: 'Company culture video'
  });
  doc.iframes = [applicationFrame, unrelatedFrame];
  doc.body.children = [...doc.fields, ...doc.iframes];
  const result = scanDocument({ doc, url: doc.location.href }).result;
  assert.equal(result.adapter, 'generic');
  assert.deepEqual(result.fields, []);
  assert.equal(result.discovery.mode, 'limited');
  assert.equal(result.discovery.unsupportedCount, 1);
  assert.deepEqual(result.discovery.contexts, [{
    kind: 'cross_origin_iframe',
    count: 1,
    status: 'unsupported'
  }]);
});

test('closed-shadow diagnostics require an explicit adapter declaration', () => {
  const doc = createFakeDocument([], { url: 'https://careers.example.test/openings/2' });
  const genericResult = scanDocument({ doc, url: doc.location.href }).result;
  assert.equal(genericResult.discovery.contexts.some(
    (context) => context.kind === 'closed_shadow_root'), false);
  const adapter = {
    id: 'known-closed-component',
    getApplicationRoot: () => null,
    getDiscoveryContexts: () => [{
      kind: 'closed_shadow_root',
      count: 1,
      status: 'unsupported'
    }],
    collectCandidates: () => [],
    getSupplementalLabel: () => '',
    getNearbyText: () => '',
    extractJobMetadata: () => ({ source: 'Known component' })
  };
  const result = scanDocument({ doc, adapter }).result;
  assert.deepEqual(result.discovery.contexts, [{
    kind: 'closed_shadow_root',
    count: 1,
    status: 'unsupported'
  }]);
  assert.equal(result.discovery.unsupportedCount, 1);
});

test('free-format discovery recurses through open shadow roots and resolves local aria labels', async () => {
  const doc = createFakeDocument([], { url: 'https://custom.example/apply', application: true });
  const outer = attachOpenShadowHost(doc, { id: 'application-shell' });
  const inner = attachOpenShadowHost(doc, {
    id: 'question-panel',
    parentRoot: outer.shadowRoot,
    fieldSpecs: [{
      tagName: 'TEXTAREA',
      type: 'textarea',
      name: 'project_answer',
      id: 'project-answer',
      label: 'What project are you most proud of?',
      useAriaLabel: true
    }]
  });
  const before = scanDocument({ doc, url: doc.location.href });
  assert.equal(before.result.adapter, 'generic');
  assert.equal(before.result.discovery.mode, 'free_format');
  assert.equal(before.result.fields.length, 1);
  assert.equal(before.result.fields[0].label, 'What project are you most proud of?');
  assert.equal(before.result.fields[0].riskClass, RISK_CLASSES.F2_REVIEW);
  assert.equal(before.records.get(before.result.fields[0].fieldId).elements[0], inner.fields[0]);

  doc.fields.unshift(new FakeField({
    tagName: 'INPUT',
    type: 'hidden',
    name: 'validation_helper',
    label: 'Validation helper'
  }, doc, 99));
  doc.body.children = [...doc.fields, ...doc.hosts];
  const after = scanDocument({ doc, url: doc.location.href });
  assert.equal(after.result.fields[0].fieldId, before.result.fields[0].fieldId);

  const runtimeApi = {
    id: 'extension-id',
    sendMessage() {
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'shadow-scan'));
  const fillResponse = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'shadow-fill', {
    fieldId: scanResponse.payload.fields[0].fieldId,
    fingerprint: scanResponse.payload.fields[0].fingerprint,
    value: 'A source-cited project response.',
    confirmed: true
  }));
  assert.equal(fillResponse.payload.status, 'filled');
  assert.equal(inner.fields[0].value, 'A source-cited project response.');
  runtime.destroy();
});


test('shadow discovery prioritizes nested field components within the root budget', () => {
  const doc = createFakeDocument([], { url: 'https://custom.example/apply', application: true });
  const phone = attachOpenShadowHost(doc, { id: 'phone-field' });
  for (let index = 0; index < 70; index += 1) {
    attachOpenShadowHost(doc, {
      id: `decoration-${index}`,
      parentRoot: phone.shadowRoot
    });
  }
  const nested = attachOpenShadowHost(doc, {
    id: 'phone-input',
    parentRoot: phone.shadowRoot,
    fieldSpecs: [{
      tagName: 'INPUT',
      type: 'tel',
      name: 'phone',
      label: 'Phone number'
    }]
  });
  const result = scanDocument({ doc, url: doc.location.href }).result;
  assert.equal(result.fields.some(field => field.label === 'Phone number'), true);
  assert.equal(result.fields.length, 1);
});
test('shadow host identity prevents collisions and the extension overlay root is never traversed', () => {
  const doc = createFakeDocument([], { url: 'https://custom.example/apply', application: true });
  attachOpenShadowHost(doc, {
    id: 'question-one',
    fieldSpecs: [{ tagName: 'INPUT', type: 'text', name: 'first_name', id: 'first-name', label: 'First name' }]
  });
  attachOpenShadowHost(doc, {
    id: 'question-two',
    fieldSpecs: [{ tagName: 'INPUT', type: 'text', name: 'first_name', id: 'first-name', label: 'First name' }]
  });
  attachOpenShadowHost(doc, {
    id: '__job_application_copilot_runtime_v1__',
    overlay: true,
    fieldSpecs: [{ tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' }]
  });
  const result = scanDocument({ doc, url: doc.location.href }).result;
  assert.equal(result.fields.length, 2);
  assert.notEqual(result.fields[0].fingerprint, result.fields[1].fingerprint);
  assert.equal(result.fields.some((field) => field.label === 'Email address'), false);
});

test('custom ARIA widgets are generated but remain copy-only without DOM mutation', async () => {
  const doc = createFakeDocument([{
    tagName: 'INPUT',
    type: 'text',
    name: 'location',
    label: 'Current location',
    role: 'combobox',
    ariaAutocomplete: 'list'
  }], { url: 'https://custom.example/apply', application: true });
  const runtimeApi = {
    id: 'extension-id',
    sendMessage() {
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const scanResponse = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'custom-widget-scan'));
  const field = scanResponse.payload.fields[0];
  assert.equal(field.fillMode, 'copy_only');
  assert.deepEqual(scanResponse.payload.discovery.contexts, [{
    kind: 'custom_aria_widget',
    count: 1,
    status: 'manual'
  }]);
  const exported = createSanitizedExport(scanResponse.payload);
  assert.equal(exported.fields[0].fillMode, 'copy_only');
  assert.equal(exported.discovery.recognizedCount, 1);
  const fillResponse = await runtime.handleMessage(envelope('FIELD_FILL_REQUEST', 'custom-widget-fill', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: 'Denver, Colorado',
    confirmed: true
  }));
  assert.equal(fillResponse.payload.status, 'copy_only');
  assert.equal(fillResponse.payload.reason, 'custom_widget_copy_only');
  assert.equal(fillResponse.payload.verified, false);
  assert.equal(doc.fields[0].value, '');
  assert.deepEqual(doc.fields[0].events, []);
  runtime.destroy();
});

test('an authoritative null adapter root stays limited and never scans the surrounding page', () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' }
  ], { url: 'https://workday.example/account' });
  const adapter = {
    id: 'workday',
    getApplicationRoot: () => null,
    getDiscoveryContexts: () => [{ kind: 'account_gate', root: doc.body }],
    collectCandidates: () => doc.fields,
    getSupplementalLabel: () => '',
    getNearbyText: () => '',
    extractJobMetadata: () => ({ source: 'Workday' })
  };
  const result = scanDocument({ doc, adapter }).result;
  assert.deepEqual(result.fields, []);
  assert.equal(result.discovery.mode, 'limited');
  assert.equal(result.discovery.recognizedCount, 0);
  assert.deepEqual(result.discovery.contexts, [{ kind: 'account_gate', count: 1, status: 'manual' }]);
  assert.deepEqual(Object.keys(createSanitizedExport(result).discovery).sort(), [
    'contexts',
    'exclusionCounts',
    'mode',
    'recognizedCount',
    'truncated',
    'unsupportedCount'
  ]);
});

test('declared account gates suppress otherwise actionable adapter candidates', () => {
  const doc = createFakeDocument([
    { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' }
  ], { url: 'https://custom.example/sign-in', application: true });
  const adapter = {
    id: 'generic',
    getApplicationRoot: () => doc,
    getDiscoveryContexts: () => [{ kind: 'account_gate', root: doc }],
    collectCandidates: () => doc.fields,
    getSupplementalLabel: () => '',
    getNearbyText: () => '',
    extractJobMetadata: () => ({ source: 'Custom application' })
  };
  const scan = scanDocument({ doc, adapter });
  assert.deepEqual(scan.result.fields, []);
  assert.equal(scan.records.size, 0);
  assert.equal(scan.result.discovery.mode, 'limited');
  assert.deepEqual(scan.result.discovery.contexts, [{ kind: 'account_gate', count: 1, status: 'manual' }]);
});
test('shadow roots added by an SPA are observed and invalidate stale proposals', async () => {
  class FakeMutationObserver {
    static latest = null;

    constructor(callback) {
      this.callback = callback;
      this.observed = [];
      this.disconnected = false;
      FakeMutationObserver.latest = this;
    }

    observe(root) {
      this.observed.push(root);
    }

    disconnect() {
      this.disconnected = true;
    }
  }

  const doc = createFakeDocument([], { url: 'https://custom.example/apply', application: true });
  doc.defaultView.MutationObserver = FakeMutationObserver;
  const sent = [];
  const runtimeApi = {
    id: 'extension-id',
    sendMessage(message) {
      sent.push(message);
      return Promise.resolve();
    },
    onMessage: {
      addListener() {},
      removeListener() {}
    }
  };
  const runtime = createPageRuntime({ doc, view: doc.defaultView, runtimeApi, settle: async () => {} });
  const initial = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'shadow-mutation-initial'));
  assert.equal(initial.payload.fields.length, 0);

  const added = attachOpenShadowHost(doc, {
    id: 'lazy-question-panel',
    fieldSpecs: [{ tagName: 'TEXTAREA', type: 'textarea', name: 'project', label: 'What project are you most proud of?' }]
  });
  FakeMutationObserver.latest.callback([{
    type: 'childList',
    target: doc.body,
    addedNodes: [added.host],
    removedNodes: []
  }]);
  await new Promise((resolve) => setTimeout(resolve, 110));
  const afterHost = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'shadow-mutation-host'));
  assert.equal(afterHost.payload.domRevision, 1);
  assert.equal(afterHost.payload.fields.length, 1);
  assert.equal(FakeMutationObserver.latest.observed.includes(added.shadowRoot), true);

  const secondField = appendFakeShadowField(added.shadowRoot, doc, {
    tagName: 'TEXTAREA',
    type: 'textarea',
    name: 'interest',
    label: 'Why are you interested in this role?'
  });
  FakeMutationObserver.latest.callback([{
    type: 'childList',
    target: added.shadowRoot,
    addedNodes: [secondField],
    removedNodes: []
  }]);
  await new Promise((resolve) => setTimeout(resolve, 110));
  const afterField = await runtime.handleMessage(envelope('PAGE_SCAN_REQUEST', 'shadow-mutation-field'));
  assert.equal(afterField.payload.domRevision, 2);
  assert.equal(afterField.payload.fields.length, 2);
  assert.equal(sent.some((message) => message.type === 'PAGE_SCAN_RESULT' && message.payload.stale), true);
  runtime.destroy();
  assert.equal(FakeMutationObserver.latest.disconnected, true);
});

test('same-origin application frames are scanned, filled, and included in CAPTCHA checks', async () => {
  const outer = createFakeDocument([], {
    url: 'https://custom.example/jobs/embedded',
    application: false
  });
  const inner = createFakeDocument([
    { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' },
    { tagName: 'TEXTAREA', type: 'textarea', name: 'interest', label: 'Why are you interested?' }
  ], {
    url: 'https://custom.example/jobs/embedded/application',
    application: true
  });
  const frame = createFakeCaptchaNode({
    src: inner.location.href,
    title: 'Candidate application'
  });
  frame.contentDocument = inner;
  frame.ownerDocument = outer;
  outer.iframes = [frame];
  outer.body.children = [frame];

  const scan = scanDocument({ doc: outer, url: outer.location.href });
  assert.equal(scan.result.discovery.unsupportedCount, 0);
  assert.equal(scan.result.discovery.mode, 'free_format');
  assert.deepEqual(scan.result.fields.map((field) => field.label), [
    'Email address',
    'Why are you interested?'
  ]);
  const email = scan.result.fields[0];
  const fill = await fillFieldRecord(scan.records.get(email.fieldId), 'candidate@example.test');
  assert.equal(fill.attempted, true);
  assert.equal(inner.fields[0].value, 'candidate@example.test');

  inner.captchaNodes = [createFakeCaptchaNode({
    src: 'https://www.google.com/recaptcha/api2/bframe?k=test-key',
    visible: true
  })];
  assert.equal(scanDocument({ doc: outer, url: outer.location.href }).result.captchaPresent, true);
});

test('bounded Yes and No button questions are proposed as copy-only without treating Submit as a field', async () => {
  const questionContainer = {};
  const doc = createFakeDocument([
    {
      tagName: 'BUTTON',
      type: 'button',
      name: 'authorization_yes',
      label: 'Yes',
      question: 'Are you authorized to work in this country?',
      container: questionContainer
    },
    {
      tagName: 'BUTTON',
      type: 'button',
      name: 'authorization_no',
      label: 'No',
      question: 'Are you authorized to work in this country?',
      container: questionContainer
    },
    {
      tagName: 'BUTTON',
      type: 'submit',
      name: 'submit',
      label: 'Submit application',
      container: {}
    }
  ], {
    url: 'https://jobs.ashbyhq.com/example/application',
    application: true
  });
  doc.fields[0].textContent = 'Yes';
  doc.fields[1].textContent = 'No';
  doc.fields[2].textContent = 'Submit application';

  const adapter = {
    id: 'ashby',
    getApplicationRoot: () => doc,
    getDiscoveryContexts: () => [{ kind: 'application', root: doc }],
    collectCandidates: () => doc.fields,
    getFieldContainer: (element) => element.spec.container,
    getSupplementalLabel: (element) => element.spec.question || '',
    getNearbyText: () => '',
    extractJobMetadata: () => ({ source: 'Ashby' })
  };
  const scan = scanDocument({ doc, adapter });
  assert.equal(scan.result.fields.length, 1);
  assert.equal(scan.result.fields[0].label, 'Are you authorized to work in this country?');
  assert.equal(scan.result.fields[0].type, 'select-one');
  assert.deepEqual(scan.result.fields[0].options, ['Yes', 'No']);
  assert.equal(scan.result.fields[0].fillMode, 'copy_only');
  const record = scan.records.get(scan.result.fields[0].fieldId);
  const fill = await fillFieldRecord(record, 'Yes');
  assert.deepEqual(fill, { attempted: false, reason: 'custom_widget_copy_only' });
  assert.deepEqual(doc.fields.flatMap((field) => field.events), []);
});
