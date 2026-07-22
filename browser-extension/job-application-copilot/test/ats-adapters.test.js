import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { ashbyAdapter } from '../src/adapters/ashby.js';
import { genericAdapter, selectGenericApplicationRoot } from '../src/adapters/generic.js';
import { greenhouseAdapter, isGreenhousePage } from '../src/adapters/greenhouse.js';
import { selectAdapter } from '../src/adapters/index.js';
import { leverAdapter } from '../src/adapters/lever.js';
import { smartRecruitersAdapter } from '../src/adapters/smartrecruiters.js';
import { workdayAdapter } from '../src/adapters/workday.js';

const readFixture = async (name) => JSON.parse(await readFile(
  new URL(`./fixtures/${name}`, import.meta.url),
  'utf8'
));

const fixtures = {
  additionalAts: await readFixture('additional-ats-structures.json'),
  lever: await readFixture('lever-live-structure.json'),
  ashby: await readFixture('ashby-live-structure.json'),
  custom: await readFixture('custom-application-structure.json'),
  greenhouse: await readFixture('greenhouse-live-structure.json'),
  smartRecruiters: await readFixture('smartrecruiters-live-structure.json'),
  workday: await readFixture('workday-live-structure.json')
};

const candidateTags = new Set(['input', 'textarea', 'select']);

const createField = (spec) => {
  const labels = spec.label ? [{ textContent: spec.label }] : [];
  const attributes = {
    ...spec,
    class: spec.className || '',
    type: spec.type || ''
  };
  return {
    ...spec,
    labels,
    getAttribute(name) {
      return attributes[name] || '';
    }
  };
};

const createRoot = (spec) => {
  const fields = (spec.fields || []).map(createField);
  const attributes = {
    ...spec,
    class: spec.className || ''
  };
  return {
    ...spec,
    fields,
    getAttribute(name) {
      return attributes[name] || '';
    },
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    },
    querySelectorAll(selector) {
      if (selector === 'form') return [];
      if ([...candidateTags].some((tag) => selector.includes(tag))) {
        return fields.filter((field) => candidateTags.has(String(field.tagName || '').toLowerCase()));
      }
      return [];
    }
  };
};

const createDocument = (rootSpecs = [], metadata = {}) => {
  const roots = rootSpecs.map(createRoot);
  return {
    title: metadata.title || '',
    roots,
    querySelector(selector) {
      const metadataSelector = selector.split(',')
        .map((part) => part.trim())
        .find((part) => metadata[part] !== undefined);
      if (metadataSelector) {
        return {
          textContent: metadata[metadataSelector],
          getAttribute(name) {
            return name === 'content' ? metadata[metadataSelector] : '';
          }
        };
      }
      return this.querySelectorAll(selector)[0] || null;
    },
    querySelectorAll(selector) {
      if (selector === 'form') {
        return roots.filter((root) => String(root.tagName || '').toLowerCase() === 'form');
      }
      return roots.filter((root) => root.selector === selector);
    }
  };
};

const names = (controls) => controls.map((control) => control.name);

test('dedicated ATS adapters win only on their stable hosts', () => {
  assert.equal(selectAdapter({ doc: createDocument(), url: fixtures.greenhouse.url }).id, 'greenhouse');
  assert.equal(selectAdapter({ doc: createDocument(), url: fixtures.lever.url }).id, 'lever');
  assert.equal(selectAdapter({ doc: createDocument(), url: fixtures.ashby.url }).id, 'ashby');
  assert.equal(selectAdapter({ doc: createDocument(), url: fixtures.smartRecruiters.url }).id, 'smartrecruiters');
  assert.equal(selectAdapter({ doc: createDocument(), url: fixtures.workday.url }).id, 'workday');
  assert.equal(selectAdapter({ doc: createDocument(), url: 'https://jobs.lever.co.example.test/apply' }).id, 'generic');
  assert.equal(selectAdapter({ doc: createDocument(), url: 'https://job-boards.greenhouse.io.example.test/jobs/1' }).id, 'generic');
  assert.equal(selectAdapter({ doc: createDocument(), url: 'https://myworkdayjobs.com.example.test/apply' }).id, 'generic');
});

test('Greenhouse supports hosted boards and structural custom-domain applications', () => {
  const hostedDoc = createDocument([fixtures.greenhouse.applicationForm]);
  assert.equal(isGreenhousePage({ doc: hostedDoc, url: fixtures.greenhouse.url }), true);
  assert.equal(greenhouseAdapter.getApplicationRoot(hostedDoc).id, 'application-form');
  const customApplicationForm = {
    ...fixtures.greenhouse.applicationForm,
    selector: 'form[action*="greenhouse.io"][action*="application"]'
  };
  const customDoc = createDocument([customApplicationForm]);
  assert.equal(isGreenhousePage({ doc: customDoc, url: 'https://careers.example.test/openings/1' }), true);
  const embeddedDoc = createDocument([{
    selector: 'iframe[src*="greenhouse.io"][src*="application"]',
    tagName: 'IFRAME'
  }]);
  assert.equal(isGreenhousePage({ doc: embeddedDoc, url: 'https://careers.example.test/openings/2' }), true);
  assert.equal(selectAdapter({ doc: customDoc, url: 'https://careers.example.test/openings/1' }).id, 'greenhouse');
  assert.deepEqual(names(greenhouseAdapter.collectCandidates(customDoc)),
    ['first_name', 'last_name', 'email', 'urls[LinkedIn]', 'motivation', 'sponsorship']);
});

test('free-format discovery selects the candidate application over auxiliary forms', () => {
  const doc = createDocument([
    fixtures.custom.applicationForm,
    ...fixtures.custom.auxiliaryForms
  ]);
  const root = selectGenericApplicationRoot(doc);
  assert.equal(root.id, 'application-form');
  assert.deepEqual(names(genericAdapter.collectCandidates(doc)), [
    'candidate_first_name',
    'candidate_last_name',
    'candidate_email',
    'work_samples',
    'relevant_experience',
    'resume'
  ]);
  assert.equal(genericAdapter.getDiscoveryContexts(doc)[0].kind, 'application');
});

test('free-format discovery rejects search, referral, and newsletter-only pages', () => {
  const doc = createDocument(fixtures.custom.auxiliaryForms);
  assert.equal(selectGenericApplicationRoot(doc), null);
  assert.deepEqual(genericAdapter.collectCandidates(doc), []);
  assert.deepEqual(genericAdapter.getDiscoveryContexts(doc), []);
});

test('free-format discovery supports a custom form-less SPA application root', () => {
  const doc = createDocument([fixtures.custom.formlessApplicationRoot]);
  const root = selectGenericApplicationRoot(doc);
  assert.equal(root.selector, '[data-application-form]');
  assert.deepEqual(names(genericAdapter.collectCandidates(doc)), [
    'fullName',
    'email',
    'motivation'
  ]);
});

test('free-format discovery rejects a form-less account surface', () => {
  const email = createField({
    tagName: 'INPUT',
    type: 'email',
    name: 'email',
    label: 'Email address'
  });
  const doc = {
    querySelector() {
      return null;
    },
    querySelectorAll(selector) {
      if (selector === 'form') return [];
      if (selector.includes('input')) return [email];
      return [];
    }
  };
  assert.equal(selectGenericApplicationRoot(doc), null);
  assert.deepEqual(genericAdapter.collectCandidates(doc), []);
  assert.deepEqual(genericAdapter.getDiscoveryContexts(doc), []);
});

test('Lever scopes collection to the application form and drops files, consent, and submit controls', () => {
  const doc = createDocument([
    fixtures.lever.applicationForm,
    ...fixtures.lever.auxiliaryForms
  ]);
  const root = leverAdapter.getApplicationRoot(doc);
  assert.equal(root.id, 'application-form');
  assert.deepEqual(names(leverAdapter.collectCandidates(doc)), [
    'name',
    'email',
    'phone',
    'urls[LinkedIn]',
    'salary_expectation',
    'work_eligibility'
  ]);
  assert.deepEqual(leverAdapter.getDiscoveryContexts(doc).map(({ kind }) => kind), ['application']);
});

test('Lever does not fall back to a job-alert form when no application form is present', () => {
  const doc = createDocument(fixtures.lever.auxiliaryForms);
  assert.equal(leverAdapter.getApplicationRoot(doc), null);
  assert.deepEqual(leverAdapter.collectCandidates(doc), []);
  assert.deepEqual(leverAdapter.getDiscoveryContexts(doc), []);
});

test('Ashby keeps safe prose questions but excludes CAPTCHA, files, consent, and auxiliary forms', () => {
  const doc = createDocument([
    fixtures.ashby.applicationForm,
    ...fixtures.ashby.auxiliaryForms
  ]);
  assert.deepEqual(names(ashbyAdapter.collectCandidates(doc)), [
    '_systemfield_name',
    '_systemfield_email',
    '_systemfield_phone',
    'project_proud_of',
    'sponsorship'
  ]);
  assert.equal(ashbyAdapter.getDiscoveryContexts(doc)[0].kind, 'application');
});


test('Ashby recognizes its current form-less React application root', () => {
  const rootSpec = { ...fixtures.ashby.applicationForm, selector: '#form', tagName: 'DIV', id: 'form' };
  const doc = createDocument([rootSpec]);
  assert.equal(ashbyAdapter.getApplicationRoot(doc).id, 'form');
  assert.equal(names(ashbyAdapter.collectCandidates(doc)).includes('project_proud_of'), true);
});
test('SmartRecruiters exposes its application host for runtime open-shadow discovery', () => {
  const host = {
    ...fixtures.smartRecruiters.applicationHost,
    fields: []
  };
  const doc = createDocument([host]);
  const root = smartRecruitersAdapter.getApplicationRoot(doc);
  assert.equal(root.tagName, 'SMARTRECRUITERS-APPLICATION');
  assert.equal(smartRecruitersAdapter.getDiscoveryContexts(doc)[0].root, root);
  assert.deepEqual(smartRecruitersAdapter.collectCandidates(doc), []);
  assert.equal(fixtures.smartRecruiters.applicationHost.shadowMode, 'open');
});


test('SmartRecruiters recognizes the live OneClick main component layout', () => {
  const applicationControl = {};
  const main = {
    querySelector(selector) {
      return selector.includes('spl-input') ? applicationControl : null;
    },
    querySelectorAll() {
      return [];
    }
  };
  const doc = {
    querySelector(selector) {
      return selector === 'main' ? main : null;
    },
    querySelectorAll() {
      return [];
    }
  };
  assert.equal(smartRecruitersAdapter.getApplicationRoot(doc), main);
});

test('Workday reports an account gate without exposing sign-in controls as application fields', () => {
  const doc = createDocument([fixtures.workday.accountGate]);
  assert.equal(workdayAdapter.getApplicationRoot(doc), null);
  assert.deepEqual(workdayAdapter.collectCandidates(doc), []);
  const contexts = workdayAdapter.getDiscoveryContexts(doc);
  assert.equal(contexts.length, 1);
  assert.equal(contexts[0].kind, 'account_gate');
  assert.equal(contexts[0].root.getAttribute('data-automation-id'), 'signInForm');
});

test('Workday discovers later application steps and excludes the robots-only honeypot', () => {
  const doc = createDocument([fixtures.workday.applicationForm]);
  assert.deepEqual(names(workdayAdapter.collectCandidates(doc)), [
    'legalNameSection_firstName',
    'legalNameSection_lastName',
    'email',
    'salaryExpectation'
  ]);
  assert.equal(workdayAdapter.getDiscoveryContexts(doc)[0].kind, 'application');
});

test('ATS metadata stays bounded and identifies the dedicated source', () => {
  const cases = [
    [greenhouseAdapter, fixtures.greenhouse.url, 'main h1', 'Staff Analyst', 'Greenhouse'],
    [leverAdapter, fixtures.lever.url, '.posting-headline h2', 'Data Analyst', 'Lever'],
    [ashbyAdapter, fixtures.ashby.url, '[data-testid="job-title"]', 'Risk Analyst', 'Ashby'],
    [smartRecruitersAdapter, fixtures.smartRecruiters.url, '[data-testid="job-title"]', 'Data Analyst', 'SmartRecruiters'],
    [workdayAdapter, fixtures.workday.url, '[data-automation-id="jobTitle"]', 'Software Engineer', 'Workday']
  ];
  cases.forEach(([adapter, url, selector, title, source]) => {
    const metadata = adapter.extractJobMetadata(createDocument([], { [selector]: title }), url);
    assert.equal(metadata.title, title);
    assert.equal(metadata.source, source);
    assert.ok(metadata.jobUrl.length <= 2048);
  });
});

test('profiled ATS adapters use stable hosts, scoped roots, and bounded metadata', () => {
  fixtures.additionalAts.cases.forEach((fixtureCase) => {
    const doc = createDocument([fixtureCase.root]);
    const adapter = selectAdapter({ doc, url: fixtureCase.url });
    assert.equal(adapter.id, fixtureCase.id);
    assert.equal(adapter.getApplicationRoot(doc)?.id, fixtureCase.root.id);
    assert.deepEqual(names(adapter.collectCandidates(doc)), [
      'candidateFirstName',
      'candidateLastName',
      'candidateEmail',
      'interest'
    ]);
    assert.deepEqual(adapter.getDiscoveryContexts(doc).map(({ kind }) => kind), ['application']);

    const metadata = adapter.extractJobMetadata(
      createDocument([], { 'main h1': 'Platform Test Analyst' }),
      fixtureCase.url
    );
    assert.equal(metadata.title, 'Platform Test Analyst');
    assert.equal(metadata.source, fixtureCase.source);

    const hostname = new URL(fixtureCase.url).hostname;
    assert.equal(selectAdapter({
      doc: createDocument(),
      url: 'https://' + hostname + '.example.test/apply'
    }).id, 'generic');
  });
});

test('profiled ATS adapters block account gates before candidate collection', () => {
  fixtures.additionalAts.cases.forEach((fixtureCase) => {
    const accountDoc = createDocument([{
      selector: '[data-testid*="sign-in" i]',
      tagName: 'FORM',
      id: fixtureCase.id + '-sign-in',
      className: 'account sign in',
      fields: [
        { tagName: 'INPUT', type: 'email', name: 'email', label: 'Email address' },
        { tagName: 'INPUT', type: 'password', name: 'password', label: 'Password' }
      ]
    }]);
    const adapter = selectAdapter({ doc: accountDoc, url: fixtureCase.url });
    assert.equal(adapter.id, fixtureCase.id);
    assert.equal(adapter.getApplicationRoot(accountDoc), null);
    assert.deepEqual(adapter.collectCandidates(accountDoc), []);
    assert.deepEqual(adapter.getDiscoveryContexts(accountDoc).map(({ kind }) => kind), ['account_gate']);
  });
});
