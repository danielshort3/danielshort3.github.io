import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { applicantProAdapter, isApplicantProPage } from '../src/adapters/applicantpro.js';
import { extractGenericJobMetadata, genericAdapter, sanitizeJobUrl } from '../src/adapters/generic.js';
import { selectAdapter } from '../src/adapters/index.js';

const fixture = JSON.parse(await readFile(new URL('./fixtures/applicantpro-structure.json', import.meta.url), 'utf8'));
const liveApplicantProFixture = JSON.parse(await readFile(new URL('./fixtures/applicantpro-live-structure.json', import.meta.url), 'utf8'));

const createMetaDocument = (values = {}) => ({
  querySelector(selector) {
    if (values[selector] === undefined) return null;
    const value = values[selector];
    return {
      textContent: value,
      getAttribute(name) {
        return name === 'content' ? value : '';
      }
    };
  },
  querySelectorAll() {
    return [];
  }
});

test('ApplicantPro detection prefers its bounded adapter', () => {
  const doc = createMetaDocument();
  assert.equal(isApplicantProPage({ doc, url: fixture.url }), true);
  assert.equal(selectAdapter({ doc, url: fixture.url }).id, 'applicantpro');
  assert.equal(selectAdapter({ doc, url: 'https://jobs.example.com/apply' }).id, 'generic');
});

test('ApplicantPro can be detected by generator metadata without reading page text', () => {
  const doc = createMetaDocument({
    'meta[name="generator"]': 'ApplicantPro'
  });
  assert.equal(isApplicantProPage({ doc, url: 'https://careers.example.com/opening/4' }), true);
});

test('ApplicantPro scopes candidates to the live apply form and ignores auxiliary widgets', () => {
  const createForm = (formFixture) => ({
    id: formFixture.id,
    querySelector(selector) {
      if (selector === 'input[name="listing_id"]') {
        return formFixture.fields.find((field) => field.name === 'listing_id') || null;
      }
      if (selector.includes('input[name="first_name"]')) {
        return formFixture.fields.find((field) => ['first_name', 'last_name', 'email', 'contact_number'].includes(field.name)) || null;
      }
      return null;
    },
    querySelectorAll() {
      return formFixture.fields;
    }
  });
  const forms = [liveApplicantProFixture.applicationForm, ...liveApplicantProFixture.auxiliaryForms]
    .map(createForm);
  const doc = {
    querySelector(selector) {
      if (selector === 'form#apply') return forms.find((form) => form.id === 'apply') || null;
      return null;
    },
    querySelectorAll(selector) {
      if (selector === 'form') return forms;
      return forms.flatMap((form) => form.querySelectorAll());
    }
  };

  const candidates = applicantProAdapter.collectCandidates(doc);
  assert.deepEqual(
    candidates.map((field) => field.name),
    liveApplicantProFixture.applicationForm.fields.map((field) => field.name)
  );
  assert.equal(candidates.some((field) => field.id === 'refer-email'), false);
  assert.equal(candidates.some((field) => field.id === 'faq_bar_searchCriteria'), false);
});

test('job URL sanitization retains only recognized identifiers', () => {
  const sanitized = sanitizeJobUrl(fixture.url.replace('utm_source=tracker', 'utm_source=tracker&token=secret'));
  const url = new URL(sanitized);
  assert.equal(url.hash, '');
  assert.equal(url.searchParams.get('jobId'), '12345');
  assert.equal(url.searchParams.has('utm_source'), false);
  assert.equal(url.searchParams.has('token'), false);
});

test('generic job metadata uses bounded headings and metadata only', () => {
  const doc = createMetaDocument({
    '[data-job-title]': fixture.job.title,
    '[data-company-name]': fixture.job.company,
    '[data-job-location]': fixture.job.location
  });
  const result = extractGenericJobMetadata(doc, fixture.url);
  assert.deepEqual(result, {
    company: fixture.job.company,
    title: fixture.job.title,
    jobUrl: 'https://acme.applicantpro.com/jobs/12345?jobId=12345',
    location: fixture.job.location,
    source: 'acme.applicantpro.com'
  });
});

test('generic metadata recognizes current Greenhouse title and location hooks', () => {
  const doc = createMetaDocument({
    'main h1': 'Staff Fullstack Engineer, Clinic',
    '.job__location': 'United States - Remote',
    '.job__description': 'Responsibilities include analytics. The posted salary range is $65,000 - $85,000 USD.'
  });
  doc.title = 'Job Application for Staff Fullstack Engineer, Clinic at Weight Watchers';
  const result = extractGenericJobMetadata(doc, 'https://job-boards.greenhouse.io/ww/jobs/5226133008');
  assert.deepEqual(result, {
    company: 'Weight Watchers',
    title: 'Staff Fullstack Engineer, Clinic',
    jobUrl: 'https://job-boards.greenhouse.io/ww/jobs/5226133008',
    location: 'United States - Remote',
    source: 'job-boards.greenhouse.io',
    description: 'Responsibilities include analytics. The posted salary range is $65,000 - $85,000 USD.'
  });
});

test('generic metadata recognizes current Lever title and company hooks', () => {
  const doc = createMetaDocument({
    '.posting-header h2': 'Software Engineer, Backend & Integrations (Remote)',
    '.location': 'Remote'
  });
  doc.title = 'Supermove - Software Engineer, Backend & Integrations (Remote)';
  const result = extractGenericJobMetadata(
    doc,
    'https://jobs.lever.co/supermove/a0d545ec-c114-45e5-8707-16e5c1997182/apply'
  );
  assert.deepEqual(result, {
    company: 'Supermove',
    title: 'Software Engineer, Backend & Integrations (Remote)',
    jobUrl: 'https://jobs.lever.co/supermove/a0d545ec-c114-45e5-8707-16e5c1997182/apply',
    location: 'Remote',
    source: 'jobs.lever.co'
  });
});

test('generic adapter reads the current Lever application-question label', () => {
  const labelNode = { textContent: 'Why Supermove / this position? ✱' };
  const container = {
    querySelector(selector) {
      const selectors = selector.split(',').map((part) => part.trim());
      return selectors.includes('.application-label') ? labelNode : null;
    }
  };
  const element = { closest: () => container, parentElement: null };
  assert.equal(genericAdapter.getSupplementalLabel(element), 'Why Supermove / this position? ✱');
});

test('generic nearby text keeps instructions and ignores validation feedback nodes', () => {
  const node = (textContent, attributes = {}) => ({
    textContent,
    hidden: false,
    getAttribute(name) {
      return attributes[name] || '';
    }
  });
  const instruction = node('Maximum 500 characters. If yes, explain.', { class: 'help-text' });
  const error = node('This field is required.', {
    id: 'experience-error',
    class: 'error-message',
    role: 'alert',
    'aria-live': 'assertive'
  });
  const status = node('Checking your answer.', {
    class: 'field-status',
    'aria-live': 'polite'
  });
  const container = {
    querySelectorAll() {
      return [instruction, error, status];
    }
  };
  const element = {
    closest: () => container,
    parentElement: null
  };
  assert.equal(
    genericAdapter.getNearbyText(element),
    'Maximum 500 characters. If yes, explain.'
  );
});

test('generic nearby text ignores validation class tokens separated by whitespace', () => {
  const node = (textContent, className) => ({
    textContent,
    hidden: false,
    getAttribute(name) {
      return name === 'class' ? className : '';
    }
  });
  const instruction = node('Describe your most relevant experience.', 'field help-text');
  const feedback = node('This field is required.', 'field validation message');
  const container = {
    querySelectorAll() {
      return [instruction, feedback];
    }
  };
  const element = {
    closest: () => container,
    parentElement: null
  };

  assert.equal(
    genericAdapter.getNearbyText(element),
    'Describe your most relevant experience.'
  );
});
test('ApplicantPro metadata reports the ATS source without descriptions', () => {
  const doc = createMetaDocument({
    '[data-job-title],.job-title,.posting-title,main h1,h1': fixture.job.title,
    '[data-company-name],.company-name,.job-company,#company-name': fixture.job.company,
    '[data-job-location],.job-location,.posting-location,.location': fixture.job.location
  });
  const result = applicantProAdapter.extractJobMetadata(doc, fixture.url);
  assert.equal(result.source, 'ApplicantPro');
  assert.equal(result.title, fixture.job.title);
  assert.equal(Object.hasOwn(result, 'description'), false);
});
