import assert from 'node:assert/strict';
import test from 'node:test';
import {
  carrySourceSelection,
  importedSourceApplicationId,
  recommendedSourceSelection,
  selectImportedSource,
  sourceCatalogueSummary
} from '../src/vault/source-catalogue.js';

const source = ({
  id,
  role,
  applicationId = null,
  updatedAt = '2026-07-01T00:00:00.000Z'
}) => ({
  id,
  updatedAt,
  value: {
    document: {
      applicationId,
      filename: `${id}.txt`,
      sourceRole: role
    }
  }
});

test('catalogue recommends current reusable candidate and style documents without old job context', () => {
  const records = [
    source({ id: 'resume-old', role: 'candidate_evidence', updatedAt: '2026-06-01T00:00:00.000Z' }),
    source({ id: 'resume-current', role: 'candidate_evidence', updatedAt: '2026-07-01T00:00:00.000Z' }),
    source({ id: 'cover-current', role: 'style_example', updatedAt: '2026-07-02T00:00:00.000Z' }),
    source({ id: 'verified-one', role: 'user_verified' }),
    source({ id: 'verified-two', role: 'user_verified' }),
    source({ id: 'old-job', role: 'job_requirement', applicationId: 'application-old' }),
    source({ id: 'old-company', role: 'company_context', applicationId: 'application-old' }),
    source({ id: 'current-job', role: 'job_requirement', applicationId: 'application-current' }),
    source({ id: 'current-company', role: 'company_context', applicationId: 'application-current' })
  ];

  assert.deepEqual([...recommendedSourceSelection(records, 'application-current')].sort(), [
    'cover-current',
    'current-company',
    'current-job',
    'resume-current',
    'verified-one',
    'verified-two'
  ]);
});

test('new application carries reusable documents and drops previous application context', () => {
  const records = [
    source({ id: 'resume', role: 'candidate_evidence' }),
    source({ id: 'cover', role: 'style_example' }),
    source({ id: 'old-job', role: 'job_requirement', applicationId: 'application-old' }),
    source({ id: 'new-job', role: 'job_requirement', applicationId: 'application-new' })
  ];
  const carried = carrySourceSelection({
    selectedIds: new Set(['resume', 'cover', 'old-job']),
    records,
    applicationId: 'application-new'
  });
  assert.deepEqual([...carried].sort(), ['cover', 'new-job', 'resume']);
});

test('first scoped application preserves explicitly selected unscoped notes but defaults omit them', () => {
  const records = [
    source({ id: 'resume', role: 'candidate_evidence' }),
    source({ id: 'manual-company-note', role: 'company_context' })
  ];
  assert.deepEqual([...recommendedSourceSelection(records, 'application-new')], ['resume']);
  assert.deepEqual(
    [...carrySourceSelection({
      selectedIds: new Set(['resume', 'manual-company-note']),
      records,
      applicationId: 'application-new',
      fromUnscoped: true
    })].sort(),
    ['manual-company-note', 'resume']
  );
});

test('job and company imports are scoped while reusable candidate imports remain global', () => {
  assert.equal(importedSourceApplicationId({
    sourceRole: 'job_requirement',
    applicationId: 'application-current'
  }), 'application-current');
  assert.equal(importedSourceApplicationId({
    sourceRole: 'company_context',
    applicationId: 'application-current'
  }), 'application-current');
  assert.equal(importedSourceApplicationId({
    sourceRole: 'candidate_evidence',
    applicationId: 'application-current'
  }), null);
  assert.equal(importedSourceApplicationId({
    sourceRole: 'style_example',
    applicationId: 'application-current'
  }), null);
});

test('new resume or style import becomes the single safe default for its role', () => {
  const oldResume = source({ id: 'resume-old', role: 'candidate_evidence' });
  const supporting = source({ id: 'verified', role: 'user_verified' });
  const newResume = source({ id: 'resume-new', role: 'candidate_evidence' });
  const next = selectImportedSource({
    selectedIds: new Set(['resume-old', 'verified']),
    records: [oldResume, supporting],
    importedRecord: newResume
  });
  assert.deepEqual([...next].sort(), ['resume-new', 'verified']);
});

test('catalogue summary exposes active and recommended counts without document contents', () => {
  const records = [
    source({ id: 'resume', role: 'candidate_evidence' }),
    source({ id: 'cover', role: 'style_example' }),
    source({ id: 'old-job', role: 'job_requirement', applicationId: 'application-old' })
  ];
  const summary = sourceCatalogueSummary(records, new Set(['resume', 'old-job']), 'application-current');
  assert.equal(summary.activeCount, 1);
  assert.equal(summary.recommendedCount, 2);
  assert.equal(summary.totalCount, 3);
  assert.deepEqual(summary.entries.map(entry => Object.keys(entry).sort()), [
    ['applicationId', 'availableForApplication', 'id', 'recommended', 'selected', 'sourceRole'],
    ['applicationId', 'availableForApplication', 'id', 'recommended', 'selected', 'sourceRole'],
    ['applicationId', 'availableForApplication', 'id', 'recommended', 'selected', 'sourceRole']
  ]);
});
