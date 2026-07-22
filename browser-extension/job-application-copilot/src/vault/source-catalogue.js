import { SOURCE_ROLES } from '../shared/schemas.js';

const SINGLE_RECOMMENDED_ROLES = Object.freeze([
  SOURCE_ROLES.CANDIDATE_EVIDENCE,
  SOURCE_ROLES.STYLE_EXAMPLE
]);
const SINGLE_RECOMMENDED_ROLE_SET = new Set(SINGLE_RECOMMENDED_ROLES);
const REUSABLE_ROLE_SET = new Set([
  ...SINGLE_RECOMMENDED_ROLES,
  SOURCE_ROLES.USER_VERIFIED
]);
const APPLICATION_CONTEXT_ROLE_SET = new Set([
  SOURCE_ROLES.JOB_REQUIREMENT,
  SOURCE_ROLES.COMPANY_CONTEXT
]);

const documentFor = record => record?.value?.document || null;
const applicationIdFor = record => String(documentFor(record)?.applicationId || '').trim();
const sourceRoleFor = record => String(documentFor(record)?.sourceRole || '').trim();
const recordTimestamp = record => String(
  record?.updatedAt
  || documentFor(record)?.importedAt
  || record?.createdAt
  || ''
);

const newestFirst = (left, right) => recordTimestamp(right).localeCompare(recordTimestamp(left))
  || String(left?.id || '').localeCompare(String(right?.id || ''));

const usableRecords = records => (records || []).filter(record => record?.id && documentFor(record));

export const isReusableSourceRecord = record => !applicationIdFor(record)
  && REUSABLE_ROLE_SET.has(sourceRoleFor(record));

export const recommendedSourceSelection = (records, applicationId = '') => {
  const targetApplicationId = String(applicationId || '').trim();
  const candidates = usableRecords(records);
  const selected = new Set();

  if (targetApplicationId) {
    candidates
      .filter(record => applicationIdFor(record) === targetApplicationId)
      .forEach(record => selected.add(record.id));
  }

  candidates
    .filter(record => !applicationIdFor(record) && sourceRoleFor(record) === SOURCE_ROLES.USER_VERIFIED)
    .forEach(record => selected.add(record.id));

  for (const role of SINGLE_RECOMMENDED_ROLES) {
    const newest = candidates
      .filter(record => !applicationIdFor(record) && sourceRoleFor(record) === role)
      .sort(newestFirst)[0];
    if (newest) selected.add(newest.id);
  }

  return selected;
};

export const carrySourceSelection = ({
  selectedIds,
  records,
  applicationId = '',
  fromUnscoped = false
} = {}) => {
  const selected = selectedIds instanceof Set ? selectedIds : new Set(selectedIds || []);
  const recordsById = new Map(usableRecords(records).map(record => [record.id, record]));
  const targetApplicationId = String(applicationId || '').trim();
  const carried = new Set([...selected].filter((id) => {
    const record = recordsById.get(id);
    if (!record) return false;
    if (applicationIdFor(record) === targetApplicationId && targetApplicationId) return true;
    return fromUnscoped ? !applicationIdFor(record) : isReusableSourceRecord(record);
  }));
  for (const id of recommendedSourceSelection(records, targetApplicationId)) carried.add(id);
  return carried;
};

export const importedSourceApplicationId = ({ sourceRole, applicationId = '' } = {}) => (
  APPLICATION_CONTEXT_ROLE_SET.has(String(sourceRole || ''))
    ? String(applicationId || '').trim() || null
    : null
);

export const selectImportedSource = ({ selectedIds, records, importedRecord } = {}) => {
  const next = selectedIds instanceof Set ? new Set(selectedIds) : new Set(selectedIds || []);
  const importedRole = sourceRoleFor(importedRecord);
  const importedApplicationId = applicationIdFor(importedRecord);
  if (!importedRecord?.id) return next;

  if (!importedApplicationId && SINGLE_RECOMMENDED_ROLE_SET.has(importedRole)) {
    for (const record of usableRecords(records)) {
      if (!applicationIdFor(record) && sourceRoleFor(record) === importedRole) next.delete(record.id);
    }
  }
  next.add(importedRecord.id);
  return next;
};

export const sourceCatalogueSummary = (records, selectedIds, applicationId = '') => {
  const selected = selectedIds instanceof Set ? selectedIds : new Set(selectedIds || []);
  const recommended = recommendedSourceSelection(records, applicationId);
  const targetApplicationId = String(applicationId || '').trim();
  const entries = usableRecords(records).slice().sort(newestFirst).map(record => ({
    applicationId: applicationIdFor(record) || null,
    availableForApplication: !applicationIdFor(record)
      || Boolean(targetApplicationId && applicationIdFor(record) === targetApplicationId),
    id: record.id,
    recommended: recommended.has(record.id),
    selected: selected.has(record.id),
    sourceRole: sourceRoleFor(record)
  }));
  return {
    activeCount: entries.filter(entry => entry.selected && entry.availableForApplication).length,
    entries,
    recommendedCount: entries.filter(entry => entry.recommended).length,
    totalCount: entries.length
  };
};
