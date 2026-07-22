'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

process.env.APPLICATIONS_TABLE = process.env.APPLICATIONS_TABLE || 'job-application-tracker-test';

const {
  MAX_CAPTURE_BODY_BYTES,
  buildCaptureApplicationResponse,
  createCaptureApplicationId,
  handleCaptureApplication,
  normalizeCaptureApplicationRequest,
  parseCaptureBody
} = require('./index').__test;

const basePayload = () => ({
  captureId: '550e8400-e29b-41d4-a716-446655440000',
  company: ' Acme Corp ',
  title: ' Data Analyst ',
  appliedDate: '2025-01-15',
  jobUrl: 'https://acme.example/jobs/123',
  status: 'applied',
  tags: ['remote', 'analytics']
});

const conditionalFailure = () => {
  const err = new Error('Conditional request failed');
  err.name = 'ConditionalCheckFailedException';
  return err;
};

test('capture body parsing enforces JSON object and byte limits', () => {
  assert.deepEqual(parseCaptureBody({ body: '{"captureId":"capture-123"}' }), { captureId: 'capture-123' });
  assert.throws(
    () => parseCaptureBody({ body: '{bad json' }),
    err => err.statusCode === 400 && /valid JSON/.test(err.message)
  );
  assert.throws(
    () => parseCaptureBody({ body: '[]' }),
    err => err.statusCode === 400 && /JSON object/.test(err.message)
  );
  assert.throws(
    () => parseCaptureBody({ body: 'x'.repeat(MAX_CAPTURE_BODY_BYTES + 1) }),
    err => err.statusCode === 413
  );
});

test('capture request normalization uses normal application fields and strict validation', () => {
  const normalized = normalizeCaptureApplicationRequest('user-1', basePayload());
  assert.equal(normalized.applicationPayload.company, 'Acme Corp');
  assert.equal(normalized.applicationPayload.title, 'Data Analyst');
  assert.equal(normalized.applicationPayload.status, 'Applied');
  assert.deepEqual(normalized.applicationPayload.tags, ['remote', 'analytics']);
  assert.deepEqual(
    Object.keys(normalized.applicationPayload).sort(),
    ['appliedDate', 'company', 'jobUrl', 'location', 'postingDate', 'source', 'status', 'tags', 'title']
  );
  assert.match(normalized.captureFingerprint, /^[a-f0-9]{64}$/);
  assert.match(normalized.applicationId, /^APP#CAPTURE#[a-f0-9]{64}$/);

  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), protocolVersion: 2 }),
    err => err.statusCode === 400 && /protocolVersion/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), captureId: 'short' }),
    err => err.statusCode === 400 && /captureId/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), appliedDate: '2025-02-30' }),
    err => err.statusCode === 400 && /valid YYYY-MM-DD/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), jobUrl: 'ftp://acme.example/job' }),
    err => err.statusCode === 400 && /HTTP or HTTPS/.test(err.message)
  );
});

test('capture tags reject coercion, truncation, and case-folded duplicates', () => {
  const tooMany = Array.from({ length: 13 }, (_, index) => `tag-${index}`);
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), tags: tooMany }),
    err => err.statusCode === 400 && /at most 12/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), tags: 'remote,analytics' }),
    err => err.statusCode === 400 && /must be an array/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), tags: ['remote', ' Remote '] }),
    err => err.statusCode === 400 && /duplicates after case-folding/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), tags: ['x'.repeat(37)] }),
    err => err.statusCode === 400 && /36 characters or fewer/.test(err.message)
  );
  assert.throws(
    () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), tags: [42] }),
    err => err.statusCode === 400 && /must be a string/.test(err.message)
  );
});

test('capture status is restricted to the six bridge statuses after normalization', () => {
  for (const status of ['applied', 'SCREENING', 'interview', 'Offer', 'rejected', 'withdrawn']) {
    const normalized = normalizeCaptureApplicationRequest('user-1', { ...basePayload(), status });
    assert.equal(normalized.applicationPayload.status, status[0].toUpperCase() + status.slice(1).toLowerCase());
  }
  for (const status of ['Draft', 'Active', 'Phone screen', 'Hired', '', null]) {
    assert.throws(
      () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), status }),
      err => err.statusCode === 400 && /status (?:is required|must be one of)/.test(err.message)
    );
  }
});

test('capture request rejects every key outside the extension metadata allowlist', () => {
  const forbiddenFields = {
    notes: '',
    customFields: {},
    batch: '',
    captureDate: '',
    statusDate: '',
    followUpDate: '',
    followUpNote: '',
    attachments: [],
    questions: [],
    answers: [],
    url: 'https://acme.example/jobs/alias'
  };

  for (const [field, value] of Object.entries(forbiddenFields)) {
    assert.throws(
      () => normalizeCaptureApplicationRequest('user-1', { ...basePayload(), [field]: value }),
      err => err.statusCode === 400 && /Unsupported capture request field/.test(err.message) && err.message.includes(field),
      `${field} must be rejected even when empty`
    );
  }
});

test('capture application IDs are stable per user and capture ID', () => {
  const first = createCaptureApplicationId('user-1', 'capture-123');
  assert.equal(first, createCaptureApplicationId('user-1', 'capture-123'));
  assert.notEqual(first, createCaptureApplicationId('user-2', 'capture-123'));
  assert.notEqual(first, createCaptureApplicationId('user-1', 'capture-456'));
});

test('new capture creates one conditional normal application record', async () => {
  let putInput;
  const dynamoClient = {
    send: async (command) => {
      assert.equal(command.constructor.name, 'PutCommand');
      putInput = command.input;
      return {};
    }
  };

  const result = await handleCaptureApplication('user-1', basePayload(), { dynamoClient });
  assert.equal(result.created, true);
  assert.equal(result.captureId, basePayload().captureId);
  assert.equal(result.item.recordType, 'application');
  assert.equal(result.item.captureId, basePayload().captureId);
  assert.equal(result.item.captureSource, 'application-capture-api');
  assert.equal(putInput.ConditionExpression, 'attribute_not_exists(applicationId)');
  assert.equal(putInput.Item.applicationId, result.item.applicationId);
});

test('identical capture retry returns the consistently read stored item', async () => {
  let stored;
  const createClient = {
    send: async (command) => {
      assert.equal(command.constructor.name, 'PutCommand');
      stored = command.input.Item;
      return {};
    }
  };
  await handleCaptureApplication('user-1', basePayload(), { dynamoClient: createClient });

  let getInput;
  const retryClient = {
    send: async (command) => {
      if (command.constructor.name === 'PutCommand') throw conditionalFailure();
      assert.equal(command.constructor.name, 'GetCommand');
      getInput = command.input;
      return { Item: stored };
    }
  };
  const result = await handleCaptureApplication('user-1', basePayload(), { dynamoClient: retryClient });
  assert.equal(result.created, false);
  assert.equal(result.captureId, basePayload().captureId);
  assert.deepEqual(result.item, stored);
  assert.equal(getInput.ConsistentRead, true);
  assert.equal(getInput.Key.applicationId, stored.applicationId);
});

test('capture responses wrap created and replay results with their respective status codes', () => {
  const item = { applicationId: 'APP#CAPTURE#abc', company: 'Acme Corp' };
  const captureId = '550e8400-e29b-41d4-a716-446655440000';

  const createdResponse = buildCaptureApplicationResponse({ item, created: true, captureId }, 'https://example.com');
  assert.equal(createdResponse.statusCode, 201);
  assert.deepEqual(JSON.parse(createdResponse.body), { item, created: true, captureId });

  const replayResponse = buildCaptureApplicationResponse({ item, created: false, captureId }, 'https://example.com');
  assert.equal(replayResponse.statusCode, 200);
  assert.deepEqual(JSON.parse(replayResponse.body), { item, created: false, captureId });
});

test('capture API Gateway route uses the existing JWT authorizer', () => {
  const template = fs.readFileSync(path.join(__dirname, 'template.yaml'), 'utf8');
  assert.match(
    template,
    /ApplicationsCaptureRoute:[\s\S]*?RouteKey: 'POST \/api\/applications\/capture'[\s\S]*?AuthorizationType: JWT[\s\S]*?AuthorizerId: !Ref HttpApiAuthorizer/
  );
});

test('capture ID reuse with different normalized data returns conflict', async () => {
  let stored;
  const createClient = {
    send: async (command) => {
      stored = command.input.Item;
      return {};
    }
  };
  await handleCaptureApplication('user-1', basePayload(), { dynamoClient: createClient });

  const conflictClient = {
    send: async (command) => {
      if (command.constructor.name === 'PutCommand') throw conditionalFailure();
      return { Item: stored };
    }
  };
  await assert.rejects(
    handleCaptureApplication('user-1', { ...basePayload(), title: 'Senior Data Analyst' }, { dynamoClient: conflictClient }),
    err => err.statusCode === 409 && /different application data/.test(err.message)
  );
});
