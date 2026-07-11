const assert = require('assert');
const Module = require('module');

class PutCommand {
  constructor(input) { this.input = input; }
}
class GetCommand {
  constructor(input) { this.input = input; }
}
class UpdateCommand {
  constructor(input) { this.input = input; }
}
class DeleteCommand {
  constructor(input) { this.input = input; }
}
class QueryCommand {
  constructor(input) { this.input = input; }
}
class PutObjectCommand {
  constructor(input) { this.input = input; }
}
class GetObjectCommand {
  constructor(input) { this.input = input; }
}

const conditionalFailure = () => {
  const err = new Error('The conditional request failed');
  err.name = 'ConditionalCheckFailedException';
  return err;
};

const records = new Map([
  ['user-a|APP#owned', {
    userId: 'user-a',
    applicationId: 'APP#owned',
    recordType: 'application',
    company: 'Canary Company',
    title: 'Canary Role',
    appliedDate: '2026-07-11',
    status: 'Applied'
  }],
  ['user-a|PROSPECT#owned', {
    userId: 'user-a',
    applicationId: 'PROSPECT#owned',
    recordType: 'prospect',
    company: 'Canary Prospect',
    title: 'Prospect Role',
    jobUrl: 'https://example.com/job'
  }]
]);
const calls = [];

const documentClient = {
  async send(command) {
    calls.push(command);
    const input = command.input;
    const key = input?.Key ? `${input.Key.userId}|${input.Key.applicationId}` : '';

    if (command instanceof UpdateCommand) {
      const item = records.get(key);
      if (!item) throw conditionalFailure();
      if (input.ExpressionAttributeValues[':expectedRecordType'] === 'prospect' && item.recordType !== 'prospect') {
        throw conditionalFailure();
      }
      if (input.ExpressionAttributeValues[':expectedRecordType'] === 'application' && item.recordType === 'prospect') {
        throw conditionalFailure();
      }
      const updated = { ...item, updatedAt: '2026-07-11T00:00:00.000Z' };
      records.set(key, updated);
      return { Attributes: updated };
    }

    if (command instanceof DeleteCommand) {
      if (!records.has(key)) throw conditionalFailure();
      records.delete(key);
      return {};
    }

    if (command instanceof GetCommand) {
      return { Item: records.get(key) };
    }
    if (command instanceof QueryCommand) return { Items: [] };
    if (command instanceof PutCommand) return {};
    throw new Error(`Unexpected command: ${command.constructor.name}`);
  }
};

const originalLoad = Module._load;
Module._load = function mockAwsModules(request, parent, isMain) {
  if (request === '@aws-sdk/client-dynamodb') {
    return { DynamoDBClient: class DynamoDBClient {} };
  }
  if (request === '@aws-sdk/lib-dynamodb') {
    return {
      DynamoDBDocumentClient: { from: () => documentClient },
      PutCommand,
      GetCommand,
      UpdateCommand,
      DeleteCommand,
      QueryCommand
    };
  }
  if (request === '@aws-sdk/client-s3') {
    return {
      S3Client: class S3Client {},
      PutObjectCommand,
      GetObjectCommand
    };
  }
  if (request === '@aws-sdk/s3-request-presigner') {
    return { getSignedUrl: async () => 'https://example.invalid/signed' };
  }
  if (request === '@aws-sdk/lib-storage') {
    return { Upload: class Upload {} };
  }
  if (request === 'archiver') {
    return () => ({ pipe() {}, append() {}, on() {}, finalize() {} });
  }
  return originalLoad.call(this, request, parent, isMain);
};

process.env.APPLICATIONS_TABLE = 'job-tracker-test';
process.env.ATTACHMENTS_BUCKET = 'job-tracker-test-attachments';
const { handler } = require('../aws/job-application-tracker/index');
Module._load = originalLoad;

const event = ({ method, route, path, userId, body }) => ({
  routeKey: `${method} ${route}`,
  rawPath: path,
  headers: {},
  requestContext: {
    http: { method },
    authorizer: { jwt: { claims: { sub: userId } } }
  },
  body: body === undefined ? undefined : JSON.stringify(body)
});

const invoke = async (input) => {
  const response = await handler(event(input));
  return {
    statusCode: response.statusCode,
    body: JSON.parse(response.body)
  };
};

async function main() {
  let checks = 0;
  const check = (condition, message) => {
    assert(condition, message);
    checks += 1;
  };

  const ownPatch = await invoke({
    method: 'PATCH',
    route: '/api/applications/{id}',
    path: '/api/applications/APP#owned',
    userId: 'user-a',
    body: { notes: 'owned update' }
  });
  check(ownPatch.statusCode === 200, 'Owner PATCH should succeed');
  check(ownPatch.body.applicationId === 'APP#owned', 'Owner PATCH should return the owned row');

  const appUpdate = calls.find((command) => command instanceof UpdateCommand && command.input.Key.applicationId === 'APP#owned');
  check(appUpdate.input.ConditionExpression.includes('attribute_exists(#ownerUserId)'), 'Application PATCH must require the owner partition key');
  check(appUpdate.input.ConditionExpression.includes('attribute_exists(#ownerApplicationId)'), 'Application PATCH must require the item sort key');
  check(appUpdate.input.ExpressionAttributeValues[':expectedRecordType'] === 'application', 'Application PATCH must reject prospect rows');

  const crossPatch = await invoke({
    method: 'PATCH',
    route: '/api/applications/{id}',
    path: '/api/applications/APP#owned',
    userId: 'user-b',
    body: { notes: 'cross-user update' }
  });
  const missingPatch = await invoke({
    method: 'PATCH',
    route: '/api/applications/{id}',
    path: '/api/applications/APP#missing',
    userId: 'user-a',
    body: { notes: 'missing update' }
  });
  check(crossPatch.statusCode === 404, 'Cross-user PATCH must return 404');
  check(missingPatch.statusCode === 404, 'Missing PATCH must return 404');
  check(crossPatch.body.error === missingPatch.body.error, 'Cross-user and missing PATCH must not reveal existence');
  check(!records.has('user-b|APP#owned'), 'Cross-user PATCH must not create a shadow row');

  const ownProspectPatch = await invoke({
    method: 'PATCH',
    route: '/api/prospects/{id}',
    path: '/api/prospects/PROSPECT#owned',
    userId: 'user-a',
    body: { notes: 'owned prospect update' }
  });
  check(ownProspectPatch.statusCode === 200, 'Owner prospect PATCH should succeed');
  const prospectUpdate = calls.find((command) => command instanceof UpdateCommand && command.input.Key.applicationId === 'PROSPECT#owned');
  check(prospectUpdate.input.ExpressionAttributeValues[':expectedRecordType'] === 'prospect', 'Prospect PATCH must require a prospect row');

  const crossProspectPatch = await invoke({
    method: 'PATCH',
    route: '/api/prospects/{id}',
    path: '/api/prospects/PROSPECT#owned',
    userId: 'user-b',
    body: { notes: 'cross-user prospect update' }
  });
  check(crossProspectPatch.statusCode === 404, 'Cross-user prospect PATCH must return 404');

  const crossDelete = await invoke({
    method: 'DELETE',
    route: '/api/applications/{id}',
    path: '/api/applications/APP#owned',
    userId: 'user-b'
  });
  const missingDelete = await invoke({
    method: 'DELETE',
    route: '/api/applications/{id}',
    path: '/api/applications/APP#missing',
    userId: 'user-a'
  });
  check(crossDelete.statusCode === 404, 'Cross-user DELETE must return 404');
  check(missingDelete.statusCode === 404, 'Missing DELETE must return 404');
  check(crossDelete.body.error === missingDelete.body.error, 'Cross-user and missing DELETE must not reveal existence');
  check(records.has('user-a|APP#owned'), 'Cross-user DELETE must not remove the owner row');

  const deleteCommand = calls.find((command) => command instanceof DeleteCommand);
  check(deleteCommand.input.ConditionExpression.includes('attribute_exists(#ownerUserId)'), 'DELETE must require the owner partition key');
  check(deleteCommand.input.ConditionExpression.includes('attribute_exists(#ownerApplicationId)'), 'DELETE must require the item sort key');

  const ownDelete = await invoke({
    method: 'DELETE',
    route: '/api/applications/{id}',
    path: '/api/applications/APP#owned',
    userId: 'user-a'
  });
  check(ownDelete.statusCode === 200 && ownDelete.body.ok === true, 'Owner DELETE should succeed');
  check(!records.has('user-a|APP#owned'), 'Owner DELETE should remove the row');

  process.stdout.write(`Job Tracker ownership boundary tests passed (${checks} checks).\n`);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exitCode = 1;
});
