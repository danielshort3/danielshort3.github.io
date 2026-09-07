'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { createRequire } = require('module');

function loadStore(relative, env) {
  const filename = path.resolve(__dirname, '../..', relative);
  const localRequire = createRequire(filename);
  const clientOptions = [];
  const credentialRegions = [];
  const requests = [];
  const module = { exports: {} };
  const overrides = {
    './aws-credentials': { resolveAwsCredentials: (options) => {
      credentialRegions.push(options.region);
      return { credentials: undefined, cacheKey: 'test' };
    } },
    '@aws-sdk/client-dynamodb': { DynamoDBClient: class {
      constructor(options) { this.options = options; clientOptions.push(options); }
    } },
    '@aws-sdk/lib-dynamodb': {
      ...localRequire('@aws-sdk/lib-dynamodb'),
      DynamoDBDocumentClient: { from: (client) => ({ send: async (command) => {
        requests.push({ region: client.options.region, input: command.input });
        return {};
      } }) }
    }
  };
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), {
    module, exports: module.exports,
    require: (name) => Object.hasOwn(overrides, name) ? overrides[name] : localRequire(name),
    process: { env }, Buffer, URL, Date, console
  }, { filename });
  return { store: module.exports, clientOptions, credentialRegions, requests };
}

async function run() {
  let checks = 0;
  const services = [
    { file: 'api/_lib/tools-store-ddb.js', override: 'TOOLS_AWS_REGION', other: 'SHORTLINKS_AWS_REGION', table: 'TOOLS_DDB_TABLE', read: (store) => store.listActivity({ sub: 'region-test', limit: 1 }) },
    { file: 'api/_lib/short-links-store.js', override: 'SHORTLINKS_AWS_REGION', other: 'TOOLS_AWS_REGION', table: 'SHORTLINKS_DDB_TABLE', read: (store) => store.getLink('region-test') }
  ];
  for (const service of services) {
    const env = {
      [service.table]: 'region-test-table',
      [service.override]: ' us-east-2 ',
      [service.other]: 'eu-west-1',
      AWS_REGION: 'us-east-1',
      AWS_DEFAULT_REGION: 'us-west-2'
    };
    const { store, clientOptions, credentialRegions, requests } = loadStore(service.file, env);
    await service.read(store);
    assert.strictEqual(requests.at(-1).region, 'us-east-2', `${service.override} must override inherited AWS_REGION and trim whitespace.`);
    assert.strictEqual(credentialRegions.at(-1), 'us-east-2', 'Credential providers must receive the same effective region as DynamoDB.');
    assert.strictEqual(requests.at(-1).input.TableName, 'region-test-table', 'Region selection must preserve the configured table.');
    assert.strictEqual(env.AWS_REGION, 'us-east-1', 'Service overrides must not mutate global AWS settings.');
    checks += 4;

    await service.read(store);
    assert.strictEqual(clientOptions.length, 1, 'An unchanged effective region should reuse the client.');
    env[service.override] = '   ';
    await service.read(store);
    assert.strictEqual(requests.at(-1).region, 'us-east-1', 'Blank overrides must fall back to AWS_REGION, ignoring another service override.');
    assert.strictEqual(clientOptions.length, 2, 'Changing the effective region must replace the cached client.');
    checks += 3;

    delete env[service.override];
    env.AWS_REGION = '   ';
    await service.read(store);
    assert.strictEqual(requests.at(-1).region, 'us-west-2', 'Missing service and global regions must fall back to AWS_DEFAULT_REGION.');
    assert.strictEqual(credentialRegions.at(-1), 'us-west-2', 'Fallback regions must also reach credential providers.');
    checks += 2;

    delete env.AWS_REGION;
    delete env.AWS_DEFAULT_REGION;
    await assert.rejects(service.read(store), (error) => error.code === 'DDB_ENV_MISSING' && error.message.includes(service.override), 'Missing all supported regions must fail with an actionable configuration error.');
    env[service.override] = 'ap-southeast-2';
    await service.read(store);
    assert.strictEqual(requests.at(-1).region, 'ap-southeast-2', 'A service region must work without any global AWS region.');
    checks += 2;
  }
  return checks;
}

module.exports = run;
if (require.main === module) run().then((checks) => console.log(`Storage region overrides: ${checks} checks passed.`)).catch((error) => { console.error(error); process.exitCode = 1; });
