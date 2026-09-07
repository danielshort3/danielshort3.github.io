'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const { once } = require('events');
const { loadLocalEnvironment } = require('../../build/lib/local-env');
const { createLocalServer } = require('../../build/dev');

async function withTempDirectory(run) {
  const tempRoot = path.resolve(os.tmpdir());
  const directory = fs.mkdtempSync(path.join(tempRoot, 'website-local-env-test-'));
  try {
    await run(directory);
  } finally {
    const resolved = path.resolve(directory);
    assert.strictEqual(path.dirname(resolved), tempRoot, 'Test cleanup must stay inside the temporary directory');
    assert(path.basename(resolved).startsWith('website-local-env-test-'));
    fs.rmSync(resolved, { recursive: true, force: true });
  }
}

async function testEnvironmentFiles() {
  await withTempDirectory(async (directory) => {
    const env = { FROM_SHELL: 'shell', EMPTY_FROM_SHELL: '' };
    fs.writeFileSync(path.join(directory, '.env.local'), [
      '# Local values take precedence over defaults',
      'FROM_SHELL=local',
      'EMPTY_FROM_SHELL=must-not-replace-empty',
      'SHARED=local',
      'EMPTY_LOCAL=',
      'QUOTED="value # preserved = yes"',
      "SINGLE_QUOTED='single quoted value'",
      'COMMENTED=value # omitted comment',
      'export EXPORTED=available',
      ''
    ].join('\n'));
    fs.writeFileSync(path.join(directory, '.env'), [
      'FROM_SHELL=default',
      'EMPTY_FROM_SHELL=default',
      'SHARED=default',
      'EMPTY_LOCAL=must-not-replace-empty',
      'DEFAULT_ONLY=default',
      ''
    ].join('\n'));
    fs.writeFileSync(path.join(directory, '.env.production.local'), 'PRODUCTION_ONLY=must-not-load\n');
    fs.mkdirSync(path.join(directory, '.vercel'));
    fs.writeFileSync(path.join(directory, '.vercel', '.env.production.local'), 'VERCEL_ONLY=must-not-load\n');

    const loaded = loadLocalEnvironment(directory, env);
    assert.deepStrictEqual(loaded.map((filename) => path.basename(filename)), ['.env.local', '.env']);
    assert.deepStrictEqual(env, {
      FROM_SHELL: 'shell',
      EMPTY_FROM_SHELL: '',
      SHARED: 'local',
      EMPTY_LOCAL: '',
      QUOTED: 'value # preserved = yes',
      SINGLE_QUOTED: 'single quoted value',
      COMMENTED: 'value',
      EXPORTED: 'available',
      DEFAULT_ONLY: 'default'
    }, 'Only the two local files should load, with shell and local-file precedence preserved');
  });

  await withTempDirectory(async (directory) => {
    const env = {};
    assert.deepStrictEqual(loadLocalEnvironment(directory, env), [], 'Missing local files are optional');
    fs.writeFileSync(path.join(directory, '.env'), 'DEFAULT_ONLY=available\n');
    assert.deepStrictEqual(loadLocalEnvironment(directory, env).map((filename) => path.basename(filename)), ['.env']);
    assert.strictEqual(env.DEFAULT_ONLY, 'available', 'Missing .env.local must not prevent loading .env');
    fs.mkdirSync(path.join(directory, '.env.local'));
    assert.throws(() => loadLocalEnvironment(directory, {}), 'Unreadable local files must not be silently ignored');
  });
}

async function stopServer(server) {
  server.closeAllConnections();
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

async function requestLocalDemo(directory, inspect) {
  const server = createLocalServer({ envDir: directory });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    await inspect(`http://127.0.0.1:${server.address().port}/api/demos/target-empty-package/data`);
  } finally {
    await stopServer(server);
  }
}

async function testLocalDemoConfiguration() {
  const controlledKeys = [
    'AWS_REGION', 'AWS_DEFAULT_REGION', 'DEMO_AWS_REGION', 'DEMO_PROXY_MODE',
    'DEMO_TARGET_EMPTY_PACKAGE_FUNCTION_ARN', 'DEMO_REQUIRE_DDB_RATE_LIMIT',
    'DEMO_RATE_LIMIT_TABLE', 'DEMO_HASH_SALT', 'DEMO_INVOKE_AWS_ROLE_ARN',
    'AWS_AUTH_MODE', 'AWS_OIDC_AUDIENCE', 'VERCEL_ENV',
    'DEMO_AWS_ACCESS_KEY_ID', 'DEMO_AWS_SECRET_ACCESS_KEY', 'DEMO_AWS_SESSION_TOKEN',
    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'
  ];
  const previous = Object.fromEntries(controlledKeys.map((key) => [key, process.env[key]]));
  const originalLoad = Module._load;
  const functionArn = 'arn:aws:lambda:us-east-2:123456789012:function:target-empty-package:live';
  const fakeAccessKey = 'AKIA_LOCAL_ENV_TEST';
  const fakeSecret = 'local-env-test-secret-do-not-return';
  const snapshot = {
    meta: { recordCount: 1, startDate: '2023-01-01', endDate: '2023-01-01' },
    rows: [{ datetime: '2023-01-01T12:00:00', value: 10, employee: 'Employee_1', location: 'Area_1' }]
  };
  const invocations = [];
  let clientOptions;
  for (const key of controlledKeys) delete process.env[key];
  try {
    Module._load = function(request, parent, isMain) {
      if (request === '@aws-sdk/client-lambda') {
        return {
          LambdaClient: class {
            constructor(options) { clientOptions = options; }
            async send(command) {
              invocations.push(command.input);
              return { Payload: Buffer.from(JSON.stringify({ statusCode: 200, body: JSON.stringify(snapshot) })) };
            }
          },
          InvokeCommand: class {
            constructor(input) { this.input = input; }
          }
        };
      }
      return originalLoad.call(this, request, parent, isMain);
    };

    await withTempDirectory(async (directory) => {
      await requestLocalDemo(directory, async (url) => {
        const response = await fetch(url);
        const body = await response.json();
        assert.strictEqual(response.status, 503, 'An unconfigured local demo should retain the real configuration error');
        assert.strictEqual(body.code, 'DEMO_PROXY_CONFIGURATION_UNAVAILABLE');
        assert.strictEqual(invocations.length, 0);
      });
      fs.writeFileSync(path.join(directory, '.env.local'), [
        `DEMO_TARGET_EMPTY_PACKAGE_FUNCTION_ARN=${functionArn}`,
        'DEMO_AWS_REGION=us-east-2',
        'DEMO_REQUIRE_DDB_RATE_LIMIT=false',
        'AWS_AUTH_MODE=static',
        `DEMO_AWS_ACCESS_KEY_ID=${fakeAccessKey}`,
        `DEMO_AWS_SECRET_ACCESS_KEY=${fakeSecret}`,
        ''
      ].join('\n'));
      await requestLocalDemo(directory, async (url) => {
        const response = await fetch(url);
        const bodyText = await response.text();
        assert.strictEqual(response.status, 200, 'Server startup should load local configuration before the real demo handler runs');
        assert.deepStrictEqual(JSON.parse(bodyText), snapshot);
        assert.strictEqual(invocations.length, 1);
        assert.strictEqual(invocations[0].FunctionName, functionArn);
        const event = JSON.parse(Buffer.from(invocations[0].Payload).toString('utf8'));
        assert.strictEqual(event.rawPath, '/data');
        assert.strictEqual(event.requestContext.http.method, 'GET');
        assert.strictEqual(clientOptions.region, 'us-east-2');
        assert.strictEqual(clientOptions.credentials.accessKeyId, fakeAccessKey);
        const publicResponse = bodyText + JSON.stringify(Object.fromEntries(response.headers));
        assert(!publicResponse.includes(fakeAccessKey) && !publicResponse.includes(fakeSecret), 'Local credentials must not appear in HTTP responses');

        const rejected = await fetch(url + '?unsupported=value');
        assert.strictEqual(rejected.status, 400, 'Local setup must preserve the production handler route validation');
        await rejected.text();
        assert.strictEqual(invocations.length, 1, 'Rejected queries must not invoke Lambda');
      });
    });
  } finally {
    Module._load = originalLoad;
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    for (const filename of Object.keys(require.cache)) {
      if (filename === require.resolve('../../api/_lib/demo-proxy') ||
        filename.startsWith(path.resolve(__dirname, '../../api/demos') + path.sep)) {
        delete require.cache[filename];
      }
    }
  }
}

(async () => {
  await testEnvironmentFiles();
  await testLocalDemoConfiguration();
  console.log('Local development environment: precedence, parsing, optional files, safe startup and real demo routing passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
