'use strict';

const {
  GetFunctionUrlConfigCommand,
  GetPolicyCommand,
  InvokeCommand,
  LambdaClient
} = require('@aws-sdk/client-lambda');

const REGION = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || 'us-east-2';
const ENVIRONMENT = String(process.env.WORKLOAD_ENVIRONMENT || 'preview').trim().toLowerCase();
if (!['preview', 'production'].includes(ENVIRONMENT)) {
  throw new Error(`WORKLOAD_ENVIRONMENT must be preview or production, received: ${ENVIRONMENT}`);
}
const EXPECT_PRIVATE = String(process.env.EXPECT_PRIVATE_LAMBDAS || 'true').trim().toLowerCase() !== 'false';
const FUNCTION_SUFFIX = ENVIRONMENT === 'preview' ? '-preview' : '';
const PIXEL = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=';

const CANARIES = Object.freeze([
  {
    name: 'shape-analyzer',
    method: 'POST',
    path: '/predict',
    body: { b64: PIXEL },
    timeoutMs: 60_000
  },
  {
    name: 'smart-sentence-finder',
    method: 'POST',
    path: '/rank',
    body: { query: 'She wonders about things.', top: 5 },
    timeoutMs: 90_000
  },
  {
    name: 'nonogram-solver',
    method: 'POST',
    path: '/solve',
    body: {},
    timeoutMs: 60_000
  },
  {
    name: 'handwriting-rating-demo',
    method: 'POST',
    path: '/score',
    body: { b64: PIXEL, image: PIXEL },
    timeoutMs: 60_000
  },
  {
    name: 'digit-generator',
    method: 'POST',
    path: '/generate',
    body: { seed: 1, mode: 'cluster', cluster_digit: 0, dim: 0, value: 0, rows: 1, cols: 1 },
    timeoutMs: 60_000
  },
  {
    name: 'covid-outbreak-drivers',
    method: 'GET',
    path: '/meta',
    timeoutMs: 60_000
  },
  {
    name: 'pizza-tips-predict',
    method: 'POST',
    path: '/predict',
    body: {
      latitude: 33.110288,
      longitude: -96.824164,
      confidenceLevel: 0.8,
      cost: 35,
      housing: 'Residential',
      orderHour: 18,
      deliveryMinutes: 40,
      rain: 0,
      maxTemp: 85,
      minTemp: 65
    },
    timeoutMs: 60_000
  },
  {
    name: 'target-empty-package',
    method: 'GET',
    path: '/data',
    timeoutMs: 60_000
  },
  {
    name: 'retail-loss-sales',
    method: 'GET',
    path: '/data',
    timeoutMs: 60_000
  },
  {
    name: 'minesweeper-solver',
    method: 'POST',
    path: '/solve',
    body: { grid: 9, mines: 10, max_steps: 1 },
    timeoutMs: 60_000
  }
].map((canary) => Object.freeze({ ...canary, name: `${canary.name}${FUNCTION_SUFFIX}` })));

function eventFor(canary) {
  const body = Object.prototype.hasOwnProperty.call(canary, 'body')
    ? JSON.stringify(canary.body)
    : null;
  return {
    version: '2.0',
    routeKey: `${canary.method} ${canary.path}`,
    rawPath: canary.path,
    rawQueryString: '',
    headers: {
      accept: 'application/json',
      'content-type': 'application/json',
      origin: 'https://danielshort.me',
      'user-agent': `website-${ENVIRONMENT}-migration-canary/1.0`
    },
    requestContext: {
      http: {
        method: canary.method,
        path: canary.path,
        protocol: 'HTTP/1.1',
        sourceIp: '127.0.0.1',
        userAgent: `website-${ENVIRONMENT}-migration-canary/1.0`
      }
    },
    body,
    isBase64Encoded: false
  };
}

function responseSummary(payload) {
  const raw = Buffer.from(payload || []).toString('utf8');
  if (!raw) throw new Error('empty Lambda response');
  const value = JSON.parse(raw);
  const statusCode = Number(value && value.statusCode) || 200;
  let body = value && Object.prototype.hasOwnProperty.call(value, 'body') ? value.body : value;
  if (value && value.isBase64Encoded && typeof body === 'string') {
    body = Buffer.from(body, 'base64').toString('utf8');
  }
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch {
      // A non-JSON response is still valid if the Lambda reports a successful status.
    }
  }
  if (statusCode < 200 || statusCode >= 300) {
    throw new Error(`Lambda returned HTTP ${statusCode}`);
  }
  const bodyBytes = Buffer.byteLength(typeof body === 'string' ? body : JSON.stringify(body ?? null));
  const keys = body && typeof body === 'object' && !Array.isArray(body)
    ? Object.keys(body).sort().slice(0, 20)
    : [];
  return { statusCode, bodyBytes, keys };
}

async function invoke(client, canary) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), canary.timeoutMs);
  try {
    const response = await client.send(new InvokeCommand({
      FunctionName: canary.name,
      Qualifier: 'live',
      InvocationType: 'RequestResponse',
      LogType: 'None',
      Payload: Buffer.from(JSON.stringify(eventFor(canary)))
    }), { abortSignal: controller.signal });
    if (response.FunctionError) throw new Error(`Lambda FunctionError: ${response.FunctionError}`);
    return responseSummary(response.Payload);
  } finally {
    clearTimeout(timeout);
  }
}

async function expectNoPublicUrl(client, canary) {
  try {
    await client.send(new GetFunctionUrlConfigCommand({ FunctionName: canary.name, Qualifier: 'live' }));
    return false;
  } catch (error) {
    if (error && (error.name === 'ResourceNotFoundException' || Number(error.$metadata && error.$metadata.httpStatusCode) === 404)) {
      return true;
    }
    throw error;
  }
}

async function expectNoResourcePolicy(client, canary) {
  try {
    const response = await client.send(new GetPolicyCommand({ FunctionName: canary.name, Qualifier: 'live' }));
    if (!response.Policy) return true;
    const policy = JSON.parse(response.Policy);
    const statements = Array.isArray(policy.Statement) ? policy.Statement : [];
    return !statements.some((statement) => {
      const principal = statement && statement.Principal;
      return principal === '*' || (principal && principal.AWS === '*');
    });
  } catch (error) {
    if (error && (error.name === 'ResourceNotFoundException' || Number(error.$metadata && error.$metadata.httpStatusCode) === 404)) {
      return true;
    }
    throw error;
  }
}

async function main() {
  const client = new LambdaClient({ region: REGION });
  const filter = String(process.env.LAMBDA_CANARY || process.env.PREVIEW_LAMBDA_CANARY || '').trim().toLowerCase();
  const canaries = filter
    ? CANARIES.filter((canary) => canary.name.toLowerCase().includes(filter))
    : CANARIES;
  if (!canaries.length) throw new Error(`No ${ENVIRONMENT} Lambda canary matched: ${filter}`);
  const results = [];
  for (const canary of canaries) {
    const startedAt = Date.now();
    try {
      const response = await invoke(client, canary);
      const noFunctionUrl = EXPECT_PRIVATE ? await expectNoPublicUrl(client, canary) : null;
      const noPublicPolicy = EXPECT_PRIVATE ? await expectNoResourcePolicy(client, canary) : null;
      const passed = !EXPECT_PRIVATE || (noFunctionUrl && noPublicPolicy);
      results.push({
        functionName: canary.name,
        operation: `${canary.method} ${canary.path}`,
        passed,
        durationMs: Date.now() - startedAt,
        noFunctionUrl,
        noPublicPolicy,
        response
      });
    } catch (error) {
      results.push({
        functionName: canary.name,
        operation: `${canary.method} ${canary.path}`,
        passed: false,
        durationMs: Date.now() - startedAt,
        error: String(error && (error.message || error))
      });
    }
  }
  const output = {
    checkedAt: new Date().toISOString(),
    region: REGION,
    environment: ENVIRONMENT,
    expectedPrivate: EXPECT_PRIVATE,
    passed: results.every((result) => result.passed),
    results
  };
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
  if (!output.passed) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
