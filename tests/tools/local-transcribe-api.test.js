'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');

const localTranscribe = require('../../api/_lib/local-transcribe');
const internal = localTranscribe._internal;

function createResponse(){
  return {
    statusCode: 200,
    headers: {},
    body: '',
    setHeader(name, value){
      this.headers[String(name).toLowerCase()] = value;
    },
    end(value){
      this.body = String(value || '');
    }
  };
}

function json(response){
  return JSON.parse(response.body);
}

function request(method, body, overrides = {}){
  return {
    method,
    socket: { encrypted: true },
    body,
    ...overrides,
    headers: {
      host: 'www.danielshort.me',
      origin: 'https://www.danielshort.me',
      ...overrides.headers
    }
  };
}

function enabledEnv(overrides = {}){
  return {
    LOCAL_TRANSCRIBE_SHARED_SECRET: 'test-local-secret-with-at-least-32-bytes',
    LOCAL_TRANSCRIBE_WORKER_ORIGIN: 'https://gpu.example.test',
    NODE_ENV: 'production',
    ...overrides
  };
}

async function run(){
  const defaults = internal.getLocalTranscribeConfig({
    LOCAL_TRANSCRIBE_SHARED_SECRET: 'x'.repeat(32)
  });
  assert.strictEqual(defaults.configured, true);
  assert.strictEqual(defaults.workerOrigin, internal.DEFAULT_WORKER_ORIGIN);
  assert.strictEqual(defaults.chunkBytes, 8 * 1024 * 1024);
  assert.strictEqual(defaults.ticketTtlSeconds, 6 * 60 * 60);
  assert(defaults.adminGroups.has('admin') && defaults.adminGroups.has('admins'));
  assert(defaults.adminEmails.has('daniel@danielshort.me'));
  assert(defaults.adminEmails.has('danielshort3@gmail.com'));

  const shortSecret = internal.getLocalTranscribeConfig({
    LOCAL_TRANSCRIBE_SHARED_SECRET: 'x'.repeat(31)
  });
  assert.strictEqual(shortSecret.configured, false);
  assert(shortSecret.disabledReason.includes('32 UTF-8 bytes'));
  const multibyteSecret = internal.getLocalTranscribeConfig({
    LOCAL_TRANSCRIBE_SHARED_SECRET: '\u00e9'.repeat(16)
  });
  assert.strictEqual(multibyteSecret.configured, true, 'secret length should be measured in UTF-8 bytes');

  const productionLoopback = internal.getLocalTranscribeConfig({
    LOCAL_TRANSCRIBE_SHARED_SECRET: 'x'.repeat(32),
    VERCEL_ENV: 'production'
  });
  assert.strictEqual(productionLoopback.configured, false);
  assert(productionLoopback.disabledReason.includes('Loopback workers are disabled in production'));
  const allowedProductionLoopback = internal.getLocalTranscribeConfig({
    LOCAL_TRANSCRIBE_SHARED_SECRET: 'x'.repeat(32),
    VERCEL_ENV: 'production',
    LOCAL_TRANSCRIBE_ALLOW_LOOPBACK: 'true'
  });
  assert.strictEqual(allowedProductionLoopback.configured, true);

  const invalidOrigin = internal.getLocalTranscribeConfig({
    LOCAL_TRANSCRIBE_SHARED_SECRET: 'x'.repeat(32),
    LOCAL_TRANSCRIBE_WORKER_ORIGIN: 'http://gpu.example.test:8765/path'
  });
  assert.strictEqual(invalidOrigin.configured, false);
  assert.strictEqual(invalidOrigin.workerOrigin, '');
  assert.strictEqual(internal.normalizeWorkerOrigin('https://gpu.example.test/'), 'https://gpu.example.test');
  assert.strictEqual(internal.normalizeWorkerOrigin('javascript:alert(1)'), '');
  assert.strictEqual(internal.normalizeWorkerOrigin('http://192.168.1.50:8765'), '');

  const overrideConfig = internal.getLocalTranscribeConfig(enabledEnv({
    LOCAL_TRANSCRIBE_ADMIN_GROUPS: 'operators',
    LOCAL_TRANSCRIBE_ADMIN_EMAILS: 'owner@example.com',
    LOCAL_TRANSCRIBE_TICKET_TTL_SECONDS: '1',
    LOCAL_TRANSCRIBE_CHUNK_BYTES: '1048576'
  }));
  assert.strictEqual(overrideConfig.ticketTtlSeconds, internal.MIN_TICKET_TTL_SECONDS);
  assert.strictEqual(overrideConfig.chunkBytes, 8 * 1024 * 1024, 'chunk size is a fixed browser/worker contract');
  assert.strictEqual(internal.isAdminClaims({ 'cognito:groups': ['Operators'] }, overrideConfig), true);
  assert.strictEqual(internal.isAdminClaims({ email: 'OWNER@EXAMPLE.COM' }, overrideConfig), false);
  assert.strictEqual(internal.isAdminClaims({ email: 'OWNER@EXAMPLE.COM', email_verified: false }, overrideConfig), false);
  assert.strictEqual(internal.isAdminClaims({ email: 'OWNER@EXAMPLE.COM', email_verified: 'false' }, overrideConfig), false);
  assert.strictEqual(internal.isAdminClaims({ email: 'OWNER@EXAMPLE.COM', email_verified: true }, overrideConfig), true);
  assert.strictEqual(internal.isAdminClaims({ email: 'OWNER@EXAMPLE.COM', email_verified: 'true' }, overrideConfig), true);
  assert.strictEqual(internal.isAdminClaims({ email: 'OWNER@EXAMPLE.COM', email_verified: 'TRUE' }, overrideConfig), false);
  assert.strictEqual(internal.isAdminClaims({ email: 'daniel@danielshort.me' }, overrideConfig), false);
  assert.strictEqual(internal.isAdminClaims({ 'cognito:groups': ['admin'] }, overrideConfig), false);
  const maxTtlConfig = internal.getLocalTranscribeConfig(enabledEnv({
    LOCAL_TRANSCRIBE_TICKET_TTL_SECONDS: String(24 * 60 * 60),
    LOCAL_TRANSCRIBE_MAX_FILE_BYTES: String(2 * 1024 * 1024 * 1024),
    LOCAL_TRANSCRIBE_MAX_DURATION_SECONDS: String(24 * 60 * 60)
  }));
  assert.strictEqual(maxTtlConfig.ticketTtlSeconds, internal.MAX_TICKET_TTL_SECONDS);
  assert.strictEqual(maxTtlConfig.maxFileBytes, 500 * 1024 * 1024);
  assert.strictEqual(maxTtlConfig.maxServiceDurationSeconds, 8 * 60 * 60);
  assert.deepStrictEqual(internal.validateTicketInput({
    filename: 'boundary.mp4',
    contentType: 'video/mp4',
    bytes: 500 * 1024 * 1024,
    durationSeconds: 8 * 60 * 60
  }, maxTtlConfig), {
    filename: 'boundary.mp4',
    format: 'mp4',
    contentType: 'video/mp4',
    bytes: 500 * 1024 * 1024,
    durationSeconds: 8 * 60 * 60
  });
  assert.throws(() => internal.validateTicketInput({
    filename: 'too-large.mp4',
    bytes: 500 * 1024 * 1024 + 1,
    durationSeconds: 8 * 60 * 60
  }, maxTtlConfig), (error) => error?.statusCode === 413);
  assert.throws(() => internal.validateTicketInput({
    filename: 'too-long.mp4',
    bytes: 500 * 1024 * 1024,
    durationSeconds: 8 * 60 * 60 + 0.001
  }, maxTtlConfig), /local processing limit/);
  const lowerWorkerLimits = internal.getLocalTranscribeConfig(enabledEnv({
    LOCAL_TRANSCRIBE_MAX_FILE_BYTES: String(100 * 1024 * 1024),
    LOCAL_TRANSCRIBE_MAX_DURATION_SECONDS: '3600'
  }));
  assert.strictEqual(lowerWorkerLimits.maxFileBytes, 100 * 1024 * 1024);
  assert.strictEqual(lowerWorkerLimits.maxServiceDurationSeconds, 3600);

  const adminClaims = {
    sub: 'admin-subject-123',
    email: 'person@example.com',
    'cognito:groups': ['Admins']
  };
  let authOptions;
  const handler = localTranscribe.createHandler({
    env: enabledEnv(),
    authenticateRequest: async (_req, options) => {
      authOptions = options;
      return { source: 'cookie', claims: adminClaims };
    },
    nowSeconds: () => 2_000_000_000,
    randomBytes: (size) => Buffer.alloc(size, 0xab)
  });

  const configResponse = createResponse();
  await handler(request('GET'), configResponse, 'config');
  assert.strictEqual(configResponse.statusCode, 200);
  assert.deepStrictEqual(authOptions, { allowBearer: true });
  assert.deepStrictEqual(json(configResponse), {
    ok: true,
    enabled: true,
    configured: true,
    disabledReason: '',
    service: 'Local GPU (RTX 5090)',
    workerOrigin: 'https://gpu.example.test',
    chunkBytes: 8 * 1024 * 1024,
    maxFilesPerRun: 10,
    maxFileBytes: 500 * 1024 * 1024,
    maxServiceDurationSeconds: 8 * 60 * 60,
    minDurationSeconds: 15,
    supportedFormats: ['amr', 'flac', 'm4a', 'mp3', 'mp4', 'ogg', 'wav', 'webm'],
    historyStored: false
  });
  assert.strictEqual(configResponse.headers['cache-control'], 'no-store');

  const ticketResponse = createResponse();
  await handler(request('POST', {
    filename: 'training video.mp4',
    format: 'mp4',
    contentType: 'video/mp4',
    bytes: 12_345_678,
    durationSeconds: 901.23456
  }), ticketResponse, 'ticket');
  assert.strictEqual(ticketResponse.statusCode, 200);
  const ticketResult = json(ticketResponse);
  assert.strictEqual(ticketResult.workerOrigin, 'https://gpu.example.test');
  assert.strictEqual(ticketResult.chunkBytes, 8 * 1024 * 1024);
  assert.strictEqual(ticketResult.job.id, 'ab'.repeat(16));
  assert.strictEqual(ticketResult.job.durationSeconds, 901.235);
  assert.strictEqual(ticketResult.expiresAt, (2_000_000_000 + 6 * 60 * 60) * 1000);

  const [body, signature] = ticketResult.ticket.split('.');
  const expectedSignature = crypto.createHmac(
    'sha256',
    Buffer.from(enabledEnv().LOCAL_TRANSCRIBE_SHARED_SECRET, 'utf8')
  ).update(body, 'utf8').digest('base64url');
  assert.strictEqual(signature, expectedSignature);
  const payload = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
  assert.deepStrictEqual(Object.keys(payload), [
    'v',
    'type',
    'aud',
    'sub',
    'jobId',
    'filename',
    'format',
    'contentType',
    'bytes',
    'durationSeconds',
    'origin',
    'iat',
    'exp'
  ]);
  assert.deepStrictEqual(payload, {
    v: 1,
    type: 'local_transcribe',
    aud: 'local-transcribe-worker',
    sub: 'admin-subject-123',
    jobId: 'ab'.repeat(16),
    filename: 'training video.mp4',
    format: 'mp4',
    contentType: 'video/mp4',
    bytes: 12_345_678,
    durationSeconds: 901.235,
    origin: 'https://www.danielshort.me',
    iat: 2_000_000_000,
    exp: 2_000_000_000 + 6 * 60 * 60
  });

  const nonAdminHandler = localTranscribe.createHandler({
    env: enabledEnv(),
    authenticateRequest: async () => ({ claims: { sub: 'normal-user', email: 'person@example.com' } })
  });
  const forbiddenResponse = createResponse();
  await nonAdminHandler(request('GET'), forbiddenResponse, 'config');
  assert.strictEqual(forbiddenResponse.statusCode, 403);

  const emailAdminEnv = enabledEnv({
    LOCAL_TRANSCRIBE_ADMIN_GROUPS: '',
    LOCAL_TRANSCRIBE_ADMIN_EMAILS: 'owner@example.com'
  });
  const emailVerificationCases = [
    [undefined, 403],
    [false, 403],
    ['false', 403],
    [true, 200],
    ['true', 200]
  ];
  for (const [verified, expectedStatus] of emailVerificationCases) {
    const claims = {
      sub: 'email-admin-subject',
      email: 'owner@example.com',
      ...(typeof verified === 'undefined' ? {} : { email_verified: verified })
    };
    const emailAdminHandler = localTranscribe.createHandler({
      env: emailAdminEnv,
      authenticateRequest: async () => ({ claims })
    });
    const response = createResponse();
    await emailAdminHandler(request('GET'), response, 'config');
    assert.strictEqual(response.statusCode, expectedStatus, `email_verified=${String(verified)} authorization mismatch`);
  }

  const legacyCookieClaims = {
    sub: 'legacy-email-admin',
    email: 'owner@example.com',
    email_verified: false
  };
  let migrationVerifierCalls = 0;
  let migrationSessionCalls = 0;
  const migrationHandler = localTranscribe.createHandler({
    env: emailAdminEnv,
    authenticateRequest: async () => ({ source: 'cookie', claims: legacyCookieClaims }),
    verifyToken: async (token) => {
      migrationVerifierCalls += 1;
      assert.strictEqual(token, 'current-cognito-id-token');
      return { ...legacyCookieClaims, email_verified: true };
    },
    createSession: (claims) => {
      migrationSessionCalls += 1;
      assert.strictEqual(claims.email_verified, true);
      return { cookie: '__Host-tools_session=rotated; Path=/; HttpOnly; Secure; SameSite=Lax' };
    }
  });
  const migratedResponse = createResponse();
  await migrationHandler(request('GET', null, {
    headers: { authorization: 'Bearer current-cognito-id-token' }
  }), migratedResponse, 'config');
  assert.strictEqual(migratedResponse.statusCode, 200);
  assert.strictEqual(migrationVerifierCalls, 1);
  assert.strictEqual(migrationSessionCalls, 1);
  assert(String(migratedResponse.headers['set-cookie']).includes('__Host-tools_session=rotated'));

  const noBearerMigrationHandler = localTranscribe.createHandler({
    env: emailAdminEnv,
    authenticateRequest: async () => ({ source: 'cookie', claims: legacyCookieClaims }),
    verifyToken: async () => {
      throw new Error('Verifier must not run without a Bearer token.');
    },
    createSession: () => {
      throw new Error('Session must not rotate without verified claims.');
    }
  });
  const noBearerMigrationResponse = createResponse();
  await noBearerMigrationHandler(request('GET'), noBearerMigrationResponse, 'config');
  assert.strictEqual(noBearerMigrationResponse.statusCode, 403);
  assert.strictEqual(noBearerMigrationResponse.headers['set-cookie'], undefined);

  for (const invalidVerifiedClaims of [
    { ...legacyCookieClaims, sub: 'different-subject', email_verified: true },
    { ...legacyCookieClaims, email: 'different@example.com', email_verified: true },
    { ...legacyCookieClaims, email_verified: false }
  ]) {
    let unsafeRotationCalls = 0;
    const rejectedMigrationHandler = localTranscribe.createHandler({
      env: emailAdminEnv,
      authenticateRequest: async () => ({ source: 'cookie', claims: legacyCookieClaims }),
      verifyToken: async () => invalidVerifiedClaims,
      createSession: () => {
        unsafeRotationCalls += 1;
        return { cookie: 'must-not-be-set' };
      }
    });
    const response = createResponse();
    await rejectedMigrationHandler(request('GET', null, {
      headers: { authorization: 'Bearer current-cognito-id-token' }
    }), response, 'config');
    assert.strictEqual(response.statusCode, 403);
    assert.strictEqual(unsafeRotationCalls, 0);
    assert.strictEqual(response.headers['set-cookie'], undefined);
  }

  const unauthenticatedHandler = localTranscribe.createHandler({
    env: enabledEnv(),
    authenticateRequest: async () => {
      const err = new Error('Unauthorized');
      err.code = 'AUTH_UNAUTHORIZED';
      throw err;
    }
  });
  const unauthorizedResponse = createResponse();
  await unauthenticatedHandler(request('GET'), unauthorizedResponse, 'config');
  assert.strictEqual(unauthorizedResponse.statusCode, 401);

  const invalidRequests = [
    [{ filename: 'file.exe', bytes: 10, durationSeconds: 20 }, 400, 'Unsupported file format'],
    [{ filename: 'file.mp4', format: 'webm', bytes: 10, durationSeconds: 20 }, 400, 'match the filename'],
    [{ filename: 'file.mp4', bytes: 500 * 1024 * 1024 + 1, durationSeconds: 20 }, 413, 'File exceeds'],
    [{ filename: 'file.mp4', bytes: 10, durationSeconds: 14.9 }, 400, 'at least 15 seconds'],
    [{ filename: 'file.mp4', bytes: 10, durationSeconds: 8 * 60 * 60 + 1 }, 400, 'local processing limit']
  ];
  for (const [bodyInput, status, message] of invalidRequests) {
    const response = createResponse();
    await handler(request('POST', bodyInput), response, 'ticket');
    assert.strictEqual(response.statusCode, status);
    assert(json(response).error.includes(message));
  }

  const crossOriginResponse = createResponse();
  await handler(request('POST', {
    filename: 'file.mp4',
    contentType: 'video/mp4',
    bytes: 10,
    durationSeconds: 20
  }, { headers: { origin: 'https://attacker.example' } }), crossOriginResponse, 'ticket');
  assert.strictEqual(crossOriginResponse.statusCode, 403);

  const disabledHandler = localTranscribe.createHandler({
    env: { LOCAL_TRANSCRIBE_SHARED_SECRET: 'short' },
    authenticateRequest: async () => ({ claims: adminClaims })
  });
  const disabledResponse = createResponse();
  await disabledHandler(request('POST', {}), disabledResponse, 'ticket');
  assert.strictEqual(disabledResponse.statusCode, 503);
  assert(json(disabledResponse).error.includes('32 UTF-8 bytes'));

  const methodResponse = createResponse();
  await handler(request('POST'), methodResponse, 'config');
  assert.strictEqual(methodResponse.statusCode, 405);
  assert.strictEqual(methodResponse.headers.allow, 'GET');

  const routerSource = fs.readFileSync('api/_lib/tools-endpoints/transcribe.js', 'utf8');
  assert(routerSource.includes("action === 'local-config'"));
  assert(routerSource.includes("action === 'local-ticket'"));

  console.log('local-transcribe-api tests passed');
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
