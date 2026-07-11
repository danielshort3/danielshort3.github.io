'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  AWS_WORKLOADS,
  DEFAULT_OIDC_AUDIENCE,
  describeAwsAuth,
  getAwsAuthConfig,
  getAwsAuthMode,
  getAwsClientConfig,
  isHostedVercelRuntime,
  maskRoleArn
} = require('../api/_lib/aws-credentials');

const root = path.resolve(__dirname, '..');
const ACCOUNT_ID = '123456789012';

function roleArn(name) {
  return `arn:aws:iam::${ACCOUNT_ID}:role/${name}`;
}

function expectCode(fn, code) {
  assert.throws(fn, (err) => err && err.code === code, `Expected error code ${code}`);
}

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), 'utf8');
}

function testModeAndRuntimeBoundaries() {
  assert.strictEqual(isHostedVercelRuntime({}), false);
  assert.strictEqual(isHostedVercelRuntime({ VERCEL_ENV: 'development' }), false);
  assert.strictEqual(isHostedVercelRuntime({ VERCEL_ENV: 'preview' }), true);
  assert.strictEqual(isHostedVercelRuntime({ VERCEL: '1' }), true);
  assert.strictEqual(getAwsAuthMode({}), 'default');
  assert.strictEqual(getAwsAuthMode({ AWS_AUTH_MODE: 'legacy' }), 'legacy');
  assert.strictEqual(getAwsAuthMode({ AWS_AUTH_MODE: 'OIDC' }), 'oidc');
  expectCode(() => getAwsAuthMode({ AWS_AUTH_MODE: 'automatic' }), 'AWS_AUTH_MODE_INVALID');
  expectCode(() => getAwsAuthMode({ VERCEL_ENV: 'production' }), 'AWS_AUTH_MODE_REQUIRED');
}

function testLocalDefaultChain() {
  const resolved = getAwsClientConfig(AWS_WORKLOADS.TOOLS_STATE, {
    env: { AWS_ACCESS_KEY_ID: 'LOCAL_CHAIN_VALUE', AWS_SECRET_ACCESS_KEY: 'LOCAL_CHAIN_SECRET' },
    region: 'us-east-2'
  });
  assert.strictEqual(resolved.auth.mode, 'default');
  assert.strictEqual(resolved.auth.credentialSource, 'sdk-default-chain');
  assert.strictEqual(resolved.clientConfig.region, 'us-east-2');
  assert.strictEqual(Object.hasOwn(resolved.clientConfig, 'credentials'), false);
}

function testLegacyMigrationMode() {
  const workload = getAwsAuthConfig(AWS_WORKLOADS.SHORT_LINKS, {
    env: {
      VERCEL: '1',
      AWS_AUTH_MODE: 'legacy',
      SHORTLINKS_AWS_ACCESS_KEY_ID: 'AKIAWORKLOAD',
      SHORTLINKS_AWS_SECRET_ACCESS_KEY: 'workload-secret'
    }
  });
  assert.strictEqual(workload.credentialSource, 'short-links');
  assert.strictEqual(workload.credentials.accessKeyId, 'AKIAWORKLOAD');

  const fallback = getAwsAuthConfig(AWS_WORKLOADS.SHORT_LINKS, {
    env: {
      VERCEL: '1',
      AWS_AUTH_MODE: 'legacy',
      AWS_ACCESS_KEY_ID: 'AKIAGLOBAL',
      AWS_SECRET_ACCESS_KEY: 'global-secret'
    }
  });
  assert.strictEqual(fallback.credentialSource, 'global');

  expectCode(() => getAwsAuthConfig(AWS_WORKLOADS.SHORT_LINKS, {
    env: { VERCEL: '1', AWS_AUTH_MODE: 'legacy' }
  }), 'AWS_LEGACY_CREDENTIALS_MISSING');
  expectCode(() => getAwsAuthConfig(AWS_WORKLOADS.SHORT_LINKS, {
    env: {
      VERCEL: '1',
      AWS_AUTH_MODE: 'legacy',
      SHORTLINKS_AWS_ACCESS_KEY_ID: 'AKIAPARTIAL'
    }
  }), 'AWS_LEGACY_CREDENTIALS_PARTIAL');
}

function testOidcWorkloadIsolation() {
  const calls = [];
  const provider = () => ({ accessKeyId: 'temporary', secretAccessKey: 'temporary' });
  const env = {
    VERCEL: '1',
    VERCEL_ENV: 'production',
    AWS_AUTH_MODE: 'oidc',
    AWS_ACCESS_KEY_ID: 'MUST_NOT_BE_USED',
    AWS_SECRET_ACCESS_KEY: 'MUST_NOT_BE_USED',
    CHATBOT_BEDROCK_AWS_ROLE_ARN: roleArn('website-chatbot-bedrock-production'),
    CHATBOT_DDB_AWS_ROLE_ARN: roleArn('website-chatbot-ddb-production')
  };
  const bedrock = getAwsAuthConfig(AWS_WORKLOADS.CHATBOT_BEDROCK, {
    env,
    region: 'us-east-2',
    oidcProviderFactory: (options) => {
      calls.push(options);
      return provider;
    }
  });
  assert.strictEqual(bedrock.credentials, provider);
  assert.strictEqual(bedrock.credentialSource, 'vercel-oidc');
  assert.strictEqual(bedrock.roleArn, env.CHATBOT_BEDROCK_AWS_ROLE_ARN);
  assert.strictEqual(calls[0].audience, DEFAULT_OIDC_AUDIENCE);
  assert.strictEqual(calls[0].roleArn, env.CHATBOT_BEDROCK_AWS_ROLE_ARN);
  assert.strictEqual(calls[0].clientConfig.region, 'us-east-2');
  assert.strictEqual(calls[0].durationSeconds, 3600);

  const ddb = getAwsAuthConfig(AWS_WORKLOADS.CHATBOT_DDB, {
    env,
    oidcProviderFactory: () => provider
  });
  assert.strictEqual(ddb.roleArn, env.CHATBOT_DDB_AWS_ROLE_ARN);
  assert.notStrictEqual(ddb.roleArn, bedrock.roleArn);

  expectCode(() => getAwsAuthConfig(AWS_WORKLOADS.TOOLS_STATE, {
    env,
    oidcProviderFactory: () => provider
  }), 'AWS_ROLE_ARN_MISSING');
  expectCode(() => getAwsAuthConfig(AWS_WORKLOADS.TOOLS_STATE, {
    env: { ...env, TOOLS_AWS_ROLE_ARN: 'not-an-arn' },
    oidcProviderFactory: () => provider
  }), 'AWS_ROLE_ARN_INVALID');
  expectCode(() => getAwsAuthConfig(AWS_WORKLOADS.TOOLS_STATE, {
    env: {
      ...env,
      TOOLS_AWS_ROLE_ARN: roleArn('website-tools-state-production'),
      AWS_OIDC_AUDIENCE: 'https://vercel.com/daniel-shorts-projects'
    },
    oidcProviderFactory: () => provider
  }), 'AWS_OIDC_AUDIENCE_INVALID');
}

async function testOidcProviderRefresh() {
  let refreshCount = 0;
  const resolved = getAwsAuthConfig(AWS_WORKLOADS.TOOLS_STATE, {
    env: {
      VERCEL: '1',
      VERCEL_ENV: 'preview',
      AWS_AUTH_MODE: 'oidc',
      TOOLS_AWS_ROLE_ARN: roleArn('website-tools-state-preview')
    },
    oidcProviderFactory: () => async () => {
      refreshCount += 1;
      return {
        accessKeyId: `temporary-${refreshCount}`,
        secretAccessKey: 'temporary-secret'
      };
    }
  });
  const first = await resolved.credentials();
  const second = await resolved.credentials();
  assert.strictEqual(first.accessKeyId, 'temporary-1');
  assert.strictEqual(second.accessKeyId, 'temporary-2');
  assert.strictEqual(refreshCount, 2, 'AWS clients must retain a refreshable OIDC provider function');
}

function testSanitizedDiagnostics() {
  const role = roleArn('website-short-links-production');
  const diagnostics = describeAwsAuth(AWS_WORKLOADS.SHORT_LINKS, {
    env: {
      VERCEL: '1',
      VERCEL_ENV: 'production',
      AWS_AUTH_MODE: 'oidc',
      SHORTLINKS_AWS_ROLE_ARN: role
    },
    oidcProviderFactory: () => async () => ({})
  });
  assert.strictEqual(diagnostics.runtime, 'vercel');
  assert.strictEqual(diagnostics.environment, 'production');
  assert.strictEqual(diagnostics.mode, 'oidc');
  assert.strictEqual(diagnostics.role, 'arn:aws:iam::********9012:role/website-short-links-production');
  assert.strictEqual(maskRoleArn(role), diagnostics.role);
  assert(!JSON.stringify(diagnostics).includes(ACCOUNT_ID));

  const health = read('api/short-links/health.js');
  ['secretFingerprint', 'accessKeyLength', 'secretLength', 'sessionTokenLength', 'maskAccessKeyId'].forEach((value) => {
    assert(!health.includes(value), `Short Links health must not expose ${value}`);
  });
}

function testCredentialConsumersUseCentralAdapter() {
  const consumers = [
    'api/chatbot.js',
    'api/_lib/chatbot-logs.js',
    'api/_lib/chatbot-rate-limit.js',
    'api/_lib/short-links-store.js',
    'api/_lib/tools-store-ddb.js',
    'api/_lib/tools-endpoints/transcribe.js',
    'api/short-links/health.js',
    'build/generate-chatbot-knowledge.js'
  ];
  consumers.forEach((file) => {
    const source = read(file);
    assert(source.includes('aws-credentials'), `${file} must use the central AWS credential adapter`);
    assert(!source.includes('AWS_ACCESS_KEY_ID'), `${file} must not implement a direct global-key fallback`);
    assert(!source.includes('getAwsCredentialsFromEnv'), `${file} must not implement a static credential helper`);
  });
}

function testCloudFormationContracts() {
  const provider = read('aws/vercel-oidc/provider.template.yaml');
  const roles = read('aws/vercel-oidc/roles.template.yaml');
  const resources = read('aws/vercel-oidc/environment-resources.template.yaml');
  const jobTracker = read('aws/job-application-tracker/template.yaml');
  assert(provider.includes("https://oidc.vercel.com/${VercelTeamSlug}"));
  assert(provider.includes('sts.amazonaws.com'));
  assert(roles.includes("'oidc.vercel.com/daniel-shorts-projects:aud': !Ref OidcAudience"));
  assert(roles.includes("'oidc.vercel.com/daniel-shorts-projects:sub': !Sub 'owner:${VercelTeamSlug}:project:${VercelProjectName}:environment:${Environment}'"));
  assert.strictEqual((roles.match(/daniel-shorts-projects:aud/g) || []).length, 7, 'Every workload role must check the OIDC audience');
  assert.strictEqual((roles.match(/daniel-shorts-projects:sub/g) || []).length, 7, 'Every workload role must check the OIDC subject');
  assert(!/^\s*[^#\n]+:\s*&[A-Za-z]/m.test(roles), 'CloudFormation does not accept YAML anchors');
  assert(!/^\s*[^#\n]+:\s*\*[A-Za-z]/m.test(roles), 'CloudFormation does not accept YAML aliases');
  assert(!roles.includes('StringLike:'), 'OIDC trust policies must use exact claim matching');
  assert(!roles.includes('environment:*'), 'OIDC trust must not wildcard deployment environments');
  assert(resources.includes('IsPreview: !Equals [!Ref Environment, preview]'));
  assert(resources.includes("TableName: !Sub 'website-demo-rate-limit-${Environment}'"));
  assert(resources.includes('BucketName: !Sub \'danielshort-transcribe-tool-preview-${AWS::AccountId}-${AWS::Region}\''));
  assert(resources.includes('ExpirationInDays: 3'));
  assert(resources.includes('BlockPublicPolicy: true'));
  assert(!resources.includes('danielshort-tools-accounts\n'), 'The environment stack must not adopt the Production Tools table');
  assert(!resources.includes('danielshort-transcribe-tool-886623862678-us-east-2'), 'The environment stack must not adopt the Production Transcribe bucket');
  assert(roles.includes("Action: transcribe:StartTranscriptionJob\n                Resource: '*'"), 'Starting a Transcribe job requires the service-level wildcard');
  assert(roles.includes("Resource: !Sub 'arn:${AWS::Partition}:transcribe:${AWS::Region}:${AWS::AccountId}:transcription-job/site-transcribe-*'"), 'Get/Delete Transcribe permissions must use the slash-form ARN and remain limited to site-owned jobs');
  assert(!/transcribe:DeleteTranscriptionJob[\s\S]{0,160}Resource: '\*'/.test(roles), 'DeleteTranscriptionJob must not apply to every job in the account');
  assert(!/transcribe:GetTranscriptionJob[\s\S]{0,160}Resource: '\*'/.test(roles), 'GetTranscriptionJob must not apply to every job in the account');
  assert(jobTracker.includes("- 'arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${AliasArn}/invocations'"), 'Job Tracker API Gateway integration must invoke the live alias');
  assert(jobTracker.includes('AliasArn: !Ref TrackerFunctionLiveAlias'), 'Job Tracker integration must resolve the live alias ARN');
  assert(jobTracker.includes('FunctionName: !Ref TrackerFunctionLiveAlias'), 'Job Tracker invoke permission must be attached to the live alias');
  assert(!jobTracker.includes('functions/${TrackerFunction.Arn}/invocations'), 'Job Tracker API Gateway must not bypass the live alias');

  [
    'TOOLS_AWS_ROLE_ARN',
    'SHORTLINKS_AWS_ROLE_ARN',
    'TRANSCRIBE_AWS_ROLE_ARN',
    'CHATBOT_BEDROCK_AWS_ROLE_ARN',
    'CHATBOT_DDB_AWS_ROLE_ARN',
    'CONTACT_AWS_ROLE_ARN',
    'DEMO_INVOKE_AWS_ROLE_ARN'
  ].forEach((name) => assert(roles.includes(`Description: ${name}`), `Missing role output ${name}`));

  [
    'dynamodb:DeleteItem',
    'dynamodb:PutItem',
    'dynamodb:DescribeTable',
    'transcribe:StartTranscriptionJob',
    'bedrock:InvokeModelWithResponseStream',
    'ses:SendEmail',
    'lambda:InvokeFunction'
  ].forEach((action) => assert(roles.includes(action), `Missing least-privilege action ${action}`));

  const production = JSON.parse(read('aws/vercel-oidc/production.parameters.example.json'));
  const preview = JSON.parse(read('aws/vercel-oidc/preview.parameters.example.json'));
  const productionResources = JSON.parse(read('aws/vercel-oidc/production-resources.parameters.example.json'));
  const previewResources = JSON.parse(read('aws/vercel-oidc/preview-resources.parameters.example.json'));
  const resolution = JSON.parse(read('aws/vercel-oidc/resource-resolution.json'));
  assert(production.some((item) => item.ParameterKey === 'Environment' && item.ParameterValue === 'production'));
  assert(preview.some((item) => item.ParameterKey === 'Environment' && item.ParameterValue === 'preview'));
  assert(productionResources.some((item) => item.ParameterKey === 'Environment' && item.ParameterValue === 'production'));
  assert(previewResources.some((item) => item.ParameterKey === 'Environment' && item.ParameterValue === 'preview'));

  const toMap = (items) => new Map(items.map((item) => [item.ParameterKey, item.ParameterValue]));
  const productionMap = toMap(production);
  const previewMap = toMap(preview);
  assert.strictEqual(productionMap.size, production.length, 'Production parameters must not contain duplicate keys');
  assert.strictEqual(previewMap.size, preview.length, 'Preview parameters must not contain duplicate keys');
  const requiredRoleParameters = [
    'Environment',
    'OidcProviderArn',
    'ToolsTableName',
    'ShortLinksTableName',
    'ShortLinksClicksTableName',
    'ChatbotTableName',
    'TranscribeBucketName',
    'BedrockResourceArns',
    'SesIdentityArns',
    'DemoFunctionArns',
    'DemoRateLimitTableName'
  ].sort();
  assert.deepStrictEqual([...productionMap.keys()].sort(), requiredRoleParameters);
  assert.deepStrictEqual([...previewMap.keys()].sort(), requiredRoleParameters);
  assert.strictEqual(resolution.accountId, '886623862678');
  assert.strictEqual(resolution.region, 'us-east-2');

  const serializedParameters = JSON.stringify({ production, preview, productionResources, previewResources });
  assert(!serializedParameters.includes('REPLACE_'), 'Parameter inventories must not contain generic REPLACE placeholders');
  assert(!serializedParameters.includes('123456789012'), 'Parameter inventories must not contain the example AWS account');

  const expectedProduction = {
    OidcProviderArn: 'arn:aws:iam::886623862678:oidc-provider/oidc.vercel.com/daniel-shorts-projects',
    ToolsTableName: 'danielshort-tools-accounts',
    ShortLinksTableName: 'danielshort-short-links',
    ShortLinksClicksTableName: 'danielshort-short-links-clicks',
    ChatbotTableName: 'danielshort-chatbot-runtime',
    TranscribeBucketName: 'danielshort-transcribe-tool-886623862678-us-east-2',
    SesIdentityArns: 'arn:aws:ses:us-east-2:886623862678:identity/noreply@danielshort.me',
    DemoRateLimitTableName: 'website-demo-rate-limit-production'
  };
  Object.entries(expectedProduction).forEach(([key, value]) => {
    assert.strictEqual(productionMap.get(key), value, `Production ${key} must match the verified AWS inventory`);
  });

  const expectedBedrock = resolution.shared.BedrockResourceArns.values.join(',');
  assert.strictEqual(productionMap.get('BedrockResourceArns'), expectedBedrock);
  assert.strictEqual(previewMap.get('BedrockResourceArns'), expectedBedrock);
  assert.strictEqual(previewMap.get('OidcProviderArn'), expectedProduction.OidcProviderArn);

  const productionDemoArns = String(productionMap.get('DemoFunctionArns') || '').split(',').filter(Boolean);
  assert.strictEqual(productionDemoArns.length, 10);
  assert.deepStrictEqual(productionDemoArns, resolution.production.resolved.DemoFunctionArns);
  productionDemoArns.forEach((arn) => assert(arn.endsWith(':live'), 'Production demo permissions must target immutable live aliases'));

  const previewDemoArns = String(previewMap.get('DemoFunctionArns') || '').split(',').filter(Boolean);
  assert.strictEqual(previewDemoArns.length, 10);
  assert.deepStrictEqual(previewDemoArns, resolution.preview.resolved.DemoFunctionArns);
  previewDemoArns.forEach((arn) => assert(arn.endsWith('-preview:live'), 'Preview demo permissions must target Preview live aliases'));

  const expectedPreview = {
    ToolsTableName: 'danielshort-tools-accounts-preview',
    ShortLinksTableName: 'danielshort-short-links-preview',
    ShortLinksClicksTableName: 'danielshort-short-links-clicks-preview',
    ChatbotTableName: 'danielshort-chatbot-runtime-preview',
    TranscribeBucketName: 'danielshort-transcribe-tool-preview-886623862678-us-east-2',
    SesIdentityArns: expectedProduction.SesIdentityArns,
    DemoRateLimitTableName: 'website-demo-rate-limit-preview'
  };
  Object.entries(expectedPreview).forEach(([key, value]) => {
    assert.strictEqual(previewMap.get(key), value, `Preview ${key} must match its environment-resource stack output`);
  });

  ['ToolsTableName', 'ShortLinksTableName', 'ShortLinksClicksTableName', 'ChatbotTableName', 'TranscribeBucketName'].forEach((key) => {
    assert.notStrictEqual(previewMap.get(key), productionMap.get(key), `Preview ${key} must not point at Production`);
  });

  const parameterSentinels = [...productionMap.values(), ...previewMap.values()]
    .filter((value) => String(value).startsWith('__UNRESOLVED_'))
    .sort();
  const documentedSentinels = [
    ...Object.keys(resolution.production.unresolved),
    ...Object.keys(resolution.preview.unresolved)
  ].sort();
  assert.deepStrictEqual(parameterSentinels, documentedSentinels, 'Every unresolved parameter must have one resolution record');
  assert.strictEqual(resolution.production.state, 'deployed-verified');
  assert.strictEqual(resolution.preview.state, 'deployed-verified');
  assert.strictEqual(resolution.shared.OidcProviderArn.state, 'live-verified');
  assert.strictEqual(resolution.shared.OidcProviderArn.stackStatus, 'CREATE_COMPLETE');
  assert.strictEqual(resolution.preview.resolved.ToolsTableName, expectedPreview.ToolsTableName);
  assert.strictEqual(resolution.preview.resolved.TranscribeBucketName, expectedPreview.TranscribeBucketName);
  assert.strictEqual(resolution.preview.resolved.DemoRateLimitTableName, expectedPreview.DemoRateLimitTableName);
  assert.strictEqual(resolution.production.resolved.DemoRateLimitTableName, 'website-demo-rate-limit-production');
  assert.strictEqual(resolution.production.deployedStacks.resources.name, 'website-production-resources');
  assert.strictEqual(resolution.production.deployedStacks.resources.status, 'CREATE_COMPLETE');
  assert.strictEqual(resolution.production.deployedStacks.roles.name, 'website-vercel-oidc-production');
  assert.strictEqual(resolution.production.deployedStacks.roles.status, 'UPDATE_COMPLETE');
  assert.deepStrictEqual(resolution.production.planned, {});
  assert.deepStrictEqual(resolution.preview.planned, {});
  assert.deepStrictEqual(resolution.production.unresolved, {});
  assert.deepStrictEqual(resolution.preview.unresolved, {});
  assert.deepStrictEqual(parameterSentinels, []);
  documentedSentinels.forEach((sentinel) => {
    const item = resolution.production.unresolved[sentinel] || resolution.preview.unresolved[sentinel];
    assert(item.owner, `${sentinel} must identify its future stack owner`);
  });
}

function testEnvironmentContract() {
  const example = read('.env.example');
  [
    'AWS_AUTH_MODE=',
    'AWS_OIDC_AUDIENCE=sts.amazonaws.com',
    'TOOLS_AWS_ROLE_ARN=',
    'SHORTLINKS_AWS_ROLE_ARN=',
    'TRANSCRIBE_AWS_ROLE_ARN=',
    'CHATBOT_BEDROCK_AWS_ROLE_ARN=',
    'CHATBOT_DDB_AWS_ROLE_ARN=',
    'CONTACT_AWS_ROLE_ARN=',
    'DEMO_INVOKE_AWS_ROLE_ARN='
  ].forEach((value) => assert(example.includes(value), `.env.example is missing ${value}`));
}

async function main() {
  testModeAndRuntimeBoundaries();
  testLocalDefaultChain();
  testLegacyMigrationMode();
  testOidcWorkloadIsolation();
  await testOidcProviderRefresh();
  testSanitizedDiagnostics();
  testCredentialConsumersUseCentralAdapter();
  testCloudFormationContracts();
  testEnvironmentContract();
  process.stdout.write('[aws-credentials] OIDC, legacy rollback, local profile, diagnostics, and template contracts passed.\n');
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exitCode = 1;
});
