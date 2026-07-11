'use strict';

const { awsCredentialsProvider } = require('@vercel/oidc-aws-credentials-provider');

const DEFAULT_OIDC_AUDIENCE = 'sts.amazonaws.com';

const AWS_WORKLOADS = Object.freeze({
  CHATBOT_BEDROCK: 'chatbot-bedrock',
  CHATBOT_DDB: 'chatbot-ddb',
  CONTACT: 'contact',
  DEMO_INVOKE: 'demo-invoke',
  SHORT_LINKS: 'short-links',
  TOOLS_STATE: 'tools-state',
  TRANSCRIBE: 'transcribe'
});

const LEGACY_GLOBAL_SOURCE = Object.freeze({
  name: 'global',
  accessKeyIdEnv: 'AWS_ACCESS_KEY_ID',
  secretAccessKeyEnv: 'AWS_SECRET_ACCESS_KEY',
  sessionTokenEnv: 'AWS_SESSION_TOKEN'
});

function legacySource(name, prefix) {
  return Object.freeze({
    name,
    accessKeyIdEnv: `${prefix}_AWS_ACCESS_KEY_ID`,
    secretAccessKeyEnv: `${prefix}_AWS_SECRET_ACCESS_KEY`,
    sessionTokenEnv: `${prefix}_AWS_SESSION_TOKEN`
  });
}

const WORKLOAD_CONFIG = Object.freeze({
  [AWS_WORKLOADS.CHATBOT_BEDROCK]: Object.freeze({
    roleArnEnv: 'CHATBOT_BEDROCK_AWS_ROLE_ARN',
    sessionName: 'website-chatbot-bedrock',
    legacySources: Object.freeze([legacySource('chatbot', 'CHATBOT'), LEGACY_GLOBAL_SOURCE])
  }),
  [AWS_WORKLOADS.CHATBOT_DDB]: Object.freeze({
    roleArnEnv: 'CHATBOT_DDB_AWS_ROLE_ARN',
    sessionName: 'website-chatbot-ddb',
    legacySources: Object.freeze([legacySource('chatbot', 'CHATBOT'), LEGACY_GLOBAL_SOURCE])
  }),
  [AWS_WORKLOADS.CONTACT]: Object.freeze({
    roleArnEnv: 'CONTACT_AWS_ROLE_ARN',
    sessionName: 'website-contact',
    legacySources: Object.freeze([legacySource('contact', 'CONTACT'), LEGACY_GLOBAL_SOURCE])
  }),
  [AWS_WORKLOADS.DEMO_INVOKE]: Object.freeze({
    roleArnEnv: 'DEMO_INVOKE_AWS_ROLE_ARN',
    sessionName: 'website-demo-invoke',
    legacySources: Object.freeze([legacySource('demo', 'DEMO_INVOKE'), LEGACY_GLOBAL_SOURCE])
  }),
  [AWS_WORKLOADS.SHORT_LINKS]: Object.freeze({
    roleArnEnv: 'SHORTLINKS_AWS_ROLE_ARN',
    sessionName: 'website-short-links',
    legacySources: Object.freeze([legacySource('short-links', 'SHORTLINKS'), LEGACY_GLOBAL_SOURCE])
  }),
  [AWS_WORKLOADS.TOOLS_STATE]: Object.freeze({
    roleArnEnv: 'TOOLS_AWS_ROLE_ARN',
    sessionName: 'website-tools-state',
    legacySources: Object.freeze([legacySource('tools', 'TOOLS'), LEGACY_GLOBAL_SOURCE])
  }),
  [AWS_WORKLOADS.TRANSCRIBE]: Object.freeze({
    roleArnEnv: 'TRANSCRIBE_AWS_ROLE_ARN',
    sessionName: 'website-transcribe',
    // Keep the historical Tools/global chain only while AWS_AUTH_MODE=legacy.
    legacySources: Object.freeze([
      legacySource('transcribe', 'TRANSCRIBE'),
      legacySource('tools', 'TOOLS'),
      LEGACY_GLOBAL_SOURCE
    ])
  })
});

function valueFromEnv(env, key) {
  const raw = env && typeof env[key] !== 'undefined' ? String(env[key]) : '';
  return raw.trim();
}

function isHostedVercelRuntime(env = process.env) {
  const vercelFlag = valueFromEnv(env, 'VERCEL').toLowerCase();
  const vercelEnvironment = valueFromEnv(env, 'VERCEL_ENV').toLowerCase();
  return vercelFlag === '1' || vercelFlag === 'true' || ['preview', 'production'].includes(vercelEnvironment);
}

function authError(code, message) {
  const err = new Error(message);
  err.code = code;
  return err;
}

function getWorkloadConfig(workload) {
  const config = WORKLOAD_CONFIG[workload];
  if (!config) {
    throw authError('AWS_WORKLOAD_INVALID', `Unknown AWS workload: ${String(workload || '')}`);
  }
  return config;
}

function getAwsAuthMode(env = process.env) {
  const configured = valueFromEnv(env, 'AWS_AUTH_MODE').toLowerCase();
  if (configured) {
    if (configured !== 'legacy' && configured !== 'oidc') {
      throw authError('AWS_AUTH_MODE_INVALID', 'AWS_AUTH_MODE must be either legacy or oidc.');
    }
    return configured;
  }
  if (isHostedVercelRuntime(env)) {
    throw authError(
      'AWS_AUTH_MODE_REQUIRED',
      'AWS_AUTH_MODE must be explicitly configured for hosted Vercel deployments.'
    );
  }
  return 'default';
}

function isRoleArn(value) {
  return /^arn:(?:aws|aws-us-gov|aws-cn):iam::\d{12}:role\/[A-Za-z0-9+=,.@_\/-]+$/.test(value);
}

function resolveLegacyCredentials(config, env, hosted) {
  for (const source of config.legacySources) {
    const accessKeyId = valueFromEnv(env, source.accessKeyIdEnv);
    const secretAccessKey = valueFromEnv(env, source.secretAccessKeyEnv);
    const sessionToken = valueFromEnv(env, source.sessionTokenEnv);
    if (!accessKeyId && !secretAccessKey && !sessionToken) continue;
    if (!accessKeyId || !secretAccessKey) {
      throw authError(
        'AWS_LEGACY_CREDENTIALS_PARTIAL',
        `Legacy AWS credentials from ${source.name} are incomplete.`
      );
    }
    return {
      credentials: {
        accessKeyId,
        secretAccessKey,
        ...(sessionToken ? { sessionToken } : {})
      },
      source: source.name
    };
  }

  if (hosted) {
    throw authError(
      'AWS_LEGACY_CREDENTIALS_MISSING',
      'Legacy AWS credentials are not configured for this workload.'
    );
  }
  return { credentials: undefined, source: 'sdk-default-chain' };
}

function getAwsAuthConfig(workload, options = {}) {
  const env = options.env || process.env;
  const config = getWorkloadConfig(workload);
  const hosted = isHostedVercelRuntime(env);
  const mode = getAwsAuthMode(env);
  const region = String(options.region || valueFromEnv(env, 'AWS_REGION') || valueFromEnv(env, 'AWS_DEFAULT_REGION') || '').trim();

  if (mode === 'oidc') {
    const roleArn = valueFromEnv(env, config.roleArnEnv);
    if (!roleArn) {
      throw authError('AWS_ROLE_ARN_MISSING', `${config.roleArnEnv} is required when AWS_AUTH_MODE=oidc.`);
    }
    if (!isRoleArn(roleArn)) {
      throw authError('AWS_ROLE_ARN_INVALID', `${config.roleArnEnv} must be a valid IAM role ARN.`);
    }
    const audience = valueFromEnv(env, 'AWS_OIDC_AUDIENCE') || DEFAULT_OIDC_AUDIENCE;
    if (audience !== DEFAULT_OIDC_AUDIENCE) {
      throw authError(
        'AWS_OIDC_AUDIENCE_INVALID',
        `AWS_OIDC_AUDIENCE must be ${DEFAULT_OIDC_AUDIENCE}.`
      );
    }
    const providerFactory = options.oidcProviderFactory || awsCredentialsProvider;
    const credentials = providerFactory({
      audience,
      roleArn,
      roleSessionName: config.sessionName,
      durationSeconds: 3600,
      ...(region ? { clientConfig: { region } } : {})
    });
    return {
      workload,
      mode,
      hosted,
      roleArn,
      roleArnEnv: config.roleArnEnv,
      audience,
      credentialSource: 'vercel-oidc',
      credentials,
      cacheKey: `oidc:${audience}:${roleArn}`
    };
  }

  if (mode === 'legacy') {
    const legacy = resolveLegacyCredentials(config, env, hosted);
    return {
      workload,
      mode,
      hosted,
      roleArn: '',
      roleArnEnv: config.roleArnEnv,
      audience: '',
      credentialSource: legacy.source,
      credentials: legacy.credentials,
      cacheKey: legacy.credentials
        ? `legacy:${legacy.source}:${legacy.credentials.accessKeyId}`
        : 'legacy:sdk-default-chain'
    };
  }

  return {
    workload,
    mode,
    hosted,
    roleArn: '',
    roleArnEnv: config.roleArnEnv,
    audience: '',
    credentialSource: 'sdk-default-chain',
    credentials: undefined,
    cacheKey: 'default:sdk-default-chain'
  };
}

function getAwsClientConfig(workload, options = {}) {
  const auth = getAwsAuthConfig(workload, options);
  const clientConfig = {
    ...(options.region ? { region: options.region } : {}),
    ...(auth.credentials ? { credentials: auth.credentials } : {})
  };
  return { auth, clientConfig, cacheKey: auth.cacheKey };
}

function maskRoleArn(roleArn) {
  const raw = String(roleArn || '').trim();
  const match = /^(arn:(?:aws|aws-us-gov|aws-cn):iam::)(\d{12})(:role\/.+)$/.exec(raw);
  if (!match) return '';
  return `${match[1]}********${match[2].slice(-4)}${match[3]}`;
}

function describeAwsAuth(workload, options = {}) {
  const env = options.env || process.env;
  const auth = getAwsAuthConfig(workload, options);
  return {
    workload,
    runtime: auth.hosted ? 'vercel' : 'local',
    environment: valueFromEnv(env, 'VERCEL_ENV') || valueFromEnv(env, 'NODE_ENV') || 'local',
    mode: auth.mode,
    credentialSource: auth.credentialSource,
    roleArnEnv: auth.roleArnEnv,
    role: maskRoleArn(auth.roleArn),
    audience: auth.audience
  };
}

function hasAwsAuthConfiguration(workload, options = {}) {
  try {
    getAwsAuthConfig(workload, options);
    return true;
  } catch {
    return false;
  }
}

module.exports = {
  AWS_WORKLOADS,
  DEFAULT_OIDC_AUDIENCE,
  describeAwsAuth,
  getAwsAuthConfig,
  getAwsAuthMode,
  getAwsClientConfig,
  hasAwsAuthConfiguration,
  isHostedVercelRuntime,
  maskRoleArn,
  _private: {
    WORKLOAD_CONFIG,
    isRoleArn,
    resolveLegacyCredentials,
    valueFromEnv
  }
};
