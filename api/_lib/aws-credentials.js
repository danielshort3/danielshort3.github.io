'use strict';

const crypto = require('crypto');
const { awsCredentialsProvider } = require('@vercel/oidc-aws-credentials-provider');

const DEFAULT_OIDC_AUDIENCE = 'sts.amazonaws.com';
const VALID_AUTH_MODES = new Set(['auto', 'oidc', 'static']);

function createConfigError(code, message){
  const err = new Error(message);
  err.code = code;
  err.statusCode = 503;
  return err;
}

function readEnv(key, env = process.env){
  if (!key || typeof env[key] === 'undefined') return '';
  return String(env[key]).trim();
}

function firstEnv(keys, env = process.env){
  for (const key of Array.isArray(keys) ? keys : []) {
    const value = readEnv(key, env);
    if (value) return { key, value };
  }
  return { key: '', value: '' };
}

function shortHash(value){
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex').slice(0, 16);
}

function normalizeServiceName(value){
  const normalized = String(value || 'aws')
    .replace(/[^A-Za-z0-9+=,.@_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48);
  return normalized || 'aws';
}

function normalizeAuthMode(env = process.env){
  const raw = readEnv('AWS_AUTH_MODE', env).toLowerCase();
  const mode = raw || 'auto';
  if (!VALID_AUTH_MODES.has(mode)) {
    throw createConfigError(
      'AWS_AUTH_MODE_INVALID',
      'AWS_AUTH_MODE must be "auto", "oidc", or "static".'
    );
  }
  return mode;
}

function assertRoleArn(roleArn){
  if (!/^arn:(?:aws|aws-us-gov|aws-cn):iam::\d{12}:role\/[A-Za-z0-9+=,.@_\/-]+$/.test(roleArn)) {
    throw createConfigError('AWS_OIDC_ROLE_INVALID', 'The configured AWS OIDC role ARN is invalid.');
  }
}

function getOidcAudience(env = process.env){
  const audience = readEnv('AWS_OIDC_AUDIENCE', env) || DEFAULT_OIDC_AUDIENCE;
  if (audience.length > 512 || /\s|[\u0000-\u001f\u007f]/.test(audience)) {
    throw createConfigError('AWS_OIDC_AUDIENCE_INVALID', 'AWS_OIDC_AUDIENCE is invalid.');
  }
  return audience;
}

function inspectStaticCredentialSets(staticCredentialSets, env = process.env){
  for (const set of Array.isArray(staticCredentialSets) ? staticCredentialSets : []) {
    const accessKeyId = readEnv(set.accessKeyId, env);
    const secretAccessKey = readEnv(set.secretAccessKey, env);
    const sessionToken = readEnv(set.sessionToken, env);
    if (!accessKeyId && !secretAccessKey) continue;
    if (!accessKeyId || !secretAccessKey) {
      throw createConfigError(
        'AWS_STATIC_CREDENTIALS_INCOMPLETE',
        `The ${set.name || 'AWS'} static credential pair is incomplete.`
      );
    }

    if (accessKeyId.startsWith('ASIA') && !sessionToken) {
      throw createConfigError(
        'AWS_STATIC_SESSION_TOKEN_MISSING',
        `The ${set.name || 'AWS'} temporary credential pair is missing its session token.`
      );
    }

    const sessionTokenUsed = Boolean(sessionToken && accessKeyId.startsWith('ASIA'));
    const credentials = {
      accessKeyId,
      secretAccessKey,
      ...(sessionTokenUsed ? { sessionToken } : {})
    };
    return {
      credentials,
      name: String(set.name || set.accessKeyId || 'static'),
      accessKeyIdSource: String(set.accessKeyId || ''),
      secretAccessKeySource: String(set.secretAccessKey || ''),
      sessionTokenSource: String(set.sessionToken || ''),
      sessionTokenConfigured: Boolean(sessionToken),
      sessionTokenUsed,
      cacheKey: `static:${shortHash(`${accessKeyId}\u0000${secretAccessKey}\u0000${sessionTokenUsed ? sessionToken : ''}`)}`
    };
  }
  return null;
}

function resolveAwsCredentials(options = {}){
  const env = options.env || process.env;
  const service = normalizeServiceName(options.service);
  const authMode = normalizeAuthMode(env);
  const role = firstEnv(options.roleArnEnvKeys, env);

  if (authMode === 'oidc' && !role.value) {
    throw createConfigError(
      'AWS_OIDC_ROLE_MISSING',
      `AWS_AUTH_MODE requires OIDC, but the ${service} role ARN is not configured.`
    );
  }

  if (role.value && authMode !== 'static') {
    assertRoleArn(role.value);
    const audience = getOidcAudience(env);
    const providerFactory = options.providerFactory || awsCredentialsProvider;
    const region = String(options.region || '').trim();
    const credentials = providerFactory({
      roleArn: role.value,
      audience,
      ...(region ? { clientConfig: { region } } : {}),
      roleSessionName: `vercel-${service}`.slice(0, 64)
    });
    return {
      credentials,
      cacheKey: `oidc:${role.key}:${shortHash(`${role.value}\u0000${audience}\u0000${region}`)}`,
      source: 'oidc',
      authMode,
      roleArnSource: role.key,
      roleArnConfigured: true,
      audience
    };
  }

  const staticConfig = inspectStaticCredentialSets(options.staticCredentialSets, env);
  if (staticConfig) {
    return {
      credentials: staticConfig.credentials,
      cacheKey: staticConfig.cacheKey,
      source: 'static',
      authMode,
      roleArnSource: role.key,
      roleArnConfigured: Boolean(role.value),
      audience: '',
      staticCredentialSource: staticConfig.name,
      staticCredentialInfo: staticConfig
    };
  }

  if (authMode === 'static') {
    throw createConfigError(
      'AWS_STATIC_CREDENTIALS_MISSING',
      `AWS_AUTH_MODE requires static credentials, but the ${service} credential pair is not configured.`
    );
  }

  return {
    credentials: undefined,
    cacheKey: 'default',
    source: 'default',
    authMode,
    roleArnSource: role.key,
    roleArnConfigured: Boolean(role.value),
    audience: ''
  };
}

module.exports = {
  DEFAULT_OIDC_AUDIENCE,
  inspectStaticCredentialSets,
  resolveAwsCredentials,
  _internal: {
    firstEnv,
    getOidcAudience,
    normalizeAuthMode,
    normalizeServiceName,
    shortHash
  }
};
