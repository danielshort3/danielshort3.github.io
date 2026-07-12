#!/usr/bin/env node
'use strict';

const { _internal } = require('../../api/_lib/demo-proxy');

const target = String(process.argv[2] || process.env.VERCEL_ENV || '').trim().toLowerCase();
if (!['production', 'preview', 'development'].includes(target)) {
  process.stderr.write('Usage: node scripts/diagnostics/check-demo-config.js <production|preview|development>\n');
  process.exitCode = 1;
} else {
  process.env.VERCEL_ENV = target;
  const demos = Object.entries(_internal.DEMO_MANIFEST);
  const functions = Object.fromEntries(demos.map(([id, config]) => [
    id,
    Boolean(process.env[config.envKey])
  ]));
  let runtime;
  try {
    const first = demos[0][1];
    const config = _internal.getRuntimeConfig(process.env, process.env[first.envKey]);
    runtime = {
      ok: true,
      authSource: config.auth.source,
      region: config.region,
      roleConfigured: Boolean(config.roleArn),
      rateTableConfigured: Boolean(config.tableName),
      hashSaltConfigured: Boolean(config.hashSalt),
      requireDdb: config.requireDdb
    };
  } catch (err) {
    runtime = {
      ok: false,
      code: String(err?.code || err?.name || 'CONFIG_ERROR')
    };
  }
  process.stdout.write(`${JSON.stringify({ target, functions, runtime })}\n`);
}
