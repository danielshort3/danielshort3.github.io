#!/usr/bin/env node
'use strict';

const { execFileSync } = require('child_process');

const DEFAULT_FN_NAME = 'whisper-transcribe';
const DEFAULT_PERIOD_SEC = 60;
const DEFAULT_WINDOW_MIN = 15;
const DEFAULT_INTERVAL_MS = 5000;
const BAR_WIDTH = 24;

const parseArgs = (argv) => {
  const args = {
    functionName: DEFAULT_FN_NAME,
    periodSec: DEFAULT_PERIOD_SEC,
    windowMin: DEFAULT_WINDOW_MIN,
    intervalMs: DEFAULT_INTERVAL_MS,
    count: 0,
    clear: true
  };

  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw) continue;
    if (raw === '--function' || raw === '--function-name') {
      args.functionName = argv[i + 1] || args.functionName;
      i += 1;
      continue;
    }
    if (raw === '--period') {
      args.periodSec = Math.max(1, parseInt(argv[i + 1] || '', 10) || DEFAULT_PERIOD_SEC);
      i += 1;
      continue;
    }
    if (raw === '--window') {
      args.windowMin = Math.max(1, parseInt(argv[i + 1] || '', 10) || DEFAULT_WINDOW_MIN);
      i += 1;
      continue;
    }
    if (raw === '--interval') {
      args.intervalMs = Math.max(1000, parseInt(argv[i + 1] || '', 10) || DEFAULT_INTERVAL_MS);
      i += 1;
      continue;
    }
    if (raw === '--count') {
      args.count = Math.max(0, parseInt(argv[i + 1] || '', 10) || 0);
      i += 1;
      continue;
    }
    if (raw === '--no-clear') {
      args.clear = false;
      continue;
    }
    if (raw === '--help' || raw === '-h') {
      args.help = true;
    }
  }

  return args;
};

const runAws = (parts) => {
  return execFileSync('aws', parts, { encoding: 'utf8' }).trim();
};

const safeJsonParse = (raw) => {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

const resolveRegion = () => {
  const envRegion = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION;
  if (envRegion) return envRegion;
  const configured = runAws(['configure', 'get', 'region']);
  if (configured) return configured;
  return 'us-east-2';
};

const isoUtc = (date) => {
  return date.toISOString().replace(/\.\d{3}Z$/, 'Z');
};

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined) return 'n/a';
  const num = Number(value);
  if (!Number.isFinite(num)) return 'n/a';
  return num.toFixed(digits).replace(/\.00$/, '');
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const bar = (value, max, width = BAR_WIDTH) => {
  const safeMax = Number.isFinite(max) && max > 0 ? max : 0;
  const safeValue = Number.isFinite(value) && value > 0 ? value : 0;
  const ratio = safeMax ? clamp(safeValue / safeMax, 0, 1) : 0;
  const filled = Math.round(ratio * width);
  const empty = Math.max(0, width - filled);
  return `[${'█'.repeat(filled)}${' '.repeat(empty)}]`;
};

const latestPoint = (result) => {
  const values = result?.Values || [];
  const timestamps = result?.Timestamps || [];
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    return { value: v, timestamp: timestamps[i] || null };
  }
  return { value: null, timestamp: null };
};

const getAccountSettings = () => safeJsonParse(runAws(['lambda', 'get-account-settings', '--output', 'json'])) || {};

const getFunctionConfig = (functionName) => {
  return safeJsonParse(runAws(['lambda', 'get-function-configuration', '--function-name', functionName, '--output', 'json'])) || {};
};

const getFunctionUrl = (functionName) => {
  const raw = runAws(['lambda', 'get-function-url-config', '--function-name', functionName, '--output', 'json']);
  return safeJsonParse(raw) || {};
};

const getFunctionConcurrency = (functionName) => {
  const raw = runAws(['lambda', 'get-function-concurrency', '--function-name', functionName, '--output', 'json']);
  return safeJsonParse(raw) || null;
};

const getMetricData = ({ region, functionName, startTime, endTime, periodSec }) => {
  const queries = [
    {
      Id: 'unres',
      MetricStat: {
        Metric: {
          Namespace: 'AWS/Lambda',
          MetricName: 'UnreservedConcurrentExecutions'
        },
        Period: periodSec,
        Stat: 'Maximum'
      },
      ReturnData: true
    },
    {
      Id: 'fnconc',
      MetricStat: {
        Metric: {
          Namespace: 'AWS/Lambda',
          MetricName: 'ConcurrentExecutions',
          Dimensions: [{ Name: 'FunctionName', Value: functionName }]
        },
        Period: periodSec,
        Stat: 'Maximum'
      },
      ReturnData: true
    },
    {
      Id: 'inv',
      MetricStat: {
        Metric: {
          Namespace: 'AWS/Lambda',
          MetricName: 'Invocations',
          Dimensions: [{ Name: 'FunctionName', Value: functionName }]
        },
        Period: periodSec,
        Stat: 'Sum'
      },
      ReturnData: true
    },
    {
      Id: 'thr',
      MetricStat: {
        Metric: {
          Namespace: 'AWS/Lambda',
          MetricName: 'Throttles',
          Dimensions: [{ Name: 'FunctionName', Value: functionName }]
        },
        Period: periodSec,
        Stat: 'Sum'
      },
      ReturnData: true
    },
    {
      Id: 'err',
      MetricStat: {
        Metric: {
          Namespace: 'AWS/Lambda',
          MetricName: 'Errors',
          Dimensions: [{ Name: 'FunctionName', Value: functionName }]
        },
        Period: periodSec,
        Stat: 'Sum'
      },
      ReturnData: true
    },
    {
      Id: 'durp95',
      MetricStat: {
        Metric: {
          Namespace: 'AWS/Lambda',
          MetricName: 'Duration',
          Dimensions: [{ Name: 'FunctionName', Value: functionName }]
        },
        Period: periodSec,
        Stat: 'p95'
      },
      ReturnData: true
    }
  ];

  const out = runAws([
    'cloudwatch',
    'get-metric-data',
    '--region',
    region,
    '--start-time',
    isoUtc(startTime),
    '--end-time',
    isoUtc(endTime),
    '--scan-by',
    'TimestampDescending',
    '--max-datapoints',
    '100',
    '--metric-data-queries',
    JSON.stringify(queries),
    '--output',
    'json'
  ]);

  return safeJsonParse(out) || {};
};

const formatDurationMs = (ms) => {
  const num = Number(ms);
  if (!Number.isFinite(num)) return 'n/a';
  if (num < 1000) return `${Math.round(num)} ms`;
  const sec = num / 1000;
  if (sec < 60) return `${formatNumber(sec, 2)} s`;
  const min = sec / 60;
  return `${formatNumber(min, 2)} min`;
};

const buildHeaderLine = (label, value) => {
  const left = `${String(label || '').trim()}:`.padEnd(28, ' ');
  return `${left}${value}`;
};

const renderSnapshot = ({ region, functionName, url, accountSettings, fnConfig, fnConcurrency, metricData }) => {
  const limit = accountSettings?.AccountLimit || {};
  const usage = accountSettings?.AccountUsage || {};
  const accountMax = Number(limit?.ConcurrentExecutions) || 0;
  const unreservedMax = Number(limit?.UnreservedConcurrentExecutions) || 0;
  const memoryMb = Number(fnConfig?.MemorySize) || 0;
  const timeoutSec = Number(fnConfig?.Timeout) || 0;
  const env = fnConfig?.Environment?.Variables || {};
  const maxAudioBytes = Number(env?.MAX_AUDIO_BYTES) || null;
  const maxAudioSeconds = Number(env?.MAX_AUDIO_SECONDS) || null;

  const results = Object.fromEntries((metricData?.MetricDataResults || []).map(r => [r.Id, r]));
  const unres = latestPoint(results.unres);
  const fnconc = latestPoint(results.fnconc);
  const inv = latestPoint(results.inv);
  const thr = latestPoint(results.thr);
  const err = latestPoint(results.err);
  const durp95 = latestPoint(results.durp95);

  const unresUsed = Number.isFinite(Number(unres.value)) ? Number(unres.value) : 0;
  const unresHeadroom = Math.max(0, unreservedMax - unresUsed);
  const durMs = Number.isFinite(Number(durp95.value)) ? Number(durp95.value) : null;
  const estTotalRps = durMs ? (unreservedMax / (durMs / 1000)) : null;
  const estExtraRps = durMs ? (unresHeadroom / (durMs / 1000)) : null;

  const reserved = Number(fnConcurrency?.ReservedConcurrentExecutions);
  const reservedText = Number.isFinite(reserved) ? String(reserved) : 'none';

  const lines = [];
  lines.push(`Whisper Transcribe Capacity Monitor (${region})`);
  lines.push('');
  lines.push(buildHeaderLine('Function', functionName));
  lines.push(buildHeaderLine('Function URL', url || 'n/a'));
  lines.push(buildHeaderLine('Memory / Timeout', `${memoryMb} MB / ${timeoutSec} s`));
  lines.push(buildHeaderLine('Reserved concurrency', reservedText));
  lines.push(buildHeaderLine('MAX_AUDIO_BYTES', maxAudioBytes ? `${maxAudioBytes} bytes` : 'n/a'));
  lines.push(buildHeaderLine('MAX_AUDIO_SECONDS', Number.isFinite(maxAudioSeconds) ? `${maxAudioSeconds} s` : 'n/a'));
  lines.push(buildHeaderLine('Account concurrency limit', `${accountMax} total / ${unreservedMax} unreserved`));
  lines.push(buildHeaderLine('Function count', String(usage?.FunctionCount ?? 'n/a')));
  lines.push('');
  lines.push('Live (CloudWatch, 1-minute granularity; expect ~1-2 min delay):');
  lines.push('');

  const concBar = bar(unresUsed, unreservedMax);
  const headroomBar = bar(unresHeadroom, unreservedMax);
  lines.push(`Unreserved concurrency used  ${concBar}  ${formatNumber(unresUsed, 0)}/${unreservedMax}`);
  lines.push(`Unreserved concurrency head  ${headroomBar}  ${formatNumber(unresHeadroom, 0)}/${unreservedMax}`);
  const fnConcUsed = Number.isFinite(Number(fnconc.value)) ? Number(fnconc.value) : 0;
  lines.push(`Function concurrent (max)    ${bar(fnConcUsed, unreservedMax)}  ${formatNumber(fnConcUsed, 0)} (max)`);
  lines.push('');
  const invCount = Number.isFinite(Number(inv.value)) ? Number(inv.value) : 0;
  const thrCount = Number.isFinite(Number(thr.value)) ? Number(thr.value) : 0;
  const errCount = Number.isFinite(Number(err.value)) ? Number(err.value) : 0;
  lines.push(buildHeaderLine('Invocations / min', formatNumber(invCount, 0)));
  lines.push(buildHeaderLine('Throttles / min', formatNumber(thrCount, 0)));
  lines.push(buildHeaderLine('Errors / min', formatNumber(errCount, 0)));
  lines.push(buildHeaderLine('Duration p95', durMs ? formatDurationMs(durMs) : 'n/a'));
  lines.push(buildHeaderLine('Est. capacity req/s', estTotalRps === null ? 'n/a' : formatNumber(estTotalRps, 2)));
  lines.push(buildHeaderLine('Est. extra req/s', estExtraRps === null ? 'n/a' : formatNumber(estExtraRps, 2)));

  const sampleTs = unres.timestamp || fnconc.timestamp || inv.timestamp || null;
  if (sampleTs) {
    lines.push('');
    lines.push(`Latest sample: ${sampleTs}`);
  }

  return lines.join('\n');
};

const main = async () => {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(`Usage: node aws/whisper-transcribe/monitor.js [options]

Options:
  --function-name <name>   Lambda name (default: ${DEFAULT_FN_NAME})
  --window <minutes>       Lookback window for metrics (default: ${DEFAULT_WINDOW_MIN})
  --period <seconds>       CloudWatch period (default: ${DEFAULT_PERIOD_SEC})
  --interval <ms>          Refresh interval (default: ${DEFAULT_INTERVAL_MS})
  --count <n>              Render N snapshots then exit (default: 0 = run forever)
  --no-clear               Do not clear the terminal each refresh
`);
    return;
  }

  const region = resolveRegion();
  const functionName = args.functionName;
  const accountSettings = getAccountSettings();
  const fnConfig = getFunctionConfig(functionName);
  const fnUrlConfig = getFunctionUrl(functionName);
  const fnConcurrency = getFunctionConcurrency(functionName);
  const url = fnUrlConfig?.FunctionUrl || '';

  let rendered = 0;
  const tick = () => {
    const end = new Date();
    const start = new Date(end.getTime() - args.windowMin * 60 * 1000);
    const metricData = getMetricData({
      region,
      functionName,
      startTime: start,
      endTime: end,
      periodSec: args.periodSec
    });

    const snapshot = renderSnapshot({
      region,
      functionName,
      url,
      accountSettings,
      fnConfig,
      fnConcurrency,
      metricData
    });

    if (args.clear) process.stdout.write('\x1Bc');
    process.stdout.write(snapshot + '\n');

    rendered += 1;
    if (args.count && rendered >= args.count) {
      process.exit(0);
    }
  };

  tick();
  setInterval(tick, args.intervalMs);
};

main().catch((err) => {
  console.error(err?.message || err);
  process.exit(1);
});
