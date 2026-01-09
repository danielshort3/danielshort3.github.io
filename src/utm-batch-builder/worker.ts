/* eslint-disable no-restricted-globals */

type WorkerMessage =
  | { type: 'generate'; requestId: string; config: unknown; limit?: number; chunkSize?: number }
  | { type: 'cancel'; requestId: string };

type WorkerResponse =
  | { type: 'start'; requestId: string; estimatedTotal: number; paramKeys: string[]; warnings: string[] }
  | { type: 'chunk'; requestId: string; rows: unknown[]; generatedCount: number }
  | { type: 'done'; requestId: string; generatedCount: number }
  | { type: 'cancelled'; requestId: string; generatedCount: number }
  | { type: 'error'; requestId: string; errors: string[]; warnings: string[] };

const core = require('./core');

const cancelled = new Set<string>();

const post = (msg: WorkerResponse) => {
  (self as any).postMessage(msg);
};

(self as any).addEventListener('message', (event: MessageEvent) => {
  const message = (event.data || {}) as WorkerMessage;
  if (!message || typeof message !== 'object') return;

  if (message.type === 'cancel') {
    cancelled.add(message.requestId);
    return;
  }

  if (message.type !== 'generate') return;
  const requestId = message.requestId;
  cancelled.delete(requestId);

  const { errors, warnings, resolved } = core.resolveAndValidateConfig(message.config);
  if (errors && errors.length) {
    post({ type: 'error', requestId, errors, warnings: warnings || [] });
    return;
  }

  const estimatedTotal = core.estimateTotalRows(resolved);
  const paramKeys = [
    ...(resolved.fields || []).map((f: any) => f.key),
    ...(resolved.customParams || []).map((f: any) => f.key),
  ];

  post({ type: 'start', requestId, estimatedTotal, paramKeys, warnings: warnings || [] });

  const limit = typeof message.limit === 'number' && message.limit >= 0 ? message.limit : Infinity;
  const chunkSize = typeof message.chunkSize === 'number' && message.chunkSize > 0 ? message.chunkSize : 500;

  let rows: unknown[] = [];
  let generatedCount = 0;

  try {
    for (const row of core.generateRows(resolved, { limit })) {
      if (cancelled.has(requestId)) {
        post({ type: 'cancelled', requestId, generatedCount });
        return;
      }
      rows.push(row);
      generatedCount += 1;
      if (rows.length >= chunkSize) {
        post({ type: 'chunk', requestId, rows, generatedCount });
        rows = [];
      }
    }
    if (rows.length) post({ type: 'chunk', requestId, rows, generatedCount });
    post({ type: 'done', requestId, generatedCount });
  } catch (err: any) {
    post({
      type: 'error',
      requestId,
      errors: [err?.message || String(err)],
      warnings: warnings || []
    });
  }
});

