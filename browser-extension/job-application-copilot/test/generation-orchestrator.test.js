import assert from 'node:assert/strict';
import test from 'node:test';
import {
  GENERATION_ERROR_CODES,
  GENERATION_STATUS_CODES,
  GenerationOrchestratorError,
  embedChunksWithLexicalFallback,
  evidenceForGenerationFields,
  orchestratePageGeneration,
  partitionPageGenerationRequest,
  runLocalGeneration
} from '../src/ollama/generation-orchestrator.js';
import { OllamaError } from '../src/ollama/client.js';
import { ValidationError } from '../src/shared/validators.js';

const PRIMARY = 'qwen3.5:27b';
const FALLBACK = 'qwen3:8b';

test('unavailable Ollama exposes a stable code and does not try another model', async () => {
  let calls = 0;
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    generate: async () => {
      calls += 1;
      throw new OllamaError('Unable to reach local Ollama.', 'CONNECTION_FAILED');
    }
  }), error => error instanceof GenerationOrchestratorError
    && error.code === GENERATION_ERROR_CODES.UNAVAILABLE);
  assert.equal(calls, 1);
});

test('missing primary model can use the configured local fallback exactly once', async () => {
  const models = [];
  const statuses = [];
  const result = await runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    onStatus: status => statuses.push(status),
    generate: async ({ model }) => {
      models.push(model);
      if (model === PRIMARY) throw new OllamaError(`model ${PRIMARY} not found`, 'HTTP_ERROR', 404);
      return { output: { ok: true }, metrics: {} };
    }
  });
  assert.deepEqual(models, [PRIMARY, FALLBACK]);
  assert.equal(result.orchestration.model, FALLBACK);
  assert.equal(result.orchestration.usedFallback, true);
  assert.ok(statuses.some(status => status.code === GENERATION_STATUS_CODES.MODEL_FALLBACK
    && status.reason === GENERATION_ERROR_CODES.MISSING_MODEL));
});

test('missing model exposes its code when no distinct fallback exists', async () => {
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: PRIMARY,
    generate: async () => { throw new OllamaError('model not found', 'HTTP_ERROR', 404); }
  }), error => error.code === GENERATION_ERROR_CODES.MISSING_MODEL && error.model === PRIMARY);
});

test('compatible local load failure falls back but arbitrary failures do not', async () => {
  const models = [];
  const result = await runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    generate: async ({ model }) => {
      models.push(model);
      if (model === PRIMARY) throw new OllamaError('model requires more system memory', 'HTTP_ERROR', 500);
      return { output: { ok: true }, metrics: {} };
    }
  });
  assert.equal(result.orchestration.model, FALLBACK);
  assert.deepEqual(models, [PRIMARY, FALLBACK]);

  let arbitraryCalls = 0;
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    generate: async () => {
      arbitraryCalls += 1;
      throw new Error('Unexpected application state');
    }
  }), error => error.code === GENERATION_ERROR_CODES.FAILED);
  assert.equal(arbitraryCalls, 1);
});

test('cold-start timeout can try the smaller model and retains timeout status if it also times out', async () => {
  const models = [];
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    generate: async ({ model }) => {
      models.push(model);
      throw new OllamaError('Ollama request timed out', 'TIMEOUT');
    }
  }), error => error.code === GENERATION_ERROR_CODES.COLD_START_TIMEOUT
    && error.attempts.length === 2);
  assert.deepEqual(models, [PRIMARY, FALLBACK]);
});

test('cancellation is never retried or sent to the fallback model', async () => {
  const controller = new AbortController();
  let calls = 0;
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    signal: controller.signal,
    generate: async () => {
      calls += 1;
      controller.abort(new DOMException('User cancelled', 'AbortError'));
      throw new OllamaError('aborted', 'ABORTED');
    }
  }), error => error.code === GENERATION_ERROR_CODES.CANCELLED);
  assert.equal(calls, 1);

  const alreadyCancelled = new AbortController();
  alreadyCancelled.abort();
  await assert.rejects(runLocalGeneration({
    signal: alreadyCancelled.signal,
    generate: async () => { calls += 1; }
  }), error => error.code === GENERATION_ERROR_CODES.CANCELLED);
  assert.equal(calls, 1);
});

test('malformed structured output gets one constrained retry only', async () => {
  const attempts = [];
  const result = await runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    generate: async ({ model, constrainedRetry }) => {
      attempts.push({ model, constrainedRetry });
      if (attempts.length === 1) throw new ValidationError('structured model output is not valid JSON');
      return { output: { valid: true }, metrics: {} };
    }
  });
  assert.deepEqual(attempts, [
    { model: PRIMARY, constrainedRetry: false },
    { model: PRIMARY, constrainedRetry: true }
  ]);
  assert.equal(result.orchestration.structuredRetryCount, 1);
  assert.equal(result.orchestration.usedFallback, false);
});

test('repeated malformed JSON stops after the constrained retry without model fallback', async () => {
  let calls = 0;
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    generate: async () => {
      calls += 1;
      throw new SyntaxError('Unexpected token in JSON');
    }
  }), error => error.code === GENERATION_ERROR_CODES.MALFORMED_OUTPUT
    && error.attempts.length === 2);
  assert.equal(calls, 2);
});

test('Ollama HTTP 400 grammar rejection has a stable code and is not retried or sent to fallback', async () => {
  let calls = 0;
  const statuses = [];
  await assert.rejects(runLocalGeneration({
    primaryModel: PRIMARY,
    fallbackModel: FALLBACK,
    onStatus: status => statuses.push(status),
    generate: async () => {
      calls += 1;
      throw new OllamaError(
        'Failed to initialize samplers: failed to parse grammar',
        'HTTP_ERROR',
        400,
        { error: 'number of repetitions exceeds sane defaults' }
      );
    }
  }), error => error instanceof GenerationOrchestratorError
    && error.code === GENERATION_ERROR_CODES.SCHEMA_REJECTED
    && /rejected the structured response format/iu.test(error.message)
    && !/local generation failed/iu.test(error.message)
    && error.model === PRIMARY
    && error.attempts.length === 1
    && error.attempts[0].outcome === GENERATION_ERROR_CODES.SCHEMA_REJECTED);
  assert.equal(calls, 1);
  assert.equal(statuses.some(status => status.code === GENERATION_STATUS_CODES.STRUCTURED_RETRY), false);
  assert.equal(statuses.some(status => status.code === GENERATION_STATUS_CODES.MODEL_FALLBACK), false);
});

test('embedding failure degrades to lexical chunks while cancellation still stops', async () => {
  const statuses = [];
  const chunks = [{ id: 'chunk-1', text: 'Python analytics', terms: ['python', 'analytics'], embedding: [1, 0] }];
  const degraded = await embedChunksWithLexicalFallback({
    client: { async embed() { throw new OllamaError('model nomic-embed-text not found', 'HTTP_ERROR', 404); } },
    chunks,
    onStatus: status => statuses.push(status)
  });
  assert.equal(degraded.mode, 'lexical');
  assert.equal(degraded.errorCode, GENERATION_ERROR_CODES.MISSING_MODEL);
  assert.equal('embedding' in degraded.chunks[0], false);
  assert.ok(statuses.some(status => status.code === GENERATION_STATUS_CODES.EMBEDDING_LEXICAL_FALLBACK));

  const controller = new AbortController();
  controller.abort();
  await assert.rejects(embedChunksWithLexicalFallback({
    client: { async embed() { throw new Error('must not run'); } },
    chunks,
    signal: controller.signal
  }), error => error.code === GENERATION_ERROR_CODES.CANCELLED);
});

test('page orchestration export calls the grounded structured generator', async () => {
  const request = {
    pageId: 'page-one',
    urlHash: 'url-one',
    domRevision: 1,
    fields: [{
      fieldId: 'field-one',
      label: 'Full name',
      type: 'text',
      options: [],
      nearbyText: '',
      riskClass: 'F1_VERIFIED'
    }]
  };
  const evidence = {
    citations: [{
      citationId: 'c1',
      sourceRole: 'candidate_evidence',
      documentId: 'doc-one',
      documentVersion: 1,
      chunkId: 'chunk-one',
      quoteHash: 'hash-one',
      text: 'Full name: Daniel Short',
      locator: { pageStart: 1 }
    }],
    byField: { 'field-one': ['c1'] }
  };
  const structuredCalls = [];
  const client = {
    async chatStructured({ model, messages }) {
      structuredCalls.push({ model, messages });
      if (structuredCalls.length === 1) return { content: 'not valid JSON', metrics: {} };
      return {
        content: JSON.stringify({
          page_id: 'page-one',
          proposals: [{
            field_id: 'field-one',
            action: 'fill',
            confidence: 'high',
            risk_class: 'F1_VERIFIED',
            value_type: 'text',
            value: 'Daniel Short',
            selected_values: [],
            checked: false,
            citation_ids: ['c1'],
            short_rationale: 'Stored candidate evidence.',
            abstain_reason: ''
          }]
        }),
        metrics: { model }
      };
    }
  };
  const result = await orchestratePageGeneration({ client, request, evidence });
  assert.equal(result.output.proposals[0].action, 'fill');
  assert.equal(result.orchestration.model, PRIMARY);
  assert.equal(structuredCalls.length, 2);
  assert.doesNotMatch(structuredCalls[0].messages[0].content, /retry constraint/iu);
  assert.match(structuredCalls[1].messages[0].content, /return exactly one schema-valid JSON object and no prose/iu);
  assert.equal(result.orchestration.structuredRetryCount, 1);
});
test('page requests are partitioned by field count with field-scoped evidence only', () => {
  const fields = Array.from({ length: 5 }, (_, index) => ({
    fieldId: `field-${index + 1}`,
    label: `Question ${index + 1}`,
    type: 'textarea',
    options: [],
    nearbyText: '',
    riskClass: 'F2_REVIEW'
  }));
  const request = { pageId: 'page-batches', urlHash: 'url-batches', domRevision: 1, fields };
  const evidence = {
    citations: fields.map((field, index) => ({
      citationId: `c${index + 1}`,
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      text: `Evidence for ${field.fieldId}.`
    })),
    byField: Object.fromEntries(fields.map((field, index) => [field.fieldId, [`c${index + 1}`]]))
  };
  const batches = partitionPageGenerationRequest({
    request,
    evidence,
    maxFieldsPerBatch: 2,
    inputTokenBudget: 10000
  });
  assert.deepEqual(batches.map(batch => batch.request.fields.length), [2, 2, 1]);
  assert.deepEqual(batches[0].evidence.citations.map(citation => citation.citationId), ['c1', 'c2']);
  assert.deepEqual(Object.keys(batches[2].evidence.byField), ['field-5']);
  assert.deepEqual(
    evidenceForGenerationFields(evidence, [fields[1]]).citations.map(citation => citation.citationId),
    ['c2']
  );
});

test('page orchestration keeps valid proposals and retries only rejected fields', async () => {
  const fields = ['analytics', 'payments', 'leadership'].map((name, index) => ({
    fieldId: name,
    label: `Describe ${name} experience`,
    type: 'textarea',
    options: [],
    nearbyText: '',
    riskClass: 'F2_REVIEW',
    maxLength: 500,
    required: index === 0
  }));
  const request = { pageId: 'page-partial', urlHash: 'url-partial', domRevision: 1, fields };
  const values = {
    analytics: 'I build analytics tools.',
    payments: 'I support payment operations.',
    leadership: 'I lead collaborative teams.'
  };
  const evidence = {
    citations: fields.map((field, index) => ({
      citationId: `c${index + 1}`,
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      documentVersion: 1,
      chunkId: `chunk-${index + 1}`,
      quoteHash: `hash-${index + 1}`,
      text: values[field.fieldId],
      locator: { section: 'Experience' }
    })),
    byField: Object.fromEntries(fields.map((field, index) => [field.fieldId, [`c${index + 1}`]]))
  };
  const calls = [];
  const client = {
    async chatStructured({ messages }) {
      const payload = JSON.parse(messages[1].content);
      const fieldIds = payload.fields.map(field => field.field_id);
      calls.push(fieldIds);
      return {
        content: JSON.stringify({
          page_id: request.pageId,
          proposals: fieldIds.map((fieldId) => ({
            field_id: fieldId,
            action: 'fill',
            value_type: 'text',
            value: values[fieldId],
            citation_ids: fieldId === 'payments' && calls.length === 1 ? ['c99'] : [evidence.byField[fieldId][0]]
          }))
        }),
        metrics: { promptEvalCount: 10, evalCount: 5 }
      };
    }
  };
  const result = await orchestratePageGeneration({
    client,
    request,
    evidence,
    maxFieldsPerBatch: 3,
    inputTokenBudget: 10000
  });
  assert.deepEqual(calls, [['analytics', 'payments', 'leadership'], ['payments']]);
  assert.deepEqual(result.output.proposals.map(proposal => proposal.field_id), fields.map(field => field.fieldId));
  assert.ok(result.output.proposals.every(proposal => proposal.action === 'fill'));
  assert.equal(result.orchestration.batchCount, 1);
  assert.equal(result.orchestration.runCount, 2);
  assert.equal(result.metrics.promptEvalCount, 20);
});
