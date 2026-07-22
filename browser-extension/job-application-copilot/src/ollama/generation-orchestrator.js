import { GroundingError } from '../grounding/postprocessor.js';
import { embedChunks } from '../rag/embedder.js';
import { DEFAULT_LOCAL_MODEL_CONFIG } from '../shared/schemas.js';
import { ValidationError } from '../shared/validators.js';
import { OllamaError } from './client.js';
import { estimatePageBatchInputTokens } from './prompts.js';
import { generatePageBatch, regenerateField } from './structured-generation.js';

export const GENERATION_ERROR_CODES = Object.freeze({
  UNAVAILABLE: 'OLLAMA_UNAVAILABLE',
  MISSING_MODEL: 'OLLAMA_MODEL_MISSING',
  COLD_START_TIMEOUT: 'OLLAMA_COLD_START_TIMEOUT',
  MALFORMED_OUTPUT: 'OLLAMA_MALFORMED_OUTPUT',
  SCHEMA_REJECTED: 'OLLAMA_SCHEMA_REJECTED',
  LOCAL_MODEL_FAILURE: 'OLLAMA_LOCAL_MODEL_FAILURE',
  CANCELLED: 'GENERATION_CANCELLED',
  FAILED: 'GENERATION_FAILED'
});

export const GENERATION_STATUS_CODES = Object.freeze({
  STARTED: 'GENERATION_STARTED',
  STRUCTURED_RETRY: 'GENERATION_STRUCTURED_RETRY',
  MODEL_FALLBACK: 'GENERATION_MODEL_FALLBACK',
  COMPLETED: 'GENERATION_COMPLETED',
  EMBEDDING_LEXICAL_FALLBACK: 'EMBEDDING_LEXICAL_FALLBACK'
});

const FALLBACK_COMPATIBLE_CODES = new Set([
  GENERATION_ERROR_CODES.MISSING_MODEL,
  GENERATION_ERROR_CODES.COLD_START_TIMEOUT,
  GENERATION_ERROR_CODES.LOCAL_MODEL_FAILURE
]);

const combinedErrorText = (error) => {
  let details = '';
  try {
    details = typeof error?.details === 'string' ? error.details : JSON.stringify(error?.details || '');
  } catch {}
  return `${String(error?.message || '')} ${details}`.trim();
};

export const classifyGenerationError = (error, signal) => {
  if (signal?.aborted
    || error?.name === 'AbortError'
    || error?.code === 'ABORTED'
    || error?.code === GENERATION_ERROR_CODES.CANCELLED) {
    return GENERATION_ERROR_CODES.CANCELLED;
  }
  if (error instanceof ValidationError
    || error instanceof GroundingError
    || error instanceof SyntaxError
    || error?.code === 'MALFORMED_JSON'
    || error?.code === 'MALFORMED_STREAM'
    || error?.code === 'INCOMPLETE_STREAM') {
    return GENERATION_ERROR_CODES.MALFORMED_OUTPUT;
  }
  const text = combinedErrorText(error);
  if (error?.status === 400
    && /failed to (?:initialize samplers|parse grammar)|number of repetitions exceeds sane defaults/iu.test(text)) {
    return GENERATION_ERROR_CODES.SCHEMA_REJECTED;
  }
  if (error?.status === 404
    || /(?:model|manifest).*(?:not found|does not exist|missing)|pull (?:the )?model|no such model/iu.test(text)) {
    return GENERATION_ERROR_CODES.MISSING_MODEL;
  }
  if (error?.code === 'TIMEOUT' || error?.name === 'TimeoutError' || /timed? out|timeout/iu.test(text)) {
    return GENERATION_ERROR_CODES.COLD_START_TIMEOUT;
  }
  if (error?.code === 'CONNECTION_FAILED'
    || /failed to fetch|networkerror|connection refused|unable to reach local ollama/iu.test(text)) {
    return GENERATION_ERROR_CODES.UNAVAILABLE;
  }
  if (/(?:failed|unable) to load (?:the )?model|runner process|requires more (?:system )?memory|out of memory|cuda|gpu|backend.*(?:failed|error)/iu.test(text)) {
    return GENERATION_ERROR_CODES.LOCAL_MODEL_FAILURE;
  }
  if (error instanceof OllamaError && (error.code === 'OVERLOADED' || error.status >= 500)) {
    return GENERATION_ERROR_CODES.UNAVAILABLE;
  }
  return GENERATION_ERROR_CODES.FAILED;
};

export class GenerationOrchestratorError extends Error {
  constructor(code, message, { model = '', attempts = [], cause } = {}) {
    super(message, cause ? { cause } : undefined);
    this.name = 'GenerationOrchestratorError';
    this.code = code;
    this.model = model;
    this.attempts = attempts;
  }
}

const throwIfCancelled = (signal, attempts = []) => {
  if (!signal?.aborted) return;
  throw new GenerationOrchestratorError(
    GENERATION_ERROR_CODES.CANCELLED,
    signal.reason?.message || 'Local generation was cancelled.',
    { attempts, cause: signal.reason }
  );
};

const publicMessage = (code, model) => {
  if (code === GENERATION_ERROR_CODES.UNAVAILABLE) return 'Local Ollama is unavailable.';
  if (code === GENERATION_ERROR_CODES.MISSING_MODEL) return `The local model ${model} is not installed.`;
  if (code === GENERATION_ERROR_CODES.COLD_START_TIMEOUT) return `The local model ${model} did not become ready before the timeout.`;
  if (code === GENERATION_ERROR_CODES.MALFORMED_OUTPUT) return 'The local model did not return valid structured output.';
  if (code === GENERATION_ERROR_CODES.SCHEMA_REJECTED) {
    return 'Ollama rejected the structured response format. Reload the updated extension and try regeneration again.';
  }
  if (code === GENERATION_ERROR_CODES.LOCAL_MODEL_FAILURE) return `The local model ${model} could not be loaded.`;
  if (code === GENERATION_ERROR_CODES.CANCELLED) return 'Local generation was cancelled.';
  return 'Local generation failed.';
};

export const runLocalGeneration = async ({
  generate,
  primaryModel = DEFAULT_LOCAL_MODEL_CONFIG.generationModel,
  fallbackModel = DEFAULT_LOCAL_MODEL_CONFIG.fallbackGenerationModel,
  signal,
  onStatus
}) => {
  if (typeof generate !== 'function') throw new TypeError('A local generation operation is required.');
  if (typeof primaryModel !== 'string' || !primaryModel.trim()) throw new TypeError('A primary local model is required.');
  throwIfCancelled(signal);

  let model = primaryModel;
  let structuredRetryUsed = false;
  let fallbackUsed = false;
  const attempts = [];
  onStatus?.({ code: GENERATION_STATUS_CODES.STARTED, model, attempt: 1 });

  while (true) {
    throwIfCancelled(signal, attempts);
    const attempt = attempts.length + 1;
    try {
      const result = await generate({ model, signal, constrainedRetry: structuredRetryUsed });
      throwIfCancelled(signal, attempts);
      attempts.push({ model, outcome: 'completed' });
      const orchestration = {
        status: GENERATION_STATUS_CODES.COMPLETED,
        model,
        usedFallback: fallbackUsed,
        structuredRetryCount: structuredRetryUsed ? 1 : 0,
        attempts: attempts.map(entry => ({ ...entry }))
      };
      onStatus?.({ code: GENERATION_STATUS_CODES.COMPLETED, model, attempt });
      return { ...result, orchestration };
    } catch (error) {
      const code = classifyGenerationError(error, signal);
      attempts.push({ model, outcome: code });
      if (code === GENERATION_ERROR_CODES.CANCELLED) {
        throw new GenerationOrchestratorError(code, publicMessage(code, model), { model, attempts, cause: error });
      }
      if (code === GENERATION_ERROR_CODES.MALFORMED_OUTPUT && !structuredRetryUsed) {
        structuredRetryUsed = true;
        onStatus?.({ code: GENERATION_STATUS_CODES.STRUCTURED_RETRY, model, attempt: attempt + 1 });
        continue;
      }
      const canFallback = FALLBACK_COMPATIBLE_CODES.has(code)
        && !fallbackUsed
        && typeof fallbackModel === 'string'
        && fallbackModel.trim()
        && fallbackModel !== model;
      if (canFallback) {
        fallbackUsed = true;
        model = fallbackModel;
        onStatus?.({
          code: GENERATION_STATUS_CODES.MODEL_FALLBACK,
          model,
          previousModel: attempts.at(-1).model,
          reason: code,
          attempt: attempt + 1
        });
        continue;
      }
      throw new GenerationOrchestratorError(code, publicMessage(code, model), {
        model,
        attempts,
        cause: error
      });
    }
  }
};

export const DEFAULT_PAGE_MICRO_BATCH_FIELDS = 8;
export const DEFAULT_PAGE_INPUT_TOKEN_BUDGET = DEFAULT_LOCAL_MODEL_CONFIG.numContextTokens
  - DEFAULT_LOCAL_MODEL_CONFIG.maxPageOutputTokens
  - 1024;

export const evidenceForGenerationFields = (evidence, fields) => {
  const fieldIds = [...new Set((fields || [])
    .map(field => typeof field === 'string' ? field : field?.fieldId)
    .filter(Boolean))];
  const byField = Object.fromEntries(fieldIds.map(fieldId => [
    fieldId,
    [...new Set(evidence?.byField?.[fieldId] || [])]
  ]));
  const allowed = new Set(Object.values(byField).flat());
  const citations = (evidence?.citations || []).filter(citation => allowed.has(citation.citationId));
  const available = new Set(citations.map(citation => citation.citationId));
  return {
    citations,
    byField: Object.fromEntries(fieldIds.map(fieldId => [
      fieldId,
      byField[fieldId].filter(citationId => available.has(citationId))
    ]))
  };
};

const createPageGenerationBatch = ({ request, evidence, fields }) => {
  const batchRequest = { ...request, fields };
  const batchEvidence = evidenceForGenerationFields(evidence, fields);
  return {
    request: batchRequest,
    evidence: batchEvidence,
    estimatedInputTokens: estimatePageBatchInputTokens({ request: batchRequest, evidence: batchEvidence })
  };
};

export const partitionPageGenerationRequest = ({
  request,
  evidence,
  maxFieldsPerBatch = DEFAULT_PAGE_MICRO_BATCH_FIELDS,
  inputTokenBudget = DEFAULT_PAGE_INPUT_TOKEN_BUDGET
}) => {
  if (!Array.isArray(request?.fields) || !request.fields.length) {
    throw new TypeError('Page generation requires at least one field.');
  }
  if (!Number.isSafeInteger(maxFieldsPerBatch) || maxFieldsPerBatch < 1) {
    throw new TypeError('maxFieldsPerBatch must be a positive integer.');
  }
  if (!Number.isSafeInteger(inputTokenBudget) || inputTokenBudget < 512) {
    throw new TypeError('inputTokenBudget must be at least 512 tokens.');
  }

  const batches = [];
  let currentFields = [];
  for (const field of request.fields) {
    const candidateFields = [...currentFields, field];
    const candidate = createPageGenerationBatch({ request, evidence, fields: candidateFields });
    if (currentFields.length
      && (currentFields.length >= maxFieldsPerBatch || candidate.estimatedInputTokens > inputTokenBudget)) {
      batches.push(createPageGenerationBatch({ request, evidence, fields: currentFields }));
      currentFields = [field];
    } else {
      currentFields = candidateFields;
    }
  }
  if (currentFields.length) batches.push(createPageGenerationBatch({ request, evidence, fields: currentFields }));
  return batches;
};

const degradedPageProposal = field => ({
  field_id: field.fieldId,
  action: 'ask_user',
  confidence: 'needs_input',
  risk_class: field.riskClass,
  value_type: 'none',
  value: '',
  selected_values: [],
  checked: false,
  citation_ids: [],
  short_rationale: 'No verified answer is available.',
  abstain_reason: 'Review this field and enter the answer yourself.'
});

const aggregateGenerationMetrics = (entries) => {
  if (entries.length === 1) return entries[0];
  const sum = key => entries.reduce((total, entry) => total + (Number(entry?.[key]) || 0), 0);
  return {
    totalDuration: sum('totalDuration'),
    loadDuration: sum('loadDuration'),
    promptEvalCount: sum('promptEvalCount'),
    promptEvalDuration: sum('promptEvalDuration'),
    evalCount: sum('evalCount'),
    evalDuration: sum('evalDuration'),
    batches: entries.map(entry => ({ ...entry }))
  };
};

export const orchestratePageGeneration = async ({
  client,
  request,
  evidence,
  primaryModel,
  fallbackModel,
  signal,
  onProgress,
  onStatus,
  maxFieldsPerBatch = DEFAULT_PAGE_MICRO_BATCH_FIELDS,
  inputTokenBudget = DEFAULT_PAGE_INPUT_TOKEN_BUDGET
}) => {
  const batches = partitionPageGenerationRequest({
    request,
    evidence,
    maxFieldsPerBatch,
    inputTokenBudget
  });
  const proposalsByField = new Map();
  const degradedFieldIds = new Set();
  const attempts = [];
  const metrics = [];
  let activePrimaryModel = primaryModel;
  let activeFallbackModel = fallbackModel;
  let lastModel = primaryModel || DEFAULT_LOCAL_MODEL_CONFIG.generationModel;
  let usedFallback = false;
  let structuredRetryCount = 0;
  let runCount = 0;

  const executeBatch = async (batch, plannedBatchIndex) => {
    throwIfCancelled(signal, attempts);
    runCount += 1;
    const currentRun = runCount;
    const statusForBatch = update => onStatus?.({
      ...update,
      batchIndex: plannedBatchIndex,
      batchCount: batches.length,
      run: currentRun
    });
    try {
      const result = await runLocalGeneration({
        primaryModel: activePrimaryModel,
        fallbackModel: activeFallbackModel,
        signal,
        onStatus: statusForBatch,
        generate: ({ model, signal: attemptSignal, constrainedRetry }) => generatePageBatch({
          client,
          request: batch.request,
          evidence: batch.evidence,
          model,
          constrainedRetry,
          signal: attemptSignal,
          onProgress: update => onProgress?.({
            ...update,
            batchIndex: plannedBatchIndex,
            batchCount: batches.length,
            run: currentRun
          }),
          allowPartial: true
        })
      });
      lastModel = result.orchestration.model;
      usedFallback ||= result.orchestration.usedFallback;
      structuredRetryCount += result.orchestration.structuredRetryCount;
      attempts.push(...result.orchestration.attempts.map(attempt => ({
        ...attempt,
        batchIndex: plannedBatchIndex,
        run: currentRun
      })));
      metrics.push(result.metrics || {});
      result.output.proposals.forEach(proposal => proposalsByField.set(proposal.field_id, proposal));
      if (result.orchestration.usedFallback) {
        activePrimaryModel = result.orchestration.model;
        activeFallbackModel = result.orchestration.model;
      }

      const rejectedIds = new Set((result.rejectedFields || []).map(entry => entry.fieldId));
      const rejectedFields = batch.request.fields.filter(field => rejectedIds.has(field.fieldId));
      if (rejectedFields.length) {
        await executeBatch(createPageGenerationBatch({ request, evidence, fields: rejectedFields }), plannedBatchIndex);
      }
    } catch (error) {
      if (error instanceof GenerationOrchestratorError
        && error.code === GENERATION_ERROR_CODES.MALFORMED_OUTPUT) {
        attempts.push(...error.attempts.map(attempt => ({
          ...attempt,
          batchIndex: plannedBatchIndex,
          run: currentRun
        })));
        structuredRetryCount += error.attempts.length > 1 ? 1 : 0;
        if (batch.request.fields.length > 1) {
          const midpoint = Math.ceil(batch.request.fields.length / 2);
          const groups = [
            batch.request.fields.slice(0, midpoint),
            batch.request.fields.slice(midpoint)
          ].filter(group => group.length);
          for (const fields of groups) {
            await executeBatch(createPageGenerationBatch({ request, evidence, fields }), plannedBatchIndex);
          }
          return;
        }
        const field = batch.request.fields[0];
        proposalsByField.set(field.fieldId, degradedPageProposal(field));
        degradedFieldIds.add(field.fieldId);
        return;
      }
      throw error;
    }
  };

  for (let index = 0; index < batches.length; index += 1) {
    await executeBatch(batches[index], index + 1);
  }
  request.fields.forEach((field) => {
    if (!proposalsByField.has(field.fieldId)) {
      proposalsByField.set(field.fieldId, degradedPageProposal(field));
      degradedFieldIds.add(field.fieldId);
    }
  });
  return {
    output: {
      page_id: request.pageId,
      proposals: request.fields.map(field => proposalsByField.get(field.fieldId))
    },
    rejectedFields: [...degradedFieldIds].map(fieldId => ({ fieldId, issues: ['Structured output was not recoverable.'] })),
    metrics: aggregateGenerationMetrics(metrics),
    orchestration: {
      status: GENERATION_STATUS_CODES.COMPLETED,
      model: lastModel,
      usedFallback,
      structuredRetryCount,
      batchCount: batches.length,
      runCount,
      attempts
    }
  };
};

export const orchestrateFieldRegeneration = ({
  client,
  field,
  priorDraft,
  feedback,
  evidence,
  primaryModel,
  fallbackModel,
  signal,
  onProgress,
  onStatus
}) => runLocalGeneration({
  primaryModel,
  fallbackModel,
  signal,
  onStatus,
  generate: ({ model, signal: attemptSignal, constrainedRetry }) => regenerateField({
    client,
    field,
    priorDraft,
    feedback,
    evidence,
    model,
    constrainedRetry,
    signal: attemptSignal,
    onProgress
  })
});

export const embedChunksWithLexicalFallback = async ({
  client,
  chunks,
  model = DEFAULT_LOCAL_MODEL_CONFIG.embeddingModel,
  batchSize,
  signal,
  onProgress,
  onStatus
}) => {
  throwIfCancelled(signal);
  try {
    const embedded = await embedChunks({ client, chunks, model, batchSize, signal, onProgress });
    throwIfCancelled(signal);
    return { mode: 'hybrid', chunks: embedded, errorCode: null };
  } catch (error) {
    const code = classifyGenerationError(error, signal);
    if (code === GENERATION_ERROR_CODES.CANCELLED) {
      throw new GenerationOrchestratorError(code, publicMessage(code, model), { model, cause: error });
    }
    onStatus?.({ code: GENERATION_STATUS_CODES.EMBEDDING_LEXICAL_FALLBACK, model, reason: code });
    return {
      mode: 'lexical',
      chunks: chunks.map(chunk => {
        const { embedding, ...lexicalChunk } = chunk;
        return lexicalChunk;
      }),
      errorCode: code
    };
  }
};
