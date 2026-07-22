const RUNTIME_CHANNEL = 'job-application-copilot';
const RUNTIME_VERSION = 1;
const FIELD_SELECTED_TYPE = 'FIELD_SELECTED';
const FIELD_SELECTED_INTENTS = new Set(['review', 'regenerate']);
const FIELD_ID_PATTERN = /^field-[a-f0-9]{16}(?:-[2-9][0-9]*)?$/u;
const FINGERPRINT_PATTERN = /^[a-f0-9]{16}$/u;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/u;

const isPlainObject = (value) => Boolean(value)
  && typeof value === 'object'
  && !Array.isArray(value)
  && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

const exactKeys = (value, expected) => {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
};

export const validateFieldSelectedMessage = (message) => {
  const validPayloadKeys = exactKeys(message?.payload, ['field'])
    || exactKeys(message?.payload, ['field', 'intent']);
  if (!exactKeys(message, ['channel', 'version', 'type', 'requestId', 'payload'])
    || message.channel !== RUNTIME_CHANNEL
    || message.version !== RUNTIME_VERSION
    || message.type !== FIELD_SELECTED_TYPE
    || typeof message.requestId !== 'string'
    || !REQUEST_ID_PATTERN.test(message.requestId)
    || !validPayloadKeys
    || !exactKeys(message.payload.field, [
      'fieldId',
      'fingerprint',
      'label',
      'type',
      'options',
      'nearbyText',
      'required',
      'riskClass'
    ])) {
    throw new Error('Invalid FIELD_SELECTED runtime message.');
  }
  const field = message.payload.field;
  if (!FIELD_SELECTED_INTENTS.has(message.payload.intent || 'review')
    || !FIELD_ID_PATTERN.test(field.fieldId)
    || !FINGERPRINT_PATTERN.test(field.fingerprint)
    || typeof field.label !== 'string'
    || field.label.length > 240
    || typeof field.type !== 'string'
    || field.type.length < 1
    || field.type.length > 40
    || !Array.isArray(field.options)
    || field.options.length > 50
    || !field.options.every(option => typeof option === 'string' && option.length <= 160)
    || typeof field.nearbyText !== 'string'
    || field.nearbyText.length > 320
    || typeof field.required !== 'boolean'
    || !['F1_VERIFIED', 'F2_REVIEW'].includes(field.riskClass)) {
    throw new Error('Invalid FIELD_SELECTED field descriptor.');
  }
  return message;
};

const assertMainFrameContentSender = (sender, extensionId) => {
  const senderUrl = new URL(String(sender?.url || ''));
  if (sender?.id !== extensionId
    || !sender?.tab
    || !Number.isSafeInteger(sender.tab.id)
    || sender.frameId !== 0
    || !['http:', 'https:'].includes(senderUrl.protocol)) {
    throw new Error('FIELD_SELECTED is only accepted from injected main-frame content.');
  }
};

export class FieldSelectionRouter {
  constructor({
    storageSession = globalThis.chrome?.storage?.session,
    sidePanel = globalThis.chrome?.sidePanel,
    extensionId = globalThis.chrome?.runtime?.id,
    now = () => new Date()
  } = {}) {
    if (!storageSession || !sidePanel?.open || !extensionId) {
      throw new Error('Field selection routing requires extension storage and side panel APIs.');
    }
    this.storageSession = storageSession;
    this.sidePanel = sidePanel;
    this.extensionId = extensionId;
    this.now = now;
  }

  ownsMessage(message) {
    return message?.channel === RUNTIME_CHANNEL && message?.type === FIELD_SELECTED_TYPE;
  }

  async handle(message, sender) {
    const validated = validateFieldSelectedMessage(message);
    assertMainFrameContentSender(sender, this.extensionId);
    const selectedField = {
      fieldId: validated.payload.field.fieldId,
      intent: validated.payload.intent || 'review',
      tabId: sender.tab.id,
      requestId: validated.requestId,
      selectedAt: this.now().toISOString()
    };
    const openPromise = this.sidePanel.open({ tabId: sender.tab.id });
    const storePromise = this.storageSession.set({ jobCopilotSelectedField: selectedField });
    await Promise.all([openPromise, storePromise]);
    return { ok: true, selectedField };
  }
}

export const FIELD_SELECTED_RUNTIME_TYPE = FIELD_SELECTED_TYPE;
