import {
  DEFAULT_KDF_PARAMETERS,
  base64ToBytes,
  bytesToBase64,
  decryptJson,
  deriveVaultKeyMaterial,
  encryptJson,
  randomBytes
} from './crypto.js';
import { createDefaultSessionKeyStore } from './session-key-store.js';

const DATABASE_VERSION = 2;
const META_STORE = 'meta';
const RECORD_STORE = 'records';
const CONFIG_KEY = 'config';
const SENTINEL_KEY = 'sentinel';
const SENTINEL_VALUE = 'job-application-copilot-vault';
const RECORD_ENVELOPE_VERSION = 2;

const requestToPromise = (request) => new Promise((resolve, reject) => {
  request.addEventListener('success', () => resolve(request.result), { once: true });
  request.addEventListener('error', () => reject(request.error || new Error('IndexedDB request failed.')), { once: true });
});

const transactionToPromise = (transaction) => new Promise((resolve, reject) => {
  transaction.addEventListener('complete', () => resolve(), { once: true });
  transaction.addEventListener('abort', () => reject(transaction.error || new Error('IndexedDB transaction aborted.')), { once: true });
  transaction.addEventListener('error', () => reject(transaction.error || new Error('IndexedDB transaction failed.')), { once: true });
});

const openDatabase = (indexedDb, name) => new Promise((resolve, reject) => {
  const request = indexedDb.open(name, DATABASE_VERSION);
  request.addEventListener('upgradeneeded', () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(META_STORE)) {
      database.createObjectStore(META_STORE, { keyPath: 'key' });
    }
    if (!database.objectStoreNames.contains(RECORD_STORE)) {
      database.createObjectStore(RECORD_STORE, { keyPath: 'id' });
    } else {
      const records = request.transaction.objectStore(RECORD_STORE);
      if (records.indexNames.contains('kind')) records.deleteIndex('kind');
    }
  });
  request.addEventListener('success', () => resolve(request.result), { once: true });
  request.addEventListener('error', () => reject(request.error || new Error('Unable to open the encrypted vault.')), { once: true });
  request.addEventListener('blocked', () => reject(new Error('Encrypted vault upgrade is blocked by another open page.')), { once: true });
});

const legacyRecordAad = (vaultId, recordId, kind, schemaVersion) => ({
  vaultId,
  store: RECORD_STORE,
  recordId,
  kind,
  schemaVersion
});

const recordAad = (vaultId, storageId) => ({
  vaultId,
  store: RECORD_STORE,
  storageId,
  envelopeVersion: RECORD_ENVELOPE_VERSION
});

const randomStorageId = () => bytesToBase64(randomBytes(32))
  .replaceAll('+', '-')
  .replaceAll('/', '_')
  .replace(/=+$/u, '');

const isLegacyRawRecord = (raw) => Boolean(raw)
  && typeof raw.id === 'string'
  && typeof raw.kind === 'string'
  && Number.isSafeInteger(raw.schemaVersion)
  && typeof raw.createdAt === 'string'
  && typeof raw.updatedAt === 'string'
  && Boolean(raw.encrypted);

const validateRecordEnvelope = (value) => {
  const expectedKeys = ['createdAt', 'envelopeVersion', 'id', 'kind', 'schemaVersion', 'updatedAt', 'value'];
  if (!value
    || typeof value !== 'object'
    || Array.isArray(value)
    || Object.keys(value).sort().some((key, index) => key !== expectedKeys[index])
    || Object.keys(value).length !== expectedKeys.length
    || value.envelopeVersion !== RECORD_ENVELOPE_VERSION
    || typeof value.id !== 'string'
    || !value.id.trim()
    || typeof value.kind !== 'string'
    || !value.kind.trim()
    || !Number.isSafeInteger(value.schemaVersion)
    || value.schemaVersion < 1
    || typeof value.createdAt !== 'string'
    || typeof value.updatedAt !== 'string'
    || !Object.hasOwn(value, 'value')) {
    throw new VaultAuthenticationError();
  }
  return value;
};

const publicRecord = record => ({
  id: record.id,
  kind: record.kind,
  schemaVersion: record.schemaVersion,
  createdAt: record.createdAt,
  updatedAt: record.updatedAt,
  value: record.value
});

const sentinelAad = (vaultId) => ({ vaultId, store: META_STORE, key: SENTINEL_KEY, schemaVersion: 1 });

export class VaultLockedError extends Error {
  constructor() {
    super('The encrypted vault is locked.');
    this.name = 'VaultLockedError';
  }
}

export class VaultAuthenticationError extends Error {
  constructor() {
    super('The vault passphrase is incorrect or the encrypted data is damaged.');
    this.name = 'VaultAuthenticationError';
  }
}

export class EncryptedIndexedDbVault {
  constructor({
    indexedDB = globalThis.indexedDB,
    databaseName = 'job-application-copilot-vault',
    vaultId = 'default',
    sessionKeyStore = createDefaultSessionKeyStore()
  } = {}) {
    if (!indexedDB) throw new Error('IndexedDB is required for the encrypted vault.');
    this.indexedDB = indexedDB;
    this.databaseName = databaseName;
    this.vaultId = vaultId;
    this.sessionKeyStore = sessionKeyStore;
    this.database = null;
    this.legacyMigrationChecked = false;
    this.migrationPromise = null;
  }

  async open() {
    if (!this.database) {
      await this.sessionKeyStore.initialize();
      this.database = await openDatabase(this.indexedDB, this.databaseName);
      this.legacyMigrationChecked = false;
    }
    return this;
  }

  async isInitialized() {
    await this.open();
    return Boolean(await this.#getMeta(CONFIG_KEY));
  }

  async initialize(passphrase) {
    await this.open();
    if (await this.#getMeta(CONFIG_KEY)) throw new Error('The encrypted vault is already initialized.');

    const salt = randomBytes(32);
    const keyMaterial = await deriveVaultKeyMaterial(passphrase, salt, DEFAULT_KDF_PARAMETERS);
    const encryptedSentinel = await encryptJson(
      { value: SENTINEL_VALUE },
      keyMaterial,
      { aad: sentinelAad(this.vaultId) }
    );
    const transaction = this.database.transaction([META_STORE], 'readwrite');
    const transactionDone = transactionToPromise(transaction);
    const store = transaction.objectStore(META_STORE);
    store.put({
      key: CONFIG_KEY,
      value: {
        version: 1,
        vaultId: this.vaultId,
        salt: bytesToBase64(salt),
        kdf: { ...DEFAULT_KDF_PARAMETERS },
        createdAt: new Date().toISOString()
      }
    });
    store.put({ key: SENTINEL_KEY, value: encryptedSentinel });
    await transactionDone;
    await this.sessionKeyStore.set(this.vaultId, keyMaterial);
    this.legacyMigrationChecked = true;
    return this;
  }

  async unlock(passphrase) {
    await this.open();
    const config = (await this.#getMeta(CONFIG_KEY))?.value;
    const sentinel = (await this.#getMeta(SENTINEL_KEY))?.value;
    if (!config || !sentinel) throw new Error('The encrypted vault has not been initialized.');
    const keyMaterial = await deriveVaultKeyMaterial(passphrase, base64ToBytes(config.salt), config.kdf);
    try {
      const plaintext = await decryptJson(sentinel, keyMaterial, { aad: sentinelAad(this.vaultId) });
      if (plaintext?.value !== SENTINEL_VALUE) throw new Error('Invalid vault sentinel.');
    } catch {
      throw new VaultAuthenticationError();
    }
    await this.#migrateLegacyRecords(keyMaterial);
    await this.sessionKeyStore.set(this.vaultId, keyMaterial);
    return this;
  }

  async isUnlocked() {
    return Boolean(await this.sessionKeyStore.get(this.vaultId));
  }

  async lock() {
    await this.sessionKeyStore.remove(this.vaultId);
  }

  async putRecord({ id, kind, value, schemaVersion = 1 }) {
    await this.open();
    if (typeof id !== 'string' || !id.trim()) throw new Error('Encrypted record id is required.');
    if (typeof kind !== 'string' || !kind.trim()) throw new Error('Encrypted record kind is required.');
    if (!Number.isSafeInteger(schemaVersion) || schemaVersion < 1) throw new Error('schemaVersion must be a positive integer.');
    const keyMaterial = await this.#requireKey();
    const records = await this.#readRecords(keyMaterial);
    const matches = records.filter(record => record.id === id);
    if (matches.length > 1) throw new VaultAuthenticationError();
    const existing = matches[0] || null;
    if (existing && existing.kind !== kind) throw new Error('Encrypted record kind cannot be changed in place.');
    const now = new Date().toISOString();
    const storageIds = new Set(records.map(record => record.storageId));
    let storageId = existing?.storageId || randomStorageId();
    while (!existing && storageIds.has(storageId)) storageId = randomStorageId();
    const record = {
      envelopeVersion: RECORD_ENVELOPE_VERSION,
      id,
      kind,
      schemaVersion,
      createdAt: existing?.createdAt || now,
      updatedAt: now,
      value
    };
    const encrypted = await encryptJson(record, keyMaterial, {
      aad: recordAad(this.vaultId, storageId)
    });
    const transaction = this.database.transaction(RECORD_STORE, 'readwrite');
    const transactionDone = transactionToPromise(transaction);
    transaction.objectStore(RECORD_STORE).put({
      id: storageId,
      encrypted
    });
    await transactionDone;
    return { id, kind, schemaVersion, createdAt: record.createdAt, updatedAt: now };
  }

  async getRecord(id) {
    await this.open();
    const keyMaterial = await this.#requireKey();
    const matches = (await this.#readRecords(keyMaterial)).filter(record => record.id === id);
    if (matches.length > 1) throw new VaultAuthenticationError();
    return matches[0] ? publicRecord(matches[0]) : null;
  }

  async listRecords(kind) {
    await this.open();
    const keyMaterial = await this.#requireKey();
    return (await this.#readRecords(keyMaterial))
      .filter(record => !kind || record.kind === kind)
      .sort((left, right) => left.id.localeCompare(right.id))
      .map(publicRecord);
  }

  async deleteRecord(id) {
    await this.open();
    const keyMaterial = await this.#requireKey();
    const matches = (await this.#readRecords(keyMaterial)).filter(record => record.id === id);
    if (matches.length > 1) throw new VaultAuthenticationError();
    if (!matches.length) return;
    const transaction = this.database.transaction(RECORD_STORE, 'readwrite');
    const transactionDone = transactionToPromise(transaction);
    transaction.objectStore(RECORD_STORE).delete(matches[0].storageId);
    await transactionDone;
  }

  async close({ lock = true } = {}) {
    if (lock) await this.lock();
    this.database?.close();
    this.database = null;
    this.legacyMigrationChecked = false;
    this.migrationPromise = null;
  }

  async reset() {
    await this.lock();
    this.database?.close();
    this.database = null;
    this.legacyMigrationChecked = false;
    this.migrationPromise = null;
    await new Promise((resolve, reject) => {
      const request = this.indexedDB.deleteDatabase(this.databaseName);
      request.addEventListener('success', () => resolve(), { once: true });
      request.addEventListener('error', () => reject(request.error || new Error('Unable to reset the encrypted vault.')), { once: true });
      request.addEventListener('blocked', () => reject(new Error('Vault reset is blocked by another open extension page.')), { once: true });
    });
  }

  async #requireKey() {
    const keyMaterial = await this.sessionKeyStore.get(this.vaultId);
    if (!keyMaterial) throw new VaultLockedError();
    await this.#migrateLegacyRecords(keyMaterial);
    return keyMaterial;
  }

  async #migrateLegacyRecords(keyMaterial) {
    if (this.legacyMigrationChecked) return;
    if (!this.migrationPromise) {
      this.migrationPromise = this.#performLegacyMigration(keyMaterial)
        .then(() => { this.legacyMigrationChecked = true; })
        .finally(() => { this.migrationPromise = null; });
    }
    await this.migrationPromise;
  }

  async #performLegacyMigration(keyMaterial) {
    const rawRecords = await this.#getAllRawRecords();
    const legacyRecords = rawRecords.filter(isLegacyRawRecord);
    if (!legacyRecords.length) return;

    const decodedRecords = await Promise.all(rawRecords.map(raw => this.#decryptRawRecord(raw, keyMaterial)));
    const logicalIds = new Set();
    decodedRecords.forEach(record => {
      if (logicalIds.has(record.id)) throw new VaultAuthenticationError();
      logicalIds.add(record.id);
    });

    const occupiedStorageIds = new Set(rawRecords.map(raw => raw.id));
    const replacements = [];
    for (const record of decodedRecords.filter(record => record.legacy)) {
      let storageId = randomStorageId();
      while (occupiedStorageIds.has(storageId)) storageId = randomStorageId();
      occupiedStorageIds.add(storageId);
      const envelope = {
        envelopeVersion: RECORD_ENVELOPE_VERSION,
        id: record.id,
        kind: record.kind,
        schemaVersion: record.schemaVersion,
        createdAt: record.createdAt,
        updatedAt: record.updatedAt,
        value: record.value
      };
      replacements.push({
        legacyStorageId: record.storageId,
        raw: {
          id: storageId,
          encrypted: await encryptJson(envelope, keyMaterial, { aad: recordAad(this.vaultId, storageId) })
        }
      });
    }

    const transaction = this.database.transaction(RECORD_STORE, 'readwrite');
    const transactionDone = transactionToPromise(transaction);
    const store = transaction.objectStore(RECORD_STORE);
    replacements.forEach(replacement => {
      store.delete(replacement.legacyStorageId);
      store.put(replacement.raw);
    });
    await transactionDone;
  }

  async #decryptRawRecord(raw, keyMaterial) {
    if (!raw || typeof raw.id !== 'string' || !raw.id || !raw.encrypted) {
      throw new VaultAuthenticationError();
    }
    try {
      if (isLegacyRawRecord(raw)) {
        const value = await decryptJson(raw.encrypted, keyMaterial, {
          aad: legacyRecordAad(this.vaultId, raw.id, raw.kind, raw.schemaVersion)
        });
        return {
          storageId: raw.id,
          legacy: true,
          id: raw.id,
          kind: raw.kind,
          schemaVersion: raw.schemaVersion,
          createdAt: raw.createdAt,
          updatedAt: raw.updatedAt,
          value
        };
      }
      if (Object.keys(raw).length !== 2 || !Object.hasOwn(raw, 'encrypted')) {
        throw new VaultAuthenticationError();
      }
      const envelope = validateRecordEnvelope(await decryptJson(raw.encrypted, keyMaterial, {
        aad: recordAad(this.vaultId, raw.id)
      }));
      return { storageId: raw.id, legacy: false, ...envelope };
    } catch (error) {
      if (error instanceof VaultAuthenticationError) throw error;
      throw new VaultAuthenticationError();
    }
  }

  async #readRecords(keyMaterial) {
    return Promise.all((await this.#getAllRawRecords()).map(raw => this.#decryptRawRecord(raw, keyMaterial)));
  }

  async #getMeta(key) {
    const transaction = this.database.transaction(META_STORE, 'readonly');
    const transactionDone = transactionToPromise(transaction);
    const result = await requestToPromise(transaction.objectStore(META_STORE).get(key));
    await transactionDone;
    return result || null;
  }

  async #getAllRawRecords() {
    const transaction = this.database.transaction(RECORD_STORE, 'readonly');
    const transactionDone = transactionToPromise(transaction);
    const result = await requestToPromise(transaction.objectStore(RECORD_STORE).getAll());
    await transactionDone;
    return result;
  }
}
