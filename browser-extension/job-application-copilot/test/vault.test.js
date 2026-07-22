import assert from 'node:assert/strict';
import test from 'node:test';
import { IDBFactory } from 'fake-indexeddb';
import {
  EncryptedIndexedDbVault,
  VaultAuthenticationError,
  VaultLockedError
} from '../src/vault/indexeddb-vault.js';
import { encryptJson } from '../src/vault/crypto.js';
import { MemorySessionKeyStore } from '../src/vault/session-key-store.js';

const requestResult = request => new Promise((resolve, reject) => {
  request.addEventListener('success', () => resolve(request.result), { once: true });
  request.addEventListener('error', () => reject(request.error), { once: true });
});

const readRawRecords = async (indexedDB, databaseName) => {
  const database = await requestResult(indexedDB.open(databaseName));
  const records = await requestResult(database.transaction('records', 'readonly').objectStore('records').getAll());
  database.close();
  return records;
};

const readRecordStoreIndexNames = async (indexedDB, databaseName) => {
  const database = await requestResult(indexedDB.open(databaseName));
  const names = Array.from(database.transaction('records', 'readonly').objectStore('records').indexNames);
  database.close();
  return names;
};

const writeRawRecord = async (indexedDB, databaseName, record) => {
  const database = await requestResult(indexedDB.open(databaseName));
  const transaction = database.transaction('records', 'readwrite');
  transaction.objectStore('records').put(record);
  await new Promise((resolve, reject) => {
    transaction.addEventListener('complete', resolve, { once: true });
    transaction.addEventListener('error', () => reject(transaction.error), { once: true });
  });
  database.close();
};

test('encrypted IndexedDB vault locks, authenticates, detects tampering, and resets', async () => {
  const indexedDB = new IDBFactory();
  const sessionKeyStore = new MemorySessionKeyStore();
  const databaseName = `vault-${crypto.randomUUID()}`;
  const vault = new EncryptedIndexedDbVault({ indexedDB, databaseName, sessionKeyStore });
  await vault.initialize('correct horse battery staple');
  await vault.putRecord({ id: 'doc:one', kind: 'document', value: { secret: 'resume data' } });
  const [firstRaw] = await readRawRecords(indexedDB, databaseName);
  assert.deepEqual(await readRecordStoreIndexNames(indexedDB, databaseName), []);
  assert.deepEqual(Object.keys(firstRaw).sort(), ['encrypted', 'id']);
  assert.match(firstRaw.id, /^[A-Za-z0-9_-]{43}$/u);
  assert.notEqual(firstRaw.id, 'doc:one');
  assert.doesNotMatch(JSON.stringify(firstRaw), /doc:one|document|resume data|createdAt|updatedAt|schemaVersion/u);
  await vault.putRecord({ id: 'doc:one', kind: 'document', value: { secret: 'resume data' } });
  const [secondRaw] = await readRawRecords(indexedDB, databaseName);
  assert.equal(secondRaw.id, firstRaw.id);
  assert.notEqual(firstRaw.encrypted.iv, secondRaw.encrypted.iv);
  assert.equal((await vault.getRecord('doc:one')).value.secret, 'resume data');
  assert.deepEqual((await vault.listRecords('document')).map(record => record.id), ['doc:one']);
  await vault.putRecord({ id: 'fact:email', kind: 'fact', value: { value: 'candidate@example.com' } });
  const opaqueRows = await readRawRecords(indexedDB, databaseName);
  assert.equal(opaqueRows.length, 2);
  opaqueRows.forEach(row => assert.deepEqual(Object.keys(row).sort(), ['encrypted', 'id']));
  assert.doesNotMatch(JSON.stringify(opaqueRows), /fact:email|candidate@example\.com|document|createdAt|updatedAt|schemaVersion/u);
  assert.deepEqual((await vault.listRecords('fact')).map(record => record.id), ['fact:email']);
  await vault.deleteRecord('fact:email');
  assert.equal(await vault.getRecord('fact:email'), null);
  assert.equal((await readRawRecords(indexedDB, databaseName)).length, 1);

  await vault.lock();
  await assert.rejects(vault.getRecord('doc:one'), VaultLockedError);
  await assert.rejects(vault.unlock('incorrect passphrase'), VaultAuthenticationError);
  await vault.unlock('correct horse battery staple');

  const [tampered] = await readRawRecords(indexedDB, databaseName);
  tampered.encrypted.ciphertext = `${tampered.encrypted.ciphertext.slice(0, -2)}AA`;
  await writeRawRecord(indexedDB, databaseName, tampered);
  await assert.rejects(vault.getRecord('doc:one'));

  await vault.reset();
  assert.equal(await sessionKeyStore.get('default'), null);
  assert.equal(await vault.isInitialized(), false);
  await vault.close();
});

test('unlock migrates legacy plaintext metadata rows to opaque encrypted envelopes', async () => {
  const indexedDB = new IDBFactory();
  const sessionKeyStore = new MemorySessionKeyStore();
  const databaseName = `vault-legacy-${crypto.randomUUID()}`;
  const passphrase = 'legacy migration passphrase';
  const legacyId = 'doc:legacy-resume';
  const legacyKind = 'document';
  const legacySchemaVersion = 3;
  const legacyCreatedAt = '2025-01-02T03:04:05.000Z';
  const legacyUpdatedAt = '2025-06-07T08:09:10.000Z';
  const vault = new EncryptedIndexedDbVault({ indexedDB, databaseName, sessionKeyStore });
  await vault.initialize(passphrase);
  const keyMaterial = await sessionKeyStore.get('default');
  const encrypted = await encryptJson({ secret: 'legacy resume text' }, keyMaterial, {
    aad: {
      vaultId: 'default',
      store: 'records',
      recordId: legacyId,
      kind: legacyKind,
      schemaVersion: legacySchemaVersion
    }
  });
  await writeRawRecord(indexedDB, databaseName, {
    id: legacyId,
    kind: legacyKind,
    schemaVersion: legacySchemaVersion,
    createdAt: legacyCreatedAt,
    updatedAt: legacyUpdatedAt,
    encrypted
  });
  await vault.close();

  const reopened = new EncryptedIndexedDbVault({ indexedDB, databaseName, sessionKeyStore });
  await reopened.unlock(passphrase);
  const rawRecords = await readRawRecords(indexedDB, databaseName);
  assert.equal(rawRecords.length, 1);
  assert.deepEqual(Object.keys(rawRecords[0]).sort(), ['encrypted', 'id']);
  assert.match(rawRecords[0].id, /^[A-Za-z0-9_-]{43}$/u);
  const serialized = JSON.stringify(rawRecords[0]);
  assert.doesNotMatch(serialized, /doc:legacy-resume|document|2025-01-02|2025-06-07|legacy resume text|schemaVersion/u);

  assert.deepEqual(await reopened.getRecord(legacyId), {
    id: legacyId,
    kind: legacyKind,
    schemaVersion: legacySchemaVersion,
    createdAt: legacyCreatedAt,
    updatedAt: legacyUpdatedAt,
    value: { secret: 'legacy resume text' }
  });
  await reopened.close();
});
