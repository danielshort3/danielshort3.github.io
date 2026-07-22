import { base64ToBytes, bytesToBase64 } from './crypto.js';

const keyName = (vaultId) => `jobCopilotVaultKey:${vaultId}`;

export class MemorySessionKeyStore {
  #keys = new Map();

  async initialize() {}

  async set(vaultId, keyMaterial) {
    this.#keys.set(vaultId, new Uint8Array(keyMaterial));
  }

  async get(vaultId) {
    const value = this.#keys.get(vaultId);
    return value ? new Uint8Array(value) : null;
  }

  async remove(vaultId) {
    this.#keys.delete(vaultId);
  }
}

export class ChromeSessionKeyStore {
  constructor(storageArea = globalThis.chrome?.storage?.session) {
    if (!storageArea) throw new Error('chrome.storage.session is unavailable.');
    this.storageArea = storageArea;
  }

  async initialize() {
    if (typeof this.storageArea.setAccessLevel === 'function') {
      await this.storageArea.setAccessLevel({ accessLevel: 'TRUSTED_CONTEXTS' });
    }
  }

  async set(vaultId, keyMaterial) {
    await this.storageArea.set({ [keyName(vaultId)]: bytesToBase64(keyMaterial) });
  }

  async get(vaultId) {
    const name = keyName(vaultId);
    const values = await this.storageArea.get(name);
    return values?.[name] ? base64ToBytes(values[name]) : null;
  }

  async remove(vaultId) {
    await this.storageArea.remove(keyName(vaultId));
  }
}

export const createDefaultSessionKeyStore = () => globalThis.chrome?.storage?.session
  ? new ChromeSessionKeyStore()
  : new MemorySessionKeyStore();

