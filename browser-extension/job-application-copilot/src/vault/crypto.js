const encoder = new TextEncoder();
const decoder = new TextDecoder();

export const DEFAULT_KDF_PARAMETERS = Object.freeze({
  name: 'PBKDF2',
  hash: 'SHA-256',
  iterations: 310000,
  keyLengthBits: 256,
  version: 1
});

const getCrypto = () => {
  if (!globalThis.crypto?.subtle || typeof globalThis.crypto.getRandomValues !== 'function') {
    throw new Error('Web Crypto is required for the encrypted vault.');
  }
  return globalThis.crypto;
};

export const bytesToBase64 = (bytes) => {
  let binary = '';
  const input = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  for (let offset = 0; offset < input.length; offset += 0x8000) {
    binary += String.fromCharCode(...input.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
};

export const base64ToBytes = (value) => {
  const binary = atob(String(value || ''));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
};

export const randomBytes = (length) => getCrypto().getRandomValues(new Uint8Array(length));

export const canonicalJson = (value) => {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const entries = Object.keys(value)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`);
    return `{${entries.join(',')}}`;
  }
  return JSON.stringify(value);
};

export const deriveVaultKeyMaterial = async (passphrase, salt, parameters = DEFAULT_KDF_PARAMETERS) => {
  if (typeof passphrase !== 'string' || passphrase.length < 8) {
    throw new Error('Vault passphrase must contain at least 8 characters.');
  }
  const cryptoApi = getCrypto();
  const baseKey = await cryptoApi.subtle.importKey(
    'raw',
    encoder.encode(passphrase),
    { name: parameters.name },
    false,
    ['deriveBits']
  );
  const bits = await cryptoApi.subtle.deriveBits({
    name: parameters.name,
    hash: parameters.hash,
    iterations: parameters.iterations,
    salt: salt instanceof Uint8Array ? salt : new Uint8Array(salt)
  }, baseKey, parameters.keyLengthBits);
  return new Uint8Array(bits);
};

const importAesKey = (keyMaterial) => getCrypto().subtle.importKey(
  'raw',
  keyMaterial instanceof Uint8Array ? keyMaterial : new Uint8Array(keyMaterial),
  { name: 'AES-GCM', length: 256 },
  false,
  ['encrypt', 'decrypt']
);

export const encryptJson = async (value, keyMaterial, { aad = {} } = {}) => {
  const iv = randomBytes(12);
  const additionalData = encoder.encode(canonicalJson(aad));
  const key = await importAesKey(keyMaterial);
  const ciphertext = await getCrypto().subtle.encrypt({
    name: 'AES-GCM',
    iv,
    additionalData,
    tagLength: 128
  }, key, encoder.encode(JSON.stringify(value)));
  return {
    version: 1,
    algorithm: 'AES-GCM',
    iv: bytesToBase64(iv),
    ciphertext: bytesToBase64(new Uint8Array(ciphertext))
  };
};

export const decryptJson = async (encrypted, keyMaterial, { aad = {} } = {}) => {
  if (!encrypted || encrypted.version !== 1 || encrypted.algorithm !== 'AES-GCM') {
    throw new Error('Unsupported encrypted vault record.');
  }
  const key = await importAesKey(keyMaterial);
  const plaintext = await getCrypto().subtle.decrypt({
    name: 'AES-GCM',
    iv: base64ToBytes(encrypted.iv),
    additionalData: encoder.encode(canonicalJson(aad)),
    tagLength: 128
  }, key, base64ToBytes(encrypted.ciphertext));
  return JSON.parse(decoder.decode(plaintext));
};

export const sha256Base64Url = async (value) => {
  const input = typeof value === 'string' ? encoder.encode(value) : value;
  const digest = new Uint8Array(await getCrypto().subtle.digest('SHA-256', input));
  return bytesToBase64(digest).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/u, '');
};

