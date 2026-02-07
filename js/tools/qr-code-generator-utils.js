(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
    return;
  }
  root.QrToolUtils = factory();
}(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  'use strict';

  const textEncoder = (() => {
    try {
      return typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;
    } catch {
      return null;
    }
  })();

  const textDecoder = (() => {
    try {
      return typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8') : null;
    } catch {
      return null;
    }
  })();

  const toUtf8Bytes = (value) => {
    const text = String(value || '');
    if (textEncoder) return textEncoder.encode(text);
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(text, 'utf8'));
    const out = [];
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      out.push(code & 0xFF);
    }
    return new Uint8Array(out);
  };

  const fromUtf8Bytes = (bytes) => {
    if (!bytes) return '';
    if (textDecoder) return textDecoder.decode(bytes);
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('utf8');
    let out = '';
    for (let i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
    return out;
  };

  const toBase64 = (bytes) => {
    if (!bytes) return '';
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    if (typeof btoa === 'function') return btoa(binary);
    return '';
  };

  const fromBase64 = (base64) => {
    const clean = String(base64 || '');
    if (!clean) return new Uint8Array(0);
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(clean, 'base64'));
    if (typeof atob === 'function') {
      const binary = atob(clean);
      const out = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
      return out;
    }
    return new Uint8Array(0);
  };

  const encodeConfig = (value) => {
    const json = JSON.stringify(value || {});
    const b64 = toBase64(toUtf8Bytes(json));
    return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
  };

  const decodeConfig = (token) => {
    const clean = String(token || '').trim();
    if (!clean) return null;
    const b64 = clean
      .replace(/-/g, '+')
      .replace(/_/g, '/')
      .padEnd(Math.ceil(clean.length / 4) * 4, '=');
    try {
      return JSON.parse(fromUtf8Bytes(fromBase64(b64)));
    } catch {
      return null;
    }
  };

  const escapeWifiField = (value) => String(value || '')
    .replace(/\\/g, '\\\\')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,')
    .replace(/:/g, '\\:');

  const buildWifiPayload = ({ ssid, password, auth, hidden } = {}) => {
    const safeSsid = escapeWifiField(ssid);
    const modeRaw = String(auth || 'WPA').trim().toUpperCase();
    const mode = ['WEP', 'WPA', 'NOPASS'].includes(modeRaw) ? modeRaw : 'WPA';
    const isOpen = mode === 'NOPASS';
    const safePassword = isOpen ? '' : escapeWifiField(password);
    const hiddenFlag = hidden ? 'true' : '';
    if (!safeSsid) return '';
    const parts = [`WIFI:T:${isOpen ? 'nopass' : mode};S:${safeSsid};`];
    if (!isOpen && safePassword) parts.push(`P:${safePassword};`);
    if (hiddenFlag) parts.push(`H:${hiddenFlag};`);
    parts.push(';');
    return parts.join('');
  };

  const escapeVcardValue = (value) => String(value || '')
    .replace(/\\/g, '\\\\')
    .replace(/\n/g, '\\n')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,');

  const buildVcardPayload = ({
    firstName,
    lastName,
    org,
    title,
    phone,
    email,
    website,
    address,
    note
  } = {}) => {
    const fn = [String(firstName || '').trim(), String(lastName || '').trim()]
      .filter(Boolean)
      .join(' ')
      .trim();
    const nField = `${escapeVcardValue(lastName)};${escapeVcardValue(firstName)};;;`;

    const lines = [
      'BEGIN:VCARD',
      'VERSION:3.0',
      `N:${nField}`
    ];

    if (fn) lines.push(`FN:${escapeVcardValue(fn)}`);
    if (org) lines.push(`ORG:${escapeVcardValue(org)}`);
    if (title) lines.push(`TITLE:${escapeVcardValue(title)}`);
    if (phone) lines.push(`TEL;TYPE=CELL:${escapeVcardValue(phone)}`);
    if (email) lines.push(`EMAIL;TYPE=INTERNET:${escapeVcardValue(email)}`);
    if (website) lines.push(`URL:${escapeVcardValue(website)}`);
    if (address) lines.push(`ADR;TYPE=WORK:;;${escapeVcardValue(address)};;;;`);
    if (note) lines.push(`NOTE:${escapeVcardValue(note)}`);
    lines.push('END:VCARD');
    return lines.join('\n');
  };

  const crcTable = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let j = 0; j < 8; j++) {
        c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      }
      table[i] = c >>> 0;
    }
    return table;
  })();

  const crc32 = (bytes) => {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) {
      crc = crcTable[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  };

  const asUint8Array = (input) => {
    if (input instanceof Uint8Array) return input;
    if (typeof input === 'string') return toUtf8Bytes(input);
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    return new Uint8Array(0);
  };

  const dosDateTime = (dateValue) => {
    const d = dateValue instanceof Date ? dateValue : new Date();
    const year = Math.min(Math.max(d.getFullYear(), 1980), 2107);
    const dosTime = (d.getSeconds() >> 1) | (d.getMinutes() << 5) | (d.getHours() << 11);
    const dosDate = d.getDate() | ((d.getMonth() + 1) << 5) | ((year - 1980) << 9);
    return { dosDate, dosTime };
  };

  const writeU16 = (view, offset, value) => {
    view.setUint16(offset, value & 0xFFFF, true);
  };

  const writeU32 = (view, offset, value) => {
    view.setUint32(offset, value >>> 0, true);
  };

  const buildZip = (entries) => {
    const safeEntries = Array.isArray(entries) ? entries : [];
    const localParts = [];
    const centralParts = [];
    let offset = 0;
    let totalCentralLength = 0;

    safeEntries.forEach((entry) => {
      const name = String(entry && entry.name ? entry.name : 'file.bin').replace(/\\/g, '/');
      const nameBytes = toUtf8Bytes(name);
      const dataBytes = asUint8Array(entry && entry.data);
      const crc = crc32(dataBytes);
      const size = dataBytes.length;
      const { dosDate, dosTime } = dosDateTime(entry && entry.date);

      const localHeader = new Uint8Array(30 + nameBytes.length);
      const localView = new DataView(localHeader.buffer);
      writeU32(localView, 0, 0x04034B50);
      writeU16(localView, 4, 20);
      writeU16(localView, 6, 0);
      writeU16(localView, 8, 0);
      writeU16(localView, 10, dosTime);
      writeU16(localView, 12, dosDate);
      writeU32(localView, 14, crc);
      writeU32(localView, 18, size);
      writeU32(localView, 22, size);
      writeU16(localView, 26, nameBytes.length);
      writeU16(localView, 28, 0);
      localHeader.set(nameBytes, 30);

      localParts.push(localHeader, dataBytes);

      const centralHeader = new Uint8Array(46 + nameBytes.length);
      const centralView = new DataView(centralHeader.buffer);
      writeU32(centralView, 0, 0x02014B50);
      writeU16(centralView, 4, 20);
      writeU16(centralView, 6, 20);
      writeU16(centralView, 8, 0);
      writeU16(centralView, 10, 0);
      writeU16(centralView, 12, dosTime);
      writeU16(centralView, 14, dosDate);
      writeU32(centralView, 16, crc);
      writeU32(centralView, 20, size);
      writeU32(centralView, 24, size);
      writeU16(centralView, 28, nameBytes.length);
      writeU16(centralView, 30, 0);
      writeU16(centralView, 32, 0);
      writeU16(centralView, 34, 0);
      writeU16(centralView, 36, 0);
      writeU32(centralView, 38, 0);
      writeU32(centralView, 42, offset);
      centralHeader.set(nameBytes, 46);

      centralParts.push(centralHeader);

      offset += localHeader.length + dataBytes.length;
      totalCentralLength += centralHeader.length;
    });

    const endRecord = new Uint8Array(22);
    const endView = new DataView(endRecord.buffer);
    writeU32(endView, 0, 0x06054B50);
    writeU16(endView, 4, 0);
    writeU16(endView, 6, 0);
    writeU16(endView, 8, safeEntries.length);
    writeU16(endView, 10, safeEntries.length);
    writeU32(endView, 12, totalCentralLength);
    writeU32(endView, 16, offset);
    writeU16(endView, 20, 0);

    const totalLength = localParts.reduce((sum, part) => sum + part.length, 0)
      + totalCentralLength
      + endRecord.length;
    const out = new Uint8Array(totalLength);

    let ptr = 0;
    localParts.forEach((part) => {
      out.set(part, ptr);
      ptr += part.length;
    });
    centralParts.forEach((part) => {
      out.set(part, ptr);
      ptr += part.length;
    });
    out.set(endRecord, ptr);
    return out;
  };

  return {
    encodeConfig,
    decodeConfig,
    buildWifiPayload,
    buildVcardPayload,
    buildZip,
    toUtf8Bytes,
    fromUtf8Bytes,
    crc32,
  };
}));
