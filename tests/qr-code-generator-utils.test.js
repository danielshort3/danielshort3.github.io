const utils = require('../js/tools/qr-code-generator-utils.js');

module.exports = function runQrCodeGeneratorUtilsTests({ assert }) {
  assert(utils && typeof utils === 'object', 'qr utils module missing');
  assert(typeof utils.buildWifiPayload === 'function', 'buildWifiPayload missing');
  assert(typeof utils.buildVcardPayload === 'function', 'buildVcardPayload missing');
  assert(typeof utils.encodeConfig === 'function', 'encodeConfig missing');
  assert(typeof utils.decodeConfig === 'function', 'decodeConfig missing');
  assert(typeof utils.buildZip === 'function', 'buildZip missing');

  const wifi = utils.buildWifiPayload({
    ssid: 'Office;Guest',
    password: 'p@ss:word',
    auth: 'WPA',
    hidden: true,
  });
  assert(wifi.startsWith('WIFI:'), 'wifi payload should start with WIFI:');
  assert(wifi.includes('S:Office\\;Guest;'), 'wifi payload should escape SSID separators');
  assert(wifi.includes('P:p@ss\\:word;'), 'wifi payload should escape password separators');
  assert(wifi.includes('H:true;'), 'wifi payload should include hidden flag');

  const openWifi = utils.buildWifiPayload({ ssid: 'Guest', auth: 'nopass' });
  assert(openWifi.includes('T:nopass;'), 'open wifi payload should use nopass mode');
  assert(!openWifi.includes('P:'), 'open wifi payload should not include password');

  const vcard = utils.buildVcardPayload({
    firstName: 'Daniel',
    lastName: 'Short',
    org: 'Acme, Inc.',
    email: 'test@example.com',
    note: 'Line 1\nLine 2',
  });
  assert(vcard.includes('BEGIN:VCARD'), 'vcard should include BEGIN');
  assert(vcard.includes('FN:Daniel Short'), 'vcard should include formatted name');
  assert(vcard.includes('ORG:Acme\\, Inc.'), 'vcard should escape commas');
  assert(vcard.includes('NOTE:Line 1\\nLine 2'), 'vcard should escape newlines');
  assert(vcard.includes('END:VCARD'), 'vcard should include END');

  const token = utils.encodeConfig({ a: 1, b: 'test', nested: { c: true } });
  const decoded = utils.decodeConfig(token);
  assert(decoded && decoded.a === 1 && decoded.b === 'test', 'config token should round-trip');
  assert(decoded.nested && decoded.nested.c === true, 'config token should preserve nested fields');

  const zipBytes = utils.buildZip([
    { name: 'a.txt', data: 'hello' },
    { name: 'b.bin', data: new Uint8Array([1, 2, 3]) },
  ]);
  assert(zipBytes instanceof Uint8Array, 'zip output should be Uint8Array');
  assert(zipBytes.length > 50, 'zip output should contain headers and file data');
  const sig0 = String.fromCharCode(zipBytes[0], zipBytes[1], zipBytes[2], zipBytes[3]);
  assert(sig0 === 'PK\u0003\u0004', 'zip should start with local file header signature');
};
