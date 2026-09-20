'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createLocalServer } = require('../../build/dev');
const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'site-release-env-'));
// An empty environment directory prevents loading personal development keys.
const server = createLocalServer({ envDir });
server.listen(Number(process.env.RELEASE_TEST_PORT || 4219), '127.0.0.1');
function close() {
  server.closeAllConnections();
  server.close(() => { fs.rmdirSync(envDir); process.exit(0); });
}
process.on('SIGTERM', close);
process.on('SIGINT', close);
