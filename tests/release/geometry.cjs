'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { createLocalServer } = require('../../build/dev');
const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'release-geometry-env-'));
const server = createLocalServer({ envDir });
server.listen(0, '127.0.0.1', () => {
  const child = spawn(process.execPath, ['tests/site/frame-geometry.browser.cjs'], {
    stdio: 'inherit',
    env: { ...process.env, FRAME_SEAM_URL: `http://127.0.0.1:${server.address().port}`, FRAME_SEAM_GROUPS: 'routes,typography', FRAME_SEAM_SIZES: '[{"width":1440,"height":900},{"width":844,"height":390}]', FRAME_SEAM_ARTIFACT_DIR: 'tmp/release-geometry' }
  });
  child.on('exit', (code) => {
    server.closeAllConnections();
    server.close(() => { fs.rmdirSync(envDir); process.exit(code || 0); });
  });
});
