'use strict';

const fs = require('fs');
const path = require('path');
const { parseEnv } = require('util');

function loadLocalEnvironment(directory, env = process.env) {
  const loaded = [];
  // Shell settings win, followed by local overrides, then shared defaults.
  for (const filename of ['.env.local', '.env']) {
    let source;
    try {
      source = fs.readFileSync(path.join(directory, filename), 'utf8');
    } catch (err) {
      if (err.code === 'ENOENT') continue;
      throw err;
    }
    for (const [key, value] of Object.entries(parseEnv(source))) {
      if (!Object.hasOwn(env, key)) env[key] = value;
    }
    loaded.push(filename);
  }
  return loaded;
}

module.exports = { loadLocalEnvironment };
