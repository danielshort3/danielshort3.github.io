'use strict';

const fs = require('node:fs');
const path = require('node:path');
const gamePage = require('../../content/pages/games.json');
const CONTENT_DIR = path.join(__dirname, '..', '..', 'content');

const GROUPS = [
  { prefix: 'tool', label: 'Tools', directory: 'tools', key: 'slug' },
  { prefix: 'project', label: 'Projects', directory: 'projects', key: 'id' }
];
const GROUP_ORDER = ['Tools', 'Projects', 'Games'];

function loadStatusCatalog(contentDir = CONTENT_DIR) {
  const games = contentDir === CONTENT_DIR ? gamePage
    : JSON.parse(fs.readFileSync(path.join(contentDir, 'pages', 'games.json'), 'utf8'));
  const entries = GROUPS.flatMap(({ prefix, label, directory, key }) => {
    const root = path.join(contentDir, directory);
    return fs.readdirSync(root)
      .filter((file) => file.endsWith('.json'))
      .map((file) => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8')))
      .filter((item) => item.published !== false && typeof item[key] === 'string' && typeof item.title === 'string')
      .map((item) => ({ id: `${prefix}:${item[key]}`, label: item.title, group: label }));
  });

  entries.push(...games.games
    .filter((item) => item.published !== false && typeof item.id === 'string' && typeof item.title === 'string')
    .map((item) => ({ id: `game:${item.id}`, label: item.title, group: 'Games' })));
  return entries.sort((left, right) =>
    GROUP_ORDER.indexOf(left.group) - GROUP_ORDER.indexOf(right.group) || left.label.localeCompare(right.label));
}

module.exports = { loadStatusCatalog };
