#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const Data = require('../js/games/project-starfall/project-starfall-data.js');

const ROOT_DIR = path.resolve(__dirname, '..');

const THEMES = Object.freeze({
  character: { accent: '#7bdff2', dark: '#14233a', mid: '#2f7dd6', motif: characterMotif },
  equipment: { accent: '#f6c75a', dark: '#23253d', mid: '#6f8cff', motif: equipmentMotif },
  partyPanel: { accent: '#8ef0a9', dark: '#163326', mid: '#2f9a67', motif: partyMotif },
  inventory: { accent: '#d9a86c', dark: '#2b2430', mid: '#895f3d', motif: inventoryMotif },
  skills: { accent: '#c794ff', dark: '#271f40', mid: '#6f56d9', motif: skillsMotif },
  quests: { accent: '#f3d878', dark: '#332918', mid: '#b88a2e', motif: questsMotif },
  worldmap: { accent: '#80e6d1', dark: '#14343b', mid: '#2f7dd6', motif: worldmapMotif },
  monsters: { accent: '#ff768c', dark: '#381f2b', mid: '#9b3456', motif: monstersMotif },
  shop: { accent: '#ffd16a', dark: '#342516', mid: '#c86d3c', motif: shopMotif },
  upgrade: { accent: '#79e0ff', dark: '#18263d', mid: '#506fd8', motif: upgradeMotif },
  cashShop: { accent: '#ffdc73', dark: '#342a17', mid: '#d49a26', motif: cashShopMotif },
  beta: { accent: '#75f0c2', dark: '#17372f', mid: '#388f7b', motif: betaMotif },
  guide: { accent: '#f4e6a0', dark: '#2d2a20', mid: '#7d8f5f', motif: guideMotif },
  log: { accent: '#d6ecff', dark: '#1c2b3d', mid: '#5c7fa6', motif: logMotif },
  settings: { accent: '#c6d4e8', dark: '#1d2734', mid: '#627990', motif: settingsMotif },
  keybinds: { accent: '#ffc987', dark: '#302217', mid: '#b46f3c', motif: keybindsMotif },
  admin: { accent: '#96ffec', dark: '#112f36', mid: '#3a8492', motif: adminMotif },
  logout: { accent: '#ff9b7d', dark: '#3a1c20', mid: '#b83232', motif: logoutMotif }
});

function iconFrame(theme, id, motif) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" role="img" aria-label="${id}">
  <defs>
    <linearGradient id="bg" x1="12" y1="8" x2="54" y2="58" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="${theme.mid}" stop-opacity="0.96"/>
      <stop offset="1" stop-color="${theme.dark}" stop-opacity="0.98"/>
    </linearGradient>
    <radialGradient id="shine" cx="21" cy="16" r="38" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.42"/>
      <stop offset="0.52" stop-color="${theme.accent}" stop-opacity="0.12"/>
      <stop offset="1" stop-color="${theme.dark}" stop-opacity="0"/>
    </radialGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#081322" flood-opacity="0.36"/>
    </filter>
  </defs>
  <rect x="7" y="6" width="50" height="52" rx="14" fill="url(#bg)" filter="url(#shadow)"/>
  <rect x="10" y="9" width="44" height="46" rx="11" fill="url(#shine)"/>
  <path d="M14 17 C22 9 42 8 50 18" fill="none" stroke="#ffffff" stroke-opacity="0.18" stroke-width="2" stroke-linecap="round"/>
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">${motif(theme)}</g>
</svg>`;
}

function characterMotif(theme) {
  return `
    <circle cx="32" cy="22" r="8" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M20 45 C22 35 42 35 44 45 V49 H20 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M25 19 C29 13 37 14 40 20" stroke="${theme.dark}" stroke-width="3"/>`;
}

function equipmentMotif(theme) {
  return `
    <path d="M32 14 L46 20 V31 C46 41 40 48 32 51 C24 48 18 41 18 31 V20 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M32 17 V47 M23 26 H41" stroke="${theme.accent}" stroke-width="3"/>
    <path d="M22 46 L43 16" stroke="#ffffff" stroke-width="3"/>`;
}

function partyMotif(theme) {
  return `
    <circle cx="25" cy="24" r="6" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <circle cx="40" cy="24" r="6" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <circle cx="32" cy="18" r="6" fill="#ffffff" stroke="${theme.dark}" stroke-width="2"/>
    <path d="M18 45 C20 35 30 35 32 45 M32 45 C34 35 44 35 46 45 M22 49 H42" stroke="#ffffff" stroke-width="3"/>`;
}

function inventoryMotif(theme) {
  return `
    <path d="M19 25 H45 L42 49 H22 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M25 25 V21 C25 14 39 14 39 21 V25" stroke="${theme.accent}" stroke-width="3"/>
    <path d="M23 34 H41 M25 41 H39" stroke="#ffffff" stroke-opacity="0.75" stroke-width="2"/>`;
}

function skillsMotif(theme) {
  return `
    <path d="M32 13 L36 27 L50 31 L37 38 L32 51 L27 38 L14 31 L28 27 Z" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M32 20 V44 M22 31 H42" stroke="${theme.dark}" stroke-width="2"/>
    <circle cx="32" cy="31" r="4" fill="#ffffff" stroke="${theme.mid}" stroke-width="2"/>`;
}

function questsMotif(theme) {
  return `
    <path d="M20 15 H40 L45 20 V49 H20 Z" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M40 15 V21 H46" stroke="${theme.dark}" stroke-width="2"/>
    <path d="M25 28 H39 M25 35 H38 M25 42 H35" stroke="${theme.dark}" stroke-width="3"/>
    <circle cx="43" cy="43" r="5" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>`;
}

function worldmapMotif(theme) {
  return `
    <path d="M18 18 L29 14 L39 20 L48 17 V46 L37 50 L27 44 L18 47 Z" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M29 14 V44 M39 20 V50" stroke="${theme.dark}" stroke-width="2"/>
    <path d="M21 34 C27 26 35 40 46 30" stroke="${theme.mid}" stroke-width="3"/>`;
}

function monstersMotif(theme) {
  return `
    <path d="M18 38 C18 25 26 17 32 17 C38 17 46 25 46 38 C46 48 18 48 18 38 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M22 22 L17 16 M42 22 L47 16" stroke="${theme.accent}" stroke-width="3"/>
    <circle cx="27" cy="33" r="4" fill="#ffffff"/>
    <circle cx="37" cy="33" r="4" fill="#ffffff"/>
    <path d="M27 42 L31 38 L35 42 L39 38" stroke="#ffffff" stroke-width="2"/>`;
}

function shopMotif(theme) {
  return `
    <path d="M17 25 H47 L43 16 H21 Z" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M20 28 H44 V49 H20 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M17 25 C19 32 25 32 27 25 C29 32 35 32 37 25 C39 32 45 32 47 25" stroke="#ffffff" stroke-width="2"/>
    <path d="M27 49 V37 H37 V49" stroke="${theme.dark}" stroke-width="3"/>`;
}

function upgradeMotif(theme) {
  return `
    <path d="M20 45 L45 20" stroke="#ffffff" stroke-width="6"/>
    <path d="M18 48 L24 42 M42 22 L48 16" stroke="${theme.accent}" stroke-width="4"/>
    <path d="M32 15 L35 23 L43 26 L35 29 L32 37 L29 29 L21 26 L29 23 Z" fill="${theme.accent}" stroke="${theme.dark}" stroke-width="2"/>`;
}

function cashShopMotif(theme) {
  return `
    <circle cx="32" cy="32" r="17" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M32 20 V44 M24 27 C25 22 39 22 40 28 C41 36 25 31 25 38 C25 44 39 43 41 37" stroke="${theme.dark}" stroke-width="4"/>
    <path d="M18 18 L14 13 M46 18 L50 13 M18 46 L14 51 M46 46 L50 51" stroke="#ffffff" stroke-width="2"/>`;
}

function betaMotif(theme) {
  return `
    <path d="M27 16 H37 M30 17 V30 L20 48 H44 L34 30 V17" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M25 39 H39" stroke="${theme.accent}" stroke-width="3"/>
    <circle cx="28" cy="43" r="2" fill="#ffffff"/>
    <circle cx="35" cy="38" r="2" fill="#ffffff"/>`;
}

function guideMotif(theme) {
  return `
    <path d="M18 18 H31 C34 18 36 20 36 23 V49 C34 47 31 46 27 46 H18 Z" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M36 23 C36 20 38 18 41 18 H46 V46 H37 C34 46 32 47 30 49" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M23 27 H31 M23 34 H31 M39 27 H44 M39 34 H44" stroke="${theme.dark}" stroke-width="2"/>`;
}

function logMotif(theme) {
  return `
    <path d="M22 14 H40 L46 20 V50 H22 Z" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M40 14 V21 H46" stroke="${theme.dark}" stroke-width="2"/>
    <path d="M27 29 H40 M27 36 H39 M27 43 H36" stroke="${theme.dark}" stroke-width="3"/>
    <path d="M20 20 H18 V54 H42" stroke="#ffffff" stroke-opacity="0.65" stroke-width="2"/>`;
}

function settingsMotif(theme) {
  return `
    <circle cx="32" cy="32" r="8" fill="${theme.accent}" stroke="#ffffff" stroke-width="2"/>
    <path d="M32 15 V21 M32 43 V49 M15 32 H21 M43 32 H49 M20 20 L24 24 M40 40 L44 44 M44 20 L40 24 M24 40 L20 44" stroke="#ffffff" stroke-width="4"/>
    <circle cx="32" cy="32" r="3" fill="${theme.dark}"/>`;
}

function keybindsMotif(theme) {
  return `
    <path d="M18 23 H46 V45 H18 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M23 29 H30 V36 H23 Z M34 29 H41 V36 H34 Z M25 39 H39" stroke="${theme.accent}" stroke-width="2"/>
    <path d="M25 17 H39" stroke="#ffffff" stroke-width="3"/>
    <path d="M28 17 V23 M36 17 V23" stroke="${theme.accent}" stroke-width="3"/>`;
}

function adminMotif(theme) {
  return `
    <rect x="18" y="18" width="28" height="22" rx="4" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M23 27 H31 M23 33 H37" stroke="${theme.accent}" stroke-width="3"/>
    <path d="M26 47 H38 M32 40 V47" stroke="#ffffff" stroke-width="3"/>
    <circle cx="41" cy="27" r="2" fill="#ffffff"/>`;
}

function logoutMotif(theme) {
  return `
    <path d="M21 17 H36 V47 H21 Z" fill="${theme.mid}" stroke="#ffffff" stroke-width="2"/>
    <path d="M34 32 H48 M42 24 L50 32 L42 40" stroke="${theme.accent}" stroke-width="5"/>
    <path d="M28 32 H29" stroke="${theme.dark}" stroke-width="4"/>`;
}

async function writeIcon(id, assetPath) {
  const theme = THEMES[id] || THEMES.settings;
  const svg = iconFrame(theme, id, theme.motif);
  const outputPath = path.join(ROOT_DIR, assetPath);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9 })
    .toFile(outputPath);
}

async function main() {
  const entries = Object.entries(Data.MENU_ICON_ASSETS || {});
  for (const [id, assetPath] of entries) await writeIcon(id, assetPath);
  console.log(`Generated ${entries.length} Project Starfall menu icons.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
