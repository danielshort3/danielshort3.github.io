#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const TERRAIN_DIR = path.join(ROOT, 'img/project-starfall/environment/terrain');
const PROP_DIR = path.join(ROOT, 'img/project-starfall/environment/props');
const STRUCTURE_DIR = path.join(ROOT, 'img/project-starfall/environment/structures');
const CELL = 64;
const STRUCTURE_CELL = 256;

const THEMES = Object.freeze([
  { id: 'town', top: '#d9a85c', body: '#8b6a46', dark: '#5a4432', light: '#f7d28a', accent: '#7ec8d8', leaf: '#6ca66a', flower: '#f0c36a' },
  { id: 'grass', top: '#77bf65', body: '#6b4f36', dark: '#2f6848', light: '#b6e37c', accent: '#91dbe8', leaf: '#4f9b58', flower: '#ffe16a' },
  { id: 'thorn', top: '#3f8f58', body: '#5b3d2d', dark: '#243c2d', light: '#81c95f', accent: '#c4475d', leaf: '#2e7a43', flower: '#e05b75' },
  { id: 'bramble', top: '#294d38', body: '#513325', dark: '#1f2f28', light: '#6aa86b', accent: '#e05b75', leaf: '#345f3f', flower: '#ff7c9b' },
  { id: 'ruins', top: '#8c6b35', body: '#565d65', dark: '#303842', light: '#d8b74a', accent: '#29b3ad', leaf: '#5a6c54', flower: '#75dacb' },
  { id: 'gearworks', top: '#665b48', body: '#7a8592', dark: '#343a42', light: '#d8b74a', accent: '#29b3ad', leaf: '#55666d', flower: '#64dcca' },
  { id: 'cinder', top: '#9b4835', body: '#2d2c30', dark: '#201c22', light: '#ff7842', accent: '#d8a531', leaf: '#5d4038', flower: '#ffbe55' },
  { id: 'ember', top: '#d8a531', body: '#3b2c32', dark: '#211c22', light: '#ff7842', accent: '#ffbe55', leaf: '#6b3835', flower: '#ffd166' },
  { id: 'ridge', top: '#c3995b', body: '#6f5132', dark: '#3f2c24', light: '#dcb978', accent: '#4f7b63', leaf: '#5e8f58', flower: '#ffe16a' },
  { id: 'quarry', top: '#b58a4a', body: '#6b6960', dark: '#3a4145', light: '#c3b48f', accent: '#62c5a2', leaf: '#4f7b63', flower: '#8cf0d4' },
  { id: 'ashglass', top: '#d8a531', body: '#4d3d47', dark: '#292630', light: '#f06b37', accent: '#ffbe55', leaf: '#6a514d', flower: '#ffd166' },
  { id: 'frost', top: '#d7f3ff', body: '#5ca8e8', dark: '#2f5878', light: '#f7fbff', accent: '#8fd7ff', leaf: '#7fb7c8', flower: '#eaf8ff' },
  { id: 'storm', top: '#ffe16a', body: '#4f6073', dark: '#2e394a', light: '#91dbe8', accent: '#7bdff2', leaf: '#6e8d84', flower: '#fff1a8' },
  { id: 'astral', top: '#28c7b7', body: '#29365f', dark: '#1c243f', light: '#7bdff2', accent: '#c794ff', leaf: '#306f85', flower: '#d9b7ff' },
  { id: 'eclipse', top: '#ffbe55', body: '#1f2330', dark: '#151926', light: '#ffe16a', accent: '#7bdff2', leaf: '#4a5c70', flower: '#b8f3ff' },
  { id: 'rift', top: '#7bdff2', body: '#191b2c', dark: '#101321', light: '#f06bff', accent: '#ffbe55', leaf: '#423a75', flower: '#ff9cff' },
  { id: 'bramble-court', top: '#6aa86b', body: '#513325', dark: '#1f2f28', light: '#9bc776', accent: '#e05b75', leaf: '#345f3f', flower: '#ff7c9b' },
  { id: 'titan-foundry', top: '#d8b74a', body: '#7a8592', dark: '#303842', light: '#f0d078', accent: '#29b3ad', leaf: '#55666d', flower: '#64dcca' },
  { id: 'deepcore-core', top: '#c3b48f', body: '#6b6960', dark: '#2a2f34', light: '#e4d6a0', accent: '#69d1a6', leaf: '#4f7b63', flower: '#8cf0d4' },
  { id: 'ember-furnace', top: '#ff7842', body: '#3b2c32', dark: '#211c22', light: '#ffd166', accent: '#ffbe55', leaf: '#6b3835', flower: '#ffe16a' },
  { id: 'rime-vault', top: '#d7f3ff', body: '#5f88a9', dark: '#223d5a', light: '#ecfbff', accent: '#79e7ff', leaf: '#7fb7c8', flower: '#eaf8ff' },
  { id: 'storm-aerie', top: '#ffe16a', body: '#4f6073', dark: '#26364f', light: '#91dbe8', accent: '#7bdff2', leaf: '#6e8d84', flower: '#fff1a8' },
  { id: 'astral-stacks', top: '#28c7b7', body: '#29365f', dark: '#1c243f', light: '#7bdff2', accent: '#c794ff', leaf: '#306f85', flower: '#d9b7ff' },
  { id: 'eclipse-throne', top: '#ffbe55', body: '#1f2330', dark: '#151926', light: '#ffe16a', accent: '#7bdff2', leaf: '#4a5c70', flower: '#b8f3ff' },
  { id: 'guardian-trial', top: '#6fa8d9', body: '#4d5968', dark: '#273141', light: '#d5ecff', accent: '#ffd166', leaf: '#5f8061', flower: '#f0c36a' },
  { id: 'berserker-trial', top: '#a94a3c', body: '#55323a', dark: '#241e26', light: '#ff8a5f', accent: '#ffd166', leaf: '#6b3d38', flower: '#ff6b5e' },
  { id: 'duelist-trial', top: '#d8b46f', body: '#7b6048', dark: '#3f2c24', light: '#ffe0a6', accent: '#68a9ff', leaf: '#5e8f58', flower: '#ffe16a' },
  { id: 'fire-mage-trial', top: '#ff7842', body: '#3b2c32', dark: '#211c22', light: '#ffd166', accent: '#ff3d2e', leaf: '#6b3835', flower: '#ffe16a' },
  { id: 'rune-mage-trial', top: '#28c7b7', body: '#29365f', dark: '#1c243f', light: '#b8fff2', accent: '#7bdff2', leaf: '#306f85', flower: '#d7fff7' },
  { id: 'storm-mage-trial', top: '#ffe16a', body: '#354861', dark: '#1f2a3b', light: '#d8f6ff', accent: '#7bdff2', leaf: '#6e8d84', flower: '#fff1a8' },
  { id: 'sniper-trial', top: '#c3995b', body: '#4f6073', dark: '#2e394a', light: '#ffe0a6', accent: '#ffd166', leaf: '#5e8f58', flower: '#f6e8a8' },
  { id: 'trapper-trial', top: '#3f8f58', body: '#513325', dark: '#1f2f28', light: '#81c95f', accent: '#c4475d', leaf: '#2e7a43', flower: '#e05b75' },
  { id: 'beast-archer-trial', top: '#77bf65', body: '#6b6960', dark: '#31452e', light: '#b6e37c', accent: '#69d1a6', leaf: '#4f9b58', flower: '#ffe16a' }
]);

const BOSS_ROOM_THEME_IDS = new Set(['bramble-court', 'titan-foundry', 'deepcore-core', 'ember-furnace', 'rime-vault', 'storm-aerie', 'astral-stacks', 'eclipse-throne']);

function ensureDirs() {
  [TERRAIN_DIR, PROP_DIR, STRUCTURE_DIR].forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
}

function svg(width, height, body) {
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="crispEdges">${body}</svg>`);
}

function rect(x, y, w, h, fill, extra) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}"${extra || ''}/>`;
}

function circle(cx, cy, r, fill, extra) {
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}"${extra || ''}/>`;
}

function ellipse(cx, cy, rx, ry, fill, extra) {
  return `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${fill}"${extra || ''}/>`;
}

function polygon(points, fill, extra) {
  return `<polygon points="${points}" fill="${fill}"${extra || ''}/>`;
}

function group(x, y, body) {
  return `<g transform="translate(${x} ${y})">${body}</g>`;
}

function tile(x, y, body) {
  return group(x * CELL, y * CELL, body);
}

function themeFamily(theme) {
  const id = String(theme && theme.id || '');
  if (id.includes('gearworks') || id.includes('ruins') || id.includes('titan') || id.includes('rust')) return 'mechanical';
  if (id.includes('cinder') || id.includes('ember') || id.includes('fire') || id.includes('ashglass') || id.includes('berserker')) return 'volcanic';
  if (id.includes('frost') || id.includes('rime')) return 'frost';
  if (id.includes('storm')) return 'storm';
  if (id.includes('astral') || id.includes('eclipse') || id.includes('rift') || id.includes('rune')) return 'astral';
  if (id.includes('quarry') || id.includes('deepcore')) return 'quarry';
  if (id.includes('town') || id.includes('duelist') || id.includes('sniper')) return 'town';
  return 'nature';
}

function terrainTopMotif(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') {
    return [
      rect(6, 6, 18, 4, theme.light, ' opacity="0.64"'),
      rect(34, 5, 20, 5, theme.accent, ' opacity="0.58"'),
      circle(14, 15, 2, theme.dark, ' opacity="0.65"'),
      circle(48, 15, 2, theme.dark, ' opacity="0.65"')
    ].join('');
  }
  if (family === 'volcanic') {
    return [
      polygon('6,16 18,9 28,17 42,8 58,15 58,18 6,18', theme.accent, ' opacity="0.5"'),
      rect(20, 5, 13, 3, theme.light, ' opacity="0.58"'),
      rect(44, 4, 8, 3, theme.light, ' opacity="0.48"')
    ].join('');
  }
  if (family === 'frost') {
    return [
      polygon('6,16 16,4 24,16', theme.light, ' opacity="0.65"'),
      polygon('38,16 49,2 59,16', theme.accent, ' opacity="0.52"'),
      rect(0, 0, 64, 3, '#ffffff', ' opacity="0.72"')
    ].join('');
  }
  if (family === 'storm') {
    return [
      polygon('10,17 22,5 18,14 34,8 24,18', theme.accent, ' opacity="0.64"'),
      rect(42, 5, 12, 4, theme.light, ' opacity="0.5"')
    ].join('');
  }
  if (family === 'astral') {
    return [
      ellipse(18, 10, 14, 5, theme.accent, ' opacity="0.42"'),
      rect(10, 9, 17, 2, theme.light, ' opacity="0.72"'),
      rect(34, 7, 4, 10, theme.accent, ' opacity="0.58"'),
      rect(44, 11, 12, 2, theme.light, ' opacity="0.58"')
    ].join('');
  }
  if (family === 'quarry') {
    return [
      polygon('8,17 18,8 26,17', theme.accent, ' opacity="0.38"'),
      rect(34, 6, 20, 4, theme.light, ' opacity="0.48"')
    ].join('');
  }
  return [
    polygon('8,18 14,5 18,18', theme.leaf, ' opacity="0.82"'),
    polygon('22,18 28,8 32,18', theme.light, ' opacity="0.62"'),
    circle(48, 10, 3, theme.flower, ' opacity="0.82"')
  ].join('');
}

function terrainBodyMotif(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') {
    return [
      rect(6, 12, 20, 7, theme.dark, ' opacity="0.3"'),
      rect(32, 32, 26, 5, theme.accent, ' opacity="0.34"'),
      circle(14, 46, 3, theme.light, ' opacity="0.46"'),
      circle(50, 14, 2, theme.light, ' opacity="0.5"')
    ].join('');
  }
  if (family === 'volcanic') {
    return [
      polygon('8,56 24,24 34,56', theme.accent, ' opacity="0.26"'),
      polygon('38,8 50,42 57,8', theme.light, ' opacity="0.18"'),
      rect(6, 44, 20, 5, theme.dark, ' opacity="0.34"')
    ].join('');
  }
  if (family === 'frost') {
    return [
      polygon('12,54 20,28 30,54', theme.light, ' opacity="0.28"'),
      polygon('42,10 51,36 58,10', theme.accent, ' opacity="0.22"'),
      rect(8, 18, 28, 5, '#ffffff', ' opacity="0.32"')
    ].join('');
  }
  if (family === 'storm') {
    return [
      polygon('16,52 28,24 22,38 42,20 30,52', theme.accent, ' opacity="0.28"'),
      rect(44, 42, 14, 4, theme.light, ' opacity="0.34"')
    ].join('');
  }
  if (family === 'astral') {
    return [
      ellipse(32, 30, 24, 8, theme.accent, ' opacity="0.22"'),
      rect(15, 28, 34, 4, theme.light, ' opacity="0.38"'),
      rect(29, 12, 4, 34, theme.accent, ' opacity="0.28"')
    ].join('');
  }
  if (family === 'quarry') {
    return [
      polygon('10,54 22,18 32,54', theme.accent, ' opacity="0.22"'),
      rect(38, 20, 18, 6, theme.light, ' opacity="0.3"')
    ].join('');
  }
  return [
    rect(8, 12, 22, 8, theme.dark, ' opacity="0.24"'),
    rect(42, 8, 12, 7, theme.light, ' opacity="0.55"'),
    rect(20, 36, 30, 10, theme.dark, ' opacity="0.28"')
  ].join('');
}

function terrainUndersideMotif(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') return rect(10, 12, 44, 5, theme.accent, ' opacity="0.48"') + rect(18, 18, 6, 12, theme.dark, ' opacity="0.5"') + rect(42, 18, 6, 10, theme.dark, ' opacity="0.5"');
  if (family === 'volcanic') return polygon('10,18 18,34 26,18', theme.accent, ' opacity="0.55"') + polygon('42,18 50,30 56,18', theme.light, ' opacity="0.42"');
  if (family === 'frost') return polygon('8,16 16,36 24,16', theme.light, ' opacity="0.55"') + polygon('40,16 48,32 56,16', '#ffffff', ' opacity="0.5"');
  if (family === 'storm') return polygon('16,18 28,34 24,24 40,36 30,18', theme.accent, ' opacity="0.44"');
  if (family === 'astral') return ellipse(32, 20, 22, 6, theme.accent, ' opacity="0.3"') + rect(18, 19, 28, 2, theme.light, ' opacity="0.55"');
  return rect(16, 18, 6, 16, theme.leaf, ' opacity="0.56"') + rect(42, 18, 5, 12, theme.leaf, ' opacity="0.46"');
}

function terrainDetailMotif(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') return rect(4, 16, 56, 10, theme.dark, ' opacity="0.36"') + circle(16, 21, 4, theme.accent, ' opacity="0.72"') + circle(48, 21, 4, theme.light, ' opacity="0.6"');
  if (family === 'volcanic') return polygon('4,30 18,12 30,30', theme.accent, ' opacity="0.7"') + polygon('28,30 42,8 58,30', theme.light, ' opacity="0.42"');
  if (family === 'frost') return polygon('8,36 24,6 36,36', theme.light, ' opacity="0.62"') + polygon('32,38 46,14 56,38', theme.accent, ' opacity="0.44"');
  if (family === 'storm') return polygon('22,8 8,34 24,28 16,54 46,20 30,24', theme.accent, ' opacity="0.62"');
  if (family === 'astral') return ellipse(32, 32, 28, 11, theme.accent, ' opacity="0.38"') + rect(12, 30, 40, 4, theme.light, ' opacity="0.58"') + rect(30, 14, 4, 36, theme.light, ' opacity="0.36"');
  if (family === 'quarry') return polygon('10,46 24,12 38,46', theme.accent, ' opacity="0.46"') + polygon('34,46 48,22 58,46', theme.light, ' opacity="0.36"');
  return rect(6, 38, 52, 7, theme.leaf, ' opacity="0.5"') + circle(18, 32, 4, theme.flower, ' opacity="0.8"') + circle(42, 30, 4, theme.accent, ' opacity="0.68"');
}

function terrainTile(theme, kind) {
  if (kind === 'top') {
    return [
      rect(0, 0, 64, 18, theme.top),
      rect(0, 18, 64, 46, theme.body),
      rect(0, 0, 64, 5, theme.light),
      rect(0, 14, 64, 5, theme.dark, ' opacity="0.35"'),
      terrainTopMotif(theme),
      rect(6, 30, 14, 8, theme.dark, ' opacity="0.35"'),
      rect(34, 42, 20, 7, theme.dark, ' opacity="0.28"')
    ].join('');
  }
  if (kind === 'body') {
    return [
      rect(0, 0, 64, 64, theme.body),
      rect(0, 0, 64, 5, theme.dark, ' opacity="0.3"'),
      terrainBodyMotif(theme),
      rect(6, 54, 18, 5, theme.light, ' opacity="0.38"')
    ].join('');
  }
  if (kind === 'left') {
    return [
      rect(16, 0, 48, 64, theme.body),
      rect(16, 0, 48, 18, theme.top),
      polygon('16,0 0,12 0,64 16,64', theme.dark),
      rect(16, 0, 48, 5, theme.light),
      rect(6, 20, 10, 8, theme.light, ' opacity="0.4"')
    ].join('');
  }
  if (kind === 'right') {
    return [
      rect(0, 0, 48, 64, theme.body),
      rect(0, 0, 48, 18, theme.top),
      polygon('48,0 64,12 64,64 48,64', theme.dark),
      rect(0, 0, 48, 5, theme.light),
      rect(48, 24, 10, 8, theme.light, ' opacity="0.4"')
    ].join('');
  }
  if (kind === 'underside') {
    return [
      rect(0, 0, 64, 18, theme.dark),
      rect(0, 18, 64, 46, 'transparent'),
      rect(8, 4, 14, 6, theme.body, ' opacity="0.7"'),
      rect(34, 6, 20, 5, theme.body, ' opacity="0.65"'),
      terrainUndersideMotif(theme)
    ].join('');
  }
  if (kind === 'topAlt') {
    return [
      rect(0, 0, 64, 18, theme.top),
      rect(0, 18, 64, 46, theme.body),
      rect(0, 0, 64, 5, theme.light),
      rect(4, 2, 8, 12, theme.leaf, ' opacity="0.85"'),
      rect(15, 5, 5, 9, theme.leaf, ' opacity="0.85"'),
      rect(45, 3, 11, 10, theme.accent, ' opacity="0.7"'),
      rect(25, 38, 22, 7, theme.dark, ' opacity="0.28"')
    ].join('');
  }
  if (kind === 'cap') {
    return [
      rect(0, 8, 64, 46, theme.body),
      rect(4, 0, 56, 12, theme.top),
      rect(4, 0, 56, 4, theme.light),
      rect(0, 16, 10, 38, theme.dark),
      rect(54, 16, 10, 38, theme.dark),
      rect(20, 28, 24, 9, theme.light, ' opacity="0.45"')
    ].join('');
  }
  return [
    rect(0, 0, 64, 64, 'transparent'),
    rect(0, 0, 64, 9, theme.light, ' opacity="0.65"'),
    terrainDetailMotif(theme)
  ].join('');
}

function themedSmallProp(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') {
    return [
      ellipse(32, 55, 22, 5, theme.dark, ' opacity="0.28"'),
      rect(18, 34, 28, 21, theme.body),
      rect(22, 28, 20, 8, theme.top),
      circle(32, 44, 8, theme.accent, ' opacity="0.82"'),
      rect(29, 36, 6, 16, theme.light, ' opacity="0.42"')
    ].join('');
  }
  if (family === 'volcanic') {
    return [
      ellipse(32, 55, 22, 5, theme.dark, ' opacity="0.3"'),
      polygon('18,54 26,30 38,30 48,54', theme.body),
      polygon('24,52 32,34 41,52', theme.accent, ' opacity="0.58"'),
      polygon('30,50 35,40 40,50', theme.light, ' opacity="0.42"')
    ].join('');
  }
  if (family === 'frost') {
    return [
      ellipse(32, 55, 22, 5, theme.dark, ' opacity="0.18"'),
      polygon('18,55 30,26 42,55', theme.light, ' opacity="0.72"'),
      polygon('34,55 45,34 54,55', theme.accent, ' opacity="0.48"'),
      rect(16, 52, 42, 4, '#ffffff', ' opacity="0.62"')
    ].join('');
  }
  if (family === 'storm') {
    return [
      ellipse(32, 55, 22, 5, theme.dark, ' opacity="0.22"'),
      rect(24, 28, 16, 28, theme.body),
      polygon('32,16 20,38 33,34 25,54 48,26 36,31', theme.accent, ' opacity="0.6"')
    ].join('');
  }
  if (family === 'astral') {
    return [
      ellipse(32, 55, 24, 6, theme.accent, ' opacity="0.2"'),
      rect(20, 38, 24, 16, theme.body),
      ellipse(32, 36, 25, 8, theme.accent, ' opacity="0.44"'),
      rect(22, 34, 20, 3, theme.light, ' opacity="0.72"')
    ].join('');
  }
  return [
    ellipse(32, 54, 22, 6, theme.dark, ' opacity="0.25"'),
    rect(20, 38, 24, 18, theme.body),
    rect(17, 33, 30, 8, theme.top),
    rect(21, 29, 7, 6, theme.light),
    rect(38, 28, 7, 7, theme.accent)
  ].join('');
}

function themedTallProp(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') {
    return [
      rect(28, 16, 8, 42, theme.body),
      rect(22, 24, 20, 8, theme.dark, ' opacity="0.5"'),
      circle(32, 14, 12, theme.accent, ' opacity="0.82"'),
      circle(32, 14, 5, theme.light),
      rect(18, 49, 28, 8, theme.dark, ' opacity="0.34"')
    ].join('');
  }
  if (family === 'volcanic') {
    return [
      rect(29, 22, 7, 36, theme.body),
      polygon('20,42 32,8 44,42', theme.accent, ' opacity="0.42"'),
      polygon('26,38 34,18 40,38', theme.light, ' opacity="0.36"'),
      rect(18, 50, 28, 8, theme.dark, ' opacity="0.3"')
    ].join('');
  }
  if (family === 'frost') {
    return [
      polygon('28,58 32,8 38,58', theme.light, ' opacity="0.72"'),
      polygon('14,58 25,26 34,58', theme.accent, ' opacity="0.44"'),
      polygon('38,58 48,30 55,58', '#ffffff', ' opacity="0.5"')
    ].join('');
  }
  if (family === 'storm') {
    return [
      rect(28, 20, 8, 38, theme.body),
      polygon('32,8 20,34 34,29 26,54 50,22 38,26', theme.accent, ' opacity="0.55"'),
      circle(32, 16, 5, theme.light, ' opacity="0.7"')
    ].join('');
  }
  if (family === 'astral') {
    return [
      rect(29, 20, 7, 38, theme.body),
      ellipse(32, 17, 23, 10, theme.accent, ' opacity="0.42"'),
      rect(20, 15, 24, 4, theme.light, ' opacity="0.64"'),
      rect(30, 5, 4, 28, theme.light, ' opacity="0.38"')
    ].join('');
  }
  return [
    rect(29, 18, 7, 40, theme.body),
    rect(23, 22, 20, 8, theme.dark, ' opacity="0.42"'),
    circle(32, 14, 11, theme.accent),
    circle(32, 14, 5, theme.light),
    rect(17, 48, 30, 8, theme.dark, ' opacity="0.28"')
  ].join('');
}

function themedSignProp(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') {
    return [
      rect(29, 27, 6, 31, theme.body),
      rect(13, 17, 38, 20, theme.dark),
      rect(16, 20, 32, 4, theme.light, ' opacity="0.58"'),
      circle(23, 30, 3, theme.accent),
      circle(42, 30, 3, theme.accent)
    ].join('');
  }
  if (family === 'volcanic') {
    return [
      rect(29, 28, 6, 30, theme.body),
      polygon('14,20 50,14 46,38 18,36', theme.dark),
      rect(19, 24, 24, 4, theme.accent, ' opacity="0.68"'),
      rect(20, 31, 20, 3, theme.light, ' opacity="0.38"')
    ].join('');
  }
  if (family === 'frost') {
    return [
      rect(29, 28, 6, 30, theme.body),
      polygon('14,18 50,18 44,38 20,38', theme.light, ' opacity="0.82"'),
      rect(19, 23, 26, 3, theme.accent, ' opacity="0.58"')
    ].join('');
  }
  if (family === 'astral') {
    return [
      rect(29, 28, 6, 30, theme.body),
      ellipse(32, 26, 24, 13, theme.accent, ' opacity="0.46"'),
      rect(18, 25, 28, 3, theme.light, ' opacity="0.68"'),
      rect(31, 17, 3, 18, theme.light, ' opacity="0.42"')
    ].join('');
  }
  return [
    rect(29, 28, 6, 30, theme.body),
    rect(14, 18, 36, 18, theme.top),
    rect(17, 21, 30, 3, theme.light),
    rect(18, 29, 26, 3, theme.dark, ' opacity="0.35"')
  ].join('');
}

function themedGlowProp(theme) {
  const family = themeFamily(theme);
  if (family === 'mechanical') {
    return [
      ellipse(32, 54, 22, 7, theme.accent, ' opacity="0.22"'),
      circle(32, 38, 13, theme.accent, ' opacity="0.7"'),
      circle(32, 38, 6, theme.light, ' opacity="0.85"'),
      rect(18, 50, 28, 5, theme.dark, ' opacity="0.35"')
    ].join('');
  }
  if (family === 'volcanic') {
    return [
      ellipse(32, 54, 24, 7, theme.accent, ' opacity="0.28"'),
      polygon('18,52 30,28 42,52', theme.accent, ' opacity="0.72"'),
      polygon('27,52 34,38 40,52', theme.light, ' opacity="0.64"')
    ].join('');
  }
  if (family === 'frost') {
    return [
      ellipse(32, 54, 22, 7, theme.accent, ' opacity="0.2"'),
      polygon('22,52 32,22 44,52', theme.light, ' opacity="0.68"'),
      polygon('30,52 38,34 49,52', '#ffffff', ' opacity="0.45"')
    ].join('');
  }
  if (family === 'storm') {
    return [
      ellipse(32, 54, 22, 7, theme.accent, ' opacity="0.24"'),
      polygon('30,20 18,46 32,40 24,58 50,30 36,34', theme.accent, ' opacity="0.68"'),
      circle(34, 28, 5, theme.light, ' opacity="0.7"')
    ].join('');
  }
  if (family === 'astral') {
    return [
      ellipse(32, 54, 24, 7, theme.accent, ' opacity="0.24"'),
      ellipse(32, 38, 24, 10, theme.accent, ' opacity="0.52"'),
      rect(17, 36, 30, 4, theme.light, ' opacity="0.68"'),
      rect(30, 22, 4, 30, theme.light, ' opacity="0.36"')
    ].join('');
  }
  return [
    ellipse(32, 54, 22, 7, theme.accent, ' opacity="0.24"'),
    circle(22, 43, 10, theme.accent),
    circle(37, 35, 14, theme.light),
    circle(48, 47, 8, theme.flower),
    rect(19, 52, 31, 5, theme.dark, ' opacity="0.35"')
  ].join('');
}

function propTile(theme, kind) {
  if (kind === 'grass') {
    return [
      rect(5, 50, 54, 8, theme.dark, ' opacity="0.28"'),
      polygon('9,54 14,34 19,54', theme.leaf),
      polygon('19,54 26,28 31,54', theme.top),
      polygon('31,54 39,32 44,54', theme.leaf),
      polygon('43,54 50,38 55,54', theme.light)
    ].join('');
  }
  if (kind === 'bush') {
    return [
      ellipse(31, 51, 26, 7, theme.dark, ' opacity="0.28"'),
      circle(18, 39, 14, theme.leaf),
      circle(32, 34, 18, theme.top),
      circle(46, 42, 13, theme.leaf),
      circle(28, 24, 9, theme.light, ' opacity="0.8"'),
      circle(43, 33, 5, theme.flower)
    ].join('');
  }
  if (kind === 'tree') {
    return [
      rect(28, 28, 9, 30, theme.body),
      rect(32, 31, 4, 26, theme.dark, ' opacity="0.35"'),
      circle(22, 24, 15, theme.leaf),
      circle(38, 18, 18, theme.top),
      circle(46, 34, 14, theme.leaf),
      circle(31, 8, 11, theme.light, ' opacity="0.72"')
    ].join('');
  }
  if (kind === 'rock') {
    return [
      ellipse(32, 54, 23, 6, theme.dark, ' opacity="0.28"'),
      polygon('12,52 20,33 44,28 55,51', theme.body),
      polygon('20,33 32,20 44,28', theme.light),
      polygon('35,34 55,51 34,51', theme.dark, ' opacity="0.42"')
    ].join('');
  }
  if (kind === 'flower') {
    return [
      rect(30, 36, 4, 20, theme.leaf),
      polygon('28,45 16,38 30,50', theme.leaf),
      polygon('35,43 51,36 36,50', theme.leaf),
      circle(32, 32, 7, theme.flower),
      circle(26, 30, 4, theme.accent),
      circle(39, 29, 4, theme.light)
    ].join('');
  }
  if (kind === 'small') {
    return themedSmallProp(theme);
  }
  if (kind === 'tall') {
    return themedTallProp(theme);
  }
  if (kind === 'crate') {
    return [
      rect(15, 30, 34, 27, theme.body),
      rect(15, 30, 34, 5, theme.light),
      rect(19, 34, 5, 23, theme.dark, ' opacity="0.45"'),
      rect(40, 34, 5, 23, theme.dark, ' opacity="0.45"'),
      polygon('17,55 47,32 49,38 22,57', theme.dark, ' opacity="0.42"')
    ].join('');
  }
  if (kind === 'crystal') {
    return [
      ellipse(32, 55, 20, 5, theme.accent, ' opacity="0.22"'),
      polygon('23,54 32,12 42,54', theme.accent),
      polygon('32,12 42,54 35,49', theme.light, ' opacity="0.55"'),
      polygon('12,56 22,28 30,56', theme.top),
      polygon('40,56 48,30 55,56', theme.flower)
    ].join('');
  }
  if (kind === 'vine') {
    return [
      rect(29, 4, 5, 56, theme.body),
      polygon('29,22 14,16 29,30', theme.leaf),
      polygon('34,35 52,30 34,44', theme.leaf),
      circle(21, 15, 4, theme.flower),
      circle(48, 29, 4, theme.accent)
    ].join('');
  }
  if (kind === 'sign') {
    return themedSignProp(theme);
  }
  if (kind === 'glow') {
    return themedGlowProp(theme);
  }
  return themedGlowProp(theme);
}

function terrainSvg(theme) {
  const cells = ['top', 'body', 'left', 'right', 'underside', 'topAlt', 'cap', 'detail'];
  return svg(CELL * 4, CELL * 2, cells.map((kind, index) => tile(index % 4, Math.floor(index / 4), terrainTile(theme, kind))).join(''));
}

function propSvg(theme) {
  const cells = ['grass', 'bush', 'tree', 'rock', 'flower', 'small', 'tall', 'crate', 'crystal', 'vine', 'sign', 'glow'];
  const tallKinds = new Set(['tree', 'tall', 'vine', 'crystal', 'sign']);
  const lowKinds = new Set(['grass', 'flower', 'rock', 'glow']);
  return svg(CELL * 6, CELL * 2, cells.map((kind, index) => {
    const scale = tallKinds.has(kind) ? 0.82 : lowKinds.has(kind) ? 0.72 : 0.76;
    const yAnchor = lowKinds.has(kind) ? 58 : 56;
    const body = `<g transform="translate(32 ${yAnchor}) scale(${scale}) translate(-32 -${yAnchor})">${propTile(theme, kind)}</g>`;
    return tile(index % 6, Math.floor(index / 6), body);
  }).join(''));
}

function structureTile(x, y, body) {
  return group(x * STRUCTURE_CELL, y * STRUCTURE_CELL, body);
}

function structureBuilding(config) {
  const colors = config || {};
  const roof = colors.roof || '#8b6a46';
  const wall = colors.wall || '#d9a85c';
  const dark = colors.dark || '#5a4432';
  const light = colors.light || '#f7d28a';
  const accent = colors.accent || '#7ec8d8';
  const window = colors.window || accent;
  return [
    ellipse(128, 234, 100, 12, dark, ' opacity="0.22"'),
    rect(42, 96, 172, 116, wall),
    rect(54, 108, 148, 92, light, ' opacity="0.12"'),
    polygon('30,102 128,38 226,102', roof),
    rect(52, 92, 152, 14, dark, ' opacity="0.42"'),
    rect(88, 142, 34, 58, dark),
    rect(96, 150, 18, 38, window, ' opacity="0.72"'),
    rect(140, 138, 44, 38, dark, ' opacity="0.72"'),
    rect(148, 146, 28, 22, window, ' opacity="0.78"'),
    rect(72, 116, 38, 26, dark, ' opacity="0.68"'),
    rect(80, 123, 22, 12, window, ' opacity="0.7"'),
    rect(122, 58, 12, 36, accent, ' opacity="0.74"'),
    rect(54, 204, 150, 10, dark, ' opacity="0.5"')
  ].join('');
}

function structureWorkshop() {
  return [
    ellipse(128, 236, 105, 12, '#303842', ' opacity="0.24"'),
    rect(36, 104, 184, 104, '#565d65'),
    rect(46, 92, 164, 18, '#d8b74a'),
    polygon('28,104 74,56 218,88 218,106', '#8c6b35'),
    rect(64, 128, 42, 48, '#303842'),
    circle(85, 150, 18, '#29b3ad', ' opacity="0.68"'),
    rect(132, 122, 48, 24, '#303842', ' opacity="0.72"'),
    rect(140, 128, 32, 10, '#d8b74a', ' opacity="0.75"'),
    rect(184, 74, 16, 38, '#303842'),
    circle(192, 72, 14, '#29b3ad', ' opacity="0.52"'),
    rect(48, 204, 164, 9, '#303842', ' opacity="0.55"')
  ].join('');
}

function structureForge() {
  return [
    ellipse(128, 236, 108, 13, '#201c22', ' opacity="0.28"'),
    rect(44, 104, 168, 104, '#3b2c32'),
    polygon('32,106 88,54 222,104', '#9b4835'),
    rect(68, 130, 54, 68, '#201c22'),
    polygon('78,194 94,150 114,194', '#ff7842', ' opacity="0.76"'),
    polygon('90,194 101,164 114,194', '#ffd166', ' opacity="0.66"'),
    rect(146, 126, 40, 32, '#201c22'),
    rect(154, 134, 24, 14, '#ff7842', ' opacity="0.68"'),
    rect(170, 58, 20, 50, '#201c22'),
    rect(52, 204, 152, 10, '#201c22', ' opacity="0.56"')
  ].join('');
}

function structureLodge() {
  return [
    ellipse(128, 236, 106, 12, '#2f5878', ' opacity="0.18"'),
    rect(42, 104, 172, 104, '#5ca8e8'),
    rect(52, 112, 152, 84, '#d7f3ff', ' opacity="0.18"'),
    polygon('28,104 128,48 228,104', '#d7f3ff'),
    rect(44, 98, 168, 10, '#ffffff', ' opacity="0.72"'),
    rect(92, 140, 42, 58, '#223d5a'),
    rect(146, 132, 42, 34, '#223d5a', ' opacity="0.68"'),
    rect(154, 140, 26, 16, '#ecfbff', ' opacity="0.78"'),
    polygon('58,210 72,184 86,210', '#ecfbff', ' opacity="0.7"'),
    polygon('188,210 202,180 216,210', '#8fd7ff', ' opacity="0.58"')
  ].join('');
}

function structureGate() {
  return [
    ellipse(128, 236, 112, 12, '#26364f', ' opacity="0.22"'),
    rect(48, 92, 30, 122, '#4f6073'),
    rect(178, 92, 30, 122, '#4f6073'),
    rect(42, 78, 42, 18, '#ffe16a'),
    rect(172, 78, 42, 18, '#ffe16a'),
    polygon('36,80 64,38 92,80', '#91dbe8'),
    polygon('164,80 192,38 220,80', '#91dbe8'),
    rect(76, 116, 104, 20, '#26364f'),
    polygon('128,88 110,126 134,118 122,168 158,106 136,114', '#7bdff2', ' opacity="0.76"'),
    rect(88, 190, 80, 16, '#ffe16a', ' opacity="0.42"')
  ].join('');
}

function structureObservatory() {
  return [
    ellipse(128, 236, 108, 12, '#1c243f', ' opacity="0.22"'),
    rect(72, 118, 112, 90, '#29365f'),
    rect(88, 90, 80, 34, '#1c243f'),
    ellipse(128, 86, 68, 28, '#28c7b7', ' opacity="0.42"'),
    rect(94, 84, 68, 5, '#7bdff2', ' opacity="0.74"'),
    rect(124, 38, 8, 70, '#7bdff2', ' opacity="0.58"'),
    ellipse(128, 42, 34, 12, '#c794ff', ' opacity="0.48"'),
    rect(100, 148, 56, 34, '#1c243f'),
    rect(110, 156, 36, 16, '#7bdff2', ' opacity="0.78"'),
    rect(66, 204, 124, 9, '#1c243f', ' opacity="0.56"')
  ].join('');
}

function structureAwning() {
  return [
    ellipse(128, 226, 96, 10, '#5a4432', ' opacity="0.2"'),
    rect(48, 102, 160, 100, '#8b6a46'),
    rect(58, 118, 140, 72, '#d9a85c', ' opacity="0.46"'),
    polygon('38,102 62,68 194,68 218,102', '#f7d28a'),
    rect(54, 96, 28, 18, '#7ec8d8', ' opacity="0.72"'),
    rect(88, 96, 28, 18, '#f0c36a', ' opacity="0.76"'),
    rect(122, 96, 28, 18, '#7ec8d8', ' opacity="0.72"'),
    rect(156, 96, 28, 18, '#f0c36a', ' opacity="0.76"'),
    rect(76, 150, 48, 44, '#5a4432'),
    rect(142, 138, 34, 28, '#5a4432', ' opacity="0.72"'),
    rect(148, 145, 22, 12, '#7ec8d8', ' opacity="0.76"')
  ].join('');
}

function structureLanternArch() {
  return [
    ellipse(128, 232, 76, 9, '#5a4432', ' opacity="0.18"'),
    rect(62, 88, 18, 128, '#8b6a46'),
    rect(176, 88, 18, 128, '#8b6a46'),
    rect(72, 82, 112, 16, '#5a4432'),
    polygon('78,82 128,46 178,82', '#d9a85c'),
    rect(116, 100, 24, 36, '#7ec8d8', ' opacity="0.42"'),
    circle(128, 148, 18, '#f7d28a', ' opacity="0.68"'),
    circle(128, 148, 8, '#ffe16a', ' opacity="0.9"'),
    rect(66, 206, 124, 10, '#5a4432', ' opacity="0.46"')
  ].join('');
}

function structureSvg() {
  const cells = [
    structureBuilding({ roof: '#8b6a46', wall: '#d9a85c', dark: '#5a4432', light: '#f7d28a', accent: '#7ec8d8' }),
    structureWorkshop(),
    structureForge(),
    structureLodge(),
    structureGate(),
    structureObservatory(),
    structureAwning(),
    structureLanternArch()
  ];
  return svg(STRUCTURE_CELL * 4, STRUCTURE_CELL * 2, cells.map((body, index) => structureTile(index % 4, Math.floor(index / 4), body)).join(''));
}

async function writeTheme(theme) {
  await sharp(terrainSvg(theme)).png().toFile(path.join(TERRAIN_DIR, `${theme.id}.png`));
  await sharp(propSvg(theme)).png().toFile(path.join(PROP_DIR, `${theme.id}.png`));
}

async function writeStructureAtlas() {
  await sharp(structureSvg()).png().toFile(path.join(STRUCTURE_DIR, 'town-landmarks.png'));
}

async function main() {
  ensureDirs();
  const onlyIndex = process.argv.indexOf('--only');
  const onlyTarget = onlyIndex >= 0 ? String(process.argv[onlyIndex + 1] || '').trim().toLowerCase() : '';
  if (onlyTarget && !['trials', 'boss-rooms', 'structures'].includes(onlyTarget)) throw new Error(`Unsupported --only target: ${onlyTarget}`);
  if (onlyTarget === 'structures') {
    await writeStructureAtlas();
    console.log('[project-starfall] generated town structure atlas');
    return;
  }
  const themes = onlyTarget === 'trials'
    ? THEMES.filter((theme) => theme.id.endsWith('-trial'))
    : onlyTarget === 'boss-rooms'
      ? THEMES.filter((theme) => BOSS_ROOM_THEME_IDS.has(theme.id))
      : THEMES;
  for (const theme of themes) {
    await writeTheme(theme);
  }
  if (!onlyTarget) await writeStructureAtlas();
  console.log(`[project-starfall] generated ${themes.length} terrain atlases and ${themes.length} prop atlases${onlyTarget ? '' : ' plus town structures'}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
