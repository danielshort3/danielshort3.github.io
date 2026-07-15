#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const { processSkillFxSheets } = require('./process-project-starfall-ai-skill-fx.js');

const DEFAULT_FRAME_SIZE = 160;
const DEFAULT_FRAMES = 6;
const SKILL_FRAME_GUTTER = 8;
const SKILL_CONTENT_SCALE = 0.82;
const SKILL_CONTENT_OFFSET = DEFAULT_FRAME_SIZE * (1 - SKILL_CONTENT_SCALE) / 2;
const SKILL_SOURCE_DIR = path.join(ROOT, 'img/project-starfall/animations/combat-fx/skills/source');
const SKILL_ACTION_PROFILES = Object.freeze([
  'melee',
  'mobility',
  'guard',
  'buff',
  'projectile',
  'trap',
  'area'
]);

const CLASS_PALETTES = Object.freeze({
  fighter: Object.freeze({ color: '#f25f4c', accent: '#ffd166' }),
  mage: Object.freeze({ color: '#4f8cff', accent: '#e7fbff' }),
  archer: Object.freeze({ color: '#3aa76d', accent: '#fff0a6' }),
  guardian: Object.freeze({ color: '#68a9ff', accent: '#d5ecff' }),
  berserker: Object.freeze({ color: '#ef3d55', accent: '#ffbe55' }),
  duelist: Object.freeze({ color: '#f0c36a', accent: '#ffffff' }),
  fireMage: Object.freeze({ color: '#ff7b3a', accent: '#ffe16a' }),
  runeMage: Object.freeze({ color: '#28c7b7', accent: '#b8fff2' }),
  stormMage: Object.freeze({ color: '#7bdff2', accent: '#ffffff' }),
  sniper: Object.freeze({ color: '#d8c25f', accent: '#ffffff' }),
  trapper: Object.freeze({ color: '#66d79a', accent: '#dbffe6' }),
  beastArcher: Object.freeze({ color: '#8ed174', accent: '#fff0a6' })
});

const ENEMY_PALETTES = Object.freeze({
  plant: Object.freeze({ color: '#4fb46a', accent: '#d8ff9b' }),
  beast: Object.freeze({ color: '#d58a42', accent: '#fff0a6' }),
  construct: Object.freeze({ color: '#8f98a3', accent: '#f3d86d' }),
  volcanic: Object.freeze({ color: '#ff6b35', accent: '#ffe16a' }),
  frost: Object.freeze({ color: '#7bdff2', accent: '#ffffff' }),
  storm: Object.freeze({ color: '#8ab4ff', accent: '#ffffff' }),
  astral: Object.freeze({ color: '#a98bff', accent: '#ffffff' }),
  void: Object.freeze({ color: '#7b5cff', accent: '#ff8ef8' }),
  humanoid: Object.freeze({ color: '#d69a64', accent: '#ffe4b8' }),
  ooze: Object.freeze({ color: '#66d79a', accent: '#dbffe6' }),
  default: Object.freeze({ color: '#d8e5ec', accent: '#ffffff' })
});

function getOnlyMode(args) {
  const equalsArg = args.find((arg) => String(arg || '').startsWith('--only='));
  if (equalsArg) return equalsArg.split('=').slice(1).join('=').trim() || 'all';
  const onlyIndex = args.indexOf('--only');
  if (onlyIndex >= 0) return String(args[onlyIndex + 1] || '').trim() || 'all';
  return 'all';
}

function getArgValue(args, flag) {
  const equalsArg = args.find((arg) => String(arg || '').startsWith(`${flag}=`));
  if (equalsArg) return equalsArg.split('=').slice(1).join('=').trim();
  const index = args.indexOf(flag);
  return index >= 0 ? String(args[index + 1] || '').trim() : '';
}

function normalizeId(value) {
  return String(value || '')
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-z0-9-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
}

function normalizeMode(mode) {
  const value = String(mode || '').trim().toLowerCase();
  if (!value || value === 'all' || value === '*') return 'all';
  if (value === 'skill') return 'skills';
  if (value === 'enemy') return 'enemies';
  return value;
}

function usage() {
  return [
    'Usage: node build/generate-project-starfall-combat-fx.js --only all [--skill <skill-id>]',
    'Modes: --only all, --only skills, --only basic, --only enemies',
    'Dedicated skill sources take priority; remaining skills use deterministic semantic profiles.'
  ].join('\n');
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function escapeXml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function hashString(value) {
  const text = String(value || '');
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hexToRgb(hex) {
  const raw = String(hex || '').replace('#', '').trim();
  const normalized = raw.length === 3
    ? raw.split('').map((part) => part + part).join('')
    : raw.padEnd(6, '0').slice(0, 6);
  return {
    r: parseInt(normalized.slice(0, 2), 16) || 0,
    g: parseInt(normalized.slice(2, 4), 16) || 0,
    b: parseInt(normalized.slice(4, 6), 16) || 0
  };
}

function rgbToHex(color) {
  return `#${[color.r, color.g, color.b].map((value) => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0')).join('')}`;
}

function mixHex(a, b, amount) {
  const colorA = hexToRgb(a);
  const colorB = hexToRgb(b);
  const t = Math.max(0, Math.min(1, Number(amount || 0)));
  return rgbToHex({
    r: colorA.r + (colorB.r - colorA.r) * t,
    g: colorA.g + (colorB.g - colorA.g) * t,
    b: colorA.b + (colorB.b - colorA.b) * t
  });
}

function shape(tag, attributes) {
  const attrs = Object.entries(attributes || {})
    .filter(([, value]) => value !== null && value !== undefined && value !== false)
    .map(([key, value]) => `${key}="${escapeXml(value)}"`)
    .join(' ');
  return `<${tag}${attrs ? ` ${attrs}` : ''}/>`;
}

function polygon(points, attributes) {
  return shape('polygon', Object.assign({}, attributes || {}, {
    points: points.map((point) => point.join(',')).join(' ')
  }));
}

function pathShape(d, attributes) {
  return shape('path', Object.assign({}, attributes || {}, { d }));
}

function enemyPalette(enemy) {
  const family = String(enemy && enemy.family || '').toLowerCase();
  const id = String(enemy && enemy.id || '').toLowerCase();
  if (family.includes('plant') || id.includes('briar') || id.includes('thorn') || id.includes('vine')) return ENEMY_PALETTES.plant;
  if (family.includes('beast') || id.includes('boar') || id.includes('stag') || id.includes('ram') || id.includes('roc')) return ENEMY_PALETTES.beast;
  if (family.includes('construct') || id.includes('clock') || id.includes('coil') || id.includes('titan')) return ENEMY_PALETTES.construct;
  if (family.includes('volcanic') || id.includes('ember') || id.includes('cinder') || id.includes('lava')) return ENEMY_PALETTES.volcanic;
  if (family.includes('frost') || id.includes('rime') || id.includes('ice') || id.includes('glacier') || id.includes('snow')) return ENEMY_PALETTES.frost;
  if (family.includes('storm') || id.includes('storm') || id.includes('thunder') || id.includes('gale')) return ENEMY_PALETTES.storm;
  if (family.includes('astral') || id.includes('astral') || id.includes('eclipse')) return ENEMY_PALETTES.astral;
  if (family.includes('void') || id.includes('void') || id.includes('rift')) return ENEMY_PALETTES.void;
  if (family.includes('humanoid') || id.includes('bandit') || id.includes('duelist')) return ENEMY_PALETTES.humanoid;
  if (family.includes('ooze') || id.includes('slime')) return ENEMY_PALETTES.ooze;
  return ENEMY_PALETTES.default;
}

function skillPalette(skill, identity) {
  const visual = skill && skill.visualId && Data.SKILL_VISUALS && Data.SKILL_VISUALS[skill.visualId];
  const classPalette = CLASS_PALETTES[skill && skill.owner] || CLASS_PALETTES.fighter;
  const baseColor = visual && visual.color || classPalette.color;
  const baseAccent = visual && visual.accent || classPalette.accent;
  const seed = Number(identity && identity.seed || 0);
  const colorMix = 0.025 + (seed % 7) * 0.008;
  const accentMix = 0.02 + ((seed >>> 5) % 6) * 0.007;
  return {
    color: mixHex(baseColor, seed & 1 ? baseAccent : '#ffffff', colorMix),
    accent: mixHex(baseAccent, seed & 2 ? baseColor : '#ffffff', accentMix)
  };
}

function inferSkillActionProfile(skill) {
  const id = String(skill && skill.id || '').toLowerCase();
  const type = String(skill && skill.type || '').toLowerCase();
  const purpose = String(skill && skill.purpose || '').toLowerCase();
  const roleTags = Array.isArray(skill && skill.roleTags) ? skill.roleTags : [];
  const targetingMode = String(skill && skill.targeting && skill.targeting.mode || '').toLowerCase();

  if (skill && skill.movementEffect || skill && skill.category === 'mobility' || /\b(mobility|movement)\b/.test(type)) return 'mobility';
  if (/snare_trap|spike_trap|tripwire|trapper_detonate|kill_zone|tactical_field/.test(id)) return 'trap';
  if (targetingMode === 'projectile' || targetingMode === 'chain' || /shot|arrow|bolt|fireball/.test(id)) return 'projectile';
  if (/fighter_guard|shield|barrier|oath|impact_guard|retaliation|last_stand/.test(id) || /\bdefense\b/.test(type)) return 'guard';
  if (skill && skill.category === 'buff' || /\bbuff\b|\bparty\b|\bstance\b|\bsustain\b/.test(type)) return 'buff';
  if (/\barea\b|\bburst\b|\bfinisher\b|\bresource\b/.test(type) ||
    purpose === 'mobbing' || purpose === 'finisher' || purpose === 'resource' ||
    roleTags.includes('Mobbing') || /slam|burst|wildfire|circle|glyph|wave|detonation|cleave/.test(id)) return 'area';
  return 'melee';
}

function buildSkillFxIdentityMap(skillIds) {
  const ids = Array.from(new Set((skillIds || []).map((id) => String(id || '')).filter(Boolean))).sort();
  const identities = new Map();
  const usedSeeds = new Set();
  ids.forEach((skillId) => {
    let seed = hashString(`project-starfall:skill-fx:${skillId}`);
    while (usedSeeds.has(seed)) seed = (seed + 0x9e3779b9) >>> 0;
    usedSeeds.add(seed);
    identities.set(skillId, Object.freeze({
      seed,
      phase: ((seed >>> 7) % 360) * Math.PI / 180,
      signatureSides: 3 + ((seed >>> 13) % 6),
      signatureInset: 22 + ((seed >>> 19) % 11)
    }));
  });
  return identities;
}

function getSkillSourcePath(entry) {
  return path.join(SKILL_SOURCE_DIR, path.basename(entry && entry.animation && entry.animation.sheet || ''));
}

function hasDedicatedSkillSource(entry) {
  const sourcePath = getSkillSourcePath(entry);
  return !!path.basename(sourcePath) && fs.existsSync(sourcePath);
}

function inferMotif(meta) {
  const text = [
    meta && meta.id,
    meta && meta.name,
    meta && meta.owner,
    meta && meta.type,
    meta && meta.category,
    meta && meta.family,
    meta && meta.behavior,
    meta && meta.visualKind,
    meta && meta.iconKind
  ].join(' ').toLowerCase();
  if (text.includes('fire') || text.includes('flame') || text.includes('inferno') || text.includes('ember') || text.includes('cinder') || text.includes('lava')) return 'fire';
  if (text.includes('storm') || text.includes('chain') || text.includes('static') || text.includes('lightning') || text.includes('thunder') || text.includes('gale')) return 'lightning';
  if (text.includes('rune') || text.includes('glyph') || text.includes('seal') || text.includes('astral') || text.includes('eclipse')) return 'rune';
  if (text.includes('void') || text.includes('rift')) return 'void';
  if (text.includes('frost') || text.includes('ice') || text.includes('rime') || text.includes('glacier') || text.includes('snow')) return 'frost';
  if (text.includes('arrow') || text.includes('shot') || text.includes('volley') || text.includes('sniper') || text.includes('archer')) return 'arrow';
  if (text.includes('trap') || text.includes('wire') || text.includes('zone')) return 'trap';
  if (text.includes('guard') || text.includes('shield') || text.includes('barrier') || text.includes('armor')) return 'shield';
  if (text.includes('boar') || text.includes('beast') || text.includes('claw') || text.includes('pack') || text.includes('companion')) return 'beast';
  if (text.includes('plant') || text.includes('briar') || text.includes('thorn') || text.includes('vine') || text.includes('moss')) return 'nature';
  if (text.includes('construct') || text.includes('clock') || text.includes('gear') || text.includes('coil') || text.includes('titan')) return 'gear';
  return 'slash';
}

function drawOrbitals(centerX, centerY, radius, count, turn, palette, seed, opacity) {
  const parts = [];
  for (let index = 0; index < count; index += 1) {
    const angle = turn + index / count * Math.PI * 2 + (seed % 13) * 0.04;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius * 0.58;
    const size = 4 + ((seed + index) % 5);
    parts.push(shape('circle', { cx: x.toFixed(1), cy: y.toFixed(1), r: size.toFixed(1), fill: index % 2 ? palette.accent : palette.color, 'fill-opacity': opacity }));
  }
  return parts.join('');
}

function drawMotifSignature(motif, rowId, frame, frames, palette, seed) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const opacity = (0.22 + pulse * 0.34).toFixed(2);
  const spin = (seed % 180 + frame * 18).toFixed(1);
  const parts = [
    shape('circle', { cx: 80, cy: 82, r: (46 + pulse * 12).toFixed(1), fill: 'none', stroke: palette.color, 'stroke-width': 2, 'stroke-dasharray': '7 9', 'stroke-opacity': opacity, transform: `rotate(${spin} 80 82)` })
  ];
  if (motif === 'fire') {
    parts.push(pathShape('M80 33 C94 58 114 68 101 105 C91 95 79 109 66 96 C50 80 71 61 80 33 Z', { fill: palette.color, 'fill-opacity': 0.14 + pulse * 0.16, stroke: palette.accent, 'stroke-width': 2, 'stroke-opacity': opacity }));
  } else if (motif === 'lightning') {
    parts.push(pathShape('M71 28 L101 29 L85 66 L113 66 L61 136 L75 88 L51 88 Z', { fill: palette.accent, 'fill-opacity': 0.16 + pulse * 0.18, stroke: palette.color, 'stroke-width': 2, 'stroke-opacity': opacity }));
  } else if (motif === 'rune' || motif === 'void') {
    parts.push(polygon([[80, 29], [126, 82], [80, 135], [34, 82]], { fill: 'none', stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': opacity }));
    parts.push(pathShape('M80 43 L80 121 M49 82 L111 82 M59 59 L101 105 M101 59 L59 105', { fill: 'none', stroke: palette.color, 'stroke-width': 2, 'stroke-opacity': opacity }));
  } else if (motif === 'arrow') {
    parts.push(pathShape('M31 83 L117 83 M99 65 L133 83 L99 101', { fill: 'none', stroke: palette.accent, 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-opacity': opacity }));
  } else if (motif === 'trap') {
    parts.push(pathShape('M42 121 L118 43 M42 43 L118 121 M37 82 L123 82', { fill: 'none', stroke: palette.accent, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-opacity': opacity }));
  } else if (motif === 'shield') {
    parts.push(pathShape('M80 31 L116 49 L107 105 Q80 132 53 105 L44 49 Z', { fill: palette.color, 'fill-opacity': 0.1 + pulse * 0.12, stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': opacity }));
  } else if (motif === 'nature' || motif === 'beast') {
    parts.push(pathShape('M80 125 C63 98 38 105 30 73 C55 75 61 55 80 38 C99 55 105 75 130 73 C122 105 97 98 80 125 Z', { fill: palette.color, 'fill-opacity': 0.12 + pulse * 0.14, stroke: palette.accent, 'stroke-width': 2, 'stroke-opacity': opacity }));
  } else if (motif === 'gear') {
    for (let index = 0; index < 8; index += 1) {
      const angle = index / 8 * Math.PI * 2 + p * Math.PI;
      parts.push(shape('rect', { x: (75 + Math.cos(angle) * 49).toFixed(1), y: (77 + Math.sin(angle) * 49).toFixed(1), width: 10, height: 10, rx: 2, fill: palette.accent, 'fill-opacity': opacity, transform: `rotate(${angle * 180 / Math.PI} ${(80 + Math.cos(angle) * 49).toFixed(1)} ${(82 + Math.sin(angle) * 49).toFixed(1)})` }));
    }
  }
  if (rowId === 'impact' || rowId === 'area') {
    parts.push(drawOrbitals(80, 82, 50 + pulse * 18, 6 + seed % 3, p * Math.PI * 2, palette, seed, 0.42));
  }
  return parts.join('');
}

function drawCast(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const rotate = (frame * 24 + seed % 90).toFixed(1);
  const radius = 22 + pulse * 30;
  const dark = mixHex(palette.color, '#071827', 0.45);
  const parts = [
    `<g transform="rotate(${rotate} 80 82)">`,
    shape('ellipse', { cx: 80, cy: 84, rx: (radius * 1.3).toFixed(1), ry: (radius * 0.46).toFixed(1), fill: 'none', stroke: palette.color, 'stroke-width': 5, 'stroke-opacity': (0.2 + pulse * 0.55).toFixed(2) }),
    shape('ellipse', { cx: 80, cy: 84, rx: (radius * 0.76).toFixed(1), ry: (radius * 0.28).toFixed(1), fill: 'none', stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': (0.45 + pulse * 0.4).toFixed(2) }),
    `</g>`,
    drawOrbitals(80, 80, 24 + pulse * 22, 5 + seed % 3, p * Math.PI * 2, palette, seed, 0.72),
    shape('circle', { cx: 80, cy: 80, r: (12 + pulse * 10).toFixed(1), fill: palette.accent, 'fill-opacity': 0.34 }),
    shape('circle', { cx: 80, cy: 80, r: (7 + pulse * 4).toFixed(1), fill: motif === 'void' ? dark : palette.color, 'fill-opacity': 0.82 })
  ];
  if (motif === 'shield') {
    parts.push(pathShape(`M80 ${34 + pulse * 8} L111 ${50 + pulse * 5} L103 101 Q80 124 57 101 L49 ${50 + pulse * 5} Z`, { fill: palette.color, 'fill-opacity': 0.18, stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': 0.72 }));
  } else if (motif === 'rune' || motif === 'trap') {
    parts.push(polygon([[80, 35], [118, 80], [80, 125], [42, 80]], { fill: 'none', stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': 0.75 }));
    parts.push(pathShape('M80 48 L80 112 M52 80 L108 80 M62 60 L98 100 M98 60 L62 100', { fill: 'none', stroke: palette.color, 'stroke-width': 3, 'stroke-opacity': 0.62 }));
  }
  return parts.join('');
}

function drawProjectile(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const wobble = Math.sin((p + (seed % 5) * 0.03) * Math.PI * 2);
  const offset = (p - 0.5) * 24;
  const parts = [
    shape('ellipse', { cx: 73 - offset * 0.3, cy: 82, rx: 58, ry: 12, fill: palette.color, 'fill-opacity': 0.12 }),
    shape('ellipse', { cx: 83 + offset * 0.4, cy: 80, rx: 34, ry: 9, fill: palette.accent, 'fill-opacity': 0.16 })
  ];
  if (motif === 'arrow') {
    parts.push(pathShape(`M28 ${80 + wobble * 5} L111 ${80 - wobble * 2}`, { fill: 'none', stroke: palette.accent, 'stroke-width': 6, 'stroke-linecap': 'round', 'stroke-opacity': 0.92 }));
    parts.push(polygon([[108, 68], [136, 80], [108, 92]], { fill: palette.color, 'fill-opacity': 0.96, stroke: palette.accent, 'stroke-width': 2 }));
    parts.push(pathShape(`M31 ${73 + wobble * 5} L12 ${64 + wobble * 2} M31 ${87 + wobble * 5} L12 ${96 + wobble * 2}`, { fill: 'none', stroke: palette.color, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-opacity': 0.7 }));
  } else if (motif === 'lightning') {
    parts.push(pathShape(`M20 ${87 + wobble * 5} L52 ${62 - wobble * 4} L70 ${79 + wobble * 5} L104 ${48 - wobble * 2} L93 ${77} L139 ${63 - wobble * 4}`, { fill: 'none', stroke: palette.accent, 'stroke-width': 9, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-opacity': 0.76 }));
    parts.push(pathShape(`M22 ${88 + wobble * 5} L53 ${64 - wobble * 4} L71 ${81 + wobble * 5} L107 ${50 - wobble * 2} L96 ${80} L139 ${65 - wobble * 4}`, { fill: 'none', stroke: palette.color, 'stroke-width': 4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-opacity': 0.96 }));
  } else if (motif === 'fire') {
    parts.push(pathShape(`M36 86 C54 ${38 + wobble * 8} 84 ${42 - wobble * 7} 124 76 C96 72 85 102 50 116 C42 108 33 99 36 86 Z`, { fill: palette.color, 'fill-opacity': 0.78 }));
    parts.push(pathShape(`M55 86 C70 ${55 + wobble * 6} 91 ${58 - wobble * 5} 110 78 C91 79 83 98 61 106 C56 100 52 93 55 86 Z`, { fill: palette.accent, 'fill-opacity': 0.74 }));
  } else if (motif === 'rune' || motif === 'void') {
    parts.push(polygon([[35, 80], [58, 57], [114, 57], [137, 80], [114, 103], [58, 103]], { fill: palette.color, 'fill-opacity': 0.2, stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': 0.9 }));
    parts.push(pathShape('M49 80 L123 80 M86 59 L86 101 M61 64 L111 96 M111 64 L61 96', { fill: 'none', stroke: palette.color, 'stroke-width': 3, 'stroke-opacity': 0.72 }));
  } else {
    parts.push(pathShape(`M30 ${88 + wobble * 5} C58 ${46 - wobble * 5} 103 ${50 + wobble * 4} 133 ${72} C104 ${89 + wobble * 7} 68 ${105 - wobble * 3} 30 ${88 + wobble * 5} Z`, { fill: palette.color, 'fill-opacity': 0.46, stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': 0.78 }));
  }
  parts.push(drawOrbitals(82, 80, 34 + wobble * 4, 4, p * Math.PI * 2, palette, seed, 0.55));
  return parts.join('');
}

function drawImpact(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const radius = 18 + p * 56;
  const spike = 28 + pulse * 42;
  const parts = [
    shape('circle', { cx: 80, cy: 82, r: radius.toFixed(1), fill: palette.color, 'fill-opacity': (0.28 * (1 - p)).toFixed(2), stroke: palette.accent, 'stroke-width': 5, 'stroke-opacity': (0.78 * (1 - p * 0.35)).toFixed(2) }),
    shape('circle', { cx: 80, cy: 82, r: (8 + pulse * 14).toFixed(1), fill: palette.accent, 'fill-opacity': 0.82 })
  ];
  const points = [];
  for (let index = 0; index < 12; index += 1) {
    const angle = index / 12 * Math.PI * 2 + (seed % 9) * 0.03;
    const length = index % 2 ? spike * 0.56 : spike;
    points.push([80 + Math.cos(angle) * length, 82 + Math.sin(angle) * length]);
  }
  parts.push(polygon(points, { fill: palette.color, 'fill-opacity': (0.18 + pulse * 0.28).toFixed(2), stroke: palette.accent, 'stroke-width': 2, 'stroke-opacity': 0.68 }));
  if (motif === 'gear') {
    for (let index = 0; index < 8; index += 1) {
      const angle = index / 8 * Math.PI * 2 + p * Math.PI;
      parts.push(shape('rect', { x: (75 + Math.cos(angle) * 38).toFixed(1), y: (77 + Math.sin(angle) * 38).toFixed(1), width: 10, height: 10, rx: 2, fill: palette.accent, 'fill-opacity': 0.72, transform: `rotate(${angle * 180 / Math.PI} ${(80 + Math.cos(angle) * 38).toFixed(1)} ${(82 + Math.sin(angle) * 38).toFixed(1)})` }));
    }
  } else if (motif === 'nature') {
    parts.push(pathShape(`M80 118 C63 94 41 99 31 74 C54 73 61 57 80 43 C99 57 106 73 129 74 C119 99 97 94 80 118 Z`, { fill: palette.color, 'fill-opacity': 0.24, stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': 0.72 }));
  }
  return parts.join('');
}

function drawArea(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const parts = [
    shape('ellipse', { cx: 80, cy: 111, rx: (34 + p * 48).toFixed(1), ry: (9 + p * 18).toFixed(1), fill: palette.color, 'fill-opacity': (0.2 + pulse * 0.26).toFixed(2), stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': (0.78 - p * 0.28).toFixed(2) }),
    shape('ellipse', { cx: 80, cy: 111, rx: (18 + pulse * 28).toFixed(1), ry: (5 + pulse * 9).toFixed(1), fill: palette.accent, 'fill-opacity': 0.34 })
  ];
  if (motif === 'trap') {
    parts.push(pathShape('M42 112 L118 112 M52 92 L108 132 M108 92 L52 132', { fill: 'none', stroke: palette.color, 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-opacity': 0.78 }));
    parts.push(polygon([[80, 68], [116, 110], [80, 142], [44, 110]], { fill: 'none', stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': 0.86 }));
  } else if (motif === 'fire') {
    for (let index = 0; index < 5; index += 1) {
      const x = 44 + index * 18 + ((seed + index) % 7);
      const height = 24 + pulse * 32 + (index % 2) * 10;
      parts.push(pathShape(`M${x} 118 C${x - 8} ${95 - height * 0.3} ${x + 8} ${92 - height} ${x + 15} ${116} C${x + 8} ${111} ${x + 3} ${126} ${x} 118 Z`, { fill: index % 2 ? palette.accent : palette.color, 'fill-opacity': 0.62 }));
    }
  } else {
    parts.push(drawOrbitals(80, 108, 38 + pulse * 22, 7, p * Math.PI * 2, palette, seed, 0.68));
  }
  return parts.join('');
}

function drawTrail(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const sway = Math.sin(p * Math.PI * 2 + seed % 5);
  const parts = [
    pathShape(`M28 ${104 + sway * 5} C53 ${38 - sway * 8} 107 ${38 + sway * 8} 133 ${100 - sway * 5}`, { fill: 'none', stroke: palette.color, 'stroke-width': 17, 'stroke-linecap': 'round', 'stroke-opacity': (0.22 + (1 - p) * 0.42).toFixed(2) }),
    pathShape(`M32 ${101 + sway * 5} C57 ${49 - sway * 8} 103 ${49 + sway * 8} 130 ${96 - sway * 5}`, { fill: 'none', stroke: palette.accent, 'stroke-width': 6, 'stroke-linecap': 'round', 'stroke-opacity': (0.45 + (1 - p) * 0.45).toFixed(2) }),
    drawOrbitals(82, 80, 32 + p * 18, 4, p * Math.PI, palette, seed, 0.52)
  ];
  if (motif === 'beast') {
    parts.push(pathShape('M54 47 L70 97 M83 42 L84 101 M112 50 L96 99', { fill: 'none', stroke: palette.color, 'stroke-width': 6, 'stroke-linecap': 'round', 'stroke-opacity': 0.76 }));
  }
  return parts.join('');
}

function drawTelegraph(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const parts = [
    shape('ellipse', { cx: 80, cy: 110, rx: (42 + pulse * 26).toFixed(1), ry: (14 + pulse * 8).toFixed(1), fill: palette.color, 'fill-opacity': 0.16, stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': (0.42 + pulse * 0.5).toFixed(2) }),
    pathShape('M80 41 L80 118 M39 110 L121 110', { fill: 'none', stroke: palette.accent, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-opacity': 0.74 }),
    shape('circle', { cx: 80, cy: 77, r: (10 + pulse * 8).toFixed(1), fill: palette.color, 'fill-opacity': 0.48 })
  ];
  if (motif === 'frost') {
    parts.push(pathShape('M80 36 L92 62 L121 66 L99 84 L105 113 L80 98 L55 113 L61 84 L39 66 L68 62 Z', { fill: 'none', stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': 0.8 }));
  }
  return parts.join('');
}

function drawBuff(row, frame, frames, palette, motif, seed) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const parts = [
    shape('ellipse', { cx: 80, cy: 116, rx: 42, ry: 12, fill: palette.color, 'fill-opacity': 0.18, stroke: palette.accent, 'stroke-width': 3, 'stroke-opacity': 0.6 }),
    pathShape(`M52 112 C49 ${74 - pulse * 14} 65 ${50 - pulse * 10} 80 38 C96 ${51 - pulse * 12} 112 ${75 - pulse * 14} 108 112`, { fill: palette.color, 'fill-opacity': 0.18, stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': 0.72 }),
    drawOrbitals(80, 78, 33 + pulse * 17, 6, p * Math.PI * 2, palette, seed, 0.68)
  ];
  if (motif === 'shield') {
    parts.push(pathShape('M80 35 L111 51 L104 101 Q80 124 56 101 L49 51 Z', { fill: palette.color, 'fill-opacity': 0.18, stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': 0.78 }));
  }
  return parts.join('');
}

function drawActionProfileSignature(profile, rowId, frame, frames, palette, identity) {
  const p = frame / Math.max(1, frames - 1);
  const pulse = Math.sin(p * Math.PI);
  const seed = Number(identity && identity.seed || 0);
  const phase = Number(identity && identity.phase || 0);
  const opacity = (0.28 + pulse * 0.38).toFixed(2);
  const parts = [];

  if (profile === 'melee') {
    parts.push(pathShape(`M35 ${112 - pulse * 28} Q80 ${34 + (seed % 9)} 129 ${102 + pulse * 8}`, {
      fill: 'none', stroke: palette.accent, 'stroke-width': 5, 'stroke-linecap': 'round', 'stroke-opacity': opacity
    }));
  } else if (profile === 'mobility') {
    for (let index = 0; index < 3; index += 1) {
      const y = 58 + index * 24 + Math.sin(phase + index + p * Math.PI * 2) * 5;
      parts.push(pathShape(`M${25 + index * 5} ${y.toFixed(1)} L${91 + pulse * 18} ${y.toFixed(1)} L${78 + pulse * 15} ${(y - 10).toFixed(1)} M${91 + pulse * 18} ${y.toFixed(1)} L${78 + pulse * 15} ${(y + 10).toFixed(1)}`, {
        fill: 'none', stroke: index === 1 ? palette.accent : palette.color, 'stroke-width': 3 + index, 'stroke-linecap': 'round', 'stroke-opacity': opacity
      }));
    }
  } else if (profile === 'guard') {
    parts.push(pathShape('M80 30 L119 50 L109 106 Q80 136 51 106 L41 50 Z', {
      fill: palette.color, 'fill-opacity': 0.08 + pulse * 0.1, stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': opacity
    }));
  } else if (profile === 'buff') {
    const rayCount = 5 + (seed % 4);
    for (let index = 0; index < rayCount; index += 1) {
      const angle = phase + index / rayCount * Math.PI * 2;
      const inner = 31 + pulse * 7;
      const outer = 51 + pulse * 11;
      parts.push(pathShape(`M${(80 + Math.cos(angle) * inner).toFixed(1)} ${(82 + Math.sin(angle) * inner).toFixed(1)} L${(80 + Math.cos(angle) * outer).toFixed(1)} ${(82 + Math.sin(angle) * outer).toFixed(1)}`, {
        fill: 'none', stroke: index % 2 ? palette.accent : palette.color, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-opacity': opacity
      }));
    }
  } else if (profile === 'projectile') {
    parts.push(pathShape(`M24 ${82 + Math.sin(phase + p * Math.PI * 2) * 5} L136 82`, {
      fill: 'none', stroke: palette.accent, 'stroke-width': rowId === 'projectile' ? 3 : 2, 'stroke-dasharray': '9 12', 'stroke-opacity': opacity
    }));
  } else if (profile === 'trap') {
    const teeth = 5 + (seed % 3);
    const points = [];
    for (let index = 0; index < teeth; index += 1) {
      const x = 42 + index * (76 / Math.max(1, teeth - 1));
      points.push([x, index % 2 ? 118 : 91 - pulse * 8]);
    }
    parts.push(polygon(points, { fill: 'none', stroke: palette.accent, 'stroke-width': 4, 'stroke-linejoin': 'round', 'stroke-opacity': opacity }));
  } else if (profile === 'area') {
    parts.push(shape('ellipse', { cx: 80, cy: 110, rx: 54 + pulse * 12, ry: 16 + pulse * 6, fill: 'none', stroke: palette.accent, 'stroke-width': 4, 'stroke-opacity': opacity }));
    parts.push(shape('ellipse', { cx: 80, cy: 110, rx: 34 + pulse * 8, ry: 9 + pulse * 4, fill: 'none', stroke: palette.color, 'stroke-width': 3, 'stroke-opacity': opacity }));
  }

  const sides = Math.max(3, Number(identity && identity.signatureSides || 5));
  const signatureRadius = Math.max(18, Number(identity && identity.signatureInset || 26));
  const signaturePoints = [];
  for (let index = 0; index < sides; index += 1) {
    const angle = phase + p * Math.PI * 0.75 + index / sides * Math.PI * 2;
    signaturePoints.push([
      (80 + Math.cos(angle) * signatureRadius).toFixed(1),
      (82 + Math.sin(angle) * signatureRadius).toFixed(1)
    ]);
  }
  parts.push(polygon(signaturePoints, {
    fill: 'none', stroke: palette.color, 'stroke-width': 2, 'stroke-dasharray': `${3 + seed % 5} ${5 + (seed >>> 4) % 6}`, 'stroke-opacity': (0.2 + pulse * 0.24).toFixed(2)
  }));
  return parts.join('');
}

function drawSkillFrame(profile, rowId, frame, state, palette, motif, seed, identity) {
  const frames = Math.max(1, Number(state.frames || DEFAULT_FRAMES));
  let core = '';
  if (profile === 'melee') {
    core = rowId === 'cast' || rowId === 'projectile'
      ? drawTrail(rowId, frame, frames, palette, motif, seed)
      : rowId === 'area'
        ? drawArea(rowId, frame, frames, palette, motif, seed)
        : drawImpact(rowId, frame, frames, palette, motif, seed);
  } else if (profile === 'mobility') {
    core = rowId === 'impact'
      ? drawImpact(rowId, frame, frames, palette, motif, seed)
      : rowId === 'area'
        ? drawArea(rowId, frame, frames, palette, motif, seed)
        : drawTrail(rowId, frame, frames, palette, motif, seed);
  } else if (profile === 'guard') {
    core = rowId === 'cast'
      ? drawCast(rowId, frame, frames, palette, 'shield', seed)
      : rowId === 'area'
        ? drawBuff(rowId, frame, frames, palette, 'shield', seed)
        : rowId === 'projectile'
          ? drawProjectile(rowId, frame, frames, palette, 'shield', seed)
          : drawImpact(rowId, frame, frames, palette, 'shield', seed);
  } else if (profile === 'buff') {
    core = rowId === 'projectile'
      ? drawCast(rowId, frame, frames, palette, motif, seed)
      : drawBuff(rowId, frame, frames, palette, motif, seed);
  } else if (profile === 'trap') {
    core = rowId === 'area'
      ? drawArea(rowId, frame, frames, palette, 'trap', seed)
      : rowId === 'impact'
        ? drawImpact(rowId, frame, frames, palette, 'trap', seed)
        : rowId === 'projectile'
          ? drawProjectile(rowId, frame, frames, palette, 'trap', seed)
          : drawCast(rowId, frame, frames, palette, 'trap', seed);
  } else if (profile === 'area') {
    core = rowId === 'area'
      ? drawArea(rowId, frame, frames, palette, motif, seed)
      : rowId === 'impact'
        ? drawImpact(rowId, frame, frames, palette, motif, seed)
        : drawCast(rowId, frame, frames, palette, motif, seed);
  } else {
    core = drawFrame(rowId, frame, state, palette, motif, seed);
  }
  return core + drawActionProfileSignature(profile, rowId, frame, frames, palette, identity);
}

function drawFrame(rowId, frame, state, palette, motif, seed) {
  const frames = Math.max(1, Number(state.frames || DEFAULT_FRAMES));
  if (rowId === 'cast') return drawCast(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'projectile') return drawProjectile(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'impact') return drawImpact(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'area') return drawArea(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'trail') return drawTrail(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'telegraph') return drawTelegraph(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'melee') return drawTrail(rowId, frame, frames, palette, motif, seed);
  if (rowId === 'buff') return drawBuff(rowId, frame, frames, palette, motif, seed);
  return drawImpact(rowId, frame, frames, palette, motif, seed);
}

function orderedRows(animation) {
  return Object.entries(animation.states || {})
    .sort((a, b) => Number(a[1].row || 0) - Number(b[1].row || 0));
}

function makeSheetSvg(entry) {
  const animation = entry.animation;
  const rows = orderedRows(animation);
  const frameSize = Number(animation.frameWidth || DEFAULT_FRAME_SIZE);
  const frameHeight = Number(animation.frameHeight || DEFAULT_FRAME_SIZE);
  const frames = Math.max(DEFAULT_FRAMES, ...rows.map(([, state]) => Number(state.frames || DEFAULT_FRAMES)));
  const width = frameSize * frames;
  const height = frameHeight * rows.length;
  const identity = entry.identity || Object.freeze({ seed: hashString(`${entry.kind}:${entry.id}`) });
  const seed = Number(identity.seed || hashString(`${entry.kind}:${entry.id}`));
  const motif = inferMotif(entry.meta || {});
  const profile = entry.profile || '';
  const contentScale = entry.kind === 'skill' ? SKILL_CONTENT_SCALE : 1;
  const contentOffset = entry.kind === 'skill' ? SKILL_CONTENT_OFFSET : 0;
  const palette = {
    color: entry.palette.color,
    accent: entry.palette.accent
  };
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`
  ];
  rows.forEach(([rowId, state], rowIndex) => {
    const rowFrames = Math.max(1, Number(state.frames || DEFAULT_FRAMES));
    for (let frame = 0; frame < frames; frame += 1) {
      const stateFrame = Math.min(rowFrames - 1, frame);
      const x = frame * frameSize + contentOffset;
      const y = rowIndex * frameHeight + contentOffset;
      const frameSeed = seed + rowIndex * 101 + frame * 37;
	      parts.push(`<g transform="translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${contentScale})">`);
	      parts.push(shape('ellipse', { cx: 80, cy: 124, rx: 52, ry: 12, fill: mixHex(palette.color, '#000000', 0.25), 'fill-opacity': 0.08 }));
	      parts.push(entry.kind === 'skill'
	        ? drawSkillFrame(profile, rowId, stateFrame, state, palette, motif, frameSeed, identity)
	        : drawFrame(rowId, stateFrame, state, palette, motif, frameSeed));
	      parts.push(drawMotifSignature(motif, rowId, stateFrame, rowFrames, palette, frameSeed));
	      parts.push('</g>');
    }
  });
  parts.push('</svg>');
  return parts.join('');
}

function buildEntries() {
  const skillsById = Object.freeze((Data.SKILLS || []).reduce((skills, skill) => {
    skills[skill.id] = skill;
    return skills;
  }, {}));
  const enemiesById = Object.freeze((Data.ENEMIES || []).reduce((enemies, enemy) => {
    enemies[enemy.id] = enemy;
    return enemies;
  }, {}));
  const skillIds = Object.keys(Data.SKILL_FX_ANIMATION_ASSETS || {});
  const skillIdentities = buildSkillFxIdentityMap(skillIds);
  const skillEntries = Object.entries(Data.SKILL_FX_ANIMATION_ASSETS || {}).map(([id, animation]) => {
    const skill = skillsById[id] || {};
    const identity = skillIdentities.get(id);
    return {
      kind: 'skill',
      id,
      animation,
      meta: Object.assign({}, skill, { visualKind: skill.visualId && Data.SKILL_VISUALS && Data.SKILL_VISUALS[skill.visualId] && Data.SKILL_VISUALS[skill.visualId].kind || '' }),
      profile: inferSkillActionProfile(skill),
      identity,
      palette: skillPalette(skill, identity)
    };
  });
  const basicEntries = Object.entries(Data.BASIC_ATTACK_FX_ANIMATION_ASSETS || {}).map(([id, animation]) => ({
    kind: 'basic',
    id,
    animation,
    meta: { id, owner: id, type: 'basic attack', iconKind: id === 'archer' ? 'arrow' : id === 'mage' ? 'magic' : 'slash' },
    palette: CLASS_PALETTES[id] || CLASS_PALETTES.fighter
  }));
  const enemyEntries = Object.entries(Data.ENEMY_COMBAT_FX_ANIMATION_ASSETS || {}).map(([id, animation]) => {
    const enemy = enemiesById[id] || {};
    return {
      kind: 'enemy',
      id,
      animation,
      meta: enemy,
      palette: enemyPalette(enemy)
    };
  });
  return { skills: skillEntries, basic: basicEntries, enemies: enemyEntries };
}

function clearTransparentRgb(data) {
  for (let offset = 0; offset < data.length; offset += 4) {
    if (data[offset + 3] !== 0) continue;
    data[offset] = 0;
    data[offset + 1] = 0;
    data[offset + 2] = 0;
  }
  return data;
}

function clearNegligibleChromaArtifacts(data) {
  const keys = [[0, 255, 0], [255, 0, 255]];
  for (let offset = 0; offset < data.length; offset += 4) {
    const alpha = data[offset + 3];
    if (alpha > 8) continue;
    const matchesKey = keys.some((key) => {
      const red = data[offset] - key[0];
      const green = data[offset + 1] - key[1];
      const blue = data[offset + 2] - key[2];
      return red * red + green * green + blue * blue <= 48 * 48;
    });
    if (!matchesKey) continue;
    data[offset] = 0;
    data[offset + 1] = 0;
    data[offset + 2] = 0;
    data[offset + 3] = 0;
  }
  return data;
}

function validateSemanticSheetData(data, width, height, label = 'semantic skill FX sheet') {
  const expectedWidth = DEFAULT_FRAME_SIZE * DEFAULT_FRAMES;
  const expectedHeight = DEFAULT_FRAME_SIZE * 4;
  if (width !== expectedWidth || height !== expectedHeight) {
    throw new Error(`${label} should be ${expectedWidth}x${expectedHeight}, got ${width}x${height}`);
  }
  let transparentRgbPixels = 0;
  const frameHashes = new Map();
  const frames = [];
  for (let offset = 0; offset < data.length; offset += 4) {
    if (data[offset + 3] === 0 && (data[offset] || data[offset + 1] || data[offset + 2])) transparentRgbPixels += 1;
  }
  if (transparentRgbPixels) throw new Error(`${label} contains ${transparentRgbPixels} transparent pixel(s) with hidden RGB data`);

  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < DEFAULT_FRAMES; column += 1) {
      const frame = Buffer.alloc(DEFAULT_FRAME_SIZE * DEFAULT_FRAME_SIZE * 4);
      let minX = DEFAULT_FRAME_SIZE;
      let minY = DEFAULT_FRAME_SIZE;
      let maxX = -1;
      let maxY = -1;
      let visiblePixels = 0;
      for (let y = 0; y < DEFAULT_FRAME_SIZE; y += 1) {
        const sourceStart = (((row * DEFAULT_FRAME_SIZE + y) * width) + column * DEFAULT_FRAME_SIZE) * 4;
        const destinationStart = y * DEFAULT_FRAME_SIZE * 4;
        data.copy(frame, destinationStart, sourceStart, sourceStart + DEFAULT_FRAME_SIZE * 4);
        for (let x = 0; x < DEFAULT_FRAME_SIZE; x += 1) {
          if (data[sourceStart + x * 4 + 3] === 0) continue;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
          visiblePixels += 1;
        }
      }
      const frameName = `row ${row + 1}, frame ${column + 1}`;
      if (!visiblePixels) throw new Error(`${label} has an empty ${frameName}`);
      if (minX < SKILL_FRAME_GUTTER || minY < SKILL_FRAME_GUTTER ||
          maxX >= DEFAULT_FRAME_SIZE - SKILL_FRAME_GUTTER || maxY >= DEFAULT_FRAME_SIZE - SKILL_FRAME_GUTTER) {
        throw new Error(`${label} ${frameName} violates the ${SKILL_FRAME_GUTTER}px safety gutter (${minX},${minY})-(${maxX},${maxY})`);
      }
      const hash = crypto.createHash('sha256').update(frame).digest('hex');
      if (frameHashes.has(hash)) throw new Error(`${label} has duplicate frames: ${frameHashes.get(hash)} and ${frameName}`);
      frameHashes.set(hash, frameName);
      frames.push({ row, column, visiblePixels, minX, minY, maxX, maxY, hash });
    }
  }
  return { frames, transparentRgbPixels };
}

async function renderEntryBuffer(entry) {
  const svg = makeSheetSvg(entry);
  const { data, info } = await sharp(Buffer.from(svg))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  clearNegligibleChromaArtifacts(data);
  clearTransparentRgb(data);
  if (entry.kind === 'skill') {
    validateSemanticSheetData(data, info.width, info.height, `generated skill FX ${entry.id}`);
  }
  return sharp(data, {
    raw: { width: info.width, height: info.height, channels: 4 }
  })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
}

async function validateSkillFxBuffer(buffer, entryOrLabel) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const label = typeof entryOrLabel === 'string'
    ? entryOrLabel
    : `skill FX ${entryOrLabel && entryOrLabel.id || 'sheet'}`;
  return validateSemanticSheetData(data, info.width, info.height, label);
}

function selectSkillEntries(entries, requestedIds) {
  const requested = (Array.isArray(requestedIds) ? requestedIds : [requestedIds])
    .map(normalizeId)
    .filter(Boolean);
  if (!requested.length) return entries.slice();
  const requestedSet = new Set(requested);
  const selected = entries.filter((entry) => {
    const fileId = path.basename(entry.animation.sheet, path.extname(entry.animation.sheet));
    return requestedSet.has(normalizeId(entry.id)) || requestedSet.has(normalizeId(fileId));
  });
  const matched = new Set();
  selected.forEach((entry) => {
    const normalizedId = normalizeId(entry.id);
    const normalizedFileId = normalizeId(path.basename(entry.animation.sheet, path.extname(entry.animation.sheet)));
    requested.forEach((id) => {
      if (id === normalizedId || id === normalizedFileId) matched.add(id);
    });
  });
  const unknown = requested.filter((id) => !matched.has(id));
  if (unknown.length) throw new Error(`Unknown skill FX id(s): ${unknown.join(', ')}`);
  return selected;
}

function outputPathFor(entry, outputDir) {
  return outputDir
    ? path.join(path.resolve(outputDir), path.basename(entry.animation.sheet))
    : path.join(ROOT, entry.animation.sheet);
}

async function writeEntry(entry, options = {}) {
  const outputPath = outputPathFor(entry, options.outputDir);
  const buffer = await renderEntryBuffer(entry);
  ensureDir(outputPath);
  await fs.promises.writeFile(outputPath, buffer);
  return outputPath;
}

async function processSemanticSkillFx(options = {}) {
  const allEntries = buildEntries().skills;
  const selected = selectSkillEntries(allEntries, options.skillIds || options.skill || []);
  const shouldWrite = options.write !== false;
  const useDedicatedSources = shouldWrite && !options.outputDir && options.preferSources !== false;
  const outputPaths = [];
  const semanticBuffers = new Map();
  const semanticHashes = new Map();

  // Establish a deterministic semantic fallback for every selected skill first.
  for (const entry of selected) {
    const buffer = await renderEntryBuffer(entry);
    const digest = crypto.createHash('sha256').update(buffer).digest('hex');
    if (semanticHashes.has(digest)) {
      throw new Error(`Semantic skill FX collision: ${semanticHashes.get(digest)} and ${entry.id}`);
    }
    semanticHashes.set(digest, entry.id);
    semanticBuffers.set(entry.id, buffer);
    const outputPath = outputPathFor(entry, options.outputDir);
    outputPaths.push(outputPath);
    if (shouldWrite) {
      ensureDir(outputPath);
      await fs.promises.writeFile(outputPath, buffer);
    }
  }

  // A dedicated source may replace only its own semantic fallback, never a
  // source-less sibling. This runs last so later build stages cannot erase it.
  const sourceBacked = [];
  if (useDedicatedSources) {
    for (const entry of selected.filter(hasDedicatedSkillSource)) {
      await processSkillFxSheets({ only: entry.id });
      sourceBacked.push(entry.id);
    }
  }

  const finalHashes = new Map();
  for (let index = 0; index < selected.length; index += 1) {
    const entry = selected[index];
    const finalBuffer = shouldWrite
      ? await fs.promises.readFile(outputPaths[index])
      : semanticBuffers.get(entry.id);
    await validateSkillFxBuffer(finalBuffer, entry);
    const digest = crypto.createHash('sha256').update(finalBuffer).digest('hex');
    if (finalHashes.has(digest)) {
      throw new Error(`Final skill FX collision: ${finalHashes.get(digest)} and ${entry.id}`);
    }
    finalHashes.set(digest, entry.id);
  }

  return {
    processed: selected.length,
    generated: selected.length - sourceBacked.length,
    sourceBacked: sourceBacked.length,
    sourceBackedIds: sourceBacked,
    outputPaths,
    entries: selected,
    buffers: options.returnBuffers ? semanticBuffers : undefined
  };
}

async function main(argv = process.argv.slice(2)) {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(usage());
    return;
  }
  const mode = normalizeMode(getOnlyMode(argv));
  const requestedSkill = getArgValue(argv, '--skill');
  const entries = buildEntries();
  if (requestedSkill || mode === 'skills') {
    if (requestedSkill && mode !== 'all' && mode !== 'skills') {
      throw new Error(`--skill cannot be combined with --only ${mode}`);
    }
    const result = await processSemanticSkillFx({ skill: requestedSkill });
    console.log(`Generated ${result.processed} semantic Project Starfall skill FX sheet(s); ${result.sourceBacked} dedicated source sheet(s) applied last.`);
    return;
  }
  const selected = mode === 'all'
    ? entries.basic.concat(entries.enemies)
    : entries[mode];
  if (!selected) throw new Error(`Unknown mode: ${mode}\n${usage()}`);

  let skillResult = null;
  if (mode === 'all') skillResult = await processSemanticSkillFx();
  for (const entry of selected) await writeEntry(entry);
  const skillSummary = skillResult
    ? ` Generated ${skillResult.processed} semantic skill FX sheets; ${skillResult.sourceBacked} dedicated source sheet(s) applied last.`
    : '';
  console.log(`Generated ${selected.length} Project Starfall combat FX sheets (${mode}).${skillSummary}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack || error);
    process.exitCode = 1;
  });
}

module.exports = {
  SKILL_ACTION_PROFILES,
  SKILL_FRAME_GUTTER,
  buildEntries,
  buildSkillFxIdentityMap,
  clearNegligibleChromaArtifacts,
  hasDedicatedSkillSource,
  inferSkillActionProfile,
  main,
  makeSheetSvg,
  processSemanticSkillFx,
  renderEntryBuffer,
  selectSkillEntries,
  validateSemanticSheetData,
  validateSkillFxBuffer,
  writeEntry
};
