'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const { buildGamesDirectoryWorkbenchData } = require('./cms-renderers');
const { normalizePathname } = require('./seo-routing');

const ROOT = path.resolve(__dirname, '../..');
const WIDTH = 1200;
const HEIGHT = 630;

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function loadSocialPreviewRecords(root = ROOT) {
  const toolDirectory = path.join(root, 'content/tools');
  const tools = fs.readdirSync(toolDirectory).filter(name => name.endsWith('.json')).sort()
    .map(name => readJson(path.join(toolDirectory, name)))
    .filter(tool => tool.slug && !tool.hidden && !tool.noindex && (tool.visibility || 'public') === 'public')
    .map(tool => ({
      id: `tool-${tool.slug}`,
      pathname: normalizePathname(tool.href || `/tools/${tool.slug}`),
      title: tool.title,
      summary: tool.summary || '',
      category: 'BROWSER TOOL',
      accent: '#087f8c',
      image: tool.iconImage || '',
      icon: true
    }));
  const gamePage = readJson(path.join(root, 'content/pages/games.json'));
  const games = buildGamesDirectoryWorkbenchData(gamePage).items.map(game => ({
    id: `game-${game.id}`,
    pathname: normalizePathname(game.href),
    title: game.title,
    summary: game.summary || '',
    category: game.type === 'Simulation' ? 'INTERACTIVE SIMULATION' : 'BROWSER GAME',
    accent: '#c94b0a',
    image: game.image || '',
    iconHtml: game.iconHtml || '',
    icon: !game.image
  }));
  return [...tools, ...games].map(record => ({
    ...record,
    file: `img/social/${record.id}.png`,
    alt: `${record.title} — ${record.category.toLowerCase()} by Daniel Short`
  }));
}

function escapeMarkup(value) {
  return String(value || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function textLayer(text, { root = ROOT, size, width, color = '#091f3b', bold = false } = {}) {
  return sharp({ text: {
    text: `<span foreground="${color}">${escapeMarkup(text)}</span>`,
    font: `Inter ${bold ? 'Bold ' : ''}${size}`,
    fontfile: path.join(root, 'build/fonts/Inter-Latin.ttf'),
    width,
    wrap: 'word',
    spacing: 8,
    rgba: true
  } }).png().toBuffer({ resolveWithObject: true });
}

async function renderSocialPreview(record, { root = ROOT } = {}) {
  const background = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">
    <rect width="1200" height="630" fill="#f7faff"/>
    <rect width="1200" height="12" fill="${record.accent}"/>
    <rect x="802" y="120" width="334" height="350" rx="32" fill="#ffffff" stroke="#dce5ef" stroke-width="2"/>
    <path d="M64 520h1072" stroke="#dce5ef" stroke-width="2"/>
    <rect x="64" y="71" width="8" height="25" rx="4" fill="${record.accent}"/>
  </svg>`);
  const layers = [];
  const addText = async (value, options, left, top) => {
    const layer = await textLayer(value, { root, ...options });
    layers.push({ input: layer.data, left, top });
    return layer.info.height;
  };
  await addText(record.category, { size: 21, width: 670, bold: true, color: record.accent }, 91, 73);
  let title = await textLayer(record.title, { root, size: 62, width: 674, bold: true });
  if (title.info.height > 225) title = await textLayer(record.title, { root, size: 50, width: 674, bold: true });
  layers.push({ input: title.data, left: 64, top: 150 });
  const summary = String(record.summary || '').trim();
  const summaryTop = 150 + title.info.height + 30;
  let description = await textLayer(summary, { root, size: 25, width: 660, color: '#475569' });
  if (description.info.height > 490 - summaryTop) {
    description = await textLayer(summary, { root, size: 21, width: 660, color: '#475569' });
  }
  if (description.info.height > 490 - summaryTop) throw new Error(`Social preview copy is too long: ${record.id}`);
  layers.push({ input: description.data, left: 64, top: summaryTop });
  await addText('Daniel Short', { size: 27, width: 450, bold: true }, 64, 553);
  await addText('danielshort.me', { size: 24, width: 340, color: '#475569' }, 824, 555);

  let media;
  if (record.image) {
    const imagePath = path.resolve(root, record.image.replace(/^\//, ''));
    if (!imagePath.startsWith(path.resolve(root) + path.sep)) throw new Error('Social preview image must be inside the repository.');
    media = await sharp(imagePath).resize(record.icon ? 274 : 314, record.icon ? 274 : 330, {
      fit: record.icon ? 'contain' : 'cover',
      background: '#ffffff'
    }).png().toBuffer();
  } else if (record.iconHtml) {
    const iconSvg = record.iconHtml.replace('<svg ', `<svg xmlns="http://www.w3.org/2000/svg" width="274" height="274" fill="none" stroke="${record.accent}" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round" `)
      .replace(/class="icon-fill"/g, `fill="${record.accent}" stroke="none"`);
    media = await sharp(Buffer.from(iconSvg)).png().toBuffer();
  } else {
    throw new Error(`No social preview artwork for ${record.id}`);
  }
  layers.push({ input: media, left: record.icon ? 832 : 812, top: record.icon ? 158 : 130 });
  return sharp(background).composite(layers).png({ compressionLevel: 9 }).toBuffer();
}

function socialPreviewMetadata(record, { root = ROOT, siteOrigin } = {}) {
  const file = path.join(root, record.file);
  if (!fs.existsSync(file)) return null;
  const hash = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex').slice(0, 12);
  return { url: `${siteOrigin}/${record.file}?v=${hash}`, width: String(WIDTH), height: String(HEIGHT), type: 'image/png', alt: record.alt };
}

module.exports = { WIDTH, HEIGHT, loadSocialPreviewRecords, renderSocialPreview, socialPreviewMetadata };
