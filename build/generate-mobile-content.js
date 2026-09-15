#!/usr/bin/env node
'use strict';

// The native app consumes public data, never rendered pages or executable CMS
// content. Keep this projection explicit: adding a CMS field does not publish it.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const net = require('net');
const { loadSiteContentAsync } = require('./lib/content-loader');

const REPOSITORY_ROOT = path.resolve(__dirname, '..');
const SITE_ORIGIN = 'https://www.danielshort.me';
const CATALOG_PATH = path.join('dist', 'app-content', 'v1', 'catalog.json');
const SITE_HOSTS = new Set(['www.danielshort.me', 'danielshort.me', 'www.dshort.me', 'dshort.me']);
const PRIVATE_PATH = /^\/(?:api|admin|_internal|professional|analytics|data-science|tourism|resume(?:-[^/]*)?)(?:\/|$)/i;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function plainText(value) {
  if (typeof value !== 'string') return '';
  let text = value;
  for (let pass = 0; pass < 2; pass += 1) {
    text = text.replace(/<!--[^]*?-->/g, '')
      .replace(/<(script|style|template)\b[^>]*>[^]*?<\/\1\s*>/gi, '')
      .replace(/<[^>]*>/g, ' ')
      .replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp|ndash|mdash|rsquo|lsquo|rdquo|ldquo);/gi, (_, entity) => {
        if (entity[0] === '#') {
          const code = entity[1].toLowerCase() === 'x' ? parseInt(entity.slice(2), 16) : Number(entity.slice(1));
          return code > 0 && code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff) ? String.fromCodePoint(code) : '';
        }
        return { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', rsquo: '’', lsquo: '‘', rdquo: '”', ldquo: '“' }[entity.toLowerCase()];
      });
  }
  return text.replace(/<(script|style|template)\b[^>]*>[^]*?<\/\1\s*>/gi, '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '').replace(/\s+/g, ' ').trim();
}

function publicHttpsUrl(value, origin = SITE_ORIGIN) {
  if (typeof value !== 'string') return '';
  const candidate = value.trim();
  if (!candidate || /[\u0000-\u0020\u007f\\]/.test(candidate) || candidate.startsWith('//')) return '';
  try {
    const url = new URL(candidate, `${origin}/`);
    if (url.protocol !== 'https:' || url.username || url.password || (url.port && url.port !== '443')) return '';
    const hostname = url.hostname.toLowerCase();
    if (net.isIP(hostname) || hostname.startsWith('[') || !hostname.includes('.') || /(?:^|\.)(localhost|local|internal|test|invalid)$/.test(hostname)) return '';
    if (SITE_HOSTS.has(hostname) && PRIVATE_PATH.test(decodeURIComponent(url.pathname))) return '';
    for (const key of url.searchParams.keys()) {
      if (/(?:token|secret|password|signature|credential|authorization|api[-_]?key)/i.test(key)) return '';
    }
    return url.href;
  } catch {
    return '';
  }
}

function imageUrl(value, root, origin, preferPreview = false) {
  const safeUrl = publicHttpsUrl(value, origin);
  if (!safeUrl) return '';
  const url = new URL(safeUrl);
  if (!SITE_HOSTS.has(url.hostname) || !/^\/img\/(?:projects|tools|games|hero|brand)\/[a-z\d_./-]+\.(?:png|jpe?g|webp|avif)$/i.test(url.pathname)) return '';
  let relative = url.pathname.slice(1);
  if (preferPreview) {
    const variant = relative.replace(/\.[^.]+$/, '-640.webp');
    if (fs.existsSync(path.join(root, variant))) relative = variant;
  }
  const absolute = path.resolve(root, relative);
  if (!absolute.startsWith(`${path.resolve(root)}${path.sep}`) || !fs.existsSync(absolute) || !fs.statSync(absolute).isFile()) return '';
  const version = crypto.createHash('sha256').update(fs.readFileSync(absolute)).digest('hex').slice(0, 12);
  return `${origin}/${relative}?v=${version}`;
}

function isPublic(record) {
  if (!record || typeof record !== 'object') return false;
  return record.published !== false && record.enabled !== false && !record.hidden && !record.noindex && !record.private && !record.internal
    && ['public', ''].includes(String(record.visibility || '').toLowerCase())
    && !['draft', 'private', 'archived', 'unpublished'].includes(String(record.status || '').toLowerCase())
    && (!record.audience || record.audience === 'personal');
}

function ordered(records, key = 'id') {
  return (Array.isArray(records) ? records : []).filter(isPublic).slice().sort((a, b) =>
    (Number(a.order) || 0) - (Number(b.order) || 0) || String(a[key] || '').localeCompare(String(b[key] || ''), 'en'));
}

function identifier(value) {
  return typeof value === 'string' && /^[a-z\d][a-z\d_-]*$/i.test(value) ? value : '';
}

function texts(values) {
  return (Array.isArray(values) ? values : []).map(plainText).filter(Boolean);
}

function monthYear(value) {
  const match = /^(\d{4})-(0[1-9]|1[0-2])(?:-\d{2})?$/.exec(String(value || ''));
  return match ? `${MONTHS[Number(match[2]) - 1]} ${match[1]}` : plainText(value);
}

function createMobileContent(content, options = {}) {
  const root = options.root || REPOSITORY_ROOT;
  const settings = content.site?.settings || {};
  const configuredOrigin = publicHttpsUrl(settings.siteOrigin);
  const origin = configuredOrigin && SITE_HOSTS.has(new URL(configuredOrigin).hostname) ? new URL(configuredOrigin).origin : SITE_ORIGIN;
  const personal = (content.audiences || []).find((audience) => audience.key === 'personal') || {};
  const home = personal.page || {};
  const about = (home.sections || []).filter((section) => section.enabled !== false)
    .flatMap((section) => section.props?.categories || []).find((category) => category.id === 'about') || {};
  const timeline = (about.timeline?.items || []).filter(isPublic).slice()
    .sort((a, b) => String(b.date || '').localeCompare(String(a.date || ''), 'en'));
  const entries = (type, withUrl) => timeline.filter((entry) => entry.type === type).map((entry) => ({
    title: plainText(entry.title),
    organization: plainText(entry.subtitle || entry.issuer),
    date: `${monthYear(entry.date)}${entry.ongoing ? '–present' : entry.endDate ? `–${monthYear(entry.endDate)}` : ''}`,
    ...(withUrl ? { url: publicHttpsUrl(entry.href, origin) } : {})
  }));
  const toolsPage = (content.pages || []).find((page) => page.id === 'tools') || content.pagesById?.tools || {};
  const gamesPage = (content.pages || []).find((page) => page.id === 'games') || content.pagesById?.games || {};
  const categoryTitles = new Map((toolsPage.categories || []).map((category) => [category.id, plainText(category.title)]));
  const libraryItems = (records, type) => ordered(records, type === 'tools' ? 'slug' : 'id').map((entry) => ({
    id: identifier(entry.slug || entry.id),
    title: plainText(entry.title),
    summary: plainText(entry.summary),
    iconUrl: imageUrl(entry.iconImage, root, origin),
    url: publicHttpsUrl(entry.href, origin),
    category: type === 'tools' ? categoryTitles.get(entry.categoryId) || 'Tools' : plainText(entry.tags?.[0]) || 'Games'
  })).filter((entry) => entry.id && entry.title && entry.url && new URL(entry.url).origin === origin && new URL(entry.url).pathname.startsWith(`/${type}/`));
  const email = String(settings.email || '').trim();
  const payload = {
    schemaVersion: 1,
    site: {
      name: plainText(settings.siteName || settings.ownerName),
      description: plainText(home.description),
      url: `${origin}/`,
      email: /^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(email) ? email : '',
      githubUrl: (settings.sameAs || []).map((value) => publicHttpsUrl(value, origin)).find((value) => value && new URL(value).hostname === 'github.com') || '',
      privacyUrl: `${origin}/privacy`
    },
    about: {
      greeting: plainText(about.title),
      location: plainText(about.context),
      intro: plainText(about.lead),
      portraitUrl: imageUrl(about.profile?.image || settings.profileImage, root, origin),
      interests: (about.aboutStory?.connections || []).filter(isPublic).map((interest) => ({
        title: plainText(interest.title), body: plainText(interest.description), imageUrl: imageUrl(interest.image, root, origin)
      })),
      experience: entries('job', false),
      education: entries('degree', true),
      credentials: entries('certification', true)
    },
    projects: ordered(content.projects).filter((project) => identifier(project.id) && plainText(project.title)).map((project) => {
      const demoUrl = publicHttpsUrl(project.embed?.type === 'iframe' ? project.embed.url : project.embed?.type === 'tableau' ? project.embed.base : '', origin);
      return {
        id: project.id,
        title: plainText(project.title),
        summary: plainText(project.subtitle || project.personalStory?.why || project.problem),
        imageUrl: imageUrl(project.image, root, origin, true),
        iconUrl: imageUrl(project.iconImage, root, origin),
        url: `${origin}/portfolio/${encodeURIComponent(project.id)}`,
        tags: [...new Set([...texts(project.tools), ...texts(project.concepts)])],
        situation: plainText(project.problem),
        task: plainText(project.task),
        actions: texts(project.actions),
        results: texts(project.results),
        resources: (project.resources || []).filter(isPublic).map((resource) => ({ label: plainText(resource.label), url: publicHttpsUrl(resource.url, origin) })).filter((resource) => resource.label && resource.url),
        ...(demoUrl ? { demoUrl } : {})
      };
    }),
    tools: isPublic(toolsPage) ? libraryItems(content.tools, 'tools') : [],
    games: isPublic(gamesPage) ? libraryItems(gamesPage.games, 'games') : []
  };
  const revision = crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
  return { schemaVersion: payload.schemaVersion, revision, ...payload };
}

async function generateMobileContent(options = {}) {
  const root = options.root || REPOSITORY_ROOT;
  const content = options.content || await loadSiteContentAsync(root);
  const catalog = createMobileContent(content, { root });
  const outputPath = options.outputPath || path.join(root, CATALOG_PATH);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(catalog, null, 2)}\n`, 'utf8');
  return { catalog, outputPath };
}

if (require.main === module) {
  generateMobileContent().then(({ catalog }) => {
    process.stdout.write(`Native app catalog: ${catalog.projects.length} projects, ${catalog.tools.length} tools, ${catalog.games.length} games (${catalog.revision.slice(0, 12)})\n`);
  }).catch((error) => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exitCode = 1;
  });
}

module.exports = { CATALOG_PATH, createMobileContent, generateMobileContent, plainText, publicHttpsUrl };
