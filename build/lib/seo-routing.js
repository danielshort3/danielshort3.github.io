'use strict';

const fs = require('fs');
const path = require('path');

function normalizePathname(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';

  const withLeadingSlash = raw.startsWith('/') ? raw : `/${raw}`;
  const withoutHash = withLeadingSlash.split('#')[0];
  const withoutQuery = withoutHash.split('?')[0];

  let normalized = withoutQuery.replace(/\/+$/, '') || '/';
  if (normalized !== '/' && normalized.endsWith('.html')) {
    normalized = normalized.slice(0, -5) || '/';
  }
  return normalized;
}

function isExactRouteSource(value) {
  const source = String(value || '').trim();
  if (!source) return false;
  return !/[:*()]/.test(source);
}

function loadNoindexPathnamesFromVercel(projectRoot) {
  const root = path.resolve(projectRoot || process.cwd());
  const noindexPathnames = new Set();
  const vercelConfigPath = path.join(root, 'vercel.json');

  try {
    if (!fs.existsSync(vercelConfigPath)) return noindexPathnames;
    const parsed = JSON.parse(fs.readFileSync(vercelConfigPath, 'utf8'));
    const headers = parsed && Array.isArray(parsed.headers) ? parsed.headers : [];

    headers.forEach((rule) => {
      const source = String(rule && rule.source ? rule.source : '').trim();
      if (!source || !isExactRouteSource(source)) return;

      const headerList = rule && Array.isArray(rule.headers) ? rule.headers : [];
      const hasNoindexHeader = headerList.some((entry) => {
        const key = String(entry && entry.key ? entry.key : '').trim().toLowerCase();
        if (key !== 'x-robots-tag') return false;
        const value = String(entry && entry.value ? entry.value : '').toLowerCase();
        return value.includes('noindex');
      });
      if (!hasNoindexHeader) return;

      const normalized = normalizePathname(source);
      if (normalized) noindexPathnames.add(normalized);
    });
  } catch (_) {}

  return noindexPathnames;
}

module.exports = {
  normalizePathname,
  loadNoindexPathnamesFromVercel
};
