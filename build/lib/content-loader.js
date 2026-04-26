#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const {
  loadFileSiteContent,
  sortByOrderThenId
} = require('../../api/_lib/cms-content-model');

function readJson(absPath) {
  const raw = fs.readFileSync(absPath, 'utf8');
  return JSON.parse(raw);
}

function loadJsonFile(root, relPath) {
  const absPath = path.join(root, relPath);
  if (!fs.existsSync(absPath)) return null;
  return readJson(absPath);
}

function loadJsonDirectory(root, relDirPath) {
  const absDir = path.join(root, relDirPath);
  if (!fs.existsSync(absDir)) return [];

  return fs.readdirSync(absDir)
    .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
    .sort((a, b) => a.localeCompare(b))
    .map((name) => {
      const relPath = path.join(relDirPath, name);
      const entry = loadJsonFile(root, relPath);
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new Error(`Expected JSON object in ${relPath}`);
      }
      return entry;
    });
}

function loadSiteContent(root) {
  return loadFileSiteContent(root);
}

function normalizeContentSource(value) {
  void value;
  return 'files';
}

async function loadSiteContentAsync(root, options = {}) {
  normalizeContentSource(options.source);
  return loadSiteContent(root);
}

module.exports = {
  loadJsonDirectory,
  loadJsonFile,
  loadSiteContent,
  loadSiteContentAsync,
  normalizeContentSource,
  sortByOrderThenId
};
