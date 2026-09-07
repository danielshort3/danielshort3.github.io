'use strict';

// These project presentations and demos are served entirely as site assets.
// Keep the public document allowlist explicit: documents/ also contains private
// and retired files that must never be copied wholesale.
const BROWSER_PROJECT_IDS = Object.freeze([
  'targetEmptyPackage', 'retailStore', 'covidAnalysis', 'pizza',
  'babynames', 'sheetMusicUpscale', 'deliveryTip'
]);

const BROWSER_PROJECT_DOCUMENTS = Object.freeze([
  'documents/Project_1.pdf',
  'documents/Project_1.xlsx',
  'documents/Project_2.pdf',
  'documents/Project_2.ipynb',
  'documents/Project_7.pdf',
  'documents/Project_7.xlsx',
  'documents/Project_10_pdf.zip',
  'documents/Project_10.zip',
  'documents/Project_11.pdf',
  'documents/Project_11.xlsx',
  'documents/Project_12.pdf',
  'documents/Project_12.ipynb'
]);

const BROWSER_PROJECT_DEMOS = Object.freeze([
  'target-empty-package', 'retail-loss-sales', 'covid-outbreak',
  'pizza-tips', 'baby-names'
]);

module.exports = { BROWSER_PROJECT_IDS, BROWSER_PROJECT_DOCUMENTS, BROWSER_PROJECT_DEMOS };
