import assert from 'node:assert/strict';
import test from 'node:test';
import { strToU8, zipSync } from 'fflate';
import { parseDocument } from '../src/parsers/document-parser.js';
import { createDocumentVaultRecord } from '../src/parsers/import-record.js';
import { chunkDocument } from '../src/rag/chunker.js';
import { buildEvidencePack, hybridRetrieve } from '../src/rag/retrieval.js';
import { ManualResearchProvider } from '../src/research/manual-research-provider.js';
import { base64ToBytes, sha256Base64Url } from '../src/vault/crypto.js';

const encoder = new TextEncoder();

const makeDocx = text => zipSync({
  '[Content_Types].xml': strToU8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>`),
  '_rels/.rels': strToU8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`),
  'word/document.xml': strToU8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>${text}</w:t></w:r></w:p></w:body></w:document>`)
});

const makePdf = (text = '') => {
  const escaped = text.replaceAll('\\', '\\\\').replaceAll('(', '\\(').replaceAll(')', '\\)');
  const stream = text ? `BT /F1 12 Tf 72 720 Td (${escaped}) Tj ET` : 'q Q';
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    `<< /Length ${Buffer.byteLength(stream)} >>\nstream\n${stream}\nendstream`
  ];
  let body = '%PDF-1.4\n';
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(Buffer.byteLength(body));
    body += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });
  const xrefOffset = Buffer.byteLength(body);
  body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach(offset => { body += `${String(offset).padStart(10, '0')} 00000 n \n`; });
  body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return new Uint8Array(Buffer.from(body, 'ascii'));
};

test('bundled parsers extract TXT, HTML, DOCX, and PDF without network services', async () => {
  const text = await parseDocument({ name: 'resume.txt', type: 'text/plain', data: encoder.encode('First paragraph.\n\nSecond paragraph.') });
  assert.equal(text.blocks.length, 2);
  assert.equal(text.document.mimeType, 'text/plain');

  const html = await parseDocument({ name: 'profile.html', type: 'text/html', data: encoder.encode('<h1>Profile</h1><script>ignore()</script><p>Local analytics</p>') });
  assert.match(html.text, /Profile/u);
  assert.doesNotMatch(html.text, /ignore/u);

  const docxBytes = makeDocx('Grounded DOCX evidence');
  const docx = await parseDocument({ name: 'resume.docx', type: '', data: docxBytes });
  assert.match(docx.text, /Grounded DOCX evidence/u);
  assert.equal(docx.document.mimeType, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');

  const pdfBytes = makePdf('Grounded PDF evidence');
  const expectedPdfHash = await sha256Base64Url(pdfBytes);
  const pdf = await parseDocument({ name: 'resume.pdf', type: 'application/pdf', data: pdfBytes });
  assert.match(pdf.text, /Grounded PDF evidence/u);
  assert.ok(pdfBytes.byteLength > 0, 'PDF import must not detach the caller-owned bytes');
  assert.equal(pdf.document.size, pdfBytes.byteLength);
  assert.equal(pdf.document.sha256, expectedPdfHash);
  assert.equal(pdf.document.id, `doc:${expectedPdfHash.slice(0, 24)}`);
  assert.notEqual(expectedPdfHash, await sha256Base64Url(new Uint8Array()));
});

test('parsers reject legacy DOC, blank inputs, and PDFs without extractable text', async () => {
  await assert.rejects(parseDocument({ name: 'resume.doc', type: 'application/msword', data: encoder.encode('legacy') }), /convert.*DOCX or PDF/iu);
  await assert.rejects(parseDocument({ name: 'blank.txt', type: 'text/plain', data: encoder.encode('   ') }), /no extractable text/iu);
  await assert.rejects(parseDocument({ name: 'scan.pdf', type: 'application/pdf', data: makePdf() }), /Scanned PDF has no extractable text/iu);
});

test('document vault records retain original bytes only when explicitly requested', async () => {
  const bytes = encoder.encode('Candidate evidence');
  const parsed = await parseDocument({ name: 'resume.txt', type: 'text/plain', data: bytes });
  const transient = createDocumentVaultRecord(parsed);
  assert.equal(transient.originalBytesBase64, null);
  const retained = createDocumentVaultRecord(parsed, { retainOriginal: true, originalBytes: bytes });
  assert.deepEqual(base64ToBytes(retained.originalBytesBase64), bytes);
});

test('chunking preserves page boundaries and retrieval isolates application evidence', async () => {
  const record = {
    document: { id: 'doc:resume', version: 1, applicationId: 'app-1', sourceRole: 'candidate_evidence' },
    blocks: [{ blockIndex: 0, paragraph: 1, section: '', page: 1, text: 'Python analytics dashboards and SQL reporting.' }, {
      blockIndex: 1, paragraph: 1, section: '', page: 2, text: 'Tourism forecasting and stakeholder presentations.'
    }]
  };
  const chunks = await chunkDocument(record, { maxTokens: 50, overlapTokens: 10 });
  assert.equal(chunks.length, 2);
  assert.deepEqual(chunks.map(chunk => chunk.locator.pageStart), [1, 2]);
  assert.deepEqual(await chunkDocument(record, { maxTokens: 50, overlapTokens: 10 }), chunks);

  const other = { ...chunks[0], id: 'chunk:other', applicationId: 'app-2' };
  const results = hybridRetrieve({ queryText: 'Python SQL analytics', chunks: [...chunks, other], filters: { applicationId: 'app-1' } });
  assert.ok(results.length >= 1);
  assert.ok(results.every(chunk => chunk.applicationId === 'app-1'));
  const pack = buildEvidencePack([{ fieldId: 'field-one', results }]);
  assert.equal(pack.citations[0].citationId, 'c1');
  assert.deepEqual(pack.byField['field-one'], pack.citations.map(citation => citation.citationId));
});

test('retrieval excludes role-only matches and preserves at least one result per field', () => {
  const baseChunk = {
    documentId: 'doc',
    documentVersion: 1,
    applicationId: 'app',
    sourceRole: 'candidate_evidence',
    text: 'Unrelated material',
    terms: ['unrelated', 'material'],
    locator: {},
    quoteHash: 'hash'
  };
  assert.deepEqual(hybridRetrieve({
    queryText: 'salary compensation',
    chunks: [{ ...baseChunk, id: 'irrelevant' }],
    filters: { applicationId: 'app' }
  }), []);

  const result = (id, score) => ({
    ...baseChunk,
    id,
    text: id,
    terms: [id],
    retrieval: { score, lexicalScore: score, vectorScore: 0, roleScore: 1 }
  });
  const pack = buildEvidencePack([{
    fieldId: 'field-a',
    results: [result('a1', 1), result('a2', 0.9)]
  }, {
    fieldId: 'field-b',
    results: [result('b1', 0.1)]
  }], { maxChunks: 2 });
  assert.equal(pack.citations.length, 2);
  assert.equal(pack.byField['field-a'].length, 1);
  assert.equal(pack.byField['field-b'].length, 1);
});
test('manual research accepts pasted notes without a title or URL', async () => {
  const provider = new ManualResearchProvider();
  const snapshot = await provider.createSnapshot({ text: 'The company emphasizes transparent analytics.' });
  assert.equal(snapshot.document.filename, 'Pasted research note');
  assert.equal(snapshot.document.sourceUrl, null);
  assert.equal(snapshot.document.sourceRole, 'company_context');
  await assert.rejects(provider.createSnapshot({ text: 'Research', url: 'file:///tmp/research' }), /HTTP or HTTPS/u);
});
