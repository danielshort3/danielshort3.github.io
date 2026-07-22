const joinPageItems = (items) => {
  const lines = [];
  let current = '';
  for (const item of items) {
    const text = typeof item?.str === 'string' ? item.str : '';
    if (!text) continue;
    current += current && !/^\s/u.test(text) ? ` ${text}` : text;
    if (item.hasEOL) {
      if (current.trim()) lines.push(current.trim());
      current = '';
    }
  }
  if (current.trim()) lines.push(current.trim());
  return lines.join('\n');
};

export const parsePdfDocument = async ({ data }) => {
  const [{ getDocument }, { WorkerMessageHandler }] = await Promise.all([
    import('pdfjs-dist/legacy/build/pdf.min.mjs'),
    import('pdfjs-dist/legacy/build/pdf.worker.min.mjs')
  ]);
  globalThis.pdfjsWorker ||= { WorkerMessageHandler };
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const loadingTask = getDocument({ data: bytes, useWorkerFetch: false, useSystemFonts: true });
  const pdf = await loadingTask.promise;
  const blocks = [];
  try {
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      try {
        const content = await page.getTextContent();
        const pageText = joinPageItems(content.items).trim();
        if (pageText) {
          const paragraphs = pageText
            .split(/\n{2,}|\n(?=[A-Z][^\n]{0,80}$)/u)
            .map(value => value.replace(/\s+/gu, ' ').trim())
            .filter(Boolean);
          paragraphs.forEach((paragraph, index) => {
            blocks.push({
              blockIndex: blocks.length,
              paragraph: index + 1,
              section: '',
              page: pageNumber,
              text: paragraph
            });
          });
        }
      } finally {
        page.cleanup();
      }
    }
  } finally {
    await loadingTask.destroy();
  }
  if (!blocks.length) {
    throw new Error('Scanned PDF has no extractable text; use OCR or paste the text.');
  }
  return {
    text: blocks.map(block => block.text).join('\n\n'),
    blocks,
    warnings: []
  };
};
