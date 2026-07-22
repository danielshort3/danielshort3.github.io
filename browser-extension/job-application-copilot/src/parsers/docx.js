export const parseDocxDocument = async ({ data }) => {
  const mammothModule = await import('mammoth');
  const mammoth = mammothModule.default || mammothModule;
  const arrayBuffer = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const nodeBuffer = globalThis.Buffer;
  const input = typeof nodeBuffer?.from === 'function'
    ? { buffer: nodeBuffer.from(arrayBuffer) }
    : { arrayBuffer };
  const result = await mammoth.extractRawText(input);
  const paragraphs = result.value
    .replaceAll('\r\n', '\n')
    .replaceAll('\r', '\n')
    .split(/\n+/u)
    .map(value => value.replace(/\s+/gu, ' ').trim())
    .filter(Boolean);
  return {
    text: paragraphs.join('\n\n'),
    blocks: paragraphs.map((paragraph, index) => ({
      blockIndex: index,
      paragraph: index + 1,
      section: '',
      page: null,
      text: paragraph
    })),
    warnings: (result.messages || []).map(message => String(message.message || message))
  };
};
