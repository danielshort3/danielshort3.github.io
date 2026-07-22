const decoder = new TextDecoder('utf-8', { fatal: false });

const splitParagraphs = (text) => text
  .replaceAll('\r\n', '\n')
  .replaceAll('\r', '\n')
  .split(/\n{2,}/u)
  .map(value => value.replace(/\s+/gu, ' ').trim())
  .filter(Boolean);

export const parseTextDocument = async ({ data }) => {
  const text = decoder.decode(data instanceof Uint8Array ? data : new Uint8Array(data));
  const paragraphs = splitParagraphs(text);
  return {
    text: paragraphs.join('\n\n'),
    blocks: paragraphs.map((paragraph, index) => ({
      blockIndex: index,
      paragraph: index + 1,
      section: '',
      page: null,
      text: paragraph
    })),
    warnings: []
  };
};

