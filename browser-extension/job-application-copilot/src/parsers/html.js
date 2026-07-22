import { parseTextDocument } from './text.js';

const decoder = new TextDecoder('utf-8', { fatal: false });

const decodeHtmlEntities = (value) => value.replace(
  /&(?:#(\d+)|#x([\da-f]+)|([a-z]+));/giu,
  (match, decimal, hexadecimal, named) => {
    if (decimal) return String.fromCodePoint(Number.parseInt(decimal, 10));
    if (hexadecimal) return String.fromCodePoint(Number.parseInt(hexadecimal, 16));
    const entities = {
      amp: '&',
      apos: "'",
      gt: '>',
      lt: '<',
      nbsp: ' ',
      quot: '"'
    };
    return entities[named.toLowerCase()] ?? match;
  }
);

export const htmlToPlainText = (html) => decodeHtmlEntities(String(html || ''))
  .replace(/<!--[\s\S]*?-->/gu, ' ')
  .replace(/<(?:script|style|template|noscript)\b[^>]*>[\s\S]*?<\/(?:script|style|template|noscript)>/giu, ' ')
  .replace(/<(?:br|hr)\s*\/?>/giu, '\n')
  .replace(/<\/(?:address|article|aside|blockquote|div|dl|fieldset|figure|footer|form|h[1-6]|header|li|main|nav|ol|p|pre|section|table|tr|ul)>/giu, '\n\n')
  .replace(/<[^>]+>/gu, ' ')
  .replace(/[\t\f\v ]+/gu, ' ')
  .replace(/ *\n */gu, '\n')
  .replace(/\n{3,}/gu, '\n\n')
  .trim();

export const parseHtmlDocument = async ({ data }) => {
  const html = decoder.decode(data instanceof Uint8Array ? data : new Uint8Array(data));
  return parseTextDocument({ data: new TextEncoder().encode(htmlToPlainText(html)) });
};

