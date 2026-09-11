(function (scope) {
  'use strict';

  // Catalog artwork keeps its original PNG as a native <picture> fallback.
  // Variants share the source fingerprint so CMS generation can precede encoding.
  const ICON_PATH = /^(\/?img\/(?:projects|tools|games)\/icons\/[a-z0-9_-]+)\.png([?#].*)?$/i;

  function webpSource(source) {
    const match = String(source || '').match(ICON_PATH);
    return match ? `${match[1]}.webp${match[2] || ''}` : '';
  }

  function render(imageMarkup) {
    const markup = String(imageMarkup || '');
    const match = markup.match(/^<img\b[^>]*\ssrc="([^"]+)"[^>]*>$/i);
    const source = match && webpSource(match[1]);
    return source ? `<picture class="catalog-icon"><source type="image/webp" srcset="${source}">${markup}</picture>` : markup;
  }

  function create(image, source) {
    const optimized = webpSource(source);
    if (!optimized) {
      image.setAttribute('src', source);
      return image;
    }
    const picture = image.ownerDocument.createElement('picture');
    picture.className = 'catalog-icon';
    const candidate = image.ownerDocument.createElement('source');
    candidate.type = 'image/webp';
    candidate.srcset = optimized;
    picture.append(candidate, image);
    // Assign the fallback after the source is present to avoid an eager PNG fetch.
    image.setAttribute('src', source);
    return picture;
  }

  const api = Object.freeze({ webpSource, render, create });
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') scope.SiteCatalogIcons = api;
})(typeof window === 'undefined' ? globalThis : window);
