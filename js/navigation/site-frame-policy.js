(function (root) {
  'use strict';

  // Standard shared routes keep one bounded desktop frame. Compact layouts
  // choose their natural document flow in CSS; immersive games opt out.
  const policy = Object.freeze({
    resolveFit: (requested) => requested === 'immersive' ? 'immersive' : 'viewport'
  });
  if (typeof module === 'object' && module.exports) module.exports = policy;
  if (root) root.SiteFramePolicy = policy;
})(typeof window === 'undefined' ? null : window);
