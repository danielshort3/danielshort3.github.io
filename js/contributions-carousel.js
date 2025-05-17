/* contributions-carousel.js
 * Wrap each .docs-grid in a horizontal scroll-snap carousel.
 * Mobile (≤768 px) keeps the stacked grid – we only apply the
 * carousel class at wider viewports or on resize.              */

(() => {
  "use strict";

  const MQ = window.matchMedia("(min-width:769px)");

  /* helper toggles the carousel class */
  function applyCarousel(enable){
    document.querySelectorAll(".docs-grid").forEach(g => {
      g.classList.toggle("docs-carousel", enable);
    });
  }

  /* run on load + viewport change */
  document.addEventListener("DOMContentLoaded", () => {
    applyCarousel(MQ.matches);
    MQ.addEventListener("change", e => applyCarousel(e.matches));
  });
})();

// Enable click-and-drag scrolling for .docs-carousel (no auto scroll)
document.addEventListener('DOMContentLoaded', () => {
  const carousels = document.querySelectorAll('.docs-carousel');

  carousels.forEach(carousel => {
    let isDown = false;
    let startX;
    let scrollLeft;

    carousel.addEventListener('mousedown', e => {
      isDown = true;
      carousel.classList.add('dragging');
      startX = e.pageX - carousel.offsetLeft;
      scrollLeft = carousel.scrollLeft;
    });

    carousel.addEventListener('mouseleave', () => {
      isDown = false;
      carousel.classList.remove('dragging');
    });

    carousel.addEventListener('mouseup', () => {
      isDown = false;
      carousel.classList.remove('dragging');
    });

    carousel.addEventListener('mousemove', e => {
      if (!isDown) return;
      e.preventDefault();
      const x = e.pageX - carousel.offsetLeft;
      const walk = (x - startX) * 1.5; // multiplier for scroll speed
      carousel.scrollLeft = scrollLeft - walk;
    });

    // Also support touch dragging
    carousel.addEventListener('touchstart', e => {
      startX = e.touches[0].pageX;
      scrollLeft = carousel.scrollLeft;
    });

    carousel.addEventListener('touchmove', e => {
      const x = e.touches[0].pageX;
      const walk = (x - startX) * 1.5;
      carousel.scrollLeft = scrollLeft - walk;
    });
  });
});
