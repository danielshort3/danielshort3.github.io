/* contributions.js - Build contributions UI components.
   Contribution data now lives in contributions-data.js */

/* ────────────────────────────────────────────────────────────
   DOM-builder
   ─────────────────────────────────────────────────────────── */
function buildContributions(){
  const root = document.getElementById('contrib-root');
  if(!root || !window.contributions) return;

  window.contributions.forEach(sec=>{
    // --- section shell -------------------------------------------------
    const section = document.createElement('section');
    section.className = 'surface-band reveal contrib-section';
    section.dataset.heading = sec.heading;

    const wrap   = document.createElement('div');
    wrap.className = 'wrapper';
    section.appendChild(wrap);

    // Heading + blurb
    wrap.insertAdjacentHTML('beforeend',
      `<h2 class="section-title">${sec.heading}</h2>
       <p class="section-desc">${sec.desc}</p>`);

    // Grid
    const grid = document.createElement('div');
    grid.className = 'grid docs-grid docs-carousel';
    wrap.appendChild(grid);

    // --- cards ---------------------------------------------------------
    sec.items.forEach(item => {
      const card = document.createElement('article');
      card.className = 'doc-card';

      card.innerHTML = `
        <div class="doc-layout">
          <h3 class="doc-title">${item.title}</h3>
          <div class="doc-footer">
            ${item.role ? `<p class="doc-role">${item.role}</p>` : ''}
            <div class="doc-links">
              ${item.pdf ? `<a href="${item.pdf}" target="_blank" rel="noopener" class="doc-link" aria-label="Open PDF" download><i class="fas fa-file-pdf" aria-hidden="true"></i></a>` : ''}
              ${item.link ? `<a href="${item.link}" target="_blank" rel="noopener" class="doc-link" aria-label="Open external link"><i class="fas fa-external-link-alt" aria-hidden="true"></i></a>` : ''}
            </div>
          </div>
        </div>
      `;

      grid.appendChild(card);
    });


    root.appendChild(section);

    /* ── NEW: 64 px surface-coloured band between sections ── */
    if (sec !== window.contributions.at(-1)){   // skip the very last block
      const gap = document.createElement('section');
      gap.className = 'contrib-gap';            // styled in styles.css
      root.appendChild(gap);
    }

  });
}

/* Collapse contributions on mobile with See More/See Less toggle.
   Desktop always shows all items and never displays the button.    */
function initContribSeeMore(){
  const mq = window.matchMedia('(max-width: 768px)');

  const teardown = btn => {
    const wrap = btn.parentElement;
    if (btn._observer)    { btn._observer.disconnect();    btn._observer = null; }
    if (btn._nextObs)     { btn._nextObs.disconnect();     btn._nextObs = null; }
    if (btn._onScroll)    { window.removeEventListener('scroll', btn._onScroll); btn._onScroll = null; }
    btn._sectionVisible = true;
    btn._nextVisible    = false;
    btn._topOverlap     = false;
    btn._lockHide       = false;
    btn._lastY          = 0;
    if (wrap) wrap.classList.remove('sticky', 'fade-out');
  };

  const setup = btn => {
    const section = btn.closest('.contrib-section');
    const grid = section && section.querySelector('.docs-grid');
    const wrap = btn.parentElement;
    if (!section || !grid || !wrap) return;
    if (grid.scrollHeight <= window.innerHeight || btn.dataset.expanded !== 'true') { teardown(btn); return; }

    wrap.classList.add('sticky');

    const updateFade = () => {
      const y         = window.scrollY;
      const goingDown = y > (btn._lastY || 0);
      btn._lastY      = y;

      // 1) Is the first card overlapping the floating-button wrapper?
      if (btn._firstCard) {
        const cRect = btn._firstCard.getBoundingClientRect();
        const bRect = wrap.getBoundingClientRect();
        btn._topOverlap = cRect.bottom > bRect.top && cRect.top < bRect.bottom;
      } else {
        btn._topOverlap = false;
      }

      /* ── fade-out trigger: scrolling UP into overlap ─────────── */
      if (!goingDown && btn._topOverlap) {
        btn._lockHide = true;          // stay hidden until we cross back down
      }

      /* ── fade-in trigger: scrolling DOWN past the same line ──── */
      if (
        goingDown &&
        btn._lockHide &&
        !btn._topOverlap &&                               // no longer overlapping
        btn._firstCard.getBoundingClientRect().bottom     // card bottom is now
          <= wrap.getBoundingClientRect().top             // above wrapper top
      ){
        btn._lockHide = false;       // release the lock → fade back in
      }

      /* Also clear lock near grid bottom (unchanged) */
      if (goingDown) {
        const gRect = grid.getBoundingClientRect();
        if (gRect.bottom <= window.innerHeight + 50) btn._lockHide = false;
      }

      const hide =
        btn._lockHide ||
        !btn._sectionVisible ||
        btn._nextVisible ||
        btn._topOverlap;

      wrap.classList.toggle('fade-out', hide);
    };

    if (!btn._observer) {
      btn._sectionVisible = true;
      const io = new IntersectionObserver(entries => {
        btn._sectionVisible = entries[0].isIntersecting;
        updateFade();
      });
      io.observe(section);
      btn._observer = io;
    }

    if (!btn._nextObs) {
      const nextWrap = section.nextElementSibling?.querySelector('.see-more-wrap');
      if (nextWrap) {
        btn._nextVisible = false;
        const io2 = new IntersectionObserver(entries => {
          btn._nextVisible = entries[0].isIntersecting;
          updateFade();
        });
        io2.observe(nextWrap);
        btn._nextObs = io2;
      }
    }

    if (!btn._onScroll) {
      btn._firstCard = grid.firstElementChild;
      btn._lastY = window.scrollY;
      btn._lockHide = false;
      btn._onScroll = () => updateFade();
      window.addEventListener('scroll', btn._onScroll, { passive: true });
      updateFade();
    } else {
      btn._firstCard = grid.firstElementChild;
      updateFade();
    }
  };

  const updateFloat = btn => {
    if (btn.dataset.expanded === 'true') setup(btn); else teardown(btn);
  };

  const apply = enable => {
    document.querySelectorAll('.contrib-section').forEach(section => {
      const grid  = section.querySelector('.docs-grid');
      if(!grid) return;
      const cards = [...grid.children];
      if(cards.length <= 1) return;

      // find existing button/wrapper if present
      let btn = section.querySelector('.see-more-btn');
      let wrap = section.querySelector('.see-more-wrap');

      if(enable){
        if(!btn){
          wrap = document.createElement('div');
          wrap.className = 'see-more-wrap';
          wrap.style.textAlign = 'center';
          wrap.style.marginTop = '12px';
          btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'btn-primary see-more-btn';
          btn.textContent = 'See More';
          btn.dataset.expanded = 'false';
          wrap.appendChild(btn);
          section.querySelector('.wrapper').appendChild(wrap);

          btn.addEventListener('click', () => {
            const expanded = btn.dataset.expanded === 'true';
            btn.dataset.expanded = expanded ? 'false' : 'true';
            const isExpanded = btn.dataset.expanded === 'true';
            btn.textContent = isExpanded ? 'See Less' : 'See More';
            cards.slice(1).forEach(c => c.classList.toggle('hide', !isExpanded));
            if (window.gaEvent) {
              window.gaEvent('contrib_see_more_toggle', {
                expanded: isExpanded,
                section: section.dataset.heading || ''
              });
            }
            updateFloat(btn);
          });
        }
        const isExpanded = btn.dataset.expanded === 'true';
        cards.slice(1).forEach(c => c.classList.toggle('hide', !isExpanded));
        updateFloat(btn);
      } else {
        // remove button and show all cards
        if(wrap) { teardown(btn); wrap.remove(); }
        cards.slice(1).forEach(c => c.classList.remove('hide'));
      }
    });
  };

  apply(mq.matches);
  mq.addEventListener('change', e => apply(e.matches));
}

function initContributions(){
  buildContributions();
  initContribSeeMore();
}

document.addEventListener('DOMContentLoaded', initContributions);
