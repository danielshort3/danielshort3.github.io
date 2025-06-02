/* ===================================================================
   File: animations.js
   Purpose: Scroll reveals and ticker animations
=================================================================== */
(() => {
  'use strict';
  const $$ = (sel, ctx=document) => [...ctx.querySelectorAll(sel)];
  const run = fn => typeof fn === 'function' && fn();

  function initReveal(){
    const io = new IntersectionObserver(
      (ents, o) => ents.forEach(e=>{
        if(e.isIntersecting){e.target.classList.add('active');o.unobserve(e.target);}
      }),
      { threshold:.15 }
    );
    $$('.reveal:not(.no-reveal)').forEach(el => io.observe(el));
  }

  const setScrollbarVar = () => {
    const sb = window.innerWidth - document.documentElement.clientWidth;
    document.documentElement.style.setProperty('--scrollbar', `${sb}px`);
  };

  const setViewportVar = () => {
    document.documentElement.style.setProperty('--vh', `${window.innerHeight}px`);
  };

  function initChevronHint(){
    const chevrons = $$('.chevron-hint');
    const hero     = document.querySelector('.hero');
    if(!chevrons.length || !hero) return;
    const MIN_RATIO = 0.9;
    const update = visible => {
      chevrons.forEach(c => c.classList.toggle('fade', !visible));
    };
    const observer = new IntersectionObserver(entries => {
      update(entries[0].intersectionRatio >= MIN_RATIO);
    }, { threshold: MIN_RATIO });
    observer.observe(hero);
    const rect  = hero.getBoundingClientRect();
    const ratio = (Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top,0)) / rect.height;
    update(ratio >= MIN_RATIO);
    $$('.scroll-indicator').forEach(ind => {
      ind.addEventListener('click', () => {
        const next = ind.closest('.hero')?.nextElementSibling;
        (next || window).scrollBy({ top: next ? 0 : window.innerHeight*0.8, behavior:'smooth' });
        ind.classList.add('hidden');
      });
    });
  }

  function initCertTicker(){
    const track = document.querySelector('.cert-track');
    if(!track) return;
    const GAP=160,BASE=90,DRAG=15; let v=BASE,target=BASE,bandW,stripW=0;
    let down=false,moved=false; let sx=0,lx=0; let paused=false; let cancelClk=false;
    const originals=[...track.children]; const tiles=[...originals];
    const setPos=(el,x)=>{el.dataset.x=x;el.style.transform=`translateX(${x}px)`;};
    const fill=()=>{tiles.slice(originals.length).forEach(el=>el.remove());tiles.length=originals.length;stripW=0;originals.forEach((t,i)=>{const w=t.offsetWidth+GAP;Object.assign(t.dataset,{w,orig:i});setPos(t,stripW);stripW+=w;});const baseW=stripW;bandW=track.getBoundingClientRect().width;while(stripW<baseW+bandW){originals.forEach(src=>{const clone=src.cloneNode(true);track.appendChild(clone);const w=clone.offsetWidth+GAP;Object.assign(clone.dataset,{w,orig:+src.dataset.orig});setPos(clone,stripW);stripW+=w;tiles.push(clone);});}};
    window.addEventListener('load', fill);window.addEventListener('resize', fill);
    const move=dx=>tiles.forEach(t=>{let x=+t.dataset.x+dx,w=+t.dataset.w;x=dx<0?(x+w<=0?x+stripW:x):(x>=bandW?x-stripW:x);setPos(t,x);});
    const endDrag=()=>{down=moved=false;paused=false;track.classList.remove('dragging');['pointermove','pointerup','pointercancel'].forEach(e=>window.removeEventListener(e,endDrag));};
    track.addEventListener('pointerdown',e=>{if(e.button)return;down=true;moved=false;paused=false;sx=lx=e.clientX;const onMove=e=>{if(!down)return;const dxT=e.clientX-sx;if(!moved&&Math.abs(dxT)>=DRAG){moved=true;paused=true;track.classList.add('dragging');}if(moved){move(e.clientX-lx);lx=e.clientX;e.preventDefault();}};window.addEventListener('pointermove',onMove);window.addEventListener('pointerup',endDrag,{once:true});window.addEventListener('pointercancel',endDrag,{once:true});});
    track.addEventListener('click',e=>{if(cancelClk){e.preventDefault();e.stopImmediatePropagation();}cancelClk=false;},true);
    let last=performance.now();const resetClock=()=>{last=performance.now();};document.addEventListener('visibilitychange',resetClock);window.addEventListener('focus',resetClock);(function loop(now){const dt=Math.min((now-last)/1000,0.25);last=now;v+=(target-v)*Math.min(1,dt*4);if(!paused){tiles.forEach(t=>{let x=+t.dataset.x-v*dt,w=+t.dataset.w;if(x<-w)x+=stripW;setPos(t,x);});}requestAnimationFrame(loop);})(last);
  }

  document.addEventListener('DOMContentLoaded', () => {
    initReveal();
    setScrollbarVar();
    setViewportVar();
    initChevronHint();
    initCertTicker();
  });
  window.addEventListener('resize', setViewportVar);
  window.addEventListener('orientationchange', setViewportVar);
})();
