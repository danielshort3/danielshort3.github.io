/* ===================================================================
   File: animations.js
   Purpose: Scroll reveal, chevron hints, ticker and viewport helpers
=================================================================== */
(() => {
  'use strict';
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const on = (n,e,f,o) => n && n.addEventListener(e,f,o);
  document.addEventListener('DOMContentLoaded', () => {
    initReveal();
    setScrollbarVar();
    setViewportVar();
    initChevronHint();
    initCertTicker();
    initScrollProgress();
    initHeroParallax();
    initBackToTop();
    initHeroAmbient();
  });
  function initReveal(){
    const io = new IntersectionObserver((ents,o)=>{
      ents.forEach(e=>{ if(e.isIntersecting){e.target.classList.add('active');o.unobserve(e.target);} });
    },{threshold:.15});
    $$('.reveal:not(.no-reveal)').forEach(el=>io.observe(el));
  }
  const setScrollbarVar = () => {
    const sb = window.innerWidth - document.documentElement.clientWidth;
    document.documentElement.style.setProperty('--scrollbar', `${sb}px`);
  };
  const setViewportVar = () => {
    document.documentElement.style.setProperty('--vh', `${window.innerHeight}px`);
  };
  window.addEventListener('resize', setViewportVar);
  window.addEventListener('orientationchange', setViewportVar);

  function initScrollProgress(){
    const update = () => {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const max = (document.documentElement.scrollHeight - window.innerHeight) || 1;
      const pct = Math.min(100, Math.max(0, (scrollTop / max) * 100));
      document.documentElement.style.setProperty('--scroll-progress', pct.toFixed(2));
    };
    update();
    window.addEventListener('scroll', update, { passive:true });
    window.addEventListener('resize', update);
  }

  function initHeroParallax(){
    const hero = document.querySelector('.hero');
    if (!hero) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return;
    const onScroll = () => {
      const rect = hero.getBoundingClientRect();
      // Only when hero is in view
      if (rect.bottom <= 0 || rect.top >= window.innerHeight) return;
      const progress = 1 - Math.min(1, Math.max(0, rect.top / window.innerHeight));
      const offset = Math.round(progress * -18); // subtle parallax (~18px)
      document.documentElement.style.setProperty('--hero-parallax', offset + 'px');
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive:true });
    window.addEventListener('resize', onScroll);
  }

  function initBackToTop(){
    const btn = document.createElement('button');
    btn.className = 'back-to-top';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Back to top');
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 5l-7 7m7-7l7 7"/></svg>';
    document.body.appendChild(btn);
    const update = () => {
      const half = window.innerHeight * 0.5;
      btn.classList.toggle('show', window.scrollY > half);
    };
    const scrollToTop = () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    btn.addEventListener('click', scrollToTop);
    update();
    window.addEventListener('scroll', update, { passive:true });
    window.addEventListener('resize', update);
  }

  // Subtle ambient light following the mouse inside the hero
  function initHeroAmbient(){
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const fine = window.matchMedia('(pointer: fine)').matches;
    if (reduce || !fine) return;
    const hero = document.querySelector('.hero');
    if (!hero) return;
    let raf = null, px = 50, py = 35;
    const update = () => {
      hero.style.setProperty('--mx', px.toFixed(1) + '%');
      hero.style.setProperty('--my', py.toFixed(1) + '%');
      raf = null;
    };
    hero.addEventListener('mousemove', (e) => {
      const rect = hero.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      px = Math.min(100, Math.max(0, x));
      py = Math.min(100, Math.max(0, y));
      if (!raf) raf = requestAnimationFrame(update);
    });
  }
  function initChevronHint(){
    const chevrons = $$('.chevron-hint');
    const hero = document.querySelector('.hero');
    if(!chevrons.length || !hero) return;
    const MIN_RATIO = 0.9;
    const update = v => chevrons.forEach(c=>c.classList.toggle('fade', !v));
    const observer = new IntersectionObserver(e=>{update(e[0].intersectionRatio>=MIN_RATIO);},{threshold:MIN_RATIO});
    observer.observe(hero);
    const rect = hero.getBoundingClientRect();
    const ratio = (Math.min(rect.bottom,window.innerHeight)-Math.max(rect.top,0))/rect.height;
    update(ratio>=MIN_RATIO);
    const scrollToNext = ind => {
      let next = ind.closest('section')?.nextElementSibling;
      while(next && next.tagName !== 'SECTION') next = next.nextElementSibling;
      if(next){
        const offset = parseFloat(getComputedStyle(document.documentElement)
                        .getPropertyValue('--nav-height')) || 0;
        const top = next.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({top, behavior:'smooth'});
      }else{
        window.scrollBy({top:window.innerHeight*0.8,behavior:'smooth'});
      }
    };
    $$('.chevron-hint,.scroll-indicator').forEach(ind=>{
      on(ind,'click',()=>scrollToNext(ind));
    });
  }
  function initCertTicker(){
    const track = document.querySelector('.cert-track');
    if(!track) return;
    const GAP=160, BASE=90, DRAG=15;
    let v=BASE, target=BASE, bandW, stripW=0;
    let down=false,moved=false; let sx=0,lx=0; let paused=false; let cancelClk=false;
    const originals=[...track.children];
    const tiles=[...originals];
    const setPos=(el,x)=>{el.dataset.x=x;el.style.setProperty("--ticker-x", `${x}px`);};
    const fill=()=>{
      tiles.slice(originals.length).forEach(el=>el.remove());
      tiles.length=originals.length; stripW=0;
      originals.forEach((t,i)=>{const w=t.offsetWidth+GAP;Object.assign(t.dataset,{w,orig:i});setPos(t,stripW);stripW+=w;});
      const baseW=stripW; bandW=track.getBoundingClientRect().width;
      while(stripW < baseW + bandW){
        originals.forEach(src=>{const clone=src.cloneNode(true);track.appendChild(clone);const w=clone.offsetWidth+GAP;Object.assign(clone.dataset,{w,orig:+src.dataset.orig});setPos(clone,stripW);stripW+=w;tiles.push(clone);});
      }
    };
    window.addEventListener('load', fill);
    window.addEventListener('resize', fill);
    const isTouch=matchMedia('(hover: none) and (pointer: coarse)').matches;
    if(!isTouch){
      track.addEventListener('mouseenter',()=>{target=0;});
      track.addEventListener('mouseleave',()=>{target=BASE;});
    }
    const move=dx=>tiles.forEach(t=>{let x=+t.dataset.x+dx,w=+t.dataset.w;x=dx<0?(x+w<=0?x+stripW:x):(x>=bandW?x-stripW:x);setPos(t,x);});
    const end=()=>{down=moved=false;paused=false;track.classList.remove('dragging');['pointermove','pointerup','pointercancel'].forEach(e=>window.removeEventListener(e,end));};
    track.addEventListener('pointerdown',e=>{if(e.button) return;down=true;moved=false;paused=false;sx=lx=e.clientX;const onMove=e=>{if(!down) return;const dxT=e.clientX-sx;if(!moved && Math.abs(dxT)>=DRAG){moved=true;paused=true;track.classList.add('dragging');}if(moved){move(e.clientX-lx);lx=e.clientX;e.preventDefault();}};window.addEventListener('pointermove',onMove);window.addEventListener('pointerup',end,{once:true});window.addEventListener('pointercancel',end,{once:true});});
    track.addEventListener('click',e=>{if(cancelClk){e.preventDefault();e.stopImmediatePropagation();}cancelClk=false;},true);
    let last=performance.now();
    const reset=()=>{last=performance.now();};
    document.addEventListener('visibilitychange',reset);window.addEventListener('focus',reset);
    (function loop(now){const dt=Math.min((now-last)/1000,0.25);last=now;v+=(target-v)*Math.min(1,dt*4);if(!paused){tiles.forEach(t=>{let x=+t.dataset.x-v*dt,w=+t.dataset.w;if(x<-w)x+=stripW;setPos(t,x);});}requestAnimationFrame(loop);})(last);
  }
})();
