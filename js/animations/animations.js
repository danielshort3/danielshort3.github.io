/* ===================================================================
   File: animations.js
   Purpose: Scroll reveal, chevron hints, ticker and viewport helpers
=================================================================== */
(() => {
  'use strict';
  const $$ = (s, c=document) => [...c.querySelectorAll(s)];
  const on = (n,e,f,o) => n && n.addEventListener(e,f,o);
  let revealObserver = null;
  document.addEventListener('DOMContentLoaded', () => {
    initReveal();
    setScrollbarVar();
    setViewportVar();
    initChevronHint();
    initCertTicker();
    initScrollProgress();
    initHeroParallax();
    initProjectPreviewVideos();
  });
  document.addEventListener('site:content-updated', () => {
    initReveal();
    setScrollbarVar();
    setViewportVar();
    initChevronHint();
    initCertTicker();
    initHeroParallax();
    initProjectPreviewVideos();
  });
  function initReveal(){
    const targets = $$('.reveal:not(.no-reveal):not(.active)').filter(el => el.dataset.revealObserved !== 'yes');
    if (!targets.length) return;
    if (!('IntersectionObserver' in window)) {
      targets.forEach(el => el.classList.add('active'));
      return;
    }
    if (!revealObserver) {
      revealObserver = new IntersectionObserver((ents,o)=>{
        ents.forEach(e=>{ if(e.isIntersecting){e.target.classList.add('active');o.unobserve(e.target);} });
      },{threshold:.15});
    }
    targets.forEach(el=>{
      el.dataset.revealObserved = 'yes';
      revealObserver.observe(el);
    });
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
    const root = document.documentElement;
    const body = document.body;
    let ticking = false;
    const largest = (...values) => Math.max(
      0,
      ...values.filter(value => Number.isFinite(value) && value > 0)
    );
    const getScrollTop = () => largest(
      window.scrollY,
      window.pageYOffset,
      document.scrollingElement?.scrollTop,
      root?.scrollTop,
      body?.scrollTop
    );
    const getScrollHeight = () => largest(
      document.scrollingElement?.scrollHeight,
      root?.scrollHeight,
      root?.offsetHeight,
      root?.clientHeight,
      body?.scrollHeight,
      body?.offsetHeight,
      body?.clientHeight
    );
    const getViewportHeight = () => largest(
      window.visualViewport?.height,
      window.innerHeight,
      root?.clientHeight
    );
    const update = () => {
      ticking = false;
      const scrollTop = getScrollTop();
      const max = Math.max(0, getScrollHeight() - getViewportHeight());
      const remaining = max - scrollTop;
      let pct = max > 0 ? (scrollTop / max) * 100 : 0;
      if (max <= 0 || scrollTop <= 1) pct = 0;
      else if (remaining <= 2) pct = 100;
      const clamped = Math.min(100, Math.max(0, pct));
      root.style.setProperty('--scroll-progress', clamped.toFixed(2));
      root.style.setProperty('--scroll-progress-ratio', (clamped / 100).toFixed(4));
    };
    const requestUpdate = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    };
    requestUpdate();
    window.addEventListener('scroll', requestUpdate, { passive:true });
    window.addEventListener('resize', requestUpdate);
    window.addEventListener('orientationchange', requestUpdate);
    window.addEventListener('pageshow', requestUpdate);
    document.addEventListener('navheightchange', requestUpdate);
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', requestUpdate);
      window.visualViewport.addEventListener('scroll', requestUpdate, { passive:true });
    }
  }

  function initHeroParallax(){
    if (document.body?.dataset?.page !== 'home') return;
    const hero = document.querySelector('.hero');
    if (!hero) return;
    if (hero.dataset.heroParallaxBound === 'yes') return;
    hero.dataset.heroParallaxBound = 'yes';
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

  function initProjectPreviewVideos(){
    if (document.body?.dataset?.page !== 'home') return;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const finePointer = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
    if (reduce || !finePointer) return;
    const cards = [...document.querySelectorAll('.project-examples-card')];
    if (!cards.length) return;
    const setupCard = (card) => {
      if (!card || card._previewAutoplayBound) return;
      const vid = card.querySelector('video.gif-video');
      if (!vid) return;
      const sources = [...vid.querySelectorAll('source[data-src]')];
      let pending = false;
      const loadSources = () => {
        if (vid.dataset.loaded === 'true') return;
        sources.forEach((source) => {
          if (!source.src && source.dataset.src) {
            source.src = source.dataset.src;
          }
        });
        vid.dataset.loaded = 'true';
        try { vid.load(); } catch {}
      };
      const showAndPlay = () => {
        card.classList.add('is-video-active');
        try {
          vid.muted = true;
          vid.playsInline = true;
          vid.autoplay = true;
          vid.preload = 'metadata';
          vid.setAttribute('muted', '');
          vid.setAttribute('playsinline', '');
          vid.setAttribute('autoplay', '');
        } catch {}
        try { vid.play && vid.play().catch(() => {}); } catch {}
      };
      const playVideo = () => {
        card._previewAutoplayActive = true;
        loadSources();
        if (vid.readyState >= 2) {
          pending = false;
          showAndPlay();
          return;
        }
        if (pending) return;
        pending = true;
        const onReady = () => {
          pending = false;
          if (card._previewAutoplayActive) showAndPlay();
        };
        ['loadeddata', 'canplay', 'canplaythrough', 'playing'].forEach((evt) => {
          vid.addEventListener(evt, onReady, { once: true });
        });
      };
      const pauseVideo = () => {
        card._previewAutoplayActive = false;
        try { vid.pause && vid.pause(); } catch {}
        card.classList.remove('is-video-active');
      };
      card._previewAutoplayBound = true;
      card.addEventListener('pointerenter', playVideo);
      card.addEventListener('focusin', playVideo);
      card.addEventListener('pointerleave', pauseVideo);
      card.addEventListener('focusout', pauseVideo);
    };
    cards.forEach(setupCard);
  }

  // Subtle ambient light following the mouse inside the hero
  function initHeroAmbient(){
    if (document.body?.dataset?.page !== 'home') return;
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
    const scrollToTarget = target => {
      if(!target) return false;
      if (typeof target.scrollIntoView === 'function'){
        target.scrollIntoView({ behavior:'smooth', block:'start' });
      } else {
        const top = target.getBoundingClientRect().top + window.scrollY;
        window.scrollTo({top, behavior:'smooth'});
      }
      return true;
    };
    const scrollToNext = ind => {
      let next = ind.closest('section')?.nextElementSibling;
      while(next && next.tagName !== 'SECTION') next = next.nextElementSibling;
      if(next){
        scrollToTarget(next);
      }else{
        window.scrollBy({top:window.innerHeight*0.8,behavior:'smooth'});
      }
    };
    const normalizePagePath = pathname => {
      let next = String(pathname || '/');
      next = next.replace(/\/index\.html$/i, '/');
      next = next.replace(/\.html$/i, '');
      next = next.replace(/\/+$/, '');
      return next || '/';
    };
    const samePageHashFromHref = href => {
      const value = String(href || '').trim();
      if(!value) return '';
      try{
        const targetUrl = new URL(value, window.location.href);
        const currentUrl = new URL(window.location.href);
        if(!targetUrl.hash || targetUrl.hash.length < 2) return '';
        if(targetUrl.origin !== currentUrl.origin) return '';
        if(normalizePagePath(targetUrl.pathname) !== normalizePagePath(currentUrl.pathname)) return '';
        if((targetUrl.search || '') !== (currentUrl.search || '')) return '';
        return targetUrl.hash;
      }catch{
        return '';
      }
    };
    $$('.chevron-hint,.scroll-indicator').forEach(ind=>{
      if(ind.dataset.chevronHintBound === 'yes') return;
      ind.dataset.chevronHintBound = 'yes';
      on(ind,'click',e=>{
        if(e.defaultPrevented) return;
        const href = typeof ind.getAttribute === 'function' ? ind.getAttribute('href') : null;
        const hash = samePageHashFromHref(href);
        if(hash){
          e.preventDefault();
          let targetId = hash.slice(1);
          try{ targetId = decodeURIComponent(targetId); }catch{}
          const target = document.getElementById(targetId);
          if(scrollToTarget(target)) return;
        }else if(ind.tagName === 'A'){
          e.preventDefault();
        }
        scrollToNext(ind);
      });
    });
  }
  function initCertTicker(){
    const track = document.querySelector('.cert-track');
    if(!track) return;
    if(track.dataset.certTickerBound === 'yes') return;
    track.dataset.certTickerBound = 'yes';
    const GAP=160, BASE=90, DRAG=15, MOBILE_SPEED=42;
    const mobileQuery = window.matchMedia('(max-width: 900px)');
    const reduceMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let v=BASE, target=BASE, bandW=0, stripW=0;
    let down=false,moved=false; let sx=0,lx=0; let paused=false; let cancelClk=false;
    const originals=[...track.children].filter(el=>el.classList && el.classList.contains('cert'));
    let tiles=[...originals];
    let mode='';
    let animationFrame=0;
    let activePointerMove=null;
    let scrollPauseTimer=0;
    let mobileFrame=0;
    let mobileLast=0;
    let mobileLoopWidth=0;
    let mobileResumeTimer=0;
    let mobileScrollRemainder=0;
    let activeMobilePointerMove=null;
    let activeMobileMouseMove=null;
    let mobileGesture=null;
    let mobileGestureClearTimer=0;
    let mobileActivatedAt=0;
    if(!originals.length) return;
    const setPos=(el,x)=>{el.dataset.x=x;el.style.setProperty("--ticker-x", `${x}px`);};
    const resetTile=el=>{
      delete el.dataset.x;
      delete el.dataset.w;
      delete el.dataset.orig;
      delete el.dataset.certClone;
      el.removeAttribute('aria-hidden');
      el.removeAttribute('tabindex');
      el.style.removeProperty('--ticker-x');
      el.classList.remove('reenter');
    };
    const cloneCert=src=>{
      const clone=src.cloneNode(true);
      clone.dataset.certClone='true';
      clone.setAttribute('aria-hidden','true');
      clone.setAttribute('tabindex','-1');
      return clone;
    };
    const removeDirectClones=()=>{
      [...track.children].forEach(el=>{
        if(el.dataset && el.dataset.certClone === 'true') el.remove();
      });
      tiles=originals.slice();
      originals.forEach(resetTile);
      stripW=0;
    };
    const restoreOriginals=()=>{
      track.classList.remove('dragging','is-scroll-paused','is-touch-paused');
      if(activePointerMove) window.removeEventListener('pointermove', activePointerMove);
      activePointerMove=null;
      if(activeMobilePointerMove) window.removeEventListener('pointermove', activeMobilePointerMove);
      activeMobilePointerMove=null;
      if(activeMobileMouseMove) window.removeEventListener('mousemove', activeMobileMouseMove);
      activeMobileMouseMove=null;
      if(mobileGestureClearTimer) window.clearTimeout(mobileGestureClearTimer);
      mobileGestureClearTimer=0;
      mobileGesture=null;
      const strips=[...track.querySelectorAll('.cert-track__mobile-strip')];
      if(strips.length){
        originals.forEach(el=>track.appendChild(el));
        strips.forEach(strip=>strip.remove());
      }
      removeDirectClones();
      track.style.removeProperty('--cert-mobile-marquee-distance');
      track.style.removeProperty('--cert-mobile-marquee-duration');
      mobileLoopWidth=0;
      down=moved=paused=false;
    };
    const fill=()=>{
      if(mode !== 'marquee') return;
      removeDirectClones();
      originals.forEach((t,i)=>{const w=t.offsetWidth+GAP;Object.assign(t.dataset,{w,orig:i});setPos(t,stripW);stripW+=w;});
      const baseW=stripW; bandW=track.getBoundingClientRect().width;
      while(stripW < baseW + bandW){
        originals.forEach(src=>{const clone=cloneCert(src);track.appendChild(clone);const w=clone.offsetWidth+GAP;Object.assign(clone.dataset,{w,orig:+src.dataset.orig});setPos(clone,stripW);stripW+=w;tiles.push(clone);});
      }
    };
    const layoutMobileScroller=()=>{
      const firstOriginal=originals[0];
      const firstClone=track.querySelector(':scope > .cert[data-cert-clone="true"]') || track.querySelector('.cert[data-cert-clone="true"]');
      if(!firstOriginal || !firstClone) return;
      let nextLoopWidth=Math.round(firstClone.offsetLeft - firstOriginal.offsetLeft);
      if(!Number.isFinite(nextLoopWidth) || nextLoopWidth <= track.clientWidth){
        const groups=Math.max(2, Math.floor(track.querySelectorAll(':scope > .cert').length / Math.max(1, originals.length)));
        nextLoopWidth=Math.round(track.scrollWidth / groups);
      }
      mobileLoopWidth=Math.max(track.clientWidth + 1, nextLoopWidth);
      track.dataset.certLoopWidth=String(mobileLoopWidth);
    };
    const buildMobileScroller=()=>{
      restoreOriginals();
      originals.forEach(src=>track.appendChild(cloneCert(src)));
      let guard=0;
      while(track.scrollWidth < track.clientWidth * 3 && guard < 4){
        originals.forEach(src=>track.appendChild(cloneCert(src)));
        guard+=1;
      }
      layoutMobileScroller();
    };
    const isTouch=matchMedia('(hover: none) and (pointer: coarse)').matches;
    if(!isTouch){
      track.addEventListener('mouseenter',()=>{if(mode === 'marquee') target=0;});
      track.addEventListener('mouseleave',()=>{if(mode === 'marquee') target=BASE;});
    }
    const move=dx=>tiles.forEach(t=>{let x=+t.dataset.x+dx,w=+t.dataset.w;x=dx<0?(x+w<=0?x+stripW:x):(x>=bandW?x-stripW:x);setPos(t,x);});
    const end=()=>{
      down=moved=false;
      paused=false;
      track.classList.remove('dragging');
      if(activePointerMove) window.removeEventListener('pointermove', activePointerMove);
      activePointerMove=null;
      ['pointerup','pointercancel'].forEach(e=>window.removeEventListener(e,end));
    };
    const getEventPoint=event=>{
      const touch=event?.changedTouches?.[0] || event?.touches?.[0];
      const source=touch || event;
      if(!source || typeof source.clientX !== 'number' || typeof source.clientY !== 'number') return null;
      return {x:source.clientX, y:source.clientY};
    };
    const getCertLink=target=>{
      if(!target || typeof target.closest !== 'function') return null;
      const link=target.closest('a.cert[href]');
      return link && track.contains(link) ? link : null;
    };
    const updateMobileGesture=event=>{
      if(!mobileGesture) return;
      const point=getEventPoint(event);
      if(!point) return;
      const dx=point.x-mobileGesture.x;
      const dy=point.y-mobileGesture.y;
      const scrollDx=track.scrollLeft-mobileGesture.scrollLeft;
      if(Math.abs(dx) > 10 || Math.abs(dy) > 10 || Math.abs(scrollDx) > 10){
        mobileGesture.moved=true;
      }
    };
    const clearMobileGestureSoon=()=>{
      if(mobileGestureClearTimer) window.clearTimeout(mobileGestureClearTimer);
      const gesture=mobileGesture;
      mobileGestureClearTimer=window.setTimeout(()=>{
        if(mobileGesture === gesture) mobileGesture=null;
        mobileGestureClearTimer=0;
      }, 450);
    };
    const beginMobileInteraction=event=>{
      const point=getEventPoint(event);
      mobileGesture=point ? {
        link:getCertLink(event.target),
        x:point.x,
        y:point.y,
        scrollLeft:track.scrollLeft,
        moved:false
      } : null;
      pauseMobileAuto(1400);
      if(activeMobilePointerMove) window.removeEventListener('pointermove', activeMobilePointerMove);
      activeMobilePointerMove=moveEvent=>{
        updateMobileGesture(moveEvent);
        pauseMobileAuto(1400);
      };
      window.addEventListener('pointermove', activeMobilePointerMove, {passive:true});
      window.addEventListener('pointerup', finishMobileInteraction, {once:true});
      window.addEventListener('pointercancel', finishMobileInteraction, {once:true});
      window.addEventListener('blur', finishMobileInteraction, {once:true});
    };
    const finishMobileInteraction=event=>{
      updateMobileGesture(event);
      const tapLink=event?.type === 'pointerup' && mobileGesture?.link && !mobileGesture.moved
        ? mobileGesture.link
        : null;
      if(activeMobilePointerMove) window.removeEventListener('pointermove', activeMobilePointerMove);
      activeMobilePointerMove=null;
      pauseMobileAuto(900);
      window.removeEventListener('pointerup', finishMobileInteraction);
      window.removeEventListener('pointercancel', finishMobileInteraction);
      window.removeEventListener('blur', finishMobileInteraction);
      if(tapLink){
        if(typeof event.preventDefault === 'function') event.preventDefault();
        mobileGesture=null;
        mobileActivatedAt=Date.now();
        activateCertLink(tapLink);
        return;
      }
      clearMobileGestureSoon();
    };
    const beginMobileMouseInteraction=event=>{
      if(event.button) return;
      if(mobileGesture) return;
      const point=getEventPoint(event);
      mobileGesture=point ? {
        link:getCertLink(event.target),
        x:point.x,
        y:point.y,
        scrollLeft:track.scrollLeft,
        moved:false
      } : null;
      pauseMobileAuto(1400);
      if(activeMobileMouseMove) window.removeEventListener('mousemove', activeMobileMouseMove);
      activeMobileMouseMove=moveEvent=>{
        updateMobileGesture(moveEvent);
        pauseMobileAuto(1400);
      };
      window.addEventListener('mousemove', activeMobileMouseMove, {passive:true});
      window.addEventListener('mouseup', finishMobileMouseInteraction, {once:true});
      window.addEventListener('blur', finishMobileMouseInteraction, {once:true});
    };
    const finishMobileMouseInteraction=event=>{
      updateMobileGesture(event);
      const tapLink=event?.type === 'mouseup' && mobileGesture?.link && !mobileGesture.moved
        ? mobileGesture.link
        : null;
      if(activeMobileMouseMove) window.removeEventListener('mousemove', activeMobileMouseMove);
      activeMobileMouseMove=null;
      pauseMobileAuto(900);
      window.removeEventListener('mouseup', finishMobileMouseInteraction);
      window.removeEventListener('blur', finishMobileMouseInteraction);
      if(tapLink){
        if(typeof event.preventDefault === 'function') event.preventDefault();
        mobileGesture=null;
        mobileActivatedAt=Date.now();
        activateCertLink(tapLink);
        return;
      }
      clearMobileGestureSoon();
    };
    const activateCertLink=link=>{
      const href=link?.href || link?.getAttribute?.('href');
      if(!href) return;
      try{
        window.location.assign(href);
      }catch{
        window.location.href=href;
      }
    };
    track.addEventListener('pointerdown',e=>{
      if(mode === 'mobile-scroll'){
        beginMobileInteraction(e);
        return;
      }
      if(mode !== 'marquee' || e.button) return;
      down=true;moved=false;paused=false;sx=lx=e.clientX;
      const onMove=e=>{if(!down) return;const dxT=e.clientX-sx;if(!moved && Math.abs(dxT)>=DRAG){moved=true;paused=true;cancelClk=true;track.classList.add('dragging');}if(moved){move(e.clientX-lx);lx=e.clientX;e.preventDefault();}};
      activePointerMove=onMove;
      window.addEventListener('pointermove',onMove);
      window.addEventListener('pointerup',end,{once:true});
      window.addEventListener('pointercancel',end,{once:true});
    });
    track.addEventListener('mousedown',e=>{
      if(mode === 'mobile-scroll') beginMobileMouseInteraction(e);
    });
    track.addEventListener('wheel',()=>pauseMobileAuto(900), {passive:true});
    track.addEventListener('click',e=>{
      if(mode === 'mobile-scroll'){
        const link=getCertLink(e.target);
        if(link){
          if(Date.now() - mobileActivatedAt < 1000){
            e.preventDefault();
            e.stopImmediatePropagation();
            return;
          }
          const gestureMatches=!mobileGesture || !mobileGesture.link || mobileGesture.link === link;
          if(mobileGesture?.moved && gestureMatches){
            e.preventDefault();
            e.stopImmediatePropagation();
            mobileGesture=null;
            return;
          }
          if(!(e.metaKey || e.ctrlKey || e.shiftKey || e.altKey)){
            e.preventDefault();
            e.stopImmediatePropagation();
            mobileGesture=null;
            activateCertLink(link);
            return;
          }
        }
      }
      if(cancelClk){e.preventDefault();e.stopImmediatePropagation();}
      cancelClk=false;
    },true);
    let last=performance.now();
    const reset=()=>{last=performance.now();};
    document.addEventListener('visibilitychange',reset);window.addEventListener('focus',reset);
    const stopLoop=()=>{
      if(animationFrame) window.cancelAnimationFrame(animationFrame);
      animationFrame=0;
    };
    const startLoop=()=>{
      if(animationFrame) return;
      last=performance.now();
      const loop=now=>{
        const dt=Math.min((now-last)/1000,0.25);
        last=now;
        v+=(target-v)*Math.min(1,dt*4);
        if(!paused){
          tiles.forEach(t=>{let x=+t.dataset.x-v*dt,w=+t.dataset.w;if(x<-w)x+=stripW;setPos(t,x);});
        }
        animationFrame=requestAnimationFrame(loop);
      };
      animationFrame=requestAnimationFrame(loop);
    };
    const stopMobileAuto=()=>{
      if(mobileFrame) window.cancelAnimationFrame(mobileFrame);
      mobileFrame=0;
      if(mobileResumeTimer) window.clearTimeout(mobileResumeTimer);
      mobileResumeTimer=0;
    };
    const isMobilePaused=()=>track.classList.contains('is-touch-paused') || track.classList.contains('is-scroll-paused');
    const startMobileAuto=()=>{
      if(mobileFrame) return;
      mobileLast=performance.now();
      mobileScrollRemainder=0;
      const loop=now=>{
        const dt=Math.min((now-mobileLast)/1000,0.25);
        mobileLast=now;
        if(mode === 'mobile-scroll' && !isMobilePaused() && mobileLoopWidth > 0){
          const advance=MOBILE_SPEED * dt + mobileScrollRemainder;
          const pixels=Math.trunc(advance);
          mobileScrollRemainder=advance - pixels;
          if(pixels > 0) track.scrollLeft += pixels;
          if(track.scrollLeft >= mobileLoopWidth){
            track.scrollLeft -= mobileLoopWidth;
          }
        }
        mobileFrame=requestAnimationFrame(loop);
      };
      mobileFrame=requestAnimationFrame(loop);
    };
    const pauseMobileAuto=(delay=900)=>{
      if(mode !== 'mobile-scroll') return;
      track.classList.add('is-touch-paused');
      if(mobileResumeTimer) window.clearTimeout(mobileResumeTimer);
      const resumeDelay=Number.isFinite(delay) && delay > 0 ? delay : 1400;
      mobileResumeTimer=window.setTimeout(()=>{
        track.classList.remove('is-touch-paused');
        mobileResumeTimer=0;
      }, resumeDelay);
    };
    const setNativeScroller=()=>{
      if(mode === 'native') return;
      stopLoop();
      stopMobileAuto();
      mode='native';
      track.dataset.certTickerMode = 'native';
      restoreOriginals();
    };
    const setMarquee=()=>{
      if(mode === 'marquee'){
        fill();
        return;
      }
      stopMobileAuto();
      restoreOriginals();
      mode='marquee';
      track.dataset.certTickerMode = 'marquee';
      v=BASE;
      target=BASE;
      fill();
      startLoop();
    };
    const setMobileScroller=()=>{
      if(mode === 'mobile-scroll'){
        layoutMobileScroller();
        startMobileAuto();
        return;
      }
      stopLoop();
      stopMobileAuto();
      mode='mobile-scroll';
      track.dataset.certTickerMode = 'mobile-scroll';
      buildMobileScroller();
      startMobileAuto();
    };
    const syncMode=()=>{
      if(reduceMotionQuery.matches) setNativeScroller();
      else if(mobileQuery.matches) setMobileScroller();
      else setMarquee();
    };
    const pauseMobileDuringPageScroll=()=>{
      if(mode !== 'mobile-scroll') return;
      track.classList.add('is-scroll-paused');
      if(scrollPauseTimer) window.clearTimeout(scrollPauseTimer);
      scrollPauseTimer=window.setTimeout(()=>track.classList.remove('is-scroll-paused'), 180);
    };
    window.addEventListener('load', syncMode);
    window.addEventListener('resize', syncMode);
    window.addEventListener('scroll', pauseMobileDuringPageScroll, {passive:true});
    if(mobileQuery.addEventListener) mobileQuery.addEventListener('change', syncMode);
    else if(mobileQuery.addListener) mobileQuery.addListener(syncMode);
    if(reduceMotionQuery.addEventListener) reduceMotionQuery.addEventListener('change', syncMode);
    else if(reduceMotionQuery.addListener) reduceMotionQuery.addListener(syncMode);
    syncMode();
  }
})();
