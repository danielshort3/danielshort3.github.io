(function(root) {
  'use strict';

  // Presentation only: committing an area selection remains a synchronous game
  // action. No animation callback may purchase, select, or save anything.
  function create(options) {
    const element = options.element;
    const parent = element.parentElement;
    const document = element.ownerDocument;
    const motionQuery = root.matchMedia?.('(prefers-reduced-motion: reduce)');
    const controller = new AbortController();
    const original = {
      transform:element.style.transform,
      willChange:element.style.willChange,
      pointerEvents:element.style.pointerEvents,
      overflow:parent.style.overflow
    };
    let offset = 0;
    let phase = 'idle';
    let ghost = null;
    let animations = [];
    let timer = 0;
    let generation = 0;
    let disposed = false;

    function quiet() {
      return disposed || document.hidden || !!motionQuery?.matches || !!options.quiet?.();
    }
    function translate(node, x) { node.style.transform = 'translate3d(' + Math.round(x * 100) / 100 + 'px,0,0)'; }
    function finish() {
      generation += 1;
      root.clearTimeout(timer);
      timer = 0;
      animations.forEach(animation => animation.cancel());
      animations = [];
      ghost?.remove();
      ghost = null;
      offset = 0;
      phase = 'idle';
      element.style.transform = original.transform;
      element.style.willChange = original.willChange;
      element.style.pointerEvents = original.pointerEvents;
      parent.style.overflow = original.overflow;
      delete element.dataset.areaMotion;
    }
    function begin() {
      finish();
      if (quiet() || element.hidden || !element.getClientRects().length) return false;
      phase = 'drag';
      element.dataset.areaMotion = phase;
      element.style.willChange = 'transform';
      parent.style.overflow = 'hidden';
      return true;
    }
    function drag(distance, edge) {
      if (quiet()) { finish(); return; }
      if (phase !== 'drag' && !begin()) return;
      const width = element.clientWidth;
      const dx = Number.isFinite(distance) ? distance : 0;
      offset = Math.sign(dx) * Math.min(Math.abs(dx) * (edge ? .18 : .8), width * (edge ? .08 : .48));
      translate(element, offset);
    }
    function animate(node, from, to, duration, settle) {
      if (typeof node.animate !== 'function') return false;
      const animation = node.animate([
        {transform:'translate3d(' + from + 'px,0,0)'},
        {transform:'translate3d(' + to + 'px,0,0)'}
      ], {duration,easing:'cubic-bezier(.22,.72,.18,1)',fill:'both'});
      animations.push(animation);
      animation.finished.then(settle, () => {});
      return true;
    }
    function cancel() {
      if (phase !== 'drag' || !offset || quiet()) { finish(); return; }
      const from = offset;
      phase = 'settling';
      element.dataset.areaMotion = phase;
      const token = ++generation;
      const settle = () => { if (token === generation) finish(); };
      if (!animate(element, from, 0, 150, settle)) { finish(); return; }
      timer = root.setTimeout(settle, 220);
    }
    function snapshot() {
      const copy = element.cloneNode(true);
      copy.classList.add('wx-area-ghost');
      copy.setAttribute('aria-hidden','true');
      copy.inert = true;
      // Never duplicate behavior hooks, focus targets, IDs or announcements.
      [copy,...copy.querySelectorAll('*')].forEach(node => {
        Array.from(node.attributes).forEach(attribute => {
          if (/^(id|tabindex|autofocus|role)$/.test(attribute.name) || /^data-(wx-|guide-|area-motion)/.test(attribute.name) || attribute.name.startsWith('aria-')) node.removeAttribute(attribute.name);
        });
      });
      copy.setAttribute('aria-hidden','true');
      const canvases = element.querySelectorAll('canvas');
      copy.querySelectorAll('canvas').forEach((canvas, index) => {
        canvas.width = canvases[index].width;
        canvas.height = canvases[index].height;
        try { canvas.getContext('2d').drawImage(canvases[index],0,0); } catch (error) { /* A missing optional canvas never blocks navigation. */ }
      });
      Object.assign(copy.style, {
        position:'absolute',left:element.offsetLeft + 'px',top:element.offsetTop + 'px',
        width:element.clientWidth + 'px',height:element.clientHeight + 'px',margin:'0',
        pointerEvents:'none',zIndex:'2',willChange:'transform'
      });
      // Scroll positions are not copied by cloneNode.
      const scrollables = element.querySelectorAll('*');
      const copyScrollables = copy.querySelectorAll('*');
      parent.append(copy);
      scrollables.forEach((node,index) => {
        copyScrollables[index].scrollTop = node.scrollTop;
        copyScrollables[index].scrollLeft = node.scrollLeft;
      });
      return copy;
    }
    function transition(direction, commit) {
      if (disposed) return undefined;
      const from = phase === 'drag' ? offset : 0;
      finish();
      if (quiet() || !direction || element.hidden || !element.getClientRects().length) return commit();
      const width = element.clientWidth;
      if (!width || typeof element.animate !== 'function') return commit();
      ghost = snapshot();
      translate(ghost, from);
      let result;
      try { result = commit(); } catch (error) { finish(); throw error; }
      if (result === false || result?.ok === false || !ghost || quiet() || element.hidden) { finish(); return result; }
      phase = 'transition';
      element.dataset.areaMotion = phase;
      element.style.willChange = 'transform';
      element.style.pointerEvents = 'none';
      parent.style.overflow = 'hidden';
      const sign = direction < 0 ? -1 : 1;
      const incoming = sign * width + from;
      const token = ++generation;
      const settle = () => { if (token === generation) finish(); };
      translate(element, incoming);
      animate(ghost, from, -sign * width, 260, () => {});
      animate(element, incoming, 0, 260, settle);
      timer = root.setTimeout(settle, 360);
      return result;
    }
    root.addEventListener('resize',finish,{signal:controller.signal});
    root.addEventListener('blur',finish,{signal:controller.signal});
    document.addEventListener('visibilitychange',() => { if (document.hidden) finish(); },{signal:controller.signal});
    motionQuery?.addEventListener('change',finish,{signal:controller.signal});
    return {
      begin,drag,cancel,transition,finish,
      isAnimating:() => phase !== 'idle',
      dispose() { finish(); disposed = true; controller.abort(); }
    };
  }
  root.WayfarersAreaMotion = {create};
})(typeof globalThis !== 'undefined' ? globalThis : this);
