(function (root) {
  'use strict';
  function create(options) {
    const abort = new AbortController();
    const dialog = document.createElement('dialog');
    dialog.className = 'wx-guide';
    dialog.setAttribute('aria-labelledby','wx-guide-heading');
    dialog.setAttribute('aria-describedby','wx-guide-body');
    dialog.innerHTML = '<svg class="wx-guide-shade" aria-hidden="true"><path fill-rule="evenodd"></path></svg><div class="wx-guide-ring" aria-hidden="true"></div><span class="wx-guide-announcement" aria-live="polite" aria-atomic="true" data-guide-announcement></span><section class="wx-guide-card"><div class="wx-guide-meta"><span data-guide-area></span><span data-guide-count></span><button type="button" data-guide-settings aria-label="Settings and recovery">⚙</button></div><h2 id="wx-guide-heading"></h2><div class="wx-guide-copy"><p id="wx-guide-body"></p><p class="wx-guide-reward" data-guide-reward hidden></p><p class="wx-guide-recovery" data-guide-recovery hidden></p></div><p id="wx-guide-quote" class="wx-guide-quote" data-guide-quote hidden></p><footer><button type="button" data-guide-leave>Close</button><button type="button" data-guide-next>Next</button></footer></section>';
    options.parent.append(dialog);
    const q = selector => dialog.querySelector(selector);
    let model = null;
    let signature = '';
    let previousFocus = null;
    let frame = 0;
    let disposed = false;
    let missing = false;
    let submitting = false;
    let pendingStepFocus = false;
    let target = null;
    let interactive = false;
    let allowedTargets = [];
    let isolated = [];
    let describedTarget = null;
    let priorDescription = null;
    let fallbackSurface = null;
    const observer = new ResizeObserver(schedule);
    observer.observe(dialog);
    observer.observe(q('.wx-guide-card'));
    function schedule() {
      if (disposed || !dialog.open || frame) return;
      frame = root.requestAnimationFrame(() => { frame = 0; layout(); });
    }
    function rect(node) {
      if (!node || !node.isConnected || !node.getClientRects().length) return null;
      const box = node.getBoundingClientRect();
      return box.width >= 1 && box.height >= 1 ? box : null;
    }
    function restoreIsolation() {
      isolated.forEach(([node,value]) => { node.inert=value; });
      isolated=[];
      if (describedTarget?.isConnected) {
        if (priorDescription == null) describedTarget.removeAttribute('aria-describedby');
        else describedTarget.setAttribute('aria-describedby',priorDescription);
      }
      describedTarget=null; priorDescription=null;
    }
    function isolate() {
      restoreIsolation();
      if (!interactive || !dialog.open) return;
      const keep=[dialog,...allowedTargets].filter(node=>node?.isConnected);
      function visit(node) {
        if (!(node instanceof HTMLElement) || keep.some(item=>item===node || item.contains(node))) return;
        if (keep.some(item=>node.contains(item))) { Array.from(node.children).forEach(visit); return; }
        if (!node.inert) { isolated.push([node,false]); node.inert=true; }
      }
      Array.from(document.body.children).forEach(visit);
      if (target) {
        describedTarget=target; priorDescription=target.getAttribute('aria-describedby');
        target.setAttribute('aria-describedby',[(priorDescription || ''),'wx-guide-heading','wx-guide-body',model.quoteText ? 'wx-guide-quote' : null].filter(Boolean).join(' '));
      }
    }
    function allowedNode(node) {
      return !!node && (dialog.contains(node) || allowedTargets.some(item=>item===node || item.contains(node)));
    }
    function focusables() {
      const selector='button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex]:not([tabindex="-1"])';
      const nodes=interactive ? allowedTargets.flatMap(node=>node.matches(selector) ? [node] : Array.from(node.querySelectorAll(selector))) : [];
      nodes.push(...dialog.querySelectorAll('button:not(:disabled)'));
      return [...new Set(nodes)].filter(node=>node.getClientRects().length && !node.hidden && !node.closest('[inert]'));
    }
    function layout() {
      if (!model || !dialog.open) return;
      const found = options.resolveTarget(model.target,model.fallbackTarget,model);
      const candidate=found && Object.prototype.hasOwnProperty.call(found,'element') ? found.element : found;
      const nextTarget=candidate instanceof Element ? candidate : null;
      if (interactive) {
        const parent=nextTarget?.closest('dialog[open]') || options.parent;
        // Some Android WebViews still clip a nested popover to a transformed
        // modal. Keep that real modal viewport-sized while its lesson is active.
        // Retain the framing once applied so removing its transform cannot
        // alternate the class on every ResizeObserver pass.
        const fallback=parent !== options.parent && (!dialog.hasAttribute('popover') || fallbackSurface===parent || root.getComputedStyle(parent).transform!=='none') ? parent : null;
        if (fallbackSurface !== fallback) {
          fallbackSurface?.classList.remove('wx-guide-surface');
          fallbackSurface=fallback;
          fallbackSurface?.classList.add('wx-guide-surface');
        }
        if (parent !== dialog && dialog.parentElement !== parent) {
          const visible=dialog.hasAttribute('popover') && dialog.matches(':popover-open');
          if (visible) dialog.hidePopover();
          parent.append(dialog);
          if (dialog.hasAttribute('popover')) dialog.showPopover();
        }
      }
      if (target !== nextTarget) {
        if (target) {observer.unobserve(target);target.removeAttribute('data-guide-target');}
        target = nextTarget;
        if (target) {observer.observe(target);target.setAttribute('data-guide-target','');}
      }
      allowedTargets=interactive ? [target,...(found?.allowed || [])].filter(node=>node?.isConnected) : [];
      isolate();
      const viewport = dialog.getBoundingClientRect();
      const visual = root.visualViewport;
      const left = Math.max(viewport.left,visual?.offsetLeft || 0);
      const top = Math.max(viewport.top,visual?.offsetTop || 0);
      const right = Math.min(viewport.right,(visual?.offsetLeft || 0) + (visual?.width || root.innerWidth));
      const bottom = Math.min(viewport.bottom,(visual?.offsetTop || 0) + (visual?.height || root.innerHeight));
      const pad = 12;
      const safe = { left:left-viewport.left+pad, top:top-viewport.top+pad, right:right-viewport.left-pad, bottom:bottom-viewport.top-pad };
      const clip = rect(found?.clip);
      let box = rect(target);
      if(box && found?.scroll && clip && (box.left<clip.left || box.right>clip.right)) {
        found.scroll.scrollLeft+=box.left<clip.left ? box.left-clip.left-5 : box.right-clip.right+5;
        box=rect(target);
      }
      if (box && found?.scroll && clip && (box.top < clip.top || box.bottom > clip.bottom)) {
        // A tall library cannot fit both edges. Keep its visible intersection
        // instead of alternating between top and bottom on every layout pass.
        if (box.height > clip.height) {
          if (box.bottom <= clip.top || box.top >= clip.bottom) found.scroll.scrollTop += box.top-clip.top;
        } else found.scroll.scrollTop += box.top < clip.top ? box.top-clip.top-5 : box.bottom-clip.bottom+5;
        box = rect(target);
      }
      if (box) {
        box = {left:Math.max(box.left,clip?.left ?? left,left)-viewport.left,top:Math.max(box.top,clip?.top ?? top,top)-viewport.top,right:Math.min(box.right,clip?.right ?? right,right)-viewport.left,bottom:Math.min(box.bottom,clip?.bottom ?? bottom,bottom)-viewport.top};
        if (box.right-box.left < 12 || box.bottom-box.top < 12) box = null;
      }
      missing = !box;
      dialog.dataset.missing = String(missing);
      const recovery = q('[data-guide-recovery]');
      recovery.hidden = !missing && !model.saveFailure;
      recovery.textContent = model.saveFailure ? 'Save needs attention. Retry to keep your place.' : missing ? 'This control is not visible yet. Retry, or open Settings to recover your progress.' : '';
      q('[data-guide-next]').textContent = model.saveFailure || missing ? 'Retry' : model.ackLabel || 'Next';
      q('[data-guide-next]').hidden = interactive && model.mode!=='currency' && !model.saveFailure && !missing;
      q('[data-guide-next]').disabled = submitting || !!model.disabled;
      q('[data-guide-leave]').hidden = !canLeave() && !model.saveFailure;
      q('[data-guide-leave]').disabled = submitting;
      q('[data-guide-settings]').hidden=canLeave() && !model.saveFailure;
      q('footer').hidden=q('[data-guide-leave]').hidden && q('[data-guide-next]').hidden;
      const shade = q('svg');
      shade.setAttribute('viewBox','0 0 ' + viewport.width + ' ' + viewport.height);
      let path = 'M0 0H' + viewport.width + 'V' + viewport.height + 'H0Z';
      const ring = q('.wx-guide-ring');
      ring.hidden = missing;
      if (box) {
        const gap = 4;
        box.left = Math.max(3,box.left-gap); box.right = Math.min(viewport.width-3,box.right+gap);
        box.top = Math.max(3,box.top-gap); box.bottom = Math.min(viewport.height-3,box.bottom+gap);
        path += 'M' + box.left + ' ' + box.top + 'H' + box.right + 'V' + box.bottom + 'H' + box.left + 'Z';
        Object.assign(ring.style,{left:box.left+'px',top:box.top+'px',width:box.right-box.left+'px',height:box.bottom-box.top+'px'});
      }
      if (interactive) allowedTargets.filter(node=>node!==target).forEach(node=>{
        const bounds=rect(node);
        if (!bounds) return;
        const x1=Math.max(0,bounds.left-viewport.left),y1=Math.max(0,bounds.top-viewport.top),x2=Math.min(viewport.width,bounds.right-viewport.left),y2=Math.min(viewport.height,bounds.bottom-viewport.top);
        if (x2>x1 && y2>y1) path+='M'+x1+' '+y1+'H'+x2+'V'+y2+'H'+x1+'Z';
      });
      shade.querySelector('path').setAttribute('d',path);
      const card = q('.wx-guide-card');
      const availableWidth = Math.max(0,safe.right-safe.left);
      card.style.width = Math.min(320,availableWidth) + 'px';
      card.style.maxHeight = Math.max(120,safe.bottom-safe.top) + 'px';
      let size = card.getBoundingClientRect();
      let x = (safe.left+safe.right-size.width)/2;
      let y = (safe.top+safe.bottom-size.height)/2;
      if (box) {
        const spaces = () => [
          {room:box.top-safe.top-12,axis:'y',x:(box.left+box.right-size.width)/2,y:box.top-size.height-12},
          {room:safe.bottom-box.bottom-12,axis:'y',x:(box.left+box.right-size.width)/2,y:box.bottom+12},
          {room:box.left-safe.left-12,axis:'x',x:box.left-size.width-12,y:(box.top+box.bottom-size.height)/2},
          {room:safe.right-box.right-12,axis:'x',x:box.right+12,y:(box.top+box.bottom-size.height)/2}
        ].flatMap(candidate=>candidate.axis==='x' ? [candidate,Object.assign({},candidate,{y:safe.top}),Object.assign({},candidate,{y:safe.bottom-size.height})] : [candidate]);
        const fits = candidate => {
          if (candidate.room < (candidate.axis === 'y' ? size.height : size.width)) return false;
          if (!interactive) return true;
          const cx=Math.max(safe.left,Math.min(candidate.x,safe.right-size.width)),cy=Math.max(safe.top,Math.min(candidate.y,safe.bottom-size.height));
          return allowedTargets.every(node=>{
            const r=rect(node);
            return !r || cx+size.width<=r.left-viewport.left-4 || cx>=r.right-viewport.left+4 || cy+size.height<=r.top-viewport.top-4 || cy>=r.bottom-viewport.top+4;
          });
        };
        let best = spaces().filter(fits).sort((a,b) => b.room-a.room)[0];
        if (!best && interactive) {
          const side=Math.max(box.left-safe.left-12,safe.right-box.right-12);
          if (side>=180) card.style.width=Math.min(size.width,side)+'px';
          else {
            const vertical=Math.max(box.top-safe.top-12,safe.bottom-box.bottom-12);
            if (vertical>=144) card.style.maxHeight=vertical+'px';
          }
          size=card.getBoundingClientRect();
          best=spaces().filter(fits).sort((a,b)=>b.room-a.room)[0];
        }
        if (best) { x=best.x; y=best.y; }
        else y = box.top > (safe.top+safe.bottom)/2 ? safe.top : safe.bottom-size.height;
      }
      card.style.left = Math.max(safe.left,Math.min(x,safe.right-size.width)) + 'px';
      card.style.top = Math.max(safe.top,Math.min(y,safe.bottom-size.height)) + 'px';
    }
    function show(next) {
      model = next;
      if (!model) { hide(); return; }
      const nextSignature = [model.guideId,model.stepId,model.index,model.replay,model.epoch,model.context].join(':');
      const changed = nextSignature !== signature;
      signature = nextSignature;
      const nextInteractive=!!model.mode && model.mode!=='currency' && !model.replay;
      if (interactive !== nextInteractive && dialog.open) {
        if (dialog.hasAttribute('popover')) { if (dialog.matches(':popover-open')) dialog.hidePopover(); dialog.removeAttribute('open'); dialog.removeAttribute('popover'); }
        else dialog.close();
      }
      interactive=nextInteractive;
      dialog.dataset.interactive=String(interactive);
      dialog.setAttribute('aria-modal',String(!interactive));
      dialog.setAttribute('role',interactive ? 'region' : 'dialog');
      dialog.dataset.guide = model.guideId || '';
      dialog.dataset.step = model.stepId || '';
      dialog.dataset.replay = String(!!model.replay);
      q('[data-guide-area]').textContent = (model.replay ? 'Replay · ' : '') + (model.title || 'Area guide');
      q('[data-guide-count]').textContent = (model.index+1) + ' / ' + model.total;
      q('#wx-guide-heading').textContent = model.heading;
      q('#wx-guide-body').textContent = model.body;
      q('[data-guide-quote]').hidden=!model.quoteText;
      q('[data-guide-quote]').textContent=model.quoteText || '';
      q('[data-guide-leave]').textContent = model.saveFailure ? 'Game options' : model.replay ? 'Close replay' : 'Close';
      const reward = q('[data-guide-reward]');
      reward.hidden = !model.rewardText;
      reward.textContent = model.rewardText || '';
      if (!dialog.open) {
        previousFocus=document.activeElement;
        if (interactive) {
          if (typeof dialog.showPopover === 'function') { dialog.setAttribute('popover','manual'); dialog.showPopover(); dialog.setAttribute('open',''); }
          else dialog.show();
        } else { options.parent.append(dialog); dialog.showModal(); }
      }
      layout();
      if (changed) {
        q('.wx-guide-copy').scrollTop=0;
        pendingStepFocus = true;
        focusStep();
        q('[data-guide-announcement]').textContent = (model.title || 'Area guide') + '. Step ' + (model.index+1) + ' of ' + model.total + '. ' + model.heading + '. ' + model.body + (model.quoteText ? '. '+model.quoteText : '');
      }
    }
    function focusStep() {
      if (!pendingStepFocus || submitting || !dialog.open) return;
      const node=interactive && !model.saveFailure ? focusables()[0] : q('[data-guide-next]');
      if (!node || node.disabled) return;
      pendingStepFocus = false;
      node.focus({preventScroll:true});
    }
    function hide() {
      if (target) {observer.unobserve(target);target.removeAttribute('data-guide-target');}
      restoreIsolation(); allowedTargets=[];
      fallbackSurface?.classList.remove('wx-guide-surface'); fallbackSurface=null;
      model = null; signature = ''; target = null; pendingStepFocus = false;
      if (frame) root.cancelAnimationFrame(frame);
      frame = 0;
      if (dialog.hasAttribute('popover')) { if (dialog.matches(':popover-open')) dialog.hidePopover(); dialog.removeAttribute('open'); dialog.removeAttribute('popover'); }
      else if (dialog.open) dialog.close();
      if (dialog.parentElement !== options.parent) options.parent.append(dialog);
      if (previousFocus?.isConnected && previousFocus.getClientRects().length) previousFocus.focus({preventScroll:true});
      previousFocus = null;
    }
    function canLeave() { return !!model && (model.replay || (model.canLeave ?? !model.mandatory)); }
    function leave() { if (model && !submitting && (canLeave() || model.saveFailure)) options.onLeave(model); }
    q('[data-guide-leave]').addEventListener('click',leave,{signal:abort.signal});
    q('[data-guide-settings]').addEventListener('click',()=>{if(model&&!submitting)options.onSettings?.(model);},{signal:abort.signal});
    q('[data-guide-next]').addEventListener('click',() => {
      if (!model || submitting) return;
      if (missing && !model.saveFailure) { if (interactive) options.onLocate?.(model); layout(); return; }
      submitting = true;
      try { model.saveFailure ? options.onRetry(model) : options.onNext(model); }
      finally {
        submitting = false;
        if (dialog.open) {
          layout();
          if (!dialog.contains(document.activeElement) || document.activeElement === dialog) pendingStepFocus = true;
          focusStep();
        }
      }
    },{signal:abort.signal});
    dialog.addEventListener('cancel',event => { event.preventDefault(); (options.onBack || leave)(model); },{signal:abort.signal});
    document.addEventListener('keydown',event => {
      if (!dialog.open) return;
      if (interactive && event.key === 'Escape') { event.preventDefault(); event.stopImmediatePropagation(); (options.onBack || leave)(model); return; }
      if (event.key !== 'Tab') return;
      const buttons=focusables();
      if (!buttons.length) { event.preventDefault(); return; }
      const first=buttons[0],last=buttons[buttons.length-1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus({preventScroll:true}); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus({preventScroll:true}); }
    },{capture:true,signal:abort.signal});
    ['pointerdown','click'].forEach(type=>document.addEventListener(type,event=>{
      if (!dialog.open || !interactive || allowedNode(event.target)) return;
      event.preventDefault(); event.stopImmediatePropagation();
    },{capture:true,signal:abort.signal}));
    dialog.addEventListener('click',event => event.stopPropagation(),{signal:abort.signal});
    root.addEventListener('resize',schedule,{signal:abort.signal});
    root.visualViewport?.addEventListener('resize',schedule,{signal:abort.signal});
    root.visualViewport?.addEventListener('scroll',schedule,{signal:abort.signal});
    document.addEventListener('scroll',schedule,{capture:true,passive:true,signal:abort.signal});
    return {show,hide,refresh:schedule,isOpen:()=>dialog.open,handleBack() { if (!dialog.open) return false; interactive && options.onBack ? options.onBack(model) : leave(); return true; },dispose() { hide(); disposed=true; observer.disconnect(); abort.abort(); dialog.remove(); }};
  }
  root.WayfarersOnboardingUI = {create};
}(typeof window !== 'undefined' ? window : globalThis));
