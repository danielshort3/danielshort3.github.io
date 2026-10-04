(function(root) {
  'use strict';

  // Station pictures describe production; they never advance or award it.
  const images = new Map();
  const failedImages = new Set();
  const renderers = new Set();
  let frame = null;
  let last = 0;
  function image(file, base) {
    if (!file) return null;
    const url = /^(?:https?:|\/|data:)/.test(file) ? file : base + file;
    if (!images.has(url)) {
      const item = new root.Image();
      item.onload = () => renderers.forEach(renderer => renderer.invalidate());
      item.onerror = () => {failedImages.add(url);renderers.forEach(renderer=>renderer.invalidate());};
      item.src = url;
      images.set(url, item);
    }
    const result = images.get(url);
    return result.complete && result.naturalWidth ? result : null;
  }
  function schedule() {
    if (frame !== null || document.hidden || !renderers.size) return;
    frame = root.requestAnimationFrame(tick);
  }
  function tick(now) {
    frame = null;
    if (document.hidden) return;
    if (now - last >= 32) {
      last = now;
      renderers.forEach(renderer => renderer.paint(now));
    }
    if ([...renderers].some(renderer => renderer.animated())) schedule();
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && frame !== null) { root.cancelAnimationFrame(frame); frame = null; }
    else renderers.forEach(renderer => renderer.invalidate());
  });
  function create(canvas, options) {
    const settings = options || {};
    const context = canvas.getContext('2d', {alpha:false});
    const base = String(settings.assetBase || 'img/wayfarers-guild/').replace(/\/?$/, '/');
    const motion = root.matchMedia?.('(prefers-reduced-motion: reduce)');
    let model = {};
    let visible = true;
    let dirty = true;
    let quiet = !!settings.quiet;
    let disposed = false;
    let paints = 0;
    let hasArt = false;
    canvas.dataset.sceneStatus='loading';
    const logicalWidth = 384;
    function reduced() { return quiet || !!motion?.matches; }
    function art() {
      const catalog = settings.art || root.WayfarersStationArt || {};
      const area = catalog.areas?.[model.areaId || settings.areaId] || {};
      let station = area.stations?.[model.id || settings.stationId] || catalog.stations?.[(model.areaId || settings.areaId) + ':' + (model.id || settings.stationId)] || {};
      if(model.legacy && (model.id || settings.stationId)==='quarry:tool-forge')station=Object.assign({},station,{background:area.underground,machine:'c-track-quarry-furnace.webp',addition:null});
      return {catalog,area,station};
    }
    function layer(spec, fallback) {
      if (!spec) return;
      const entry = typeof spec === 'string' ? {file:spec} : spec;
      const source = image(entry.file || entry.image || entry.src, base);
      if (!source) return;
      const target = entry.destination || entry.bounds || fallback;
      const crop = entry.crop || entry.source;
      if (crop) context.drawImage(source, ...crop, ...target);
      else context.drawImage(source, ...target);
    }
    function paint(now) {
      if (disposed || !visible || !context || (!dirty && reduced())) return;
      const box = canvas.getBoundingClientRect();
      if (!box.width || !box.height) return;
      const logicalHeight = settings.surface ? 320 : 208;
      if (canvas.width !== logicalWidth || canvas.height !== logicalHeight) { canvas.width = logicalWidth; canvas.height = logicalHeight; }
      context.imageSmoothingEnabled = false;
      const {catalog,area,station} = art();
      context.fillStyle = settings.surface ? '#58b8ed' : '#102b46';
      context.fillRect(0,0,logicalWidth,logicalHeight);
      const full = [0,0,logicalWidth,logicalHeight];
      const offset = settings.surface ? 112 : 0;
      const stationBounds = [0,offset,384,208];
      const floor = offset + 174;
      const variant = Math.min((station.variants || []).length - 1, Math.floor((Number(model.mastery) || 0) / 25));
      // The station already carries its mountain horizon. Extend only the open
      // sky above it, so the first tall scene never stacks two landscapes.
      const sky=settings.surface ? image(typeof area.sky==='string' ? area.sky : area.sky?.file,base) : null;
      if(settings.surface) {
        if(sky)context.drawImage(sky,0,0,sky.naturalWidth,sky.naturalHeight,0,0,384,117);
      }
      layer(station.background || station.scene || area.underground, stationBounds);
      if(settings.surface) {
        const background=image(typeof station.background==='string' ? station.background : station.background?.file,base);
        // A short sky overlap also hides tiny sheet-export seams at the top.
        if(background)context.drawImage(background,0,5,384,203,0,offset+5,384,203);
        if(sky)context.drawImage(sky,0,0,sky.naturalWidth,sky.naturalHeight,0,0,384,117);
      }
      layer(station.terrain || area.terrain, stationBounds);
      function anchored(spec, anchor, maxWidth, maxHeight) {
        if(!spec)return;
        const entry=typeof spec==='string' ? {file:spec} : spec;
        const source=image(entry.file || entry.image,base);
        if(!source)return;
        const scale=Math.min(maxWidth/source.naturalWidth,maxHeight/source.naturalHeight,1);
        const w=Math.round(source.naturalWidth*scale),h=Math.round(source.naturalHeight*scale);
        layer(entry,[Math.round(anchor[0]-w/2),Math.round(offset+172-h),w,h]);
      }
      const cartStation=!!catalog.movingCart && /(?:hauling|porter-camp)$/.test(model.id || settings.stationId || '');
      if(!cartStation)anchored(station.variants?.[Math.max(0,variant)] || station.machine,station.anchors?.machine || [244,150],180,112);
      const showAddition=(Number(model.mastery) || 0)>=15 || model.visualTier>1;
      if(showAddition)anchored(station.addition,station.anchors?.addition || [290,156],84,86);
      (station.layers || []).forEach(entry => layer(entry, full));
      const working = model.working !== false && Number(model.rate ?? model.output ?? 0) > 0;
      const boosted = model.boost===true || !!model.boost?.active;
      const speed = Math.min(2.5, Math.max(.4, Math.sqrt(Math.max(1,Number(model.rate) || 1)) * (boosted ? 1.35 : 1)));
      const phase = reduced() || !working ? .25 : now / 1000 * speed;
      if(cartStation) {
        const moving=catalog.movingCart,p=reduced() || !working ? .35 : phase*.13%1,t=p<.5?p*2:(1-p)*2;
        const w=moving.width || 82,h=moving.height || 54;
        layer(moving.file,[Math.round(186+t*106),offset+174-h,w,h]);
      }
      const workerName = station.workerRole || station.worker || settings.worker || 'miner';
      const sharedWorkers=catalog.workers;
      const worker = sharedWorkers?.[workerName] || sharedWorkers?.default || (sharedWorkers?.file ? Object.assign({},sharedWorkers,{row:(sharedWorkers.roles || ['miner','porter','scout','artisan','scholar','sailor']).indexOf(workerName),frames:[1,2],idle:[0],columns:4}) : null);
      if (worker) {
        const source = image(worker.file || worker.image,base);
        if (source) {
          const w = worker.cellWidth || 64, h = worker.cellHeight || 96;
          const sequence = working ? worker.work || worker.frames || [0] : worker.idle || [0];
          const pose = sequence[Math.floor(phase * (worker.fps || 6)) % sequence.length];
          const cell = typeof pose === 'object' ? pose : {x:(pose % (worker.columns || Math.floor(source.naturalWidth/w))) * w,y:(worker.row ?? Math.floor(pose / (worker.columns || Math.floor(source.naturalWidth/w)))) * h,w,h};
          const sourceAnchor = station.anchors?.worker || [134,156];
          const anchor = [sourceAnchor[0],offset+sourceAnchor[1]+18];
          const size = station.workerSize || [64,96];
          context.drawImage(source,cell.x,cell.y,cell.w || w,cell.h || h,Math.round(anchor[0]-size[0]/2),Math.round(anchor[1]-size[1]),size[0],size[1]);
        }
      }
      if (station.moving) {
        const moving = station.moving;
        const track = moving.path || [[120,floor-35],[300,floor-35]];
        const t = reduced() || !working ? .3 : (phase*.17)%1;
        const distance = t < .5 ? t*2 : (1-t)*2;
        layer(moving.file || moving.image,[Math.round(track[0][0]+(track[1][0]-track[0][0])*distance),Math.round(track[0][1]+(track[1][1]-track[0][1])*distance),moving.width || 70,moving.height || 45]);
      }
      if (working && !reduced()) {
        const effect = station.anchors?.effect || [82,floor-50];
        context.fillStyle = boosted ? '#ffe777' : '#ffd05d';
        for (let i=0;i<(boosted ? 7 : 3);i+=1) {
          const p = (phase*.8+i*.31)%1;
          context.globalAlpha = 1-p;
          context.fillRect(Math.round(effect[0]+Math.sin(i*2.7)*p*22),Math.round(effect[1]-p*30),2,2);
        }
        context.globalAlpha = 1;
      }
      const required=[station.background,cartStation ? catalog.movingCart.file : station.machine,worker?.file || worker?.image,...(showAddition ? [station.addition] : []),...(settings.surface ? [area.sky] : [])].filter(Boolean);
      hasArt=!!station.background && required.every(file=>!!image(typeof file==='string' ? file : file.file,base));
      const failed=required.some(file=>failedImages.has(/^(?:https?:|\/|data:)/.test(file) ? file : base+file));
      canvas.dataset.sceneStatus = failed ? 'error' : hasArt ? 'ready' : 'loading';
      canvas.dataset.sceneKind = model.areaId || settings.areaId;
      canvas.dataset.stationId = model.id || settings.stationId;
      canvas.dataset.sceneDraws = String(++paints);
      canvas.dataset.sceneReducedMotion = String(reduced());
      canvas.dataset.sceneWorking = String(working);
      dirty = false;
    }
    const renderer = {
      paint,
      animated:() => !disposed && visible && !reduced() && model.working !== false,
      invalidate() {dirty=true;schedule();}
    };
    renderers.add(renderer);
    const observer = root.IntersectionObserver ? new root.IntersectionObserver(entries => {visible=entries.some(entry=>entry.isIntersecting);renderer.invalidate();},{root:settings.scrollRoot || null,rootMargin:'40px'}) : null;
    observer?.observe(canvas);
    const resize = root.ResizeObserver ? new root.ResizeObserver(renderer.invalidate) : null;
    resize?.observe(canvas);
    motion?.addEventListener('change',renderer.invalidate);
    schedule();
    return {
      update(value) {model=Object.assign({},value);renderer.invalidate();},
      setQuiet(value) {quiet=!!value;renderer.invalidate();},
      getStatus() {return {state:disposed?'destroyed':canvas.dataset.sceneStatus,draws:paints,animated:renderer.animated(),hasArt};},
      dispose() {disposed=true;observer?.disconnect();resize?.disconnect();motion?.removeEventListener('change',renderer.invalidate);renderers.delete(renderer);canvas.dataset.sceneStatus='destroyed';}
    };
  }
  root.WayfarersStationScene = {create};
})(typeof globalThis !== 'undefined' ? globalThis : this);
