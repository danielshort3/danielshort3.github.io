export function setupAreaDetail(data) {
  const dialog = document.getElementById('area-dialog');
  const select = document.getElementById('area-select');
  const svg = document.getElementById('detail-map');
  const amount = document.getElementById('area-count');
  const unit = document.getElementById('area-unit');
  const source = document.getElementById('area-source');
  const launch = document.getElementById('explore-areas');
  const catalog = new Map(data.areas.map(area => [area.id, area]));
  const ns = 'http://www.w3.org/2000/svg';
  let active = null;
  let circles = [];
  let pulses;
  let visible = 0;
  let frame = 0;
  let opener = launch;

  function element(tag, attributes = {}, message) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, value));
    if (message !== undefined) node.textContent = message;
    return node;
  }

  function choose(id) {
    active = catalog.get(id);
    select.value = id;
    source.href = active.infoUrl;
    svg.replaceChildren();
    for (const [name, path] of Object.entries(active.paths)) {
      if (path) svg.append(element('path', {d: path, class: 'district-' + name, 'fill-rule': 'evenodd'}));
    }
    if (active.label) {
      const [street, x, y, angle] = active.label;
      svg.append(element('text', {class: 'road-label', transform: 'translate(' + x + ' ' + y + ') rotate(' + angle + ')', y: -4, 'text-anchor': 'middle'}, street));
    }
    const marks = element('g', {class: 'district-marks'});
    circles = active.points.map(([x, y]) => {
      const dot = element('circle', {cx: x, cy: y, r: active.radius, display: 'none'});
      marks.append(dot);
      return dot;
    });
    pulses = element('g', {class: 'district-pulse'});
    svg.append(marks, pulses);
    const [length, label] = active.scale;
    const end = active.width - 17;
    const base = active.height - 15;
    svg.append(element('path', {class: 'scale-bar', d: 'M' + (end - length) + ',' + (base-5) + 'v5h' + length + 'v-5'}));
    svg.append(element('text', {class: 'scale-label', x: end - length/2, y: base-8, 'text-anchor': 'middle'}, label));
    visible = 0;
    update(frame);
  }

  function update(nextFrame) {
    frame = nextFrame;
    if (!active || !dialog.open) return;
    const points = active.points;
    while (visible && points[visible-1][2] > frame) circles[--visible].setAttribute('display', 'none');
    while (visible < points.length && points[visible][2] <= frame) circles[visible++].removeAttribute('display');
    const value = String(visible);
    if (amount.textContent !== value || svg.dataset.area !== active.id) {
      amount.textContent = value;
      unit.textContent = visible === 1 ? 'simulated visit' : 'simulated visits';
      svg.setAttribute('aria-label', active.name + ', ' + value + ' simulated visits');
      svg.dataset.area = active.id;
    }
    const rings = [];
    const life = 27;
    for (let i = visible-1; i >= 0 && frame-points[i][2] < life; --i) {
      const age = (frame-points[i][2]) / life;
      rings.push(element('circle', {cx: points[i][0], cy: points[i][1], r: active.radius+2+10*age, opacity: (.9*(1-age)).toFixed(3)}));
    }
    pulses.replaceChildren(...rings);
  }

  function open(id, button) {
    opener = button;
    if (!dialog.open) dialog.showModal();
    choose(id);
    dialog.dispatchEvent(new Event('areachange'));
    select.focus({preventScroll: true});
  }

  function putTargets(layer, originalBox) {
    const [x, y, width, height] = originalBox;
    for (const area of data.areas) {
      const box = area.labelBounds;
      const target = document.createElement('button');
      const label = 'Zoom in on ' + area.name;
      target.type = 'button';
      target.className = 'area-target';
      target.dataset.area = area.id;
      target.setAttribute('aria-label', label);
      target.setAttribute('aria-haspopup', 'dialog');
      target.title = label;
      target.style.left = 100*(box[0]-x)/width + '%';
      target.style.top = 100*(box[1]-y)/height + '%';
      target.style.width = 100*(box[2]-box[0])/width + '%';
      target.style.height = 100*(box[3]-box[1])/height + '%';
      target.addEventListener('click', () => open(area.id, target));
      layer.append(target);
    }
  }

  data.areas.forEach(area => select.add(new Option(area.name, area.id)));
  putTargets(document.getElementById('desktop-areas'), [0, 0, data.videoSize[0]/data.nativeScale, 437]);
  const local = data.panels.city.map(coordinate => coordinate/data.nativeScale);
  putTargets(document.getElementById('mobile-areas'), [local[0], local[1], local[2]-local[0], local[3]-local[1]]);
  launch.addEventListener('click', () => open(active ? active.id : data.areas[0].id, launch));
  select.addEventListener('change', () => choose(select.value));
  document.getElementById('close-detail').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => {
    dialog.dispatchEvent(new Event('areachange'));
    (opener.getClientRects().length ? opener : launch).focus({preventScroll: true});
  });
  return {update, isOpen: () => dialog.open};
}
