import { setupAreaDetail } from './ad-verification-hotspots-areas.js';

const data = JSON.parse(document.getElementById('campaign-data').textContent);
const movie = document.getElementById('movie');
const toggle = document.getElementById('toggle');
const action = document.getElementById('action');
const position = document.getElementById('position');
const clock = document.getElementById('time');
const narrow = matchMedia('(max-width: 700px)');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const formatNumber = new Intl.NumberFormat('en-US');
const formatTime = (seconds) => '0:' + String(Math.floor(seconds)).padStart(2, '0');
const elements = ['ads', 'visits'].map(id => document.getElementById(id));
const dayCounter = document.getElementById('day');
const reading = document.getElementById('reading');
const errorBox = document.getElementById('error');
const errorMessage = document.getElementById('error-message');
const retry = document.getElementById('retry');
const areaDialog = document.getElementById('area-dialog');
const areas = setupAreaDetail(data);
const views = Object.entries(data.panels).map(([name, box]) => {
  const canvas = document.getElementById(name);
  canvas.width = box[2] - box[0];
  canvas.height = box[3] - box[1];
  return { context: canvas.getContext('2d', {alpha: false}), box, width: canvas.width, height: canvas.height };
});
const duration = data.frames / data.fps;
const enhanced = views.every(view => view.context);
position.max = duration;
document.querySelector('.skip-link').href = location.pathname + location.search + '#main';
let lastFrame = -1;
let callback = null;
let userRetry = false;

function render(seconds, force = false) {
  const frame = Math.min(data.frames - 1, Math.max(0, Math.floor(seconds * data.fps + 0.0001)));
  if (!force && frame === lastFrame) return;
  lastFrame = frame;
  position.value = seconds;
  const displayedTime = movie.ended ? Math.round(duration) : seconds;
  const time = formatTime(displayedTime) + ' / ' + formatTime(Math.round(duration));
  if (clock.textContent !== time) clock.textContent = time;
  const day = data.day[frame];
  const dayText = day >= 32 && day <= 36 ? 'Days 32–36 of 37' : 'Day ' + day + ' of 37';
  if (dayCounter.textContent !== dayText) dayCounter.textContent = dayText;
  position.setAttribute('aria-valuetext', dayText + ', ' + formatTime(displayedTime));
  const labels = [formatNumber.format(data.ads[frame]), formatNumber.format(data.visits[frame])];
  const summary = labels[0] + ' phones were shown an ad. ' + labels[1] + ' visited Grand Junction.';
  if (reading.textContent !== summary) reading.textContent = summary;
  areas.update(frame);
  if (!narrow.matches || movie.readyState < 2) return;
  labels.forEach((value, index) => {
    if (elements[index].textContent !== value) elements[index].textContent = value;
  });
  for (const view of views) {
    view.context.drawImage(movie, view.box[0], view.box[1], view.width, view.height, 0, 0, view.width, view.height);
  }
}

function schedule() {
  if (movie.paused || movie.ended || movie.error || callback !== null) return;
  if (movie.requestVideoFrameCallback) {
    callback = movie.requestVideoFrameCallback((now, metadata) => {
      callback = null;
      render(metadata.mediaTime);
      schedule();
    });
  } else {
    callback = requestAnimationFrame(() => {
      callback = null;
      render(movie.currentTime);
      schedule();
    });
  }
}

function cancelFrame() {
  if (callback === null) return;
  if (movie.cancelVideoFrameCallback) movie.cancelVideoFrameCallback(callback);
  else cancelAnimationFrame(callback);
  callback = null;
}

function reflect() {
  if (movie.error) return;
  const playing = !movie.paused && !movie.ended;
  const label = playing ? 'Pause' : movie.ended ? 'Replay' : 'Play';
  document.body.dataset.playback = playing ? 'playing' : movie.ended ? 'ended' : 'paused';
  action.textContent = label;
  toggle.setAttribute('aria-label', label + ' campaign');
  document.getElementById('icon').setAttribute('d', playing ? 'M3 2h3.5v12H3zm6.5 0H13v12H9.5z' : 'M4 1.5 14 8 4 14.5Z');
  if (playing) schedule();
  else cancelFrame();
}

function refreshLayout() {
  const controls = document.querySelector('.transport');
  const explore = document.getElementById('explore-areas');
  if (areas.isOpen()) document.getElementById('detail-playback').append(controls);
  else if (narrow.matches) document.querySelector('.totals').after(controls);
  else document.querySelector('.mobile').after(controls);
  if (narrow.matches) document.getElementById('city-figure').append(explore);
  else if (!areas.isOpen()) controls.after(explore);
  document.getElementById('desktop-heading').hidden = narrow.matches;
  movie.setAttribute('aria-hidden', String(narrow.matches));
  render(movie.currentTime, true);
}

function start() {
  if (movie.ended) movie.currentTime = 0;
  movie.play().catch(error => {
    if (movie.error || error.name === 'NotSupportedError') showError();
    else reflect();
  });
}

function showError() {
  const focusPlayer = document.activeElement.closest('.screen, .transport, .detail-dialog, .error');
  cancelFrame();
  if (areaDialog.open) areaDialog.close();
  document.body.classList.remove('ready');
  document.body.classList.add('video-error');
  document.body.dataset.playback = 'error';
  errorMessage.textContent = 'The campaign video could not load.';
  errorBox.hidden = false;
  retry.disabled = false;
  if (focusPlayer) requestAnimationFrame(() => retry.focus({preventScroll: true}));
}

function initialize() {
  const restoreFocus = errorBox.contains(document.activeElement) || userRetry;
  userRetry = false;
  document.body.classList.remove('video-error');
  errorBox.hidden = true;
  retry.disabled = false;
  if (!enhanced) {
    if (restoreFocus) movie.focus({preventScroll: true});
    return;
  }
  document.body.classList.add('ready');
  movie.controls = false;
  refreshLayout();
  reflect();
  if (restoreFocus) toggle.focus({preventScroll: true});
  if (!reduced.matches) start();
}

movie.addEventListener('error', showError);
movie.addEventListener('loadeddata', initialize);
retry.addEventListener('click', () => {
  userRetry = true;
  retry.disabled = true;
  errorMessage.textContent = 'Loading the campaign video…';
  movie.load();
});

if (enhanced) {
  movie.addEventListener('play', reflect);
  movie.addEventListener('pause', reflect);
  movie.addEventListener('ended', () => { render(duration, true); reflect(); });
  movie.addEventListener('seeked', () => { render(movie.currentTime, true); reflect(); });
  toggle.addEventListener('click', () => { if (movie.paused || movie.ended) start(); else movie.pause(); });
  position.addEventListener('input', () => { movie.currentTime = Number(position.value); });
  narrow.addEventListener('change', refreshLayout);
  areaDialog.addEventListener('areachange', refreshLayout);
  reduced.addEventListener('change', event => { if (event.matches) movie.pause(); });
}
if (movie.error) showError();
else if (movie.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) initialize();
