(function(){
  try {
    var KEY = 'site-theme';
    var prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
    var stored = localStorage.getItem(KEY);
    var theme = stored || (prefersLight ? 'light' : 'dark');
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.setProperty('color-scheme', theme === 'light' ? 'light dark' : 'dark light');
  } catch (err) {}
})();
