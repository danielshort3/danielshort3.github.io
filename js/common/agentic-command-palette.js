(function () {
  'use strict';

  var PALETTE_ID = 'agentic-command-palette';
  var OPEN_EVENT = 'agentic:open-command-palette';
  var COMMANDS = [
    {
      label: 'Resume',
      href: '/resume',
      description: 'Experience, impact metrics, and downloadable CV.'
    },
    {
      label: 'Projects',
      href: '/projects',
      description: 'Bento overview of ETL, NLP, and computer vision work.'
    },
    {
      label: 'Contact',
      href: '/contact',
      description: 'Email and professional networking channels.'
    },
    {
      label: 'GitHub',
      href: 'https://github.com/danielshort3',
      description: 'Code samples and deployed demos.'
    }
  ];

  function byId(id) {
    return document.getElementById(id);
  }

  function getStateNodes() {
    var palette = byId(PALETTE_ID);
    if (!palette) return null;
    var input = palette.querySelector('[data-agentic-search]');
    var list = palette.querySelector('[data-agentic-results]');
    return { palette: palette, input: input, list: list };
  }

  function renderResults(nodes, query) {
    if (!nodes || !nodes.list) return;
    var q = String(query || '').trim().toLowerCase();
    var matches = COMMANDS.filter(function (item) {
      if (!q) return true;
      return (item.label + ' ' + item.description).toLowerCase().indexOf(q) !== -1;
    });

    if (!matches.length) {
      nodes.list.innerHTML = '<li class="agentic-palette-link"><p class="agentic-palette-label">No match</p><p class="agentic-palette-description">Try Resume, Projects, Contact, or GitHub.</p></li>';
      return;
    }

    nodes.list.innerHTML = matches
      .map(function (item) {
        return (
          '<li>' +
          '<a class="agentic-palette-link" href="' + item.href + '">' +
          '<p class="agentic-palette-label">' + item.label + '</p>' +
          '<p class="agentic-palette-description">' + item.description + '</p>' +
          '</a>' +
          '</li>'
        );
      })
      .join('');

    Array.prototype.forEach.call(nodes.list.querySelectorAll('a'), function (anchor) {
      anchor.addEventListener('click', closePalette);
    });
  }

  function openPalette() {
    var nodes = getStateNodes();
    if (!nodes) return;
    nodes.palette.hidden = false;
    renderResults(nodes, '');
    if (nodes.input) {
      nodes.input.value = '';
      nodes.input.focus();
    }
  }

  function closePalette() {
    var nodes = getStateNodes();
    if (!nodes) return;
    nodes.palette.hidden = true;
  }

  function onGlobalKeydown(event) {
    var key = String(event.key || '').toLowerCase();
    if ((event.metaKey || event.ctrlKey) && key === 'k') {
      event.preventDefault();
      var nodes = getStateNodes();
      if (!nodes) return;
      if (nodes.palette.hidden) {
        openPalette();
      } else {
        closePalette();
      }
      return;
    }

    if (key === 'escape') {
      closePalette();
    }
  }

  function init() {
    var nodes = getStateNodes();
    if (!nodes) return;

    var openButton = document.querySelector('[data-agentic-open-palette]');
    if (openButton) {
      openButton.addEventListener('click', openPalette);
    }

    var closeButton = nodes.palette.querySelector('[data-agentic-close-palette]');
    if (closeButton) {
      closeButton.addEventListener('click', closePalette);
    }

    if (nodes.input) {
      nodes.input.addEventListener('input', function () {
        renderResults(nodes, nodes.input.value);
      });
    }

    nodes.palette.addEventListener('click', function (event) {
      if (event.target === nodes.palette) {
        closePalette();
      }
    });

    window.addEventListener('keydown', onGlobalKeydown);
    window.addEventListener(OPEN_EVENT, openPalette);
    renderResults(nodes, '');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
