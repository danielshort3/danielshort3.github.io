/* ===================================================================
   project-page.js
   Purpose: Render standalone project detail pages using PROJECTS data
=================================================================== */
(() => {
  'use strict';

  const ready = () => {
    const slot = document.querySelector('[data-project-slot]');
    if (!slot || !Array.isArray(window.PROJECTS)) return;
    const projectId = getProjectId();
    const project = window.PROJECTS.find(p => p.id === projectId);

    if (!project) {
      slot.innerHTML = missingMarkup();
      return;
    }

    hydrateModal(project);
    slot.innerHTML = buildMarkup(project);
    if (typeof window.activateGifVideo === 'function') {
      const visual = slot.querySelector('.project-hero-visual');
      window.activateGifVideo(visual);
    }
    bindModalTrigger(project.id);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }

  function getProjectId() {
    const fromDataset = document.body?.dataset?.projectId;
    if (fromDataset) return fromDataset;
    const searchId = new URLSearchParams(window.location.search).get('project');
    if (searchId) return searchId;
    const match = window.location.pathname.match(/\/projects\/(.+?)(?:\.html)?$/);
    return match ? match[1] : null;
  }

  function buildMarkup(project) {
    const metric = getMetric(project);
    const media = typeof window.projectMedia === 'function'
      ? window.projectMedia(project)
      : `<img src="${project.image || 'img/projects/placeholder.png'}" alt="${project.title}">`;
    const tools = (project.tools || []).map(t => `<span class="project-badge">${t}</span>`).join('');
    const actions = (project.actions || []).map(item => `<li>${item}</li>`).join('');
    const results = (project.results || []).map(item => `<li>${item}</li>`).join('');
    const resources = buildResources(project);

    return `
      <section class="project-hero-section">
        <div class="wrapper project-hero-grid">
          <div class="project-hero-copy">
            <p class="project-eyebrow">Case Study</p>
            <h1>${project.title}</h1>
            <p class="project-subtitle">${project.subtitle || ''}</p>
            ${metric ? `<div class="project-metric-pill">${metric}</div>` : ''}
            <div class="project-tool-badges" aria-label="Tools">${tools}</div>
            <div class="project-hero-ctas cta-group">
              <button type="button" class="btn-primary" data-open-project="${project.id}">View walkthrough</button>
              <a class="btn-secondary" href="portfolio.html?project=${project.id}">Back to portfolio</a>
            </div>
          </div>
          <div class="project-hero-visual">
            ${media}
          </div>
        </div>
      </section>
      <section class="project-body surface-band">
        <div class="wrapper project-detail-grid">
          <article>
            <h2>Problem</h2>
            <p>${project.problem || ''}</p>
          </article>
          <article>
            <h2>Actions</h2>
            <ul>${actions}</ul>
          </article>
          <article>
            <h2>Results</h2>
            <ul>${results}</ul>
          </article>
          <article class="project-resource-card">
            <h2>Evidence locker</h2>
            <ul>${resources}</ul>
          </article>
        </div>
      </section>`;
  }

  function buildResources(project) {
    if (!Array.isArray(project.resources) || !project.resources.length) {
      return '<li>No public artifacts available.</li>';
    }
    return project.resources.map(resource => {
      const label = resource.label || 'View resource';
      return `<li>
        <a href="${resource.url}" target="_blank" rel="noopener">${label}</a>
        <small>${project.title}</small>
      </li>`;
    }).join('');
  }

  function bindModalTrigger(projectId) {
    const btn = document.querySelector('[data-open-project]');
    if (!btn || typeof window.openModal !== 'function') return;
    btn.addEventListener('click', () => {
      window.openModal(projectId);
    });
  }

  function hydrateModal(project) {
    if (typeof window.generateProjectModal !== 'function') return;
    const host = document.getElementById('modals') || (() => {
      const div = document.createElement('div');
      div.id = 'modals';
      document.body.appendChild(div);
      return div;
    })();
    if (document.getElementById(`${project.id}-modal`)) return;
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.id = `${project.id}-modal`;
    modal.innerHTML = window.generateProjectModal(project);
    host.appendChild(modal);
  }

  function getMetric(project) {
    if (project.metric || project.metricHeadline) return (project.metricHeadline || project.metric).trim();
    const pool = Array.isArray(project.results) ? project.results.filter(Boolean) : [];
    const raw = pool[0] || '';
    if (!raw) return '';
    const cleaned = raw.replace(/^[-•\s]+/, '').trim();
    if (!cleaned) return '';
    const parts = cleaned.split(/\s[–—-]\s/);
    let metric = parts[0]?.trim() || cleaned;
    if (!metric && parts.length > 1) metric = parts[1].trim();
    if (metric.length > 90) {
      metric = `${metric.slice(0, 87)}…`;
    }
    return metric;
  }

  function missingMarkup() {
    return `
      <div class="wrapper project-missing">
        <h1>Project not found</h1>
        <p>The requested case study is unavailable. Head back to the <a href="portfolio.html">portfolio</a> to browse current work.</p>
      </div>`;
  }
})();
