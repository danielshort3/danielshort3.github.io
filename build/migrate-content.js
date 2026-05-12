#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function writeJson(relPath, value) {
  const absPath = path.join(root, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function normalizeManagedAssetPath(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  return raw
    .replace(/dist\/(site-[a-z-]+)\.[0-9a-f]{8}\.js/gi, 'dist/$1.js')
    .replace(/dist\/(styles(?:-tools)?)\.[0-9a-f]{8}\.css/gi, 'dist/$1.css');
}

function decodeHtml(value) {
  return String(value || '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function stripTags(value) {
  return decodeHtml(String(value || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim());
}

function parseAttributes(fragment) {
  const attrs = {};
  const re = /([:\w-]+)(?:="([^"]*)")?/g;
  let match;
  while ((match = re.exec(String(fragment || '')))) {
    const key = String(match[1] || '').trim();
    if (!key) continue;
    attrs[key] = match[2] == null ? true : decodeHtml(match[2]);
  }
  return attrs;
}

function extractHeadInner(html) {
  const match = /<head\b[^>]*>([\s\S]*?)<\/head>/i.exec(html);
  return match ? match[1] : '';
}

function extractTitle(headInner) {
  const match = /<title>([\s\S]*?)<\/title>/i.exec(headInner);
  return match ? decodeHtml(match[1].trim()) : '';
}

function extractMetaContent(headInner, attrName, attrValue) {
  const re = new RegExp(`<meta\\s+[^>]*${attrName}="${attrValue.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}"[^>]*content="([^"]*)"[^>]*>`, 'i');
  const match = re.exec(headInner);
  return match ? decodeHtml(match[1].trim()) : '';
}

function extractCanonicalPath(headInner) {
  const match = /<link\s+[^>]*rel="canonical"[^>]*href="([^"]+)"[^>]*>/i.exec(headInner);
  if (!match) return '/';
  try {
    const url = new URL(match[1]);
    return url.pathname || '/';
  } catch {
    const raw = String(match[1] || '').trim();
    return raw.startsWith('/') ? raw : `/${raw.replace(/^https?:\/\/[^/]+/i, '')}`;
  }
}

function extractStylesheets(headInner) {
  return [...headInner.matchAll(/<link\s+[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/gi)]
    .map((match) => normalizeManagedAssetPath(match[1]))
    .filter(Boolean);
}

function extractScriptObjects(fragment) {
  return [...String(fragment || '').matchAll(/<script\b([^>]*)><\/script>/gi)]
    .map((match) => {
      const attrs = parseAttributes(match[1]);
      const src = normalizeManagedAssetPath(attrs.src);
      if (!src) return null;
      delete attrs.src;
      const defer = Boolean(attrs.defer);
      delete attrs.defer;
      const otherAttrs = {};
      Object.entries(attrs).forEach(([key, value]) => {
        otherAttrs[key] = typeof value === 'string' ? normalizeManagedAssetPath(value) : value;
      });
      return {
        src,
        ...(defer ? { defer: true } : {}),
        ...(Object.keys(otherAttrs).length ? { attributes: otherAttrs } : {})
      };
    })
    .filter(Boolean);
}

function extractBodyAttributes(html) {
  const match = /<body\b([^>]*)>/i.exec(html);
  return match ? parseAttributes(match[1]) : {};
}

function extractBodyHtml(html) {
  const match = /<\/header>([\s\S]*?)<footer\b/i.exec(html);
  return match ? match[1].trim() : '';
}

function extractTailScripts(html) {
  const match = /<footer\b[\s\S]*?<\/footer>([\s\S]*?)<\/body>/i.exec(html);
  return extractScriptObjects(match ? match[1] : '');
}

function extractHeadScripts(headInner) {
  return extractScriptObjects(headInner);
}

function extractPageDefinition(relPath) {
  const html = read(relPath);
  const headInner = extractHeadInner(html);
  return {
    template: 'raw-body',
    outputPath: relPath,
    title: extractTitle(headInner),
    canonicalPath: extractCanonicalPath(headInner),
    description: extractMetaContent(headInner, 'name', 'description') || extractMetaContent(headInner, 'property', 'og:description'),
    robots: extractMetaContent(headInner, 'name', 'robots') || undefined,
    ogTitle: extractMetaContent(headInner, 'property', 'og:title'),
    ogDescription: extractMetaContent(headInner, 'property', 'og:description'),
    siteName: extractMetaContent(headInner, 'property', 'og:site_name') || undefined,
    ogType: extractMetaContent(headInner, 'property', 'og:type') || 'website',
    bodyAttributes: extractBodyAttributes(html),
    stylesheets: extractStylesheets(headInner),
    headScripts: extractHeadScripts(headInner),
    bottomScripts: extractTailScripts(html),
    bodyHtml: extractBodyHtml(html)
  };
}

function loadProjectsData() {
  const sandbox = { window: {} };
  vm.createContext(sandbox);
  vm.runInContext(read(path.join('js', 'portfolio', 'projects-data.js')), sandbox, { filename: 'js/portfolio/projects-data.js' });
  return {
    projects: Array.isArray(sandbox.window.PROJECTS) ? sandbox.window.PROJECTS : [],
    featuredIds: Array.isArray(sandbox.window.FEATURED_IDS) ? sandbox.window.FEATURED_IDS : []
  };
}

function loadAudienceConfig() {
  const modulePath = path.join(root, 'js', 'common', 'audience-config.js');
  delete require.cache[modulePath];
  return require(modulePath);
}

function extractHeaderFeaturedProjectIds() {
  const headerHtml = read(path.join('build', 'templates', 'header.partial.html'));
  return [...headerHtml.matchAll(/data-project-id="([^"]+)"/g)].map((match) => String(match[1] || '').trim()).filter(Boolean);
}

function extractToolCollections() {
  const html = read(path.join('pages', 'tools.html'));
  const page = extractPageDefinition(path.join('pages', 'tools.html'));
  const heroEyebrowMatch = /<p class="hero-eyebrow">([\s\S]*?)<\/p>/i.exec(page.bodyHtml);
  const resumePanel = /<section class="tools-resume-panel"[\s\S]*?<p class="tools-resume-kicker">([\s\S]*?)<\/p>[\s\S]*?<h2 class="tools-resume-title"[^>]*>([\s\S]*?)<\/h2>[\s\S]*?<a class="btn-secondary tools-resume-dashboard-link" href="([^"]+)">([\s\S]*?)<\/a>/i.exec(page.bodyHtml);

  const categories = [];
  const tools = [];
  const sectionRe = /<section class="tools-category"([^>]*)>([\s\S]*?)<\/section>/gi;
  let sectionMatch;
  let categoryOrder = 0;
  while ((sectionMatch = sectionRe.exec(html))) {
    categoryOrder += 1;
    const sectionAttrs = parseAttributes(sectionMatch[1]);
    const sectionBody = sectionMatch[2];
    const titleMatch = /<h2 class="section-title"[^>]*>([\s\S]*?)<\/h2>/i.exec(sectionBody);
    const descMatch = /<header class="tools-category-head">[\s\S]*?<p>([\s\S]*?)<\/p>/i.exec(sectionBody);
    const category = {
      id: String(sectionAttrs.id || `tools-category-${categoryOrder}`).trim(),
      title: stripTags(titleMatch ? titleMatch[1] : ''),
      description: stripTags(descMatch ? descMatch[1] : ''),
      order: categoryOrder
    };
    categories.push(category);

    const cardRe = /<article class="tool-card"([^>]*)>([\s\S]*?)<\/article>/gi;
    let cardMatch;
    let toolOrder = 0;
    while ((cardMatch = cardRe.exec(sectionBody))) {
      toolOrder += 1;
      const cardAttrs = parseAttributes(cardMatch[1]);
      const cardBody = cardMatch[2];
      const linkMatch = /<a href="([^"]+)">/i.exec(cardBody);
      const titleCardMatch = /<h3>([\s\S]*?)<\/h3>/i.exec(cardBody);
      const summaryMatch = /<\/div>\s*<\/div>\s*<p>([\s\S]*?)<\/p>/i.exec(cardBody);
      const iconMatch = /<span class="tool-icon"[^>]*>([\s\S]*?)<\/span>/i.exec(cardBody);
      const pills = [...cardBody.matchAll(/<span class="tool-pill([^"]*)">([\s\S]*?)<\/span>/gi)].map((pillMatch) => ({
        label: stripTags(pillMatch[2]),
        ...(String(pillMatch[1] || '').includes('tool-pill-local') ? { variant: 'local' } : {})
      }));
      const href = String(linkMatch ? linkMatch[1] : '').trim();
      const slug = href.replace(/\/+$/, '').split('/').pop();
      tools.push({
        slug,
        title: stripTags(titleCardMatch ? titleCardMatch[1] : slug),
        href,
        categoryId: category.id,
        summary: stripTags(summaryMatch ? summaryMatch[1] : ''),
        iconHtml: (iconMatch ? iconMatch[1] : '').trim(),
        pills,
        visibility: cardAttrs['data-tools-visibility'] ? String(cardAttrs['data-tools-visibility']).trim() : 'public',
        hidden: Boolean(cardAttrs.hidden),
        order: toolOrder,
        noindex: Boolean(cardAttrs['data-tools-visibility'])
      });
    }
  }

  return {
    page: {
      id: 'tools',
      template: 'tools-directory',
      outputPath: path.join('pages', 'tools.html'),
      title: page.title,
      canonicalPath: page.canonicalPath,
      description: page.description,
      ogTitle: page.ogTitle,
      ogDescription: page.ogDescription,
      siteName: page.siteName,
      ogType: page.ogType,
      bodyAttributes: page.bodyAttributes,
      stylesheets: page.stylesheets,
      headScripts: page.headScripts,
      bottomScripts: page.bottomScripts,
      heroEyebrow: stripTags(heroEyebrowMatch ? heroEyebrowMatch[1] : 'Tools'),
      heroTitle: 'Tools',
      resumePanel: {
        kicker: stripTags(resumePanel ? resumePanel[1] : 'Resume work'),
        title: stripTags(resumePanel ? resumePanel[2] : 'Continue where you left off'),
        dashboardHref: String(resumePanel ? resumePanel[3] : 'tools-dashboard').trim(),
        dashboardLabel: stripTags(resumePanel ? resumePanel[4] : 'Open dashboard')
      },
      categories
    },
    tools
  };
}

function buildSiteSettings(audienceApi) {
  return {
    siteOrigin: 'https://www.danielshort.me',
    siteName: 'Daniel Short – Data Science & Analytics',
    ownerName: 'Daniel Short',
    themeColor: '#091F3B',
    twitterSite: '@danielshort3',
    email: 'daniel@danielshort.me',
    defaultAudience: audienceApi.defaultAudience || 'analytics',
    ogImage: {
      url: 'https://www.danielshort.me/img/hero/head.png',
      width: '558',
      height: '558',
      alt: 'Portrait photo of Daniel Short'
    }
  };
}

function buildNavigation(featuredProjectIds) {
  return {
    brand: {
      homePath: '/',
      defaultTagline: 'Data Analytics',
      logoSrc: 'img/ui/logo-64.png',
      logoSrcSet: 'img/ui/logo-64.png 1x, img/ui/logo-192.png 3x',
      logoSizes: '64px',
      logoAlt: 'DS logo',
      logoWidth: 64,
      logoHeight: 64
    },
    portfolio: {
      label: 'Portfolio',
      href: 'portfolio',
      header: 'Featured Projects',
      featuredProjectIds,
      links: [
        {
          title: 'View full portfolio',
          subtitle: 'Browse the complete project library',
          href: 'portfolio'
        }
      ]
    },
    resume: {
      label: 'Resume',
      href: 'resume',
      ariaLabel: 'Resume download',
      header: 'Resume shortcuts',
      links: [
        {
          title: 'View Digital Resume',
          subtitle: 'Open the digital resume page',
          href: 'resume',
          dataAttributes: {
            'data-resume-home-link': 'true'
          }
        },
        {
          title: 'Preview PDF',
          subtitle: 'Open the PDF resume page',
          href: 'resume-pdf',
          dataAttributes: {
            'data-resume-preview-link': 'true'
          }
        },
        {
          title: 'Download Resume',
          subtitle: 'Save the latest PDF copy',
          href: 'documents/Resume.pdf',
          download: true,
          dataAttributes: {
            'data-resume-download-link': 'true'
          }
        }
      ]
    },
    contact: {
      label: 'Contact',
      href: 'contact',
      header: 'Get in touch',
      links: [
        {
          title: 'Message through website',
          subtitle: 'Send a message via website',
          href: 'contact#contact-modal',
          badge: 'Recommended',
          dataAttributes: {
            'data-contact-modal-link': 'true'
          }
        },
        {
          title: 'Email',
          subtitle: 'daniel@danielshort.me',
          href: 'mailto:daniel@danielshort.me'
        },
        {
          title: 'LinkedIn',
          subtitle: 'linkedin.com/in/danielshort3',
          href: 'https://www.linkedin.com/in/danielshort3/',
          target: '_blank',
          rel: 'noopener noreferrer'
        },
        {
          title: 'GitHub',
          subtitle: 'github.com/danielshort3',
          href: 'https://github.com/danielshort3',
          target: '_blank',
          rel: 'noopener noreferrer'
        }
      ]
    },
    search: {
      action: 'search',
      label: 'Search site',
      placeholder: 'Search…'
    }
  };
}

function buildFooter() {
  return {
    copyrightName: 'Daniel Short',
    cookieSettingsLabel: 'Cookie settings',
    cookieSettingsButtonId: 'privacy-settings-link',
    columns: [
      {
        id: 'explore',
        title: 'Explore',
        links: [
          { label: 'Home', href: '/' },
          { label: 'Search', href: 'search' },
          { label: 'Tools', href: 'tools' },
          {
            label: 'Portfolio',
            href: 'portfolio',
            dataAttributes: {
              'data-portfolio-home-link': 'true'
            }
          }
        ]
      },
      {
        id: 'work',
        title: 'Work',
        links: [
          {
            label: 'Resume',
            href: 'resume',
            dataAttributes: {
              'data-resume-home-link': 'true'
            }
          },
          { label: 'Contact', href: 'contact' }
        ]
      },
      {
        id: 'connect',
        title: 'Connect',
        links: [
          {
            label: 'LinkedIn',
            href: 'https://www.linkedin.com/in/danielshort3/',
            target: '_blank',
            rel: 'noopener noreferrer'
          },
          {
            label: 'GitHub',
            href: 'https://github.com/danielshort3',
            target: '_blank',
            rel: 'noopener noreferrer'
          },
          { label: 'Email', href: 'mailto:daniel@danielshort.me' }
        ]
      },
      {
        id: 'site',
        title: 'Site',
        links: [
          { label: 'Sitemap', href: 'sitemap-pretty' },
          {
            label: 'Ask the site assistant',
            type: 'button',
            dataAttributes: {
              'data-site-chatbot-open': true
            },
            hidden: true
          },
          { label: 'Cookie settings', type: 'button', id: 'privacy-settings-link-footer', ariaHaspopup: 'dialog' },
          {
            label: 'Back to top',
            href: '#main',
            dataAttributes: {
              'data-smooth-scroll': 'true'
            }
          }
        ]
      }
    ],
    speedDial: {
      menuId: 'speed-dial-menu',
      menuLabel: 'Contact options',
      toggleLabel: 'Open contact options',
      items: [
        {
          label: 'Direct Message',
          href: '#contact-modal',
          ariaLabel: 'Send a direct message',
          iconType: 'direct-message',
          variant: 'direct',
          dataAttributes: {
            'data-contact-modal-link': 'true'
          }
        },
        {
          label: 'Send Email',
          href: 'mailto:daniel@danielshort.me',
          ariaLabel: 'Send Email',
          iconType: 'email'
        },
        {
          label: 'View LinkedIn',
          href: 'https://www.linkedin.com/in/danielshort3/',
          target: '_blank',
          rel: 'noopener noreferrer',
          ariaLabel: 'View LinkedIn',
          iconType: 'linkedin'
        },
        {
          label: 'View GitHub',
          href: 'https://github.com/danielshort3',
          target: '_blank',
          rel: 'noopener noreferrer',
          ariaLabel: 'View GitHub',
          iconType: 'github'
        }
      ]
    }
  };
}

function main() {
  const { projects } = loadProjectsData();
  const headerFeaturedIds = extractHeaderFeaturedProjectIds();
  const audienceApi = loadAudienceConfig();
  const toolCollections = extractToolCollections();

  writeJson(path.join('content', 'site', 'settings.json'), buildSiteSettings(audienceApi));
  writeJson(path.join('content', 'site', 'navigation.json'), buildNavigation(headerFeaturedIds));
  writeJson(path.join('content', 'site', 'footer.json'), buildFooter());
  writeJson(path.join('content', 'pages', 'tools.json'), toolCollections.page);
  writeJson(path.join('content', 'pages', 'contact.json'), {
    id: 'contact',
    ...extractPageDefinition(path.join('pages', 'contact.html'))
  });
  writeJson(path.join('content', 'pages', 'resume-directory.json'), {
    id: 'resume-directory',
    ...extractPageDefinition(path.join('pages', 'resume.html'))
  });
  writeJson(path.join('content', 'pages', 'resume-pdf-directory.json'), {
    id: 'resume-pdf-directory',
    ...extractPageDefinition(path.join('pages', 'resume-pdf.html'))
  });

  projects.forEach((project, index) => {
    writeJson(path.join('content', 'projects', `${project.id}.json`), {
      ...project,
      order: index + 1
    });
  });

  toolCollections.tools.forEach((tool) => {
    writeJson(path.join('content', 'tools', `${tool.slug}.json`), tool);
  });

  Object.keys(audienceApi.audiences || {}).forEach((key, index) => {
    const audience = audienceApi.audiences[key];
    writeJson(path.join('content', 'audiences', `${key}.json`), {
      order: index + 1,
      ...audience,
      page: {
        id: `audience-${key}`,
        ...extractPageDefinition(path.join('pages', `${key}.html`))
      }
    });
  });

  const resumeVariants = [
    { key: 'analytics', audience: 'analytics' },
    { key: 'data-science', audience: 'data-science' },
    { key: 'tourism', audience: 'tourism' }
  ];
  resumeVariants.forEach((resume, index) => {
    writeJson(path.join('content', 'resumes', `${resume.key}.json`), {
      key: resume.key,
      audience: resume.audience,
      order: index + 1,
      digitalPage: {
        id: `resume-${resume.key}`,
        ...extractPageDefinition(path.join('pages', `resume-${resume.key}.html`))
      },
      pdfPage: {
        id: `resume-${resume.key}-pdf`,
        ...extractPageDefinition(path.join('pages', `resume-${resume.key}-pdf.html`))
      }
    });
  });

  process.stdout.write('[cms] Migrated current site content into ./content.\n');
}

main();
