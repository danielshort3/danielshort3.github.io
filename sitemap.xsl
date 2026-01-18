<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:s="http://www.sitemaps.org/schemas/sitemap/0.9"
  exclude-result-prefixes="s">

  <xsl:output method="html" encoding="UTF-8" indent="yes" />

  <xsl:template match="/">
    <xsl:variable name="all" select="s:urlset/s:url" />
    <xsl:variable name="tools" select="$all[contains(s:loc, '/tools')]" />
    <xsl:variable name="portfolio" select="$all[contains(s:loc, '/portfolio')]" />
    <xsl:variable name="site" select="$all[not(contains(s:loc, '/tools')) and not(contains(s:loc, '/portfolio'))]" />
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
        <title>XML Sitemap | Daniel Short</title>
        <style>
          :root{
            --bg:#0D1117;
            --surface:#161B22;
            --primary:#2396AD;
            --text:#F1F4F8;
            --muted:#BFC8D3;
            --border:#21262D;
          }
          *{box-sizing:border-box}
          html,body{height:100%}
          body{
            margin:0;
            background:var(--bg);
            color:var(--text);
            font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
            line-height:1.45;
          }
          a{color:var(--primary);text-decoration:none}
          a:hover{text-decoration:underline}
          .wrapper{max-width:1068px;margin:0 auto;padding:24px}
          .hero{
            border-bottom:1px solid var(--border);
            background:linear-gradient(180deg, rgba(35,150,173,.12), rgba(13,17,23,0));
          }
          .kicker{
            display:inline-block;
            padding:6px 12px;
            border:1px solid rgba(35,150,173,.35);
            border-radius:999px;
            color:var(--muted);
            letter-spacing:.12em;
            text-transform:uppercase;
            font-size:.74rem;
          }
          h1{margin:14px 0 8px;font-size:2rem}
          .sub{margin:0;color:var(--muted);max-width:70ch}
          .meta{
            margin-top:16px;
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            align-items:center;
          }
          .pill{
            display:inline-flex;
            gap:10px;
            align-items:center;
            padding:10px 12px;
            border:1px solid var(--border);
            background:rgba(22,27,34,.85);
            border-radius:14px;
          }
          .pill .label{color:var(--muted)}
          .pill .value{font-weight:700}
          .navlinks{margin-top:14px;display:flex;gap:14px;flex-wrap:wrap}
          .navlinks a{
            display:inline-flex;
            padding:8px 10px;
            border:1px solid rgba(33,38,45,.75);
            border-radius:999px;
            background:rgba(22,27,34,.55);
          }
          .navlinks a:hover{border-color:rgba(35,150,173,.6)}
          .filters{
            margin-top:18px;
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            align-items:center;
          }
          .filter-label{color:var(--muted);font-size:.9rem}
          .filter-input{
            flex:1 1 280px;
            min-width:220px;
            padding:10px 12px;
            border-radius:12px;
            border:1px solid var(--border);
            background:rgba(22,27,34,.85);
            color:var(--text);
            outline:none;
          }
          .filter-input:focus{
            border-color:rgba(35,150,173,.75);
            box-shadow:0 0 0 3px rgba(35,150,173,.18);
          }
          .filter-btn{
            padding:10px 12px;
            border-radius:12px;
            border:1px solid var(--border);
            background:rgba(22,27,34,.85);
            color:var(--text);
            cursor:pointer;
          }
          .filter-btn:hover{border-color:rgba(35,150,173,.6)}
          .filter-status{color:var(--muted);font-size:.9rem;white-space:nowrap}
          .sectionlinks{
            margin:0 0 12px;
            display:flex;
            flex-wrap:wrap;
            gap:10px;
          }
          .sectionlinks a{
            display:inline-flex;
            gap:10px;
            align-items:center;
            padding:10px 12px;
            border:1px solid var(--border);
            background:rgba(22,27,34,.85);
            border-radius:14px;
            color:var(--text);
            text-decoration:none;
          }
          .sectionlinks a:hover{border-color:rgba(35,150,173,.6)}
          .sectionlinks .count{color:var(--muted);font-weight:600}
          .card{
            margin-top:20px;
            border:1px solid var(--border);
            background:var(--surface);
            border-radius:18px;
            overflow:hidden;
            box-shadow:0 14px 38px rgba(0,0,0,.35);
          }
          .card-head{
            padding:14px 12px;
            border-bottom:1px solid var(--border);
          }
          .card-head h2{margin:0;font-size:1.2rem}
          .card-head p{margin:6px 0 0;color:var(--muted)}
          table{width:100%;border-collapse:collapse}
          th,td{padding:14px 12px;border-top:1px solid var(--border);vertical-align:top}
          th{
            position:sticky;
            top:0;
            background:rgba(22,27,34,.97);
            border-top:none;
            text-align:left;
            font-size:.9rem;
            color:var(--muted);
            letter-spacing:.02em;
            z-index:2;
          }
          tbody tr:hover{background:rgba(35,150,173,.08)}
          .col-url{width:56%}
          .url{
            display:block;
            word-break:break-word;
            font-weight:600;
            color:var(--text);
          }
          .url:hover{color:var(--primary)}
          .empty{color:var(--muted)}
          .foot{
            padding:14px 12px;
            color:var(--muted);
            border-top:1px solid var(--border);
            font-size:.9rem;
          }
          @media (max-width:720px){
            th:nth-child(3),td:nth-child(3){display:none}
            .col-url{width:auto}
          }
        </style>
      </head>
      <body>
        <header class="hero">
          <div class="wrapper">
            <span class="kicker">Sitemap</span>
            <h1>XML Sitemap</h1>
            <p class="sub">This is an XML sitemap for search engines. It lists canonical, indexable URLs on <strong>danielshort.me</strong>.</p>
            <div class="meta">
              <div class="pill">
                <span class="label">Total URLs</span>
                <span class="value"><xsl:value-of select="count($all)" /></span>
              </div>
              <div class="pill">
                <span class="label">Site</span>
                <span class="value"><xsl:value-of select="count($site)" /></span>
              </div>
              <div class="pill">
                <span class="label">Tools</span>
                <span class="value"><xsl:value-of select="count($tools)" /></span>
              </div>
              <div class="pill">
                <span class="label">Portfolio</span>
                <span class="value"><xsl:value-of select="count($portfolio)" /></span>
              </div>
              <div class="pill">
                <span class="label">Last updated</span>
                <span class="value">
                  <xsl:for-each select="$all[string-length(normalize-space(s:lastmod)) &gt; 0]">
                    <xsl:sort select="s:lastmod" order="descending" />
                    <xsl:if test="position() = 1">
                      <xsl:value-of select="s:lastmod" />
                    </xsl:if>
                  </xsl:for-each>
                </span>
              </div>
            </div>
            <nav class="navlinks" aria-label="Quick links">
              <a href="/">Home</a>
              <a href="/portfolio">Portfolio</a>
              <a href="/tools">Tools</a>
              <a href="/resume">Resume</a>
              <a href="/contact">Contact</a>
            </nav>
            <div class="filters" role="search" aria-label="Filter sitemap URLs">
              <label class="filter-label" for="sitemap-filter">Filter</label>
              <input
                id="sitemap-filter"
                class="filter-input"
                type="search"
                autocomplete="off"
                spellcheck="false"
                placeholder="Filter URLs (e.g. tools, portfolio, resume)" />
              <button id="sitemap-clear" class="filter-btn" type="button">Clear</button>
              <span class="filter-status">
                Showing <span data-sitemap-shown=""><xsl:value-of select="count($all)" /></span> of
                <span data-sitemap-total=""><xsl:value-of select="count($all)" /></span>
              </span>
            </div>
          </div>
        </header>

        <main class="wrapper">
          <nav class="sectionlinks" aria-label="Sitemap sections">
            <a href="#section-site">Site <span class="count">(<span data-sitemap-count="site"><xsl:value-of select="count($site)" /></span>)</span></a>
            <a href="#section-tools">Tools <span class="count">(<span data-sitemap-count="tools"><xsl:value-of select="count($tools)" /></span>)</span></a>
            <a href="#section-portfolio">Portfolio <span class="count">(<span data-sitemap-count="portfolio"><xsl:value-of select="count($portfolio)" /></span>)</span></a>
          </nav>

          <section class="card sitemap-section" id="section-site" data-sitemap-section="site">
            <header class="card-head">
              <h2>Site</h2>
              <p>Core pages like home, resume, contact, and privacy.</p>
            </header>
            <table aria-label="Site URLs">
              <thead>
                <tr>
                  <th class="col-url">URL</th>
                  <th>Last Modified</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                <xsl:for-each select="$site">
                  <xsl:sort select="s:loc" order="ascending" />
                  <tr data-sitemap-row="1" data-url="{s:loc}">
                    <td class="col-url">
                      <a class="url" href="{s:loc}">
                        <xsl:value-of select="s:loc" />
                      </a>
                    </td>
                    <td>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space(s:lastmod)) &gt; 0">
                          <xsl:value-of select="s:lastmod" />
                        </xsl:when>
                        <xsl:otherwise><span class="empty">—</span></xsl:otherwise>
                      </xsl:choose>
                    </td>
                    <td>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space(s:priority)) &gt; 0">
                          <xsl:value-of select="s:priority" />
                        </xsl:when>
                        <xsl:otherwise><span class="empty">—</span></xsl:otherwise>
                      </xsl:choose>
                    </td>
                  </tr>
                </xsl:for-each>
              </tbody>
            </table>
          </section>

          <section class="card sitemap-section" id="section-tools" data-sitemap-section="tools">
            <header class="card-head">
              <h2>Tools</h2>
              <p>Privacy-first utilities (plus the tools dashboard).</p>
            </header>
            <table aria-label="Tools URLs">
              <thead>
                <tr>
                  <th class="col-url">URL</th>
                  <th>Last Modified</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                <xsl:for-each select="$tools">
                  <xsl:sort select="s:loc" order="ascending" />
                  <tr data-sitemap-row="1" data-url="{s:loc}">
                    <td class="col-url">
                      <a class="url" href="{s:loc}">
                        <xsl:value-of select="s:loc" />
                      </a>
                    </td>
                    <td>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space(s:lastmod)) &gt; 0">
                          <xsl:value-of select="s:lastmod" />
                        </xsl:when>
                        <xsl:otherwise><span class="empty">—</span></xsl:otherwise>
                      </xsl:choose>
                    </td>
                    <td>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space(s:priority)) &gt; 0">
                          <xsl:value-of select="s:priority" />
                        </xsl:when>
                        <xsl:otherwise><span class="empty">—</span></xsl:otherwise>
                      </xsl:choose>
                    </td>
                  </tr>
                </xsl:for-each>
              </tbody>
            </table>
          </section>

          <section class="card sitemap-section" id="section-portfolio" data-sitemap-section="portfolio">
            <header class="card-head">
              <h2>Portfolio</h2>
              <p>Shareable project pages under <code>/portfolio/&lt;id&gt;</code>.</p>
            </header>
            <table aria-label="Portfolio URLs">
              <thead>
                <tr>
                  <th class="col-url">URL</th>
                  <th>Last Modified</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                <xsl:for-each select="$portfolio">
                  <xsl:sort select="s:loc" order="ascending" />
                  <tr data-sitemap-row="1" data-url="{s:loc}">
                    <td class="col-url">
                      <a class="url" href="{s:loc}">
                        <xsl:value-of select="s:loc" />
                      </a>
                    </td>
                    <td>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space(s:lastmod)) &gt; 0">
                          <xsl:value-of select="s:lastmod" />
                        </xsl:when>
                        <xsl:otherwise><span class="empty">—</span></xsl:otherwise>
                      </xsl:choose>
                    </td>
                    <td>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space(s:priority)) &gt; 0">
                          <xsl:value-of select="s:priority" />
                        </xsl:when>
                        <xsl:otherwise><span class="empty">—</span></xsl:otherwise>
                      </xsl:choose>
                    </td>
                  </tr>
                </xsl:for-each>
              </tbody>
            </table>
          </section>

          <div class="card">
            <div class="foot">
              Tip: this view is for humans; crawlers read the raw XML.
            </div>
          </div>
        </main>

        <script>
          (function() {
            var input = document.getElementById('sitemap-filter');
            var clear = document.getElementById('sitemap-clear');
            var shownEl = document.querySelector('[data-sitemap-shown]');
            var totalEl = document.querySelector('[data-sitemap-total]');
            var rows = Array.prototype.slice.call(document.querySelectorAll('[data-sitemap-row]'));
            var sections = Array.prototype.slice.call(document.querySelectorAll('[data-sitemap-section]'));

            if (totalEl) totalEl.textContent = String(rows.length);

            function update() {
              var q = (input && input.value ? input.value : '').trim().toLowerCase();
              var shown = 0;

              rows.forEach(function(row) {
                var url = row.getAttribute('data-url') || '';
                var match = !q || url.toLowerCase().indexOf(q) !== -1;
                row.hidden = !match;
                if (match) shown++;
              });

              if (shownEl) shownEl.textContent = String(shown);

              sections.forEach(function(section) {
                var name = section.getAttribute('data-sitemap-section');
                var visible = section.querySelectorAll('[data-sitemap-row]:not([hidden])').length;
                section.hidden = visible === 0;
                if (!name) return;
                var countEl = document.querySelector('[data-sitemap-count=\"' + name + '\"]');
                if (countEl) countEl.textContent = String(visible);
              });
            }

            if (input) {
              input.addEventListener('input', update);
              input.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                  input.value = '';
                  update();
                  input.blur();
                }
              });
            }

            if (clear) {
              clear.addEventListener('click', function() {
                if (!input) return;
                input.value = '';
                update();
                input.focus();
              });
            }

            update();
          })();
        </script>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
