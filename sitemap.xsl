<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:s="http://www.sitemaps.org/schemas/sitemap/0.9"
  exclude-result-prefixes="s">

  <xsl:output method="html" encoding="UTF-8" indent="yes" />

  <xsl:template match="/">
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
          .card{
            margin-top:20px;
            border:1px solid var(--border);
            background:var(--surface);
            border-radius:18px;
            overflow:hidden;
            box-shadow:0 14px 38px rgba(0,0,0,.35);
          }
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
            th:nth-child(3),td:nth-child(3),
            th:nth-child(4),td:nth-child(4){display:none}
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
                <span class="value"><xsl:value-of select="count(s:urlset/s:url)" /></span>
              </div>
              <div class="pill">
                <span class="label">Last updated</span>
                <span class="value">
                  <xsl:for-each select="s:urlset/s:url">
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
          </div>
        </header>

        <main class="wrapper">
          <div class="card">
            <table aria-label="Sitemap URLs">
              <thead>
                <tr>
                  <th class="col-url">URL</th>
                  <th>Last Modified</th>
                  <th>Change</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                <xsl:for-each select="s:urlset/s:url">
                  <tr>
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
                        <xsl:when test="string-length(normalize-space(s:changefreq)) &gt; 0">
                          <xsl:value-of select="s:changefreq" />
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
            <div class="foot">
              Tip: this view is for humans; crawlers read the raw XML.
            </div>
          </div>
        </main>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
