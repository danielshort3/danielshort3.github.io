/* portfolio.js  •  Projects / Portfolio page only
   Update or extend the PROJECTS array. Optionally set FEATURED_IDS below to
   control which projects appear in the top carousel.
   ──────────────────────────────────────────────────────────── */
/* ❶ MASTER CATALOG
   ------------------------------------------------------------------
   • id        – any unique string (used for modal id & deeplinks)
   • tools     – used for filter buttons (one or more per project)
   • resources – array of { icon , url , label } objects
   ------------------------------------------------------------------ */
  /* ➊ MASTER CATALOG  (trimmed) */
/* ================================
   DANIEL SHORT – FEATURED PROJECTS
   ================================ */
window.PROJECTS = [
  {
    id: "pizza",
    title: "Pizza-Tips Regression Model",
    subtitle: "Excel Analytics & Forecasting",
    image: "images/project_1.png",
    tools: ["Excel", "Statistics"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_1.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_1.xlsx", label: "Excel" }
    ],
    problem : "Tip revenue fluctuated  ±35 % week-to-week with no data-driven explanation.",
    actions : [
      "Profiled 12 k delivery rows; cleaned weather / customer-type fields in Excel Power Query.",
      "Built multi-variate regression isolating dwelling-type, distance, and order size effects."
    ],
    results : [
      "Identified apartments tipping **12 % less** than single-family homes.",
      "Informed route scheduling that lifted average driver earnings **+8 %** in the next quarter."
    ]
  },

  {
    id: "babynames",
    title: "Baby-Name Popularity Predictor",
    subtitle: "Python ML Pipeline",
    image: "images/project_2.png",
    tools: ["Python", "scikit-learn"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Baby-Names", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_2_pdf.zip",               label: "PDFs"  },
      { icon: "images/jupyter-icon.png",url: "documents/Project_2.zip",                   label: "Notebook"}
    ],
    problem : "Expectant parents lacked objective data on long-term name ‘stickiness’.",
    actions : [
      "Aggregated & cleaned **140 years** of SSA records; engineered trend and saturation features.",
      "Trained linear-regression and random-forest models; automated top-10 recommendation service."
    ],
    results : [
      "Model scored unseen yearly slices at **R² 0.82**.",
      "Generated personalized short-lists that cut name-selection time **−50 %** in user testing."
    ]
  },

  {
    id: "pizzaDashboard",
    title: "30-Second Delivery KPI Dashboard",
    subtitle: "Tableau Storytelling",
    image: "images/project_3.png",
    tools: ["Tableau"],
    resources: [
      { icon: "images/tableau-icon.png",
        url : "https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard?:language=en-US&:display_count=n&:origin=viz_share_link",
        label:"Interactive Dashboard"
      }
    ],
    problem : "Managers spent 15 min per shift reading static reports to spot delivery issues.",
    actions : [
      "Reshaped 12 k rows for Tableau; built map, histogram & 12-month forecast with date/zone filters.",
      "Embedded dashboard on intranet with auto-refresh via Tableau Server extracts."
    ],
    results : [
      "Cut KPI review time **15 min → 2 min**.",
      "Surfaced three zones averaging **+18 %** higher tips → targeted marketing campaign."
    ]
  },

  {
    id: "nonogram",
    title: "Adaptive Nonogram Solver",
    subtitle: "Reinforcement Learning (RL)",
    image: "images/project_4.png",
    tools: ["Python", "PyTorch"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/nonogram", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_4.pdf",                  label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_4.ipynb",                label: "Notebook"}
    ],
    problem : "No existing AI generalized across Nonogram sizes; manual play was tedious.",
    actions : [
      "Designed CNN + Transformer RL agent with curriculum (5×5 → 25×25); trained on 200 k puzzles.",
      "Implemented policy-gradient rewards and dynamic masking for valid moves."
    ],
    results : [
      "Reached **92 % solve-rate** on unseen 15×15 boards.",
      "Generalized to new clue sets with **0 retraining** required."
    ]
  },

  {
    id: "ufoDashboard",
    title: "UFO Sightings Explorer",
    subtitle: "Tableau Geospatial Analytics",
    image: "images/project_5.png",
    tools: ["Tableau", "Data Blending"],
    resources: [
      { icon: "images/tableau-icon.png",
        url  : "https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013?:language=en-US&:display_count=n&:origin=viz_share_link",
        label:"Interactive Dashboard"
      }
    ],
    problem : "Researchers couldn’t detect macro-patterns across fragmented UFO datasets.",
    actions : [
      "Merged **3 national** databases (80 k records); normalized shape, duration, geo fields.",
      "Built heat-map, time-slider, and anomaly filters for rapid exploratory analysis."
    ],
    results : [
      "Single dashboard covers **10+ KPIs**; attracts ~500 monthly views from academia.",
      "Uncovered July seasonal spike and Southwest corridor hotspot → cited in 2 blog articles."
    ]
  },

  {
    id: "covidAnalysis",
    title: "COVID-19 Mortality Drivers",
    subtitle: "Python XGBoost & SHAP",
    image: "images/project_6.png",
    tools: ["Python", "XGBoost", "SHAP"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Covid-Analysis", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_6.pdf",                        label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_6.ipynb",                      label: "Notebook"}
    ],
    problem : "Hospitals lacked evidence on which country-level factors drove mortality rates.",
    actions : [
      "Joined **30 datasets** (demography, comorbidity, policy); engineered lag features.",
      "Trained XGBoost; used SHAP for interpretability; validated via 5-fold RMSE."
    ],
    results : [
      "Isolation index & diabetes prevalence ranked top; findings shared with **3 clinics**.",
      "Notebook enabled scenario testing that cut epidemiology analysis time **−60 %**."
    ]
  },

  {
    id: "targetEmptyPackage",
    title: "Empty-Package Shrink Dashboard",
    subtitle: "Excel Forecasting & BI",
    image: "images/project_7.png",
    tools: ["Excel", "Time-Series"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_7.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_7.xlsx", label: "Excel" }
    ],
    problem : "Retail execs lacked visibility into rising empty-package theft across 200 stores.",
    actions : [
      "Cleansed loss-prevention logs; built pivot KPIs by department/month.",
      "Forecasted 12-month shrink using Holt-Winters triple exponential smoothing."
    ],
    results : [
      "Flagged Beauty & Electronics causing **43 % of losses**.",
      "Secured **$380 k** mitigation budget based on 12-month projection."
    ]
  },

  {
    id: "handwritingRating",
    title: "Numeral Legibility Scorer",
    subtitle: "PyTorch CNN Fine-Tuning",
    image: "images/project_8.png",
    tools: ["Python", "PyTorch", "CNN"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Handwriting-Rating", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_8.pdf",                            label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_8.ipynb",                          label: "Notebook"}
    ],
    problem : "Teachers needed an objective, fast way to grade handwritten numerals.",
    actions : [
      "Fine-tuned ResNet-18 on MNIST + 3 k custom samples; optimized depth vs. latency for tablets.",
      "Built scoring rubric 0–100; packaged as lightweight ONNX model."
    ],
    results : [
      "Delivered **80 % overall accuracy** at 25 ms inference on low-power devices.",
      "Dashboard flagged ‘2’ as lowest-quality digit (35 % accuracy) → targeted practice."
    ]
  },

  {
    id: "digitGenerator",
    title: "Synthetic Digit Generator",
    subtitle: "Variational Autoencoder",
    image: "images/project_9.png",
    tools: ["Python", "VAE"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Handwritten-Digit-Generator", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_9.pdf",                                   label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_9.ipynb",                                 label: "Notebook"}
    ],
    problem : "Researchers needed extra handwriting samples to balance small datasets.",
    actions : [
      "Built VAE with conv & deconv layers; visualized latent space via t-SNE & UMAP.",
      "Sampled latent vectors to create novel digits; tuned β-VAE for disentanglement."
    ],
    results : [
      "Generated **10 k** high-fidelity digits (BCE < 0.04).",
      "Augmented training sets improved downstream model accuracy **+3.2 pp**."
    ]
  },

  {
    id: "sheetMusicUpscale",
    title: "Sheet-Music Clean-&-Upscale",
    subtitle: "UNet + VDSR Pipeline",
    image: "images/project_10.png",
    tools: ["Python", "Computer Vision"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Watermark-Remover", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_10_pdf.zip",                      label: "PDFs"  },
      { icon: "images/jupyter-icon.png",url: "documents/Project_10.zip",                          label: "Notebook"}
    ],
    problem : "Musicians struggled with low-res, watermark-covered public-domain scores.",
    actions : [
      "Trained UNet on **20 k** paired pages for watermark removal.",
      "Upscaled 612×792 scans to 1700×2200 with VDSR; wrapped GUI for one-click pipeline."
    ],
    results : [
      "Delivered print-ready scores in < 10 s; **PSNR +9 dB** post-processing.",
      "Adopted by **120+** musicians within first month of release."
    ]
  },

  {
    id: "deliveryTip",
    title: "Tip Hot-Spot Heat-Map",
    subtitle: "Excel Geo-Analytics",
    image: "images/project_11.png",
    tools: ["Excel", "Power Query"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_11.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_11.xlsx", label: "Excel" }
    ],
    problem : "Drivers wanted optimal shift times & zones for higher tips.",
    actions : [
      "Built geospatial heat-map and pivot filters on **4 k** deliveries.",
      "Compared tip averages by daypart, zone, and order size."
    ],
    results : [
      "Identified Saturday 18:00–21:00 as peak (+18 % tips).",
      "Pilot driver increased weekly earnings **+12 %** following insights."
    ]
  },

  {
    id: "retailStore",
    title: "Store-Level Loss & Sales ETL",
    subtitle: "SQL + Python Viz",
    image: "images/project_12.png",
    tools: ["SQL", "Python"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/target-packaging-analysis-mssql", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_12.pdf",                                        label: "PDF"    },
      { icon: "images/jupyter-icon.png",url: "documents/Project_12.ipynb",                                      label: "Notebook"}
    ],
    problem : "Corporate lacked granular insight into security & sales by store format.",
    actions : [
      "Normalized MSSQL tables; wrote views & stored procedures for automated KPI extraction.",
      "Visualized theft vs. sales trends in Python (Matplotlib, Seaborn)."
    ],
    results : [
      "Flagged StoreFormat_47 + 3 states with **27 % higher losses**.",
      "Linked boycott events to **15 % sales dip**, informing crisis plan."
    ]
  },

  {
    id: "smartSentence",
    title: "Smart Sentence Retriever",
    subtitle: "Transformer Embeddings",
    image: "images/project_13.png",
    tools: ["Python", "NLP"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Smart-Sentence-Finder", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_13.pdf",                              label: "PDF"    },
      { icon: "images/jupyter-icon.png",url: "documents/Project_13.ipynb",                            label: "Notebook"}
    ],
    problem : "Researchers needed < 1 s extraction of key lines from large PDFs.",
    actions : [
      "Embedded sentences using sentence-transformers; indexed vectors with FAISS.",
      "Benchmarked 5 models on top-k precision vs. latency."
    ],
    results : [
      "Returned top-5 sentences in **0.4 s** on 10 k-sentence docs.",
      "Outperformed BM25 baseline by **+18 pp MAP@5**."
    ]
  },

  {
    id: "website",
    title: "danielshort.me",
    subtitle: "Responsive Portfolio Site",
    image: "images/project_14.png",
    tools: ["HTML", "CSS", "JavaScript"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/danielshort3.github.io", label: "GitHub"  },
      { icon: "images/website-icon.png",url: "https://danielshort.dev/",                               label: "Live Site"}
    ],
    problem : "Needed a mobile-fast hub to showcase analytics & ML work to recruiters.",
    actions : [
      "Built semantic, Lighthouse-90+ static site; dynamic JS loads projects via JSON.",
      "Integrated Google Analytics 4, structured-data schema, and lazy-loaded assets."
    ],
    results : [
      "First-contentful paint **1.2 s** (mobile).",
      "Drove **2 freelance leads** & **71 % higher** recruiter response vs. résumé-only."
    ]
  }
];

// IDs (in order) of projects shown in the top carousel
window.FEATURED_IDS = [
  "pizza",
  "babynames",
  "pizzaDashboard",
  "nonogram",
  "ufoDashboard"
];

window.generateProjectModal = function(p){
  return `
    <div class="modal-content" tabindex="0">
      <button class="modal-close" aria-label="Close dialog">&times;</button>
      <div class="modal-title-strip"><h3 class="modal-title">${p.title}</h3></div>

      <div class="modal-header-details">
        <div class="modal-half">
          <p class="header-label">Tools</p>
          <div class="tool-badges">
            ${p.tools.map(t => `<span class="badge">${t}</span>`).join("")}
          </div>
        </div>
        <div class="modal-half">
          <p class="header-label">Downloads / Links</p>
          <div class="icon-row">
            ${p.resources.map(r => `
              <a href="${r.url}" target="_blank" title="${r.label}">
                <img src="${r.icon}" alt="${r.label}" class="icon">
              </a>`).join("")}
          </div>
        </div>
      </div>

      <div class="modal-body">
        <div class="modal-text">
          <p class="modal-subtitle">${p.subtitle}</p>
          <h4>Problem</h4><p>${p.problem}</p>
          <h4>Action</h4><ul>${p.actions.map(a=>`<li>${a}</li>`).join("")}</ul>
          <h4>Result</h4><ul>${p.results.map(r=>`<li>${r}</li>`).join("")}</ul>
        </div>
        <div class="modal-image"><img src="${p.image}" alt="${p.title}"></div>
      </div>`;
};

/* ────────────────────────────────────────────────────────────
   Portfolio Carousel (top of page)
   ------------------------------------------------------------------ */
function buildPortfolioCarousel(){
  const container = document.getElementById("portfolio-carousel");
  if (!container || !window.PROJECTS) return;

  const track = container.querySelector(".carousel-track");
  const dots  = container.querySelector(".carousel-dots");

  let projects = [];
  if (Array.isArray(window.FEATURED_IDS) && window.FEATURED_IDS.length) {
    window.FEATURED_IDS.forEach(id => {
      const p = window.PROJECTS.find(pr => pr.id === id);
      if (p) projects.push(p);
    });
  } else {
    projects = window.PROJECTS.slice(0,5);
  }
  track.innerHTML = "";
  dots.innerHTML  = "";

  projects.forEach((p,i)=>{
    const card = document.createElement("div");
    card.className = "project-card carousel-card";
    card.innerHTML = `
      <div class="overlay"></div>
      <div class="project-title">${p.title}</div>
      <div class="project-subtitle">${p.subtitle}</div>
      <img src="${p.image}" alt="${p.title}">
    `;
    card.addEventListener("click", ()=>{ if(!moved) openModal(p.id); });
    track.appendChild(card);

    const dot = document.createElement("button");
    dot.className = "carousel-dot";
    dot.type = "button";
    dot.setAttribute("aria-label", `Show ${p.title}`);
    dot.addEventListener("click", ()=>{ goTo(i); pause=true; });
    dots.appendChild(dot);
  });

  let current = 0;
  let pause = false;

  const update = () => {
    const cardW = track.children[0].offsetWidth + 24; // gap
    const offset = (container.offsetWidth - cardW)/2;
    track.style.transform = `translateX(${-current*cardW + offset}px)`;
    [...track.children].forEach((c,i)=>c.classList.toggle("active", i===current));
    [...dots.children].forEach((d,i)=>d.classList.toggle("active", i===current));
  };

  const goTo = i => {
    const n = projects.length;
    current = (i + n) % n;
    update();
  };
  const next = () => goTo(current + 1);

  container.addEventListener("mouseenter",()=> pause = true);
  container.addEventListener("mouseleave",()=> pause = false);

  /* ── click & drag to switch slides ───────────────────────────── */
  let dragStart = 0;
  let dragging  = false;
  let moved     = false;

  const getX = e => e.touches ? e.touches[0].clientX : e.clientX;

  const onDown = e => { dragging = true; moved = false; dragStart = getX(e); container.classList.add("dragging"); };
  const onMove = e => {
    if (!dragging) return;
    const diff = getX(e) - dragStart;
    if (Math.abs(diff) > 40) {
      dragging = false;
      moved = true;
      diff < 0 ? next() : goTo(current - 1);
    }
  };
  const onUp = () => { dragging = false; container.classList.remove("dragging"); };

  container.addEventListener("mousedown", onDown);
  container.addEventListener("touchstart", onDown, { passive: true });
  container.addEventListener("mousemove", onMove);
  container.addEventListener("touchmove", onMove, { passive: true });
  container.addEventListener("mouseup", onUp);
  container.addEventListener("mouseleave", onUp);
  container.addEventListener("touchend", onUp);

  setInterval(()=>{ if(!pause) next(); }, 5000);
  window.addEventListener("resize", update);

  update();
}

function initSeeMore(){
  const btn = document.getElementById("see-more");
  const filters  = document.getElementById("filters");
  const grid     = document.getElementById("projects");
  const carousel = document.getElementById("portfolio-carousel-section");
  if(!btn || !filters || !grid) return;
  btn.addEventListener("click", () => {
    const expanded = btn.dataset.expanded === "true";
    btn.dataset.expanded = expanded ? "false" : "true";
    btn.textContent = expanded ? "See More" : "See Less";

    if (expanded) {
      // collapse grid and filters smoothly
      const gStart = grid.offsetHeight;
      const fStart = filters.offsetHeight;

      grid.style.height = `${gStart}px`;
      filters.style.height = `${fStart}px`;
      filters.classList.add("grid-fade");
      grid.classList.add("grid-fade");

      requestAnimationFrame(() => {
        grid.style.height = "0px";
        filters.style.height = "0px";
        filters.style.paddingTop = "0px";
        filters.style.paddingBottom = "0px";
        filters.style.marginTop = "0px";
      });

      setTimeout(() => {
        grid.classList.add("hide");
        filters.classList.add("hide");
        grid.classList.remove("grid-fade");
        filters.classList.remove("grid-fade");
        grid.style.height = "";
        filters.style.height = "";
        filters.style.paddingTop = "";
        filters.style.paddingBottom = "";
        filters.style.marginTop = "";
        carousel?.scrollIntoView({ behavior: "smooth" });
      }, 450); // height transition duration
    } else {
      // expand grid and filters smoothly
      filters.classList.remove("hide");
      grid.classList.remove("hide");
      const gTarget = grid.scrollHeight;
      const fTarget = filters.scrollHeight;

      grid.style.height = "0px";
      filters.style.height = "0px";
      filters.style.paddingTop = "0px";
      filters.style.paddingBottom = "0px";
      filters.style.marginTop = "0px";
      filters.classList.add("grid-fade");
      grid.classList.add("grid-fade");

      requestAnimationFrame(() => {
        grid.style.height = `${gTarget}px`;
        filters.style.height = `${fTarget}px`;
        filters.style.paddingTop = "";
        filters.style.paddingBottom = "";
        filters.style.marginTop = "";
        grid.classList.remove("grid-fade");
        filters.classList.remove("grid-fade");
      });

      setTimeout(() => {
        grid.style.height = "";
        filters.style.height = "";
      }, 450);
    }
  });
}

/* ────────────────────────────────────────────────────────────
   DOM-builder  (loads all projects immediately)
   ------------------------------------------------------------------
   • Builds cards inside  #projects
   • Builds modals inside #modals
   • Populates #filter-menu counts & click-to-filter behaviour
   ------------------------------------------------------------------ */
function buildPortfolio() {
  console.log("[DEBUG] buildPortfolio start");
  const grid   = document.getElementById("projects");
  const modals = document.getElementById("modals");
  const menu   = document.getElementById("filter-menu");
  if (!grid || !modals || !menu || !window.PROJECTS) return;

  /* helper – create & return element */
  const el = (tag, cls = "", html = "") => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html) n.innerHTML = html;
    return n;
  };

  /* ➊ Build cards & modals ----------------------------------------- */
  window.PROJECTS.forEach((p, i) => {
    /* card */
    const card = el("div", "project-card", `
      <div class="overlay"></div>
      <div class="project-title">${p.title}</div>
      <div class="project-subtitle">${p.subtitle}</div>
      <img src="${p.image}" alt="${p.title}" loading="eager">`);
    card.dataset.index = i;
    card.dataset.tags  = p.tools.join(",");
    card.addEventListener("click", () => openModal(p.id));
    grid.appendChild(card);

    /* modal */
    const modal = el("div","modal");
    modal.id = `${p.id}-modal`;
    modal.innerHTML = window.generateProjectModal(p);
    modals.appendChild(modal);
  });

  /* ➋ Animate cards right away (no IntersectionObserver) ----------- */
  [...grid.children].forEach((c, i) => {
    c.style.animationDelay = `${i * 80}ms`;
    c.classList.add("ripple-in");
  });

  /* ── auto-scroll to first card once it fades in (mobile only) ── */
  const isMobileInitial = window.matchMedia("(max-width: 768px)").matches;
  console.log("[DEBUG] initial auto-scroll, isMobile:", isMobileInitial);
  if (isMobileInitial) {
    const first = grid.firstElementChild;
    if (first) {
      const offset =
        parseFloat(
          getComputedStyle(document.documentElement).getPropertyValue(
            "--nav-height"
          )
        ) || 0;
      const y = first.getBoundingClientRect().top + window.scrollY - offset;
      console.log("[DEBUG] first card offset", { offset, y });
      setTimeout(() => {
        window.scrollTo({ top: y, behavior: "smooth" });
      }, 600); // ripple-in animation ≈550ms
    } else {
      console.log("[DEBUG] first card not found");
    }
  }

  /* ➌ Build filter-button counts ----------------------------------- */
  const counts = { all: window.PROJECTS.length };
  window.PROJECTS.forEach(p => p.tools.forEach(t => counts[t] = (counts[t] || 0) + 1));
  [...menu.children].forEach(btn => {
    const tag = btn.dataset.filter;
    btn.innerHTML = `${btn.textContent.trim()} ${(counts[tag] || 0)}/${counts.all}`;
  });

  /* ➍ Filter behaviour (fade-out → update → fade-in) --------------- */
  menu.addEventListener("click", e => {
    if (!e.target.dataset.filter) return;

    /* button UI */
    [...menu.children].forEach(b => {
      b.classList.replace("btn-primary", "btn-secondary");
      b.setAttribute("aria-selected", "false");
    });
    e.target.classList.replace("btn-secondary", "btn-primary");
    e.target.setAttribute("aria-selected", "true");

    const tag   = e.target.dataset.filter;
    const start = grid.offsetHeight;
    grid.style.height = `${start}px`;
    grid.classList.add("grid-fade");

    setTimeout(() => {
      [...grid.children].forEach(card => {
        card.classList.toggle("hide", tag !== "all" && !card.dataset.tags.includes(tag));
      });

      const visible = [...grid.children].filter(c => !c.classList.contains("hide"));
      grid.style.height = `${grid.scrollHeight}px`;

      visible.forEach((card, i) => {
        card.classList.remove("ripple-in");
        void card.offsetWidth;           // restart animation
        card.style.animationDelay = `${i * 80}ms`;
        card.classList.add("ripple-in");
      });

      grid.classList.remove("grid-fade");

      setTimeout(() => {
        grid.style.height = "";

        /* ─── ensure first visible card is flush on mobile ─── */
        const isMobileFilter = window.matchMedia("(max-width: 768px)").matches;
        console.log("[DEBUG] filter scroll isMobile:", isMobileFilter);
        if (isMobileFilter) {
          const first = visible[0];
          if (first) {
            const offset =
              parseFloat(
                getComputedStyle(document.documentElement).getPropertyValue(
                  "--nav-height"
                )
              ) || 0;
            const y = first.getBoundingClientRect().top + window.scrollY - offset;
            console.log("[DEBUG] filter first card offset", { offset, y });
            window.scrollTo({ top: y, behavior: "smooth" });
          } else {
            console.log("[DEBUG] filter first card not found");
          }
        }
      }, 450); // height transition duration
    }, 350);   // grid fade duration
  });
}

/* ➍ Modal open / focus-trap / close --------------------------------- */
function openModal(id){
  if (window.trackProjectView) trackProjectView(id);

  const modal = document.getElementById(`${id}-modal`);
  if (!modal) return;

  /* put the project hash in the URL (so it’s linkable / back-able) */
  const pushed = location.hash !== `#${id}`;
  if (pushed) history.pushState({ modal:id }, "", `#${id}`);

  modal.classList.add("active");
  document.body.classList.add("modal-open");

  /* focus-trap setup */
  const focusable = modal.querySelectorAll("a,button,[tabindex]:not([tabindex='-1'])");
  focusable[0]?.focus();

  const trap = e=>{
    if (e.key === "Escape"){ close(); return; }
    if (e.key !== "Tab" || !focusable.length) return;

    const first = focusable[0],
          last  = focusable[focusable.length-1];

    if (e.shiftKey ? document.activeElement === first
                   : document.activeElement === last){
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
    }
  };

  const clickClose = e=>{
    if (e.target.classList.contains("modal") ||
        e.target.classList.contains("modal-close")) close();
  };

  const close = ()=>{
    modal.classList.remove("active");
    document.body.classList.remove("modal-open");
    document.removeEventListener("keydown", trap);
    modal.removeEventListener("click",  clickClose);

    /* clean the address bar */
    if (pushed){
      history.back();                                    // removes #id
    } else {
      history.replaceState(null, "", location.pathname + location.search);
    }
  };

  document.addEventListener("keydown", trap);
  modal.addEventListener("click",    clickClose);
}




/* ─── handle direct links + Back/Forward buttons ─────────────── */
function routeModal(){
  const id = location.hash.slice(1);

  /* close any open modal */
  document.querySelectorAll(".modal.active").forEach(m=>{
    m.classList.remove("active");
    document.body.classList.remove("modal-open");
  });

  /* if there’s a hash, open the matching modal */
  if (id) openModal(id);
}

window.addEventListener("DOMContentLoaded", routeModal);
window.addEventListener("popstate",        routeModal);
