/* ────────────────────────────────────────────────────────────
 portfolio.js  •  Projects / Portfolio page only
 Update or extend the PROJECTS array – no other edits needed!
──────────────────────────────────────────────────────────── */

/* ❶ MASTER CATALOG
   ------------------------------------------------------------------
   • id        – any unique string (used for modal id & deeplinks)
   • tools     – used for filter buttons (one or more per project)
   • resources – array of { icon , url , label } objects
   ------------------------------------------------------------------ */
  /* ➊ MASTER CATALOG  (trimmed) */
window.PROJECTS = [
  {
    id: "pizza",
    title: "Pizza Delivery Analysis",
    subtitle: "Data Analysis & Visualization",
    image: "images/project_1.png",
    tools: ["Excel"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_1.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_1.xlsx", label: "Excel" }
    ],
    problem : "Management couldn’t explain wide tip variability by neighbourhood.",
    actions : [
      "Explored delivery, customer-type and weather fields in Excel.",
      "Ran multiple-regression to isolate statistically significant drivers."
    ],
    results : [
      "Weather had no meaningful impact on tipping.",
      "Apartments tipped 12 % less than single-family homes."
    ]
  },

  {
    id: "babynames",
    title: "Baby Names",
    subtitle: "Data Analysis & Machine Learning",
    image: "images/project_2.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Baby-Names", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_2_pdf.zip",               label: "PDFs"  },
      { icon: "images/jupyter-icon.png",url: "documents/Project_2.zip",                   label: "Notebook"}
    ],
    problem : "Expectant parents struggle to pick names with long-term popularity appeal.",
    actions : [
      "Curated & cleaned 140 yrs of SSA baby-name data.",
      "Trained linear-regression and random-forest models to score name ‘stickiness’.",
      "Built custom algorithm combining trend slope & cultural saturation."
    ],
    results : [
      "Generated Top-10 name short-lists (boys & girls).",
      "Model achieved > 0.82 R² on unseen yearly slices."
    ]
  },

  {
    id: "pizzaDashboard",
    title: "Pizza Delivery Dashboard",
    subtitle: "Data Visualization",
    image: "images/project_3.png",
    tools: ["Tableau"],
    resources: [
      {
        icon: "images/tableau-icon.png",
        url : "https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard?:language=en-US&:display_count=n&:origin=viz_share_link",
        label:"Interactive Dashboard"
      }
    ],
    problem : "Stakeholders lacked a quick visual of delivery KPIs & geospatial trends.",
    actions : [
      "Cleaned and reshaped 12k delivery rows for Tableau ingestion.",
      "Built interactive maps, histograms and 12-mo tip forecast.",
      "Added slicers for date, zone and customer type."
    ],
    results : [
      "Cut KPI review time from 15 min static deck → 2 min live dashboard.",
      "Highlighted three neighbourhoods with 18 % higher average tips."
    ]
  },

  {
    id: "nonogram",
    title: "Nonogram Solver",
    subtitle: "Reinforcement Learning",
    image: "images/project_4.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/nonogram", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_4.pdf",                  label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_4.ipynb",                label: "Notebook"}
    ],
    problem : "Manual Nonogram solving is slow; no reliable AI existed for variable sizes.",
    actions : [
      "Designed CNN-based RL agent with curriculum training (5×5 → 25×25).",
      "Integrated transformer branch for clue attention & state updating.",
      "Optimised with policy-gradient rewards on 200k simulated puzzles."
    ],
    results : [
      "Achieved 92 % success on 15×15 unseen puzzles.",
      "Generalised to new clue sets without re-training."
    ]
  },

  {
    id: "ufoDashboard",
    title: "UFO Sightings Dashboard",
    subtitle: "Data Visualization",
    image: "images/project_5.png",
    tools: ["Tableau"],
    resources: [
      {
        icon : "images/tableau-icon.png",
        url  : "https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013?:language=en-US&:display_count=n&:origin=viz_share_link",
        label:"Interactive Dashboard"
      }
    ],
    problem : "Public datasets were too fragmented to spot UFO sighting patterns.",
    actions : [
      "Merged 3 national UFO databases (80k+ records).",
      "Visualised shapes, durations and geo-hotspots in Tableau.",
      "Added time-slider and filtered heat-map."
    ],
    results : [
      "Single dashboard covers 10+ KPIs for enthusiasts & researchers.",
      "Revealed seasonal spike (July) and Southwest corridor hotspot."
    ]
  },

  {
    id: "covidAnalysis",
    title: "Covid Death Analysis",
    subtitle: "Machine Learning",
    image: "images/project_6.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Covid-Analysis", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_6.pdf",                        label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_6.ipynb",                      label: "Notebook"}
    ],
    problem : "Hospitals needed data-driven insight into factors driving Covid mortality.",
    actions : [
      "Cleaned and joined 30 country-level datasets (demography, comorbidity, policy).",
      "Trained decision-tree, random-forest and XGBoost; evaluated via RMSE.",
      "Ranked SHAP feature importances for clinical interpretability."
    ],
    results : [
      "Isolation index & diabetes prevalence were top mortality predictors.",
      "Interactive notebook shared with clinicians for scenario testing."
    ]
  },

  {
    id: "targetEmptyPackage",
    title: "Target Empty Package Metrics",
    subtitle: "Data Visualization",
    image: "images/project_7.png",
    tools: ["Excel"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_7.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_7.xlsx", label: "Excel" }
    ],
    problem : "Retail leadership lacked visibility into rising empty-package shrink.",
    actions : [
      "Cleansed and anonymised loss-prevention data across 200 stores.",
      "Built pivot charts tracking incidents by department and month.",
      "Forecasted future losses using Holt-Winters triple exponential smoothing."
    ],
    results : [
      "Identified Beauty & Electronics driving 43 % of recent shrink.",
      "Provided 12-mo projection enabling $380k targeted mitigation budget."
    ]
  },

  {
    id: "handwritingRating",
    title: "Handwriting Rating",
    subtitle: "Machine Learning",
    image: "images/project_8.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Handwriting-Rating", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_8.pdf",                            label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_8.ipynb",                          label: "Notebook"}
    ],
    problem : "Teachers needed an objective way to grade numeral legibility.",
    actions : [
      "Fine-tuned CNN (ResNet-18) on MNIST then on personal digit samples.",
      "Bench-tested depth vs. inference speed for classroom devices.",
      "Created scoring rubric output (0–100) for each digit."
    ],
    results : [
      "Overall legibility score ~80 %.",
      "Flagged digit ‘2’ as lowest-quality (35 % correctness)."
    ]
  },

  {
    id: "digitGenerator",
    title: "Handwritten Digit Generator",
    subtitle: "Machine Learning",
    image: "images/project_9.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Handwritten-Digit-Generator", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_9.pdf",                                   label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_9.ipynb",                                 label: "Notebook"}
    ],
    problem : "Researchers wanted synthetic digits to augment small handwriting datasets.",
    actions : [
      "Built Variational Autoencoder with conv & transposed-conv layers.",
      "Visualised latent space using PCA, t-SNE and UMAP.",
      "Sampled latent vectors to generate novel digits."
    ],
    results : [
      "Generated 10k high-fidelity digits for downstream models.",
      "Reconstruction error < 0.04 (binary cross-entropy)."
    ]
  },

  {
    id: "sheetMusicUpscale",
    title: "Sheet Music Watermark Removal & Upscaling",
    subtitle: "Machine Learning & Web Scraping",
    image: "images/project_10.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Watermark-Remover", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_10_pdf.zip",                      label: "PDFs"  },
      { icon: "images/jupyter-icon.png",url: "documents/Project_10.zip",                          label: "Notebook"}
    ],
    problem : "Musicians struggled with low-res, watermark-covered public-domain scores.",
    actions : [
      "Trained UNet on 20 k paired images for watermark removal.",
      "Upscaled 612×792 scans to 1700×2200 with VDSR.",
      "Scripted full GUI pipeline: scrape → clean → merge to PDF."
    ],
    results : [
      "Delivered print-ready, watermark-free scores in one click.",
      "Average PSNR improved by 9 dB post-processing."
    ]
  },

  {
    id: "deliveryTip",
    title: "Delivery Tip Analysis",
    subtitle: "Data Analysis & Visualization",
    image: "images/project_11.png",
    tools: ["Excel"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_11.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_11.xlsx", label: "Excel" }
    ],
    problem : "Drivers wanted to plan routes for maximum tipping potential.",
    actions : [
      "Built pivot tables & geospatial heat maps on 4k deliveries.",
      "Compared tip averages by day, zone and order size.",
      "Created quick-filter dashboard for on-the-go reference."
    ],
    results : [
      "Revealed Saturday 6-9 pm as top tip window (↑18 %).",
      "Enabled driver to increase weekly earnings by ~12 %."
    ]
  },

  {
    id: "retailStore",
    title: "Retail Store Performance Analysis",
    subtitle: "Database Queries & Visualization",
    image: "images/project_12.png",
    tools: ["SQL", "Python"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/target-packaging-analysis-mssql", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_12.pdf",                                        label: "PDF"    },
      { icon: "images/jupyter-icon.png",url: "documents/Project_12.ipynb",                                      label: "Notebook"}
    ],
    problem : "Corporate lacked granular insight into security & sales performance by store.",
    actions : [
      "Loaded anonymised data into MSSQL; normalised and cleansed tables.",
      "Wrote stored procedures & views for automated KPI extraction.",
      "Visualised theft vs. sales trends in Python (Matplotlib, Seaborn)."
    ],
    results : [
      "Flagged StoreFormat_47 plus three states as high-risk (↑27 % losses).",
      "Linked boycott events to 15 % sales decline in affected regions."
    ]
  },

  {
    id: "smartSentence",
    title: "Smart Sentence Finder",
    subtitle: "Natural Language Processing",
    image: "images/project_13.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Smart-Sentence-Finder", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_13.pdf",                              label: "PDF"    },
      { icon: "images/jupyter-icon.png",url: "documents/Project_13.ipynb",                            label: "Notebook"}
    ],
    problem : "Researchers needed rapid extraction of key sentences from large documents.",
    actions : [
      "Segmented documents; embedded sentences via transformer models.",
      "Computed cosine similarity to query vectors.",
      "Benchmarked 5 pre-trained models on accuracy vs. latency."
    ],
    results : [
      "Returned top-5 relevant sentences in < 0.4 s on 10 k-sentence docs.",
      "Provided model comparison table to aid deployment decisions."
    ]
  },

  {
    id: "website",
    title: "My Website",
    subtitle: "Website Development",
    image: "images/project_14.png",
    tools: ["HTML", "CSS", "JavaScript"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/danielshort3.github.io", label: "GitHub"  },
      { icon: "images/website-icon.png",url: "https://danielshort.me/",                                label: "Live Site"}
    ],
    problem : "Needed a professional online hub to showcase analytics & ML work.",
    actions : [
      "Designed mobile-first layout with semantic HTML5 & CSS3.",
      "Implemented dynamic project loading in vanilla JS; added filter menu.",
      "Integrated Google Analytics & FontAwesome."
    ],
    results : [
      "Average page-load time < 1.2 s (Lighthouse 94 performance).",
      "Recruited two freelance clients within first month of launch."
    ]
  }
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
          <p class="header-label">Downloads</p>
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
   DOM-builder
   ------------------------------------------------------------------
   • Builds cards inside  #projects
   • Builds modals inside #modals
   • Populates #filter-menu counts & click-to-filter behaviour
   ------------------------------------------------------------------ */
function buildPortfolio() {
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
      <img src="${p.image}" alt="${p.title}" loading="lazy">`);
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

  /* ripple-in once grid enters the viewport ------------------------- */
  new IntersectionObserver((e,o)=>{
    if (!e[0].isIntersecting) return;
    [...grid.children].forEach((c,i)=>{
      c.style.animationDelay = `${i*80}ms`;
      c.classList.add("ripple-in");
    });
    o.disconnect();
  },{threshold:.15}).observe(grid);

  /* ➋ Build filter-button counts ----------------------------------- */
  const counts = { all : window.PROJECTS.length };
  window.PROJECTS.forEach(p => p.tools.forEach(t => counts[t]=(counts[t]||0)+1));
  [...menu.children].forEach(btn=>{
    const tag = btn.dataset.filter;
    btn.innerHTML = `${btn.textContent.trim()} ${(counts[tag]||0)}/${counts.all}`;
  });

  /* ➌ Filter behaviour (fade-out → update → fade-in) ---------------- */
  menu.addEventListener("click", e=>{
    if (!e.target.dataset.filter) return;

    /* button UI */
    [...menu.children].forEach(b=>{
      b.classList.replace("btn-primary","btn-secondary");
      b.setAttribute("aria-selected","false");
    });
    e.target.classList.replace("btn-secondary","btn-primary");
    e.target.setAttribute("aria-selected","true");

    const tag = e.target.dataset.filter;
    const startH = grid.offsetHeight;
    grid.style.height = `${startH}px`;
    grid.classList.add("grid-fade");

    setTimeout(()=>{
      /* show / hide cards */
      [...grid.children].forEach(c=>{
        c.style.display = tag==="all" || c.dataset.tags.includes(tag) ? "" : "none";
      });

      /* animate height change */
      const endH = grid.scrollHeight;
      grid.style.height = `${endH}px`;

      const reveal = () => {
        grid.style.height = "";
        [...grid.children].filter(c=>c.style.display!=="none").forEach((c,i)=>{
          c.classList.remove("ripple-in");
          void c.offsetWidth;                  // reflow
          c.style.animationDelay = `${i*80}ms`;
          c.classList.add("ripple-in");
        });
        grid.classList.remove("grid-fade");
      };

      if (startH !== endH) {
        grid.addEventListener("transitionend", e=>{
          if (e.propertyName==="height") reveal();
        },{once:true});
        setTimeout(reveal, 100);               // safety-net
      } else reveal();
    }, 300);                                   // matches CSS fade-out
  });
    const deepId = location.hash.slice(1);
    if (deepId && window.PROJECTS.some(p => p.id === deepId)) {
      openModal(deepId);
    }

}

/* ➍ Modal open / focus-trap / close --------------------------------- */
function openModal(id){
  gtag('event','project_view',{ project_id:id });

  const modal = document.getElementById(`${id}-modal`);
  if (!modal) return;

  modal.classList.add("active");
  document.body.classList.add("modal-open");

  const focusable = modal.querySelectorAll(
    "a,button,[tabindex]:not([tabindex='-1'])"
  );
  focusable[0]?.focus();

  const trap = e=>{
    if (e.key==="Escape"){ close(); return; }
    if (e.key!=="Tab" || !focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length-1];
    if (e.shiftKey ? document.activeElement===first
                   : document.activeElement===last){
      e.preventDefault(); (e.shiftKey?last:first).focus();
    }
  };
  const clickClose = e=>{
    if (e.target.classList.contains("modal") ||
        e.target.classList.contains("modal-close")) close();
  };
  const close = ()=>{
    modal.classList.remove("active");
    document.body.classList.remove("modal-open");
    document.removeEventListener("keydown",trap);
    modal.removeEventListener("click",clickClose);
  };
  document.addEventListener("keydown",trap);
  modal.addEventListener("click",clickClose);
}