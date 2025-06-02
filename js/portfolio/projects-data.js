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
    title: "Pizza Tips Regression Analysis",
    subtitle: "Excel Analytics & Forecasting",
    image: "images/project_1.png",
    tools: ["Excel", "Statistics"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_1.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_1.xlsx", label: "Excel" }
    ],
    problem : "Tip income swung wildly across neighbourhoods and housing types, but drivers had no data-backed story to explain the variation.",
    actions : [
      "Merged 1,251 delivery tickets with NOAA weather, then cleaned the data in Power Query.",
      "Ran a multiple-regression in Excel: Tip = f(cost, delivery-time, rain, max/min temp)."
    ],
    results : [
      "Order cost explains about 38% of tip variance; every extra $10 on the bill lifts the tip by about $1.10.",
      "Apartment customers tip 28% less than house residents (p < 0.001).",
      "Weather and delivery time showed no significant effect on tip size."
    ]
  },

  {
    id: "babynames",
    title: "Baby Name Predictor",
    subtitle: "Python ML Pipeline",
    image: "images/project_2.png",
    tools: ["Python", "scikit-learn"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Baby-Names", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_2_pdf.zip",               label: "PDFs"  },
      { icon: "images/jupyter-icon.png",url: "documents/Project_2.zip",                   label: "Notebook"}
    ],
    problem : "My wife wanted me to come up with new baby names to suggest to her. I wanted to use data-backed insights to solve this problem.",
    actions : [
      "Aggregated & cleaned over 140 years of SSA records, engineering trend and saturation features.",
      "Created a script to suggest names so I could quiz my wife.",
      "Engineered multiple models then averaged results to provide recommendations."
    ],
    results : [
      "Generated personalized top-50 names for boys and girls for my wife.",
      "Successfully named my child with recommendations."
    ]
  },

  {
    id: "pizzaDashboard",
    title: "Pizza Delivery Dashboard",
    subtitle: "Tableau Delivery Storytelling",
    image: "images/project_3.png",
    tools: ["Tableau"],
    resources: [
      { icon: "images/tableau-icon.png",
        url : "https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard?:language=en-US&:display_count=n&:origin=viz_share_link",
        label:"Interactive Dashboard"
      }
    ],
    embed : {
      type : "tableau",
      base : "https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard"
    },
    problem : "As a delivery driver, I wanted to visualize my performance and where I was the most successful.",
    actions : [
      "Reshaped 12,000 rows for Tableau, built map, histogram & 12-month forecast with date/zone filters.",
      "Enabled live updates on current performance and anticipate future performance."
    ],
    results : [
      "Able to review potential deliveries and choose which to take.",
      "Improved tip revenue per delivery by more than 10%."
    ]
  },

  {
    id: "nonogram",
    title: "Nonogram Solver",
    subtitle: "Reinforcement Learning (RL)",
    image: "images/project_4.gif",
    tools: ["Python", "PyTorch"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Nonogram-Solver", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_4.pdf",                  label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_4.ipynb",                label: "Notebook"}
    ],
    problem : "I wanted to create a machine learning model to automatically solve Nonogram puzzles for me.",
    actions : [
      "Built a hybrid CNN + Transformer policy network and trained it on more than 25 million 5×5 puzzles (52,000 episodes × 512-board batches).",
      "Shaped the reward signal around unique guesses, row/column completions, and full-board solves to speed up exploration."
    ],
    results : [
      "Reached 94% accuracy on unseen 5×5 boards.",
    ]
  },

  {
    id: "ufoDashboard",
    title: "UFO Sightings Dashboard",
    subtitle: "Tableau Geospatial Analytics",
    image: "images/project_5.png",
    tools: ["Tableau"],
    resources: [
      { icon: "images/tableau-icon.png",
        url  : "https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013?:language=en-US&:display_count=n&:origin=viz_share_link",
        label:"Interactive Dashboard"
      }
    ],
    embed : {
      type : "tableau",
      base : "https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013"
    },
    problem : "I was curious what trends could be found in UFO sightings across the United States.",
    actions : [
      "Cleaned and standardized hundreds of UFO sightings.",
      "Built heat-map, bar charts, and line charts for rapid exploratory analysis."
    ],
    results : [
      "Determined that most UFO sightings tend to occur just after sunset (earlier in the winter months and later in the summer months).",
      "Found that California is the most common state for UFO sightings, with less in the central U.S."
    ]
  },

  {
    id: "covidAnalysis",
    title: "COVID-19 Outbreak Drivers",
    subtitle: "Python XGBoost & SHAP",
    image: "images/project_6.png",
    tools: ["Python"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Covid-Analysis", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_6.pdf",                        label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_6.ipynb",                      label: "Notebook"}
    ],
    problem : "I used covid historical data to predict future outbreaks.",
    actions : [
      "Cleaned & enriched more than 50k records from the HHS hospital-capacity time-series; added rolling means, trends, and 1/3/7/14-day lag features.",
      "Built an XGBoost classifier with class-imbalance weighting and a strict time-based train/test split.",
    ],
    results : [
      "Used SHAP to surface the 7 most influential drivers and embedded the interactive plot in the report.",
      "The most significant driver of COVID outbreaks was the percentage of ICU beds with COVID.",
      "Utah was likely the next location of COVID outbreak at 6.1%."
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
    problem : "Empty-package theft had ballooned—recovered retail value jumped 5× from Q1 2021 to Q2 2023—yet leaders had no single view to see which employees, locations, or departments were driving the losses.",
    actions : [
      "Consolidated 5,900+ loss-prevention records (2021-2023).",
      "Cleaned employee IDs, DPCI codes, dates, and dollar values.",
      "Built an interactive Excel sheet with drill-downs by employee, department, and recovery location.",
      "Compiled results in a report for management."
    ],
    results : [
      "Revealed Locations 02 & 03 as hot-spots.",
      "Location 03’s shrink doubled in <12 months, with Location 02 doubling in a single quarter.",
      "Flagged Departments 52, 80, 87 (especially Dept 52, up 4x in two quarters).",
      "Identified just three associates (IDs 002, 015, 045) who together accounted for ~47% of recovered value."
    ]
  },

  {
    id: "handwritingRating",
    title: "Handwriting Legibility Scoring",
    subtitle: "PyTorch CNN Fine-Tuning",
    image: "images/project_8.gif",
    tools: ["Python", "PyTorch", "CNN"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Handwriting-Rating", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_8.pdf",                            label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_8.ipynb",                          label: "Notebook"}
    ],
    problem : "My wife would say my handwriting is illegible. I wanted objective thoughts about this.",
    actions : [
      "Created three models of varying complexity to learn to read handwritten digits.",
      "Trained models on 60,000 handwritten digits."
    ],
    results : [
      "Model 3 (the most complex) was the most accurate, with 99.1% accuracy.",
      "My handwriting was determined to be 72.5% legible, with digits 0, 3, 5, and 8 the most illegible.",
      "My wife was correct in her assessment of my poor handwriting."
    ]
  },

  {
    id: "digitGenerator",
    title: "Synthetic Digit Generator",
    subtitle: "Variational Autoencoder",
    image: "images/project_9.gif",
    tools: ["Python", "VAE"],
    resources: [
      { icon: "images/github-icon.png",  url: "https://github.com/danielshort3/Handwritten-Digit-Generator", label: "GitHub" },
      { icon: "images/pdf-icon.png",     url: "documents/Project_9.pdf",                                   label: "PDF"    },
      { icon: "images/jupyter-icon.png", url: "documents/Project_9.ipynb",                                 label: "Notebook"}
    ],
    problem : "I wanted to learn how to generate completely new handwritten digits based on a sample.",
    actions : [
      "Built Variational Autoencoder trained on 60,000 handwritten digits."
    ],
    results : [
      "Successfully visualized latent digits through the trained model.",
      "I have the ability to generate unique handwritten digits when needed."
    ]
  },

  {
    id: "sheetMusicUpscale",
    title: "Sheet Music Watermark Removal & Upscale",
    subtitle: "UNet & VDSR Pipeline",
    image: "images/project_10.gif",
    tools: ["Python", "Computer Vision"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Watermark-Remover", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_10_pdf.zip",                      label: "PDFs"  },
      { icon: "images/jupyter-icon.png",url: "documents/Project_10.zip",                          label: "Notebook"}
    ],
    problem : "I wanted the ability to participate in playing music as my church but did not have any sheet music.",
    actions : [
      "Trained a UNet model on more 20,000 paired pages for watermark removal.",
      "Upscaled 612×792 scans to 1700×2200 with Very Deep Super-Resolution (VDSR).", 
      "Wrapped functionality in a GUI for a simple pipeline."
    ],
    results : [
      "Delivers cleaned and legible sheet music in <10 seconds.",
    ]
  },

  {
    id: "deliveryTip",
    title: "Delivery Tip",
    subtitle: "Excel Geo-Analytics",
    image: "images/project_11.png",
    tools: ["Excel", "Power Query"],
    resources: [
      { icon: "images/pdf-icon.png",   url: "documents/Project_11.pdf",  label: "PDF"   },
      { icon: "images/excel-icon.png", url: "documents/Project_11.xlsx", label: "Excel" }
    ],
    problem : "Drivers wanted optimal shift times & zones for higher tips.",
    actions : [
      "Built geospatial heat-map and pivot filters on over 2,000 deliveries.",
      "Compared tip averages by daypart, zone, and order size."
    ],
    results : [
      "Identified Wednesday as the day tips average the highest dollar amount ($8.07/delivery).",
      "However, Friday is the best day in terms of tips/hour ($10.34/hour).",
      "Increased weekly earnings +12% following insights."
    ]
  },

  {
    id: "retailStore",
    title: "Store Level Loss & Sales ETL",
    subtitle: "MSSQL + Python Viz",
    image: "images/project_12.png",
    tools: ["SQL", "Python"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/target-packaging-analysis-mssql", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_12.pdf",                                        label: "PDF"    },
      { icon: "images/jupyter-icon.png",url: "documents/Project_12.ipynb",                                      label: "Notebook"}
    ],
    problem : "Our store lacked visibility into security incidents, theft hot-spots, and boycott-driven sales swings.",
    actions : [
      "Merged incident, sales & HR tables in MSSQL; automated KPIs via views and stored procedures.",
      "Built Python dashboards mapping theft vs. sales, tagged by format, state and boycott timeline.",
      "Applied anomaly detection to spotlight outlier stores and employees."
    ],
    results : [
      "Identified StoreFormat 47 averaging 14 incidents/store (~4-5× higher than peers).",
      "Flagged State 38 ($991/store/day) plus States 03 & 20 as top theft hot-spots.",
      "Quantified boycott hit: –28.7% (May ’23), –11.6% (Jun ’23), –60.2% (Jul ’23) YoY sales.",
      "Surfaced Employees 231, 098, 196 as suspicious for most empty-package reports (e.g., ID 098: $249/item on just 2 items)."
    ]
  },

  {
    id: "smartSentence",
    title: "Smart Sentence Retriever",
    subtitle: "Embeddings Cosine Comparison",
    image: "images/project_13.png",
    tools: ["Python", "NLP"],
    resources: [
      { icon: "images/github-icon.png", url: "https://github.com/danielshort3/Smart-Sentence-Finder", label: "GitHub" },
      { icon: "images/pdf-icon.png",    url: "documents/Project_13.pdf",                              label: "PDF"    },
      { icon: "images/jupyter-icon.png",url: "documents/Project_13.ipynb",                            label: "Notebook"}
    ],
    problem : "I wanted to have a quick way of finding relevant information from documents based on semantics, not exact wording.",
    actions : [
      "Extracted popular texts into sentences from popular open-source texts such as Alice in Wonderland.",
      "Embedded sentences using 8 embedding models and chose one as the most efficient.",
      "Created a pipeline to automate the process of finding similar context in text."
    ],
    results : [
      "Identified the top sentences similar to the phrase \"She wonders about things\"",
      "Found that model `paraphrase-multilingual-MiniLM-L12-v2` performed the best in my testing."
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
      { icon: "images/website-icon.png",url: "https://danielshort.me/",                                label: "Live Site"}
    ],
    problem : "Needed a fast, mobile-friendly hub to showcase analytics and ML projects.",
    actions : [
      "Built a semantic static site with dynamic project loading via JSON.",
      "Integrated Google Analytics 4, structured data, and lazy-loaded assets."
    ],
    results : [
      "First-contentful paint 1.2s (mobile).",
      "Website and content confirmed selection for my current employment."
    ]
  }
];

// IDs (in order) of projects shown in the top carousel
window.FEATURED_IDS = [
  "sheetMusicUpscale",  
  "nonogram",
  "babynames",
  "handwritingRating",
  "digitGenerator"
];
