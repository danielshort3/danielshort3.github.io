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
  /* ➊ MASTER CATALOG */
/* ================================
   DANIEL SHORT – FEATURED PROJECTS
   ================================ */
window.PROJECTS = [
  {
    id: "smartSentence",
    title: "Smart Sentence Retriever",
    subtitle: "NLP Embeddings & Serverless Retrieval",
    image: "img/projects/smartSentence.png",
    imageWidth: 1280,
    imageHeight: 720,
    videoWebm: "img/projects/smartSentence.webm",
    videoMp4: "img/projects/smartSentence.mp4",
    tools: ["Python", "AWS", "Docker", "NLP"],
    concepts: ["Machine Learning", "Automation"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/Smart-Sentence-Finder", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png", url: "documents/Project_13.pdf", label: "PDF" },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_13.ipynb", label: "Notebook" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/sentence-demo.html", label: "Live Demo" }
    ],
    embed: {
      type: "iframe",
      url: "sentence-demo.html"
    },
    problem: "I wanted a quick way to retrieve relevant sentences by meaning, not exact wording.",
    actions: [
      "Prepared the corpus (Alice in Wonderland), split into sentences, and precomputed embeddings.",
      "Benchmarked 6+ embedding models on 800 sentences (k=2–6), tracking both silhouette score and efficiency (silhouette per million parameters).",
      "Chose the best silhouette‑score model and deployed it as a stateless AWS Lambda endpoint with CORS for top‑k ranking."
    ],
    results: [
      "Best absolute silhouette: 0.313 – Snowflake/snowflake‑arctic‑embed‑l‑v2.0 (k=2, 1024‑d, ~568M params).",
      "Best efficiency: 0.0116 per M params – jinaai/jina‑embeddings‑v3 (~12.9M params, k=6, 1024‑d).",
      "Deployed AWS model: Snowflake/snowflake‑arctic‑embed‑l‑v2.0 (prioritizing quality); live demo runs on a lightweight, scalable Lambda API."
    ]
  },

  {
    id: "chatbotLora",
    title: "Chatbot (LoRA + RAG)",
    subtitle: "RAG Chatbot Fine-Tuned with LoRA",
    image: "img/projects/chatbotLora.png",
    imageWidth: 1280,
    imageHeight: 720,
    videoWebm: "img/projects/chatbotLora.webm",
    videoMp4: "img/projects/chatbotLora.mp4",
    tools: ["Python", "Ollama", "AWS", "Docker"],
    concepts: ["Machine Learning", "Automation"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/Chatbot-LoRA-RAG", label: "GitHub" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/chatbot-demo.html", label: "Live Demo" }
    ],
    embed: {
      type: "iframe",
      url: "chatbot-demo.html"
    },
    problem: "Generic chatbots lacked Visit Grand Junction's tone and rarely suggested our content.",
    actions: [
      "Scraped Visit Grand Junction pages and created a FAISS retrieval index.",
      "Automated a fine-tuning dataset with GPT-OSS 20B via Ollama.",
      "Fine-tuned Mistral 7B on the generated QA set and deployed it to AWS SageMaker.",
      "Created Lambda endpoints so the website can interact with the model."
    ],
    results: [
      "Serverless RAG chatbot scales on demand and returns grounded answers with references after a 10‑minute server warm‑up."
    ]
  },

  {
    id: "shapeClassifier",
    title: "Shape Classifier Demo",
    subtitle: "Handwritten Shape Recognition",
    image: "img/projects/shapeClassifier.png",
    imageWidth: 1280,
    imageHeight: 720,
    videoWebm: "img/projects/shapeClassifier.webm",
    videoMp4: "img/projects/shapeClassifier.mp4",
    tools: ["Python", "PyTorch", "AWS", "Docker"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/Shape-Analyzer", label: "GitHub" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/shape-demo.html", label: "Live Demo" }
    ],
    embed: {
      type: "iframe",
      url: "shape-demo.html"
    },
    problem: "I wanted to create a model that recognizes handwritten shapes.",
    actions: [
      "Downloaded images from Google's QuickDraw dataset to build training and validation splits.",
      "Trained a compact ResNet18 using PyTorch Lightning.",
      "Deployed a minimal AWS Lambda handler for serverless CPU inference from the browser."
    ],
    results: [
      "Predicts circle, triangle, square, hexagon, or octagon from a single drawing with about 90% accuracy.",
      "Demo shows responses return in under a second after a 10‑second warm‑up."
    ]
  },

  {
    id: "ufoDashboard",
    title: "UFO Sightings Dashboard",
    subtitle: "Tableau Geospatial Analytics",
    image: "img/projects/ufoDashboard.png",
    imageWidth: 2008,
    imageHeight: 1116,
    tools: ["Tableau"],
    concepts: ["Visualization", "Analytics"],
    resources: [
      { icon: "img/icons/tableau-icon.png",
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
      "Built heat maps, bar charts, and line charts for rapid exploratory analysis."
    ],
    results : [
      "Determined that most UFO sightings tend to occur just after sunset (earlier in the winter months and later in the summer months).",
      "Found that California is the most common state for UFO sightings, with fewer in the central U.S."
    ]
  },

  {
    id: "covidAnalysis",
    title: "COVID-19 Outbreak Drivers",
    subtitle: "Python XGBoost & SHAP",
    image: "img/projects/covidAnalysis.png",
    imageWidth: 792,
    imageHeight: 524,
    tools: ["Python"],
    concepts: ["Analytics"],
    resources: [
      { icon: "img/icons/github-icon.png",  url: "https://github.com/danielshort3/Covid-Analysis", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_6.pdf",                        label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_6.ipynb",                      label: "Notebook"}
    ],
    problem : "I used COVID historical data to predict future outbreaks.",
    actions : [
      "Cleaned & enriched more than 50k records from the HHS hospital-capacity time-series; added rolling means, trends, and 1/3/7/14-day lag features.",
      "Built an XGBoost classifier with class-imbalance weighting and a strict time-based train/test split.",
    ],
    results : [
      "Used SHAP to surface the 7 most influential drivers and embedded the interactive plot in the report.",
      "The most significant driver of COVID outbreaks was the percentage of ICU beds with COVID.",
      "Utah was the most likely next location for a COVID outbreak (6.1%)."
    ]
  },

  {
    id: "targetEmptyPackage",
    title: "Empty-Package Shrink Dashboard",
    subtitle: "Excel Forecasting & BI",
    image: "img/projects/targetEmptyPackage.png",
    imageWidth: 896,
    imageHeight: 480,
    tools: ["Excel", "Time-Series"],
    concepts: ["Automation", "Analytics"],
    resources: [
      { icon: "img/icons/pdf-icon.png",   url: "documents/Project_7.pdf",  label: "PDF"   },
      { icon: "img/icons/excel-icon.png", url: "documents/Project_7.xlsx", label: "Excel" }
    ],
    problem : "Empty‑package theft had ballooned: recovered retail value jumped 5× from Q1 2021 to Q2 2023, yet leaders had no single view to see which employees, locations, or departments were driving the losses.",
    actions : [
      "Consolidated 5,900+ loss-prevention records (2021-2023).",
      "Cleaned employee IDs, DPCI codes, dates, and dollar values.",
      "Built an interactive Excel sheet with drill-downs by employee, department, and recovery location.",
      "Compiled results in a report for management."
    ],
    results : [
      "Revealed Locations 02 & 03 as hot spots.",
      "Location 03’s shrink doubled in <12 months, with Location 02 doubling in a single quarter.",
      "Flagged Departments 52, 80, 87 (especially Dept 52, up 4x in two quarters).",
      "Identified just three associates (IDs 002, 015, 045) who together accounted for ~47% of recovered value."
    ]
  },

  {
    id: "handwritingRating",
    title: "Handwriting Legibility Scoring",
    subtitle: "PyTorch CNN Fine-Tuning",
    image: "img/projects/handwritingRating.png",
    imageWidth: 600,
    imageHeight: 960,
    videoWebm: "img/projects/handwritingRating.webm",
    videoMp4:  "img/projects/handwritingRating.mp4",
    videoOnly: true,
    tools: ["Python", "PyTorch", "CNN"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png",  url: "https://github.com/danielshort3/Handwriting-Rating", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_8.pdf",                            label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_8.ipynb",                          label: "Notebook"}
    ],
    problem : "My wife would say my handwriting is illegible. I wanted an objective assessment.",
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
    image: "img/projects/digitGenerator.png",
    imageWidth: 400,
    imageHeight: 400,
    videoWebm: "img/projects/digitGenerator.webm",
    videoMp4:  "img/projects/digitGenerator.mp4",
    tools: ["Python", "VAE"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png",  url: "https://github.com/danielshort3/Handwritten-Digit-Generator", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_9.pdf",                                   label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_9.ipynb",                                 label: "Notebook"}
    ],
    problem : "I wanted to learn how to generate completely new handwritten digits based on samples.",
    actions : [
      "Built Variational Autoencoder trained on 60,000 handwritten digits."
    ],
    results : [
      "Successfully visualized latent digits through the trained model.",
      "I can generate unique handwritten digits when needed."
    ]
  },

  {
    id: "sheetMusicUpscale",
    title: "Sheet Music Watermark Removal & Upscale",
    subtitle: "UNet & VDSR Pipeline",
    image: "img/projects/sheetMusicUpscale.png",
    imageWidth: 1604,
    imageHeight: 1230,
    videoWebm: "img/projects/sheetMusicUpscale.webm",
    videoMp4:  "img/projects/sheetMusicUpscale.mp4",
    tools: ["Python", "Computer Vision"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/Watermark-Remover", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",    url: "documents/Project_10_pdf.zip",                      label: "PDFs"  },
      { icon: "img/icons/jupyter-icon.png",url: "documents/Project_10.zip",                          label: "Notebook"}
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
    subtitle: "Excel Geo-Analytics & Optimization",
    image: "img/projects/deliveryTip.png",
    imageWidth: 960,
    imageHeight: 794,
    tools: ["Excel", "Power Query"],
    concepts: ["Analytics"],
    resources: [
      { icon: "img/icons/pdf-icon.png",   url: "documents/Project_11.pdf",  label: "PDF"   },
      { icon: "img/icons/excel-icon.png", url: "documents/Project_11.xlsx", label: "Excel" }
    ],
    problem : "Drivers wanted optimal shift times & zones for higher tips.",
    actions : [
      "Built a geospatial heat map and pivot filters on over 2,000 deliveries.",
      "Compared tip averages by daypart, zone, and order size."
    ],
    results : [
      "Identified Wednesday as the day tips average the highest dollar amount ($8.07/delivery).",
      "However, Friday is the best day in terms of tips per hour ($10.34/hour).",
      "Increased weekly earnings by 12% following the insights."
    ]
  },

  {
    id: "retailStore",
    title: "Store-Level Loss & Sales ETL",
    subtitle: "SQL ETL + Anomaly Detection",
    image: "img/projects/retailStore.png",
    imageWidth: 1423,
    imageHeight: 947,
    tools: ["SQL", "Python"],
    concepts: ["Automation", "Analytics"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/target-packaging-analysis-mssql", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",    url: "documents/Project_12.pdf",                                        label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png",url: "documents/Project_12.ipynb",                                      label: "Notebook"}
    ],
    problem : "Our store lacked visibility into security incidents, theft hot-spots, and boycott-driven sales swings.",
    actions : [
      "Merged incident, sales & HR tables in SQL; automated KPIs via views and stored procedures.",
      "Built Python dashboards mapping theft vs. sales, tagged by format, state and boycott timeline.",
      "Applied anomaly detection to spotlight outlier stores and employees."
    ],
    results : [
      "Identified StoreFormat 47 averaging 14 incidents/store (~4–5× higher than peers).",
      "Flagged State 38 ($991/store/day) plus States 03 & 20 as top theft hot spots.",
      "Quantified boycott hit: –28.7% (May ’23), –11.6% (Jun ’23), –60.2% (Jul ’23) YoY sales.",
      "Surfaced Employees 231, 098, 196 as suspicious for most empty-package reports (e.g., ID 098: $249/item on just 2 items)."
    ]
  },

  {
    id: "pizza",
    title: "Pizza Tips Regression Modeling",
    subtitle: "Excel Analytics & Regression Modeling",
    image: "img/projects/pizza.png",
    imageWidth: 1726,
    imageHeight: 1054,
    tools: ["Excel", "Statistics"],
    concepts: ["Analytics"],
    resources: [
      { icon: "img/icons/pdf-icon.png",   url: "documents/Project_1.pdf",  label: "PDF"   },
      { icon: "img/icons/excel-icon.png", url: "documents/Project_1.xlsx", label: "Excel" }
    ],
    problem : "Tip income swung wildly across neighborhoods and housing types, but drivers had no data-backed story to explain the variation.",
    actions : [
      "Merged 1,251 delivery tickets with NOAA weather, then cleaned the data in Power Query.",
      "Ran a multiple regression in Excel: Tip = f(cost, delivery time, rain, max/min temperature)."
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
    image: "img/projects/babynames.png",
    imageWidth: 1420,
    imageHeight: 940,
    tools: ["Python", "scikit-learn"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/Baby-Names", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",    url: "documents/Project_2.pdf",                    label: "PDFs"  },
      { icon: "img/icons/jupyter-icon.png",url: "documents/Project_2.ipynb",                  label: "Notebook"}
    ],
    problem : "My wife wanted me to come up with new baby names to suggest to her. I wanted to use data-backed insights to solve this problem.",
    actions : [
      "Aggregated & cleaned over 140 years of SSA records, engineering trend and saturation features.",
      "Created a script to suggest names so I could quiz my wife.",
      "Engineered multiple models then averaged results to provide recommendations."
    ],
    results : [
      "Generated personalized top 50 names for boys and girls for my wife.",
      "Successfully named my child with recommendations."
    ]
  },

  {
    id: "pizzaDashboard",
    title: "Pizza Delivery Dashboard",
    subtitle: "Tableau Analytics & Forecasting",
    image: "img/projects/pizzaDashboard.png",
    imageWidth: 1250,
    imageHeight: 1092,
    tools: ["Tableau"],
    concepts: ["Visualization", "Analytics"],
    resources: [
      { icon: "img/icons/tableau-icon.png",
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
      "Enabled live updates on current performance and the ability to anticipate future performance."
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
    image: "img/projects/nonogram.png",
    imageWidth: 1080,
    imageHeight: 1080,
    videoWebm: "img/projects/nonogram.webm",
    videoMp4:  "img/projects/nonogram.mp4",
    tools: ["Python", "PyTorch"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png",  url: "https://github.com/danielshort3/Nonogram-Solver", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_4.pdf",                  label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_4.ipynb",                label: "Notebook"}
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
    id: "website",
    title: "danielshort.me",
    subtitle: "Responsive Portfolio Site",
    image: "img/projects/website.png",
    imageWidth: 1240,
    imageHeight: 1456,
    tools: ["HTML", "CSS", "JavaScript"],
    concepts: ["Product", "Visualization"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/danielshort3.github.io", label: "GitHub" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/", label: "Live Site" }
    ],
    problem: "Needed a fast, mobile-friendly hub to showcase analytics and ML projects.",
    actions: [
      "Built a semantic static site with dynamic project loading via JSON.",
      "Integrated Google Analytics 4, structured data, and lazy-loaded assets."
    ],
    results: [
      "First Contentful Paint: 1.2s (mobile).",
      "This site and its content helped secure my current role."
    ],
  }
];

// IDs (in order) of projects shown in the top carousel
window.FEATURED_IDS = [
  "shapeClassifier",
  "chatbotLora",
  "sheetMusicUpscale",
  "digitGenerator",
  "nonogram"
];
