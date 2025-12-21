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
    ],
    caseStudy: [
      {
        title: "System Design",
        lead: "This is a retrieval system for a fixed corpus (Alice in Wonderland): precompute sentence embeddings once, then embed each query at runtime and rank by cosine similarity.",
        bullets: [
          "Offline: chunk raw text, segment into sentences, normalize whitespace, and precompute embeddings.",
          "Online: embed the query, score against the cached embedding matrix, and return top-k sentences with similarity scores.",
          "Frontend: a lightweight demo that calls `/health` and `/rank` and renders ranked results with confidence meters."
        ]
      },
      {
        title: "Model Selection",
        bullets: [
          "Benchmarked multiple embedding models and used silhouette score (plus silhouette per million parameters) to balance quality vs. cost.",
          "Selected Snowflake Arctic Embed L v2 for highest silhouette score; tracked Jina v3 as the best efficiency baseline.",
          "Kept the evaluation reproducible by fixing the corpus, sampling strategy, and k-range for clustering."
        ]
      },
      {
        title: "Serverless Deployment",
        bullets: [
          "Built an AWS Lambda container image that bundles CPU-only PyTorch, FastAPI + Mangum, the cached embedding model, and precomputed corpus artifacts.",
          "Forced the Hugging Face backend (no sentence-transformers) to keep the cold start smaller and avoid multiprocess warnings on Lambda.",
          "Exposed a Lambda Function URL with CORS so the website demo can call it directly from the browser."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add support for user-provided documents (upload + background embedding job) instead of a fixed corpus.",
          "Introduce ANN search (HNSW/FAISS) for larger corpora and faster top-k retrieval.",
          "Move beyond silhouette score with a small labeled relevance set and offline evaluation metrics (e.g., nDCG@k)."
        ]
      }
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
    role: [
      "Led the end-to-end prototype: data ingestion, retrieval, fine-tuning, deployment, and web integration."
    ],
    notes: "Built from public Visit Grand Junction web content; tuned to return grounded answers with references.",
    problem: "Generic chatbots lacked Visit Grand Junction's tone and rarely suggested our content.",
    actions: [
      "Scraped Visit Grand Junction pages and created a FAISS retrieval index.",
      "Automated a fine-tuning dataset with GPT-OSS 20B via Ollama.",
      "Fine-tuned Mistral 7B on the generated QA set and deployed it to AWS SageMaker.",
      "Created Lambda endpoints so the website can interact with the model."
    ],
    results: [
      "Serverless RAG chatbot scales on demand and returns grounded answers with references after a 10‑minute server warm‑up."
    ],
    caseStudy: [
      {
        title: "System Design",
        lead: "A RAG pipeline tuned for a single domain: crawl → index → generate training data → fine-tune with LoRA → deploy behind an API the website can call.",
        bullets: [
          "Ingestion: crawl Visit Grand Junction pages and store cleaned text chunks.",
          "Retrieval: build a FAISS index so responses can cite relevant source passages.",
          "Generation: fine-tune Mistral 7B with LoRA on auto-generated domain Q&A pairs."
        ]
      },
      {
        title: "Dataset Generation",
        bullets: [
          "Used GPT-OSS 20B via Ollama to generate Q&A pairs from the crawled content, keeping the dataset aligned to the site’s tone and facts.",
          "Automated the end-to-end pipeline with scripts (crawl, build index, build dataset, finetune, merge LoRA).",
          "Optimized for iteration: run locally or in Docker, then deploy the same artifacts for hosting."
        ]
      },
      {
        title: "Serving and Deployment",
        bullets: [
          "Merged the LoRA adapter into a 4-bit model and exposed a FastAPI inference endpoint.",
          "Deployed the model behind AWS (SageMaker + Lambda endpoints) so the website can query it without shipping model code to the client.",
          "Added a status check and warm-up UX in the demo to handle cold starts gracefully."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add automated RAG evaluation (retrieval hit-rate, groundedness, and citation accuracy).",
          "Introduce caching for common questions and streaming tokens for a faster perceived response.",
          "Add guardrails for prompt injection and a tighter citation-first response format."
        ]
      }
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
    ],
    caseStudy: [
      {
        title: "Data and Training",
        lead: "I trained a compact classifier on Google’s QuickDraw sketches to recognize five basic shapes from a single monochrome drawing.",
        bullets: [
          "Built class-balanced train/validation splits from QuickDraw categories (circle, triangle, square, hexagon, octagon).",
          "Used a ResNet18 backbone in PyTorch Lightning to keep training and reproducibility simple.",
          "Exported lightweight weights (`model.pt`) so inference doesn’t require Lightning."
        ]
      },
      {
        title: "Inference API",
        bullets: [
          "Implemented an AWS Lambda handler that accepts a base64-encoded image and returns the predicted class + confidence.",
          "Designed the API to be browser-friendly (CORS + small JSON payloads) for a responsive demo experience.",
          "Kept inference CPU-friendly so it can run in serverless environments without GPUs."
        ]
      },
      {
        title: "Demo UX",
        bullets: [
          "Canvas-based drawing UI with a clear action to submit and a confidence bar to interpret predictions.",
          "Graceful endpoint fallbacks and clear status messaging during warm-up/cold starts.",
          "Tuned the preprocessing so messy real-world strokes still map to the training distribution."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add calibration (e.g., temperature scaling) so confidence is more trustworthy.",
          "Expand to more shapes and train with augmentation to handle stroke width and incomplete outlines.",
          "Add a small human-drawn validation set to measure real-world generalization beyond QuickDraw."
        ]
      }
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
    ],
    caseStudy : [
      {
        title: "Dashboard Design",
        lead: "I built a single-screen Tableau dashboard that makes it easy to explore UFO sightings by where, when, and what people reported seeing.",
        bullets: [
          "Geospatial views: state and city maps to compare hotspots at different levels of detail.",
          "Time patterns: a month/hour heatmap to spot seasonal and daypart effects.",
          "Shape analysis: top shapes and a month-by-month prevalence view to see how reports shift over the year."
        ]
      },
      {
        title: "Data Preparation",
        bullets: [
          "Standardized location fields so records could be mapped consistently.",
          "Normalized timestamp fields to compare reports by month and hour.",
          "Cleaned categorical fields (shape labels) to reduce fragmentation from spelling/formatting variants."
        ]
      },
      {
        title: "Insights and Interpretation",
        bullets: [
          "Reports cluster around dusk/night hours, supporting the 'after sunset' effect.",
          "Seasonality shifts the peak hour later in summer and earlier in winter (longer vs. shorter daylight).",
          "California stands out as the highest-volume state, while central regions are comparatively sparse."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Normalize by population (per-capita rates) and add confidence intervals for low-count areas.",
          "Add dashboard parameters for filtering by year, duration, and report quality.",
          "Flag potential data-quality issues (missing locations, ambiguous shapes) directly in the UI."
        ]
      }
    ]
  },

  {
    id: "covidAnalysis",
    title: "COVID-19 Outbreak Drivers",
    subtitle: "Python XGBoost & SHAP",
    image: "img/projects/covidAnalysis.png",
    imageWidth: 792,
    imageHeight: 524,
    tools: ["Python", "AWS"],
    concepts: ["Analytics"],
    resources: [
      { icon: "img/icons/github-icon.png",  url: "https://github.com/danielshort3/Covid-Analysis", label: "GitHub" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/covid-outbreak-demo",    label: "Live Demo" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_6.pdf",                        label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_6.ipynb",                      label: "Notebook"}
    ],
    embed: {
      type: "iframe",
      url: "covid-outbreak-demo.html"
    },
    problem : "I used COVID historical data to predict future outbreaks.",
    actions : [
      "Cleaned & enriched more than 50k records from the HHS hospital-capacity time-series; added rolling means, trends, and 1/3/7/14-day lag features.",
      "Built an XGBoost classifier with class-imbalance weighting and a strict time-based train/test split.",
    ],
    results : [
      "Used SHAP to surface the 7 most influential drivers and embedded the interactive plot in the report.",
      "The most significant driver of COVID outbreaks was the percentage of ICU beds with COVID.",
      "Utah was the most likely next location for a COVID outbreak (6.1%)."
    ],
    caseStudy : [
      {
        title: "Problem Framing",
        lead: "I framed this as a short-horizon risk prediction: estimate the probability that a state will breach 90% ICU occupancy within the next 7 days.",
        bullets: [
          "Target label: max(adult ICU bed utilization) over the next 7 days ≥ 0.90.",
          "Constraint: extreme class imbalance (breaches are rare), so metrics must focus on ranking and precision/recall tradeoffs.",
          "Goal: provide an interpretable early-warning signal rather than a black-box forecast."
        ]
      },
      {
        title: "Data and Feature Engineering",
        bullets: [
          "Started from the HHS hospital capacity time series (state × day) and cleaned missingness with forward-filling and column pruning.",
          "Created rolling-window features and lag features (1/3/7/14 days) to capture trend and momentum.",
          "Built ratio features like ICU beds with COVID (%) and inpatient COVID share to normalize across states."
        ]
      },
      {
        title: "Modeling and Evaluation",
        bullets: [
          "Trained an XGBoost classifier with a time-based train/test split and class-imbalance handling.",
          "Measured ranking quality with AUROC (0.606) and PR-AUC (0.060); at a 0.5 threshold the confusion matrix was [[17601, 276], [728, 31]].",
          "Used the model as a risk scorer (probability output) rather than a hard yes/no classifier."
        ]
      },
      {
        title: "Explainability (SHAP)",
        bullets: [
          "Used SHAP to identify which daily metrics raise or lower the chance of an ICU crisis in the next week.",
          "Key takeaway: a high % of ICU beds already filled by COVID patients is the clearest early-warning signal; lower overall utilization reduces risk.",
          "Renamed feature labels in the SHAP plot to keep it stakeholder-friendly."
        ]
      },
      {
        title: "Operational Output",
        bullets: [
          "Exported a per-state, per-day CSV of 7-day breach probabilities for monitoring and reporting.",
          "Most likely breach location in the final snapshot: UT on 2023-06-11 with a 7-day breach probability of 6.1%.",
          "Designed the workflow to be rerun as new days arrive."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Calibrate probabilities and tune thresholds for a target precision/recall operating point.",
          "Add exogenous signals (vaccination, policy, mobility, variant waves) to improve early warning power.",
          "Evaluate stability across time (drift) and add monitoring for feature shifts."
        ]
      }
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
    role: [
      "Built the Excel BI workflow (data cleanup, drill-down views, and reporting)."
    ],
    notes: "Employee and location identifiers are anonymized in the write-up.",
    problem : "Empty‑package theft had ballooned: recovered retail value jumped 5× from Q1 2021 to Q2 2023, and leaders needed a single view to see where losses were concentrated.",
    actions : [
      "Consolidated 5,900+ loss-prevention records (2021-2023).",
      "Cleaned and anonymized employee IDs, DPCI codes, dates, and dollar values.",
      "Built an interactive Excel dashboard with drill-downs by associate, department, and recovery location.",
      "Compiled results in a report for management."
    ],
    results : [
      "Two recovery locations (anonymized) emerged as the primary hot spots.",
      "Shrink doubled in under 12 months at one hot spot, with the second doubling in a single quarter.",
      "Three departments (anonymized) drove most recoveries; the top department jumped ~4× in two quarters.",
      "Three associates (anonymized) accounted for ~47% of recovered value."
    ],
    caseStudy : [
      {
        title: "Data Cleanup and Governance",
        lead: "This analysis turns messy operational loss-prevention logs into a decision-ready dashboard while preserving privacy through anonymization.",
        bullets: [
          "Consolidated 5,900+ records (2021–2023) and standardized dates, locations, departments, and retail values.",
          "Anonymized employee and store identifiers to keep the write-up shareable without exposing internal details.",
          "Created consistent fields for quarter/time-series trend analysis."
        ]
      },
      {
        title: "Dashboard and KPIs",
        bullets: [
          "Built pivot-based drill-downs by associate, department, and recovery location to identify concentration of losses.",
          "Tracked both retail value and item count so volume vs. value trends are visible together.",
          "Added trend views to highlight acceleration (doubling patterns) rather than only totals."
        ]
      },
      {
        title: "Key Findings",
        bullets: [
          "Recovered retail value increased five-fold from Q1 2021 to Q2 2023 (no sign of slowing).",
          "Two recovery locations emerged as hot spots; one doubled in <12 months and another doubled in a single quarter.",
          "A small set of associates and departments drove a disproportionate share of recovered value (~47% from three associates)."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add normalization (per-store traffic or shipments) to separate growth from volume changes.",
          "Introduce control charts / anomaly alerts to flag sudden spikes automatically.",
          "Automate refresh via scheduled exports so leaders always see current quarter performance."
        ]
      }
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
    tools: ["Python", "PyTorch", "AWS", "CNN"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/github-icon.png",  url: "https://github.com/danielshort3/Handwriting-Rating", label: "GitHub" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/handwriting-rating-demo.html", label: "Live Demo" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_8.pdf",                            label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_8.ipynb",                          label: "Notebook"}
    ],
    embed: {
      type: "iframe",
      url: "handwriting-rating-demo.html"
    },
    problem : "My wife would say my handwriting is illegible. I wanted an objective assessment.",
    actions : [
      "Created three models of varying complexity to learn to read handwritten digits.",
      "Trained models on 60,000 handwritten digits.",
      "Deployed the best model behind a serverless scoring API for the live demo."
    ],
    results : [
      "Model 3 (the most complex) was the most accurate, with 99.1% accuracy.",
      "My handwriting was determined to be 72.5% legible, with digits 0, 3, 5, and 8 the most illegible.",
      "My wife was correct in her assessment of my poor handwriting."
    ],
    caseStudy : [
      {
        title: "Experiment Setup",
        lead: "I treated MNIST as a controllable baseline and then tested whether the model judged my own handwriting by the same standards.",
        bullets: [
          "Dataset: MNIST (60,000 train / 10,000 test) for digit recognition.",
          "Goal: compare model families and quantify personal legibility with a consistent metric.",
          "Output: per-digit accuracy and a confusion matrix to see which digits I write most ambiguously."
        ]
      },
      {
        title: "Model Iteration",
        bullets: [
          "Model 1: simple linear classifier to establish a quick baseline.",
          "Model 2: a small CNN (TinyVGG-style) to capture local stroke structure.",
          "Model 3: a deeper CNN (VGG16-style) that achieved the best test performance (99.1% accuracy)."
        ]
      },
      {
        title: "Custom Handwriting Evaluation",
        bullets: [
          "Built a small custom dataset loader so new images could be preprocessed and scored consistently.",
          "Computed overall legibility (72.5%) and identified the hardest digits for me (0, 3, 5, and 8).",
          "Used the error breakdown to make the result actionable (which digits to practice)."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Train on EMNIST or other real handwriting datasets to reduce domain shift from MNIST.",
          "Add augmentation (stroke thickness, rotation, blur) to better match camera/scanner variation.",
          "Calibrate confidence so the model can say 'uncertain' instead of forcing a guess."
        ]
      }
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
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/digit-generator-demo", label: "Live Demo" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_9.pdf",                                   label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_9.ipynb",                                 label: "Notebook"}
    ],
    embed: {
      type: "iframe",
      url: "/digit-generator-demo"
    },
    problem : "I wanted to learn how to generate completely new handwritten digits based on samples.",
    actions : [
      "Built Variational Autoencoder trained on 60,000 handwritten digits."
    ],
    results : [
      "Successfully visualized latent digits through the trained model.",
      "I can generate unique handwritten digits when needed."
    ],
    caseStudy : [
      {
        title: "Model Architecture (VAE)",
        lead: "I trained a Variational Autoencoder to learn a compact latent representation of handwritten digits and generate new samples by sampling that latent space.",
        bullets: [
          "Encoder: convolutional stack that maps an image to a latent mean and variance.",
          "Reparameterization trick to sample latents during training while keeping gradients stable.",
          "Decoder: transposed convolutions that reconstruct images from latent vectors."
        ]
      },
      {
        title: "Training and Sampling",
        bullets: [
          "Trained on MNIST and monitored the reconstruction vs. KL-divergence tradeoff.",
          "Validated by comparing original vs. reconstructed digits and by sampling random latent vectors.",
          "Saved the model so generation is a one-command inference step, not a re-train."
        ]
      },
      {
        title: "Latent Space Analysis",
        bullets: [
          "Used PCA, t-SNE, and UMAP to visualize how digits cluster in latent space.",
          "Applied clustering (HDBSCAN/DBSCAN) to study structure and identify outliers.",
          "Performed reconstruction-error analysis to see which digits are hardest to model."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Build a conditional VAE so I can generate a specific digit on demand.",
          "Improve sample sharpness with stronger decoders, perceptual losses, or diffusion-based refinement.",
          "Add quantitative generative metrics (FID-like proxies) to compare models over time."
        ]
      }
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
    ],
    caseStudy : [
      {
        title: "End-to-End Pipeline",
        lead: "This project combines web scraping, image restoration, super-resolution, and PDF compilation into a single workflow so sheet music is playable again.",
        bullets: [
          "Input: low-resolution, watermarked page images (scanned sheet music).",
          "Step 1: watermark removal with a UNet segmentation/restoration model.",
          "Step 2: upscaling with VDSR to produce print-friendly pages.",
          "Output: cleaned images compiled into a ready-to-use PDF."
        ]
      },
      {
        title: "Modeling Choices",
        bullets: [
          "UNet works well for structured, local artifacts (watermarks) while preserving staff lines and note heads.",
          "VDSR adds detail and readability when starting from low-resolution scans.",
          "Packaged model weights as state dicts so the pipeline can run without retraining."
        ]
      },
      {
        title: "GUI and Automation",
        bullets: [
          "Built a PyQt5 GUI to scrape, process, and compile pages without needing to run notebook cells manually.",
          "Used background worker threads to keep the UI responsive while processing batches.",
          "Designed the workflow for repeat use (new songs in, clean PDFs out)."
        ]
      },
      {
        title: "Performance and Reliability",
        bullets: [
          "Optimized for practical speed: the pipeline produces legible output in under 10 seconds for typical use cases.",
          "Added sensible defaults and a simple 'one path' experience so non-ML users can run it.",
          "Kept the pipeline modular so either model can be swapped or improved independently."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add tiled inference for very large pages and better memory usage.",
          "Measure quality with objective metrics (PSNR/SSIM) and small human readability tests.",
          "Explore more modern restoration approaches for better note/staff preservation."
        ]
      }
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
      "Built a geospatial heat map and pivot filters on 1,251 deliveries.",
      "Compared tip averages by daypart, zone, and order size."
    ],
    results : [
      "Identified Wednesday as the day tips average the highest dollar amount ($8.07/delivery).",
      "However, Friday is the best day in terms of tips per hour ($10.34/hour).",
      "Increased weekly earnings by 12% following the insights."
    ],
    caseStudy : [
      {
        title: "Dataset and Cleaning",
        lead: "I analyzed delivery tickets (orders, tips, timing, city, and housing attributes) and built an Excel workflow that stays usable for non-technical drivers.",
        bullets: [
          "Normalized timestamps and derived metrics like total delivery time (minutes) and tip percentage.",
          "Standardized location fields (city and neighborhood) for mapping and rollups.",
          "Used Power Query so the dataset can be refreshed without redoing manual steps."
        ]
      },
      {
        title: "Geo-Analytics Dashboard",
        bullets: [
          "Built a tip heatmap by housing/neighborhood to identify the best delivery zones.",
          "Created pivot filters to compare performance by housing type, gated communities, city, and order size.",
          "Added weekday and shift-level summaries to support scheduling decisions."
        ]
      },
      {
        title: "Key Findings",
        bullets: [
          "Wednesday had the highest average tip per delivery ($8.07), while Friday produced the best tips per hour ($10.34/hour).",
          "Tip behavior varied by housing type and neighborhood, which helped prioritize zones during peak hours.",
          "Documented personal baseline stats (e.g., average tip $7.14 and ~42-minute average delivery time) to track improvement over time."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add distance and drive-time estimates to separate 'better zones' from 'shorter routes'.",
          "Control for order size to avoid confusing high tips with high bills.",
          "Turn insights into a simple 'where to go next' recommendation view for live shifts."
        ]
      }
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
    role: [
      "Owned the analysis end-to-end: SQL data modeling/ETL, anomaly detection, and reporting."
    ],
    notes: "Store, state, and employee identifiers are anonymized in the case study.",
    problem : "Our store lacked visibility into security incidents, theft hot-spots, and boycott-driven sales swings.",
    actions : [
      "Merged incident, sales & HR tables in SQL; automated KPIs via views and stored procedures.",
      "Built Python dashboards mapping theft vs. sales, tagged by format, state and boycott timeline.",
      "Applied anomaly detection to spotlight outlier stores and employees."
    ],
    results : [
      "Identified an outlier store cluster averaging 14 incidents/store (~4–5× higher than peers).",
      "Flagged multiple regions as top theft hot spots (up to ~$991/store/day).",
      "Quantified boycott hit: –28.7% (May ’23), –11.6% (Jun ’23), –60.2% (Jul ’23) YoY sales.",
      "Surfaced a small set of high-risk associates (anonymized) for empty-package reports; one outlier averaged $249/item on just 2 items."
    ],
    caseStudy : [
      {
        title: "Data Modeling and ETL",
        lead: "I built a SQL-first analytics layer that consolidates incidents, theft, sales, and HR attributes so leaders can answer 'where is risk growing?' with a single source of truth.",
        bullets: [
          "Joined incident, sales, and employee tables and created reusable views for KPI rollups.",
          "Standardized date keys and dimensions (store format, region/state) to support consistent slicing.",
          "Kept identifiers anonymized so insights can be shared without exposing sensitive internal data."
        ]
      },
      {
        title: "Analysis Workflow",
        bullets: [
          "Implemented targeted queries to find high-incident store formats, top-theft states, and high-risk associates.",
          "Measured boycott impact with year-over-year comparisons around the boycott timeline.",
          "Used Python visualizations to communicate findings clearly (hotspots, trends, and outliers)."
        ]
      },
      {
        title: "Anomaly Detection",
        bullets: [
          "Flagged outlier stores and associates by comparing against peer baselines rather than raw totals.",
          "Highlighted both 'high frequency' (incidents/store) and 'high severity' (value per item) anomalies.",
          "Focused outputs on investigations leaders can act on (where to send resources next)."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add automated alerting (weekly anomaly reports) and an audit trail for investigations.",
          "Normalize by traffic/shipments to separate volume effects from true risk changes.",
          "Expand to causal analysis for boycott drivers and mitigation experiments."
        ]
      }
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
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/pizza-tips-demo", label: "Live Demo" },
      { icon: "img/icons/pdf-icon.png",   url: "documents/Project_1.pdf",  label: "PDF"   },
      { icon: "img/icons/excel-icon.png", url: "documents/Project_1.xlsx", label: "Excel" }
    ],
    embed: {
      type: "iframe",
      url: "pizza-tips-demo.html"
    },
    problem : "Tip income swung wildly across neighborhoods and housing types, but drivers had no data-backed story to explain the variation.",
    actions : [
      "Merged 1,251 delivery tickets with NOAA weather, then cleaned the data in Power Query.",
      "Ran a multiple regression in Excel: Tip = f(cost, delivery time, rain, max/min temperature)."
    ],
    results : [
      "Order cost explains about 38% of tip variance; every extra $10 on the bill lifts the tip by about $1.10.",
      "Apartment customers tip 28% less than house residents (p < 0.001).",
      "Weather and delivery time showed no significant effect on tip size."
    ],
    caseStudy : [
      {
        title: "Data Integration",
        lead: "I merged delivery tickets with local NOAA weather to test common hypotheses about what drives tipping behavior.",
        bullets: [
          "Combined 1,251 deliveries with daily weather features (rainfall, max/min temperature, wind).",
          "Cleaned the dataset in Power Query and created derived fields like tip percentage and delivery time (minutes).",
          "Separated housing types (apartment vs. residential) to test neighborhood effects."
        ]
      },
      {
        title: "Exploratory Analysis",
        bullets: [
          "Found a strong positive correlation (0.6181) between order cost and tip amount.",
          "Observed only a mild relationship between rainfall and delivery duration (correlation 0.1410).",
          "Identified clear seasonal demand swings (order counts more than doubled in summer/early fall)."
        ]
      },
      {
        title: "Regression and Hypothesis Tests",
        bullets: [
          "Ran a multiple regression: Tip = f(cost, delivery time, rain, max/min temperature).",
          "Result: order cost was the primary driver, explaining ~38% of tip variance (≈ +$1.10 tip per +$10 bill).",
          "Validated housing differences with a two-sample t-test: apartment customers tipped ~28% less than house residents (p < 0.001)."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add distance, time-of-day, and driver controls to reduce omitted-variable bias.",
          "Model tip percentage and tip amount separately to avoid conflating larger orders with generosity.",
          "Use mixed-effects models to capture repeated customers or neighborhood-level variance."
        ]
      }
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
    ],
    caseStudy : [
      {
        title: "Data and Labeling",
        lead: "I combined Social Security Administration name data (national + Colorado) with my own preference labels to build a personalized recommender.",
        bullets: [
          "Aggregated 140+ years of SSA records and engineered 'recency' and trend features to avoid only recommending historically popular names.",
          "Focused on Colorado to keep suggestions relevant to the names my wife actually encounters.",
          "Collected preference labels through a simple 'quiz' workflow so the model learns her taste over time."
        ]
      },
      {
        title: "Feature Engineering",
        bullets: [
          "Name morphology: length, vowels/consonants, syllable count, start/end vowel flags, and entropy.",
          "Popularity dynamics: total count, most popular year, and recent-count features to capture saturation and momentum.",
          "Language-root signal via `langdetect` as a lightweight proxy for name origin."
        ]
      },
      {
        title: "Modeling Strategy",
        bullets: [
          "Trained multiple models (Random Forest, XGBoost, SVM, KNN, and a deep learning baseline) with randomized hyperparameter search.",
          "Optimized for weighted F1 to handle class imbalance in 'liked' vs. 'not liked' labels.",
          "Ensembled predictions across models to reduce single-model brittleness."
        ]
      },
      {
        title: "Recommendation Workflow",
        bullets: [
          "Generated a ranked Top 50 list for boys and girls and exported results for easy review.",
          "Designed the loop so new feedback becomes new training data (continual improvement).",
          "Kept the system explainable by surfacing which features correlated with higher predicted preference."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Collect more labels and add calibration so probabilities map to real acceptance rates.",
          "Use name embeddings (character-level or phonetic) instead of hand-crafted features alone.",
          "Add diversity constraints so recommendations aren’t overly similar to each other."
        ]
      }
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
    role: [
      "Built the dataset and Tableau dashboard end-to-end (data shaping, KPIs, and forecasting)."
    ],
    problem : "I needed a fast way to track earnings drivers, compare delivery zones, and forecast performance.",
    actions : [
      "Reshaped 12,000 rows for Tableau, built map, histogram & 12-month forecast with date/zone filters.",
      "Enabled live updates on current performance and the ability to anticipate future performance."
    ],
    results : [
      "Able to review potential deliveries and choose which to take.",
      "Improved tip revenue per delivery by more than 10%."
    ],
    caseStudy : [
      {
        title: "Data Shaping for Tableau",
        lead: "I built a Tableau-ready dataset that supports both exploration and operational decisions (where to deliver, when to work, and what to expect).",
        bullets: [
          "Reshaped ~12,000 rows into a tidy format with consistent date fields, zones/cities, and housing types.",
          "Defined core KPIs (tip $, tip %, delivery time) and ensured they work across filters and aggregations.",
          "Created a dashboard layout that fits 'one glance' decisions during a shift."
        ]
      },
      {
        title: "Dashboard UX",
        bullets: [
          "Maps to compare zones by average tip, tip %, and delivery time.",
          "Histograms to understand the distribution of tips and delivery times (not just averages).",
          "Breakdowns by housing type and city to explain why zones perform differently."
        ]
      },
      {
        title: "Forecasting",
        bullets: [
          "Added a 12-month tip forecast to anticipate seasonal changes and plan shift strategies.",
          "Used filtering (date/zone) so forecasts and comparisons stay apples-to-apples.",
          "Designed the view to help decide between 'high tip per delivery' vs. 'high tip per hour' tradeoffs."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add distance/drive-time estimates to better model tips per hour in real conditions.",
          "Track acceptance rates and deadhead time to capture full earnings efficiency.",
          "Publish a mobile-first layout for on-shift use."
        ]
      }
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
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/nonogram-demo.html", label: "Live Demo" },
      { icon: "img/icons/pdf-icon.png",     url: "documents/Project_4.pdf",                  label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png", url: "documents/Project_4.ipynb",                label: "Notebook"}
    ],
    embed: {
      type: "iframe",
      url: "nonogram-demo.html"
    },
    problem : "I wanted to create a machine learning model to automatically solve Nonogram puzzles for me.",
    actions : [
      "Built a hybrid CNN + Transformer policy network and trained it on more than 25 million 5×5 puzzles (52,000 episodes × 512-board batches).",
      "Shaped the reward signal around unique guesses, row/column completions, and full-board solves to speed up exploration."
    ],
    results : [
      "Reached 94% accuracy on unseen 5×5 boards.",
    ],
    caseStudy : [
      {
        title: "Environment Design",
        lead: "I built a reinforcement learning environment for Nonograms, including puzzle generation, clue computation, and reward shaping.",
        bullets: [
          "Generated large batches of unique 5×5 puzzles and computed row/column clues automatically.",
          "Represented state as (board + clue embeddings) so the agent can condition actions on constraints.",
          "Designed rewards to encourage progress (row/column completions) and discourage repeated guesses."
        ]
      },
      {
        title: "Policy Network",
        bullets: [
          "Combined a CNN (board perception) with a Transformer (clue encoding) to capture both spatial and sequential structure.",
          "Output an action distribution over grid cells so the agent learns a strategy, not a fixed solver rule set.",
          "Kept the architecture small enough to train over millions of puzzles efficiently."
        ]
      },
      {
        title: "Training",
        bullets: [
          "Trained with policy gradients at scale (25M+ puzzles; large batched episodes) to learn general solving behavior.",
          "Used reward shaping around unique guesses, row/column completions, and full-board solves to speed exploration.",
          "Tracked learning curves and saved checkpoints to compare policy improvements over time."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Scale beyond 5×5 with curriculum learning (5×5 → 10×10) and stronger value baselines.",
          "Add search (e.g., MCTS) on top of the learned policy for harder puzzles.",
          "Evaluate on benchmark Nonogram sets to compare against classical solvers."
        ]
      }
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
    caseStudy: [
      {
        title: "Goals",
        lead: "I built this as a fast, mobile-first portfolio that showcases projects clearly and stays easy to maintain as the project list grows.",
        bullets: [
          "Performance: keep page weight low and avoid blocking resources.",
          "Accessibility: semantic landmarks, keyboard-friendly components, and predictable navigation.",
          "Maintainability: centralized project data + build scripts to generate deployable pages."
        ]
      },
      {
        title: "Architecture",
        bullets: [
          "Static HTML pages with a shared JS layer for navigation, lazy-loading, and UI behaviors.",
          "Portfolio projects defined in one data file and rendered dynamically (modals + filtering).",
          "Progressive enhancement: the portfolio page works without JS via pre-rendered cards."
        ]
      },
      {
        title: "SEO and Sharing",
        bullets: [
          "Generated `/portfolio/<id>` pages for each project (canonical URLs, Open Graph metadata, JSON-LD).",
          "Kept `sitemap.xml` in sync via a build step so projects are discoverable by search engines.",
          "Configured Vercel rewrites so clean URLs map to generated pages under `pages/`."
        ]
      },
      {
        title: "Build and Deployment",
        bullets: [
          "CSS is bundled into a hashed file for caching while still allowing quick invalidation.",
          "Build scripts generate project pages and copy deployable artifacts into `public/`.",
          "Lightweight tests enforce markup contracts and ensure project pages exist for every project ID."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add richer case studies (diagrams, metrics, and lessons learned) for the top projects.",
          "Reduce third-party dependencies in demos and harden CSP as the site grows.",
          "Automate screenshots/preview generation so project cards stay consistent."
        ]
      }
    ],
  }
];

// IDs (in order) of projects shown in the top carousel
window.FEATURED_IDS = [
  "shapeClassifier",
  "retailStore",
  "sheetMusicUpscale",
  "digitGenerator",
  "nonogram"
];
