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
    problem: "I wanted a fast way to find sentences that match a question, even when the wording is different.",
    actions: [
      "Cleaned the text of Alice in Wonderland, split it into sentences, and precomputed embeddings.",
      "Tested 6+ embedding models on 800 sentences (k=2–6) and tracked both quality (silhouette score) and model size.",
      "Deployed the best-quality model behind an AWS Lambda API (with CORS) that returns the top-k matches."
    ],
    results: [
      "Best silhouette score: 0.313 with Snowflake Arctic Embed L v2.0 (k=2).",
      "Best score per million parameters: 0.0116 with Jina Embeddings v3 (k=6).",
      "Deployed Arctic Embed L v2.0; the demo calls a Lambda endpoint to rank sentences by meaning."
    ],
    caseStudy: [
      {
        title: "System Design",
        lead: "This uses a fixed corpus (Alice in Wonderland): embed each sentence once, then embed each query and rank by cosine similarity.",
        bullets: [
          "Offline: clean the text, split into sentences, and store embeddings.",
          "Online: embed the query, score against the cached matrix, and return the top-k with scores.",
          "Frontend: a small demo that calls `/health` and `/rank` and shows the top matches."
        ]
      },
      {
        title: "Model Selection",
        bullets: [
          "Compared several embedding models using silhouette score.",
          "Tracked silhouette score per million parameters to keep size and cost in mind.",
          "Kept the setup fixed (same corpus, same sample, same k range) so results are comparable."
        ]
      },
      {
        title: "Serverless Deployment",
        bullets: [
          "Built a Lambda container with CPU PyTorch, FastAPI/Mangum, and the precomputed artifacts.",
          "Used the plain Hugging Face stack (no sentence-transformers) to keep cold starts smaller.",
          "Exposed a Lambda Function URL with CORS so the website can call it from the browser."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Let users upload documents and build embeddings in the background.",
          "Add ANN search (HNSW/FAISS) for bigger corpora and faster top-k.",
          "Add a small labeled set and evaluate with metrics like nDCG@k."
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
      "Built the prototype from crawl to deployment, including the web demo."
    ],
    notes: "Uses public Visit Grand Junction pages. Answers include citations.",
    problem: "Off-the-shelf chatbots didn't sound like Visit Grand Junction, and they rarely pointed people to our pages.",
    actions: [
      "Crawled Visit Grand Junction pages and built a FAISS retrieval index.",
      "Generated a fine-tuning dataset with GPT-OSS 20B (via Ollama).",
      "Fine-tuned Mistral 7B with LoRA on the Q&A set and deployed it to AWS SageMaker.",
      "Added Lambda endpoints so the website can talk to the model."
    ],
    results: [
      "The demo answers with citations, but it needs a warm-up after idle time (about 10 minutes)."
    ],
    caseStudy: [
      {
        title: "System Design",
        lead: "Pipeline: crawl → index → generate training data → LoRA fine-tune → deploy behind an API the site can call.",
        bullets: [
          "Ingestion: crawl pages and store cleaned text chunks.",
          "Retrieval: use FAISS so answers can cite the right source passages.",
          "Fine-tuning: train Mistral 7B with LoRA on auto-generated Q&A pairs."
        ]
      },
      {
        title: "Dataset Generation",
        bullets: [
          "Used GPT-OSS 20B (Ollama) to turn crawled content into Q&A pairs.",
          "Automated crawl → index → dataset → fine-tune so runs are repeatable.",
          "Kept it Docker-friendly so I could run locally and deploy the same artifacts."
        ]
      },
      {
        title: "Serving and Deployment",
        bullets: [
          "Merged the LoRA adapter into a 4-bit model and served it with FastAPI.",
          "Hosted it behind AWS (SageMaker + Lambda) so the browser only calls an API.",
          "Added a status check and a warm-up message to handle cold starts."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add basic eval for retrieval quality and citation accuracy.",
          "Cache common questions and stream tokens to make replies feel faster.",
          "Add stronger guardrails against prompt injection."
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
    problem: "I wanted a model that can tell what shape someone drew.",
    actions: [
      "Used Google's QuickDraw sketches and built train/validation splits.",
      "Trained a small ResNet18 classifier in PyTorch.",
      "Deployed a CPU-only AWS Lambda endpoint so the browser can request predictions."
    ],
    results: [
      "About 90% accuracy on five shapes: circle, triangle, square, hexagon, and octagon.",
      "After warm-up (~10 seconds), predictions return in under a second."
    ],
    caseStudy: [
      {
        title: "Data and Training",
        lead: "I trained a small classifier on Google’s QuickDraw sketches to recognize five basic shapes from a simple black-and-white drawing.",
        bullets: [
          "Built class-balanced train/validation splits from QuickDraw categories.",
          "Used ResNet18 to keep the model small and fast.",
          "Exported plain weights (`model.pt`) so inference stays simple."
        ]
      },
      {
        title: "Inference API",
        bullets: [
          "Implemented an AWS Lambda handler that takes a base64 image and returns a class plus confidence.",
          "Kept the API browser-friendly (CORS and small JSON payloads).",
          "Ran CPU-only inference so it works well in serverless environments."
        ]
      },
      {
        title: "Demo UX",
        bullets: [
          "Canvas drawing UI with a clear submit step and a confidence bar.",
          "Clear status messaging during warm-up and cold starts.",
          "Preprocessing tuned so messy real-world strokes still work."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Calibrate confidence (for example, temperature scaling) so scores are more reliable.",
          "Expand to more shapes and add augmentation for stroke width and incomplete outlines.",
          "Create a small human-drawn validation set to measure real-world performance."
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
    problem : "I wanted to see patterns in UFO sighting reports across the U.S.",
    actions : [
      "Cleaned and standardized a UFO sightings dataset.",
      "Built a Tableau dashboard with maps and time-based charts."
    ],
    results : [
      "Most reports happen just after sunset (earlier in winter, later in summer).",
      "California has the most reports, while central states have fewer."
    ],
    caseStudy : [
      {
        title: "Dashboard Design",
        lead: "I built a one-page Tableau dashboard to explore sightings by place, time, and reported shape.",
        bullets: [
          "Maps: state and city views to compare hotspots at different levels.",
          "Time: a month/hour heatmap to spot seasonal and time-of-day patterns.",
          "Shapes: a breakdown of common shapes and how they change over the year."
        ]
      },
      {
        title: "Data Preparation",
        bullets: [
          "Standardized location fields so points map consistently.",
          "Normalized timestamps so reports compare cleanly by month and hour.",
          "Cleaned shape labels to reduce duplicates from spelling and formatting."
        ]
      },
      {
        title: "Insights and Interpretation",
        bullets: [
          "Reports cluster around dusk and nighttime.",
          "The peak hour shifts with daylight (later in summer, earlier in winter).",
          "California stands out in volume; central regions are lower."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Normalize by population (per-capita rates), especially for smaller areas.",
          "Add filters for year, duration, and report quality.",
          "Surface data-quality issues (missing locations, unclear shapes) in the UI."
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
    videoWebm: "img/projects/covidAnalysis.webm",
    videoMp4: "img/projects/covidAnalysis.mp4",
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
    problem : "I built an early-warning model to flag states at risk of crossing 90% ICU utilization in the next 7 days.",
    actions : [
      "Cleaned and enriched 50k+ rows from the HHS hospital-capacity time series; added rolling stats, trends, and 1/3/7/14-day lag features.",
      "Trained an XGBoost classifier with class-imbalance weighting and a strict time-based train/test split.",
    ],
    results : [
      "Used SHAP to highlight the top drivers and embedded an interactive plot in the report.",
      "Top driver was the share of ICU beds occupied by COVID patients.",
      "In the final snapshot, Utah had the highest predicted risk (6.1%)."
    ],
    caseStudy : [
      {
        title: "Problem Framing",
        lead: "Goal: estimate the probability a state will breach 90% ICU utilization within the next 7 days.",
        bullets: [
          "Target label: max(adult ICU bed utilization) over the next 7 days ≥ 0.90.",
          "Breaches are rare, so I focused on ranking and precision/recall tradeoffs.",
          "Output is a risk score meant to support decisions, not a perfect forecast."
        ]
      },
      {
        title: "Data and Feature Engineering",
        bullets: [
          "Started with the HHS hospital-capacity time series (state × day) and handled missing data with forward-fills and pruning.",
          "Added rolling-window features and lag features (1/3/7/14 days) to capture trend and momentum.",
          "Built ratio features like ICU beds with COVID (%) to normalize across states."
        ]
      },
      {
        title: "Modeling and Evaluation",
        bullets: [
          "Trained an XGBoost classifier with a time-based train/test split and class-imbalance handling.",
          "Measured ranking quality with AUROC (0.606) and PR-AUC (0.060).",
          "Used the model as a risk scorer (probability output) rather than a hard yes/no classifier."
        ]
      },
      {
        title: "Explainability (SHAP)",
        bullets: [
          "Used SHAP to identify which daily metrics raise or lower the chance of an ICU crisis in the next week.",
          "Key takeaway: when a high share of ICU beds are already filled by COVID patients, risk rises quickly.",
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
          "Calibrate probabilities and tune thresholds for a clear precision/recall target.",
          "Add outside signals (vaccination, policy, mobility, variants) to improve early warning.",
          "Monitor drift and retrain when feature patterns shift."
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
    videoWebm: "img/projects/targetEmptyPackage.webm",
    videoMp4: "img/projects/targetEmptyPackage.mp4",
    tools: ["Excel", "Time-Series", "AWS"],
    concepts: ["Automation", "Analytics"],
    resources: [
      { icon: "img/icons/pdf-icon.png",   url: "documents/Project_7.pdf",  label: "PDF"   },
      { icon: "img/icons/excel-icon.png", url: "documents/Project_7.xlsx", label: "Excel" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/target-empty-package-demo.html", label: "Live Demo" }
    ],
    embed: {
      type: "iframe",
      url: "target-empty-package-demo.html"
    },
    role: [
      "Built the Excel dashboard workflow (cleanup, drill-downs, and reporting)."
    ],
    notes: "Employee and location identifiers are anonymized in the write-up.",
    problem : "Empty-package theft was rising fast. Recovered retail value was 5× higher in Q2 2023 than Q1 2021, and leaders needed one view of where it was happening.",
    actions : [
      "Consolidated 5,900+ loss-prevention records (2021–2023).",
      "Cleaned the data and anonymized employee IDs, DPCI codes, dates, locations, and retail values.",
      "Built an interactive Excel dashboard with drill-downs by associate, department, and recovery location.",
      "Summarized the findings in a short report."
    ],
    results : [
      "Two recovery locations (anonymized) stood out as hotspots.",
      "Shrink doubled in under 12 months at one hotspot, with the second doubling in a single quarter.",
      "Three departments (anonymized) drove most recoveries; the top department jumped ~4× in two quarters.",
      "Three associates (anonymized) accounted for ~47% of recovered value."
    ],
    caseStudy : [
      {
        title: "Data Cleanup and Governance",
        lead: "I turned messy loss-prevention logs into a usable dashboard while keeping people and locations anonymous.",
        bullets: [
          "Consolidated 5,900+ records (2021–2023) and standardized dates, locations, departments, and retail values.",
          "Anonymized employee and store identifiers so the write-up is shareable.",
          "Added quarter fields so trends are easy to track over time."
        ]
      },
      {
        title: "Dashboard and KPIs",
        bullets: [
          "Built pivot-based drill-downs by associate, department, and recovery location.",
          "Tracked both retail value and item count to separate volume from severity.",
          "Added trend views to spot fast growth (doubling patterns), not just totals."
        ]
      },
      {
        title: "Key Findings",
        bullets: [
          "Recovered retail value increased five-fold from Q1 2021 to Q2 2023.",
          "Two recovery locations were hotspots; one doubled in <12 months and another doubled in a single quarter.",
          "A small set of associates and departments drove a disproportionate share of recovered value (~47% from three associates)."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Normalize by store traffic or shipments to separate growth from volume changes.",
          "Add control charts or anomaly alerts to flag spikes automatically.",
          "Automate refresh via scheduled exports so the dashboard stays current."
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
    tools: ["Python", "PyTorch", "AWS", "Docker", "CNN"],
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
    problem : "My wife says my handwriting is hard to read. I wanted an objective score.",
    actions : [
      "Built three digit-recognition models (simple baseline → CNN).",
      "Trained on MNIST (60,000 digits) and selected the best model.",
      "Deployed it behind a serverless scoring API for the live demo."
    ],
    results : [
      "The best model reached 99.1% accuracy on MNIST.",
      "My digits scored 72.5% legible; 0, 3, 5, and 8 were the toughest.",
      "My wife was right."
    ],
    caseStudy : [
      {
        title: "Experiment Setup",
        lead: "I used MNIST as a baseline, then scored my own handwriting with the same setup.",
        bullets: [
          "Dataset: MNIST (60,000 train / 10,000 test) for digit recognition.",
          "Goal: compare model types and quantify my own legibility with a consistent metric.",
          "Output: per-digit accuracy and a confusion matrix to see which digits I write most ambiguously."
        ]
      },
      {
        title: "Model Iteration",
        bullets: [
          "Model 1: simple linear classifier as a baseline.",
          "Model 2: small CNN (TinyVGG-style) to capture strokes and curves.",
          "Model 3: deeper CNN (VGG16-style) with the best test accuracy (99.1%)."
        ]
      },
      {
        title: "Custom Handwriting Evaluation",
        bullets: [
          "Built a loader so new images can be preprocessed and scored consistently.",
          "Computed overall legibility (72.5%) and found my hardest digits (0, 3, 5, and 8).",
          "Used the breakdown to make it actionable (which digits to practice)."
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
    tools: ["Python", "VAE", "AWS", "Docker"],
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
    problem : "I wanted to generate new handwritten digits, not just recognize them.",
    actions : [
      "Trained a Variational Autoencoder (VAE) on MNIST (60,000 handwritten digits)."
    ],
    results : [
      "Generated new digits by sampling the learned latent space.",
      "Saved the trained model so generation is a quick inference step."
    ],
    caseStudy : [
      {
        title: "Model Architecture (VAE)",
        lead: "I trained a VAE to learn a compact latent space for handwritten digits and generate new samples from it.",
        bullets: [
          "Encoder: CNN that maps an image to a latent mean and variance.",
          "Reparameterization trick to sample during training while keeping gradients stable.",
          "Decoder: transposed convolutions that reconstruct an image from a latent vector."
        ]
      },
      {
        title: "Training and Sampling",
        bullets: [
          "Trained on MNIST and balanced reconstruction loss vs. KL divergence.",
          "Validated by comparing original vs. reconstructed digits and sampling random latent vectors.",
          "Saved the model so generation doesn't require retraining."
        ]
      },
      {
        title: "Latent Space Analysis",
        bullets: [
          "Used PCA, t-SNE, and UMAP to visualize how digits cluster in latent space.",
          "Applied clustering (HDBSCAN/DBSCAN) to study structure and spot outliers.",
          "Checked reconstruction error to see which digits are hardest to model."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Build a conditional VAE so I can generate a specific digit on demand.",
          "Improve sample sharpness with stronger decoders or better loss functions.",
          "Add quantitative metrics (FID-like proxies) to compare models over time."
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
    problem : "I needed clean, readable sheet music. Most of what I could find was low-res and watermarked.",
    actions : [
      "Trained a UNet model on 20,000+ paired pages for watermark removal.",
      "Upscaled 612×792 scans to 1700×2200 with Very Deep Super-Resolution (VDSR).", 
      "Wrapped it in a simple GUI so I could run the full pipeline."
    ],
    results : [
      "Typical runs produce clean, readable output in under 10 seconds.",
    ],
    caseStudy : [
      {
        title: "End-to-End Pipeline",
        lead: "One workflow: pull pages, clean them up, upscale them, and rebuild the PDF.",
        bullets: [
          "Input: low-resolution, watermarked page images.",
          "Step 1: remove watermark artifacts with a UNet model.",
          "Step 2: upscale with VDSR for print-friendly pages.",
          "Output: cleaned images compiled into a PDF."
        ]
      },
      {
        title: "Modeling Choices",
        bullets: [
          "UNet handles local watermark artifacts while preserving staff lines and note heads.",
          "VDSR improves readability when starting from low-resolution scans.",
          "Saved model weights so the pipeline runs without retraining."
        ]
      },
      {
        title: "GUI and Automation",
        bullets: [
          "Built a PyQt5 GUI to scrape, process, and compile pages without running a notebook.",
          "Used background worker threads so the UI stays responsive during batches.",
          "Designed it for repeat use: new songs in, clean PDFs out."
        ]
      },
      {
        title: "Performance and Reliability",
        bullets: [
          "Kept it fast enough for real use (often under 10 seconds).",
          "Added sensible defaults and a simple flow so it runs without tuning.",
          "Kept steps modular so either model can be swapped or improved later."
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
    problem : "I wanted to know which shifts and neighborhoods lead to better tips.",
    actions : [
      "Built a geospatial heat map and pivot filters from 1,251 deliveries.",
      "Compared tips by daypart, zone, and order size."
    ],
    results : [
      "Wednesday had the highest average tip per delivery ($8.07).",
      "Friday had the best tips per hour ($10.34/hour).",
      "Using the changes, I increased my weekly earnings by about 12%."
    ],
    caseStudy : [
      {
        title: "Dataset and Cleaning",
        lead: "I analyzed delivery tickets and built an Excel workflow that drivers can keep updating.",
        bullets: [
          "Normalized timestamps and derived delivery time (minutes) and tip percentage.",
          "Standardized location fields (city and neighborhood) for mapping and rollups.",
          "Used Power Query so refresh doesn't require manual cleanup."
        ]
      },
      {
        title: "Geo-Analytics Dashboard",
        bullets: [
          "Built a tip heatmap by neighborhood to spot strong zones.",
          "Added pivot filters for housing type, gated communities, city, and order size.",
          "Added weekday and shift-level summaries for scheduling."
        ]
      },
      {
        title: "Key Findings",
        bullets: [
          "Wednesday had the highest average tip per delivery ($8.07); Friday had the best tips per hour ($10.34/hour).",
          "Tips varied a lot by housing type and neighborhood, which helped with zone choices.",
          "Tracked baseline stats (average tip and delivery time) to measure changes over time."
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
    tools: ["SQL", "Python", "AWS"],
    concepts: ["Automation", "Analytics"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/target-packaging-analysis-mssql", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",    url: "documents/Project_12.pdf",                                        label: "PDF"    },
      { icon: "img/icons/jupyter-icon.png",url: "documents/Project_12.ipynb",                                      label: "Notebook"},
      { icon: "img/icons/website-icon.png",url: "https://danielshort.me/retail-loss-sales-demo.html",              label: "Live Demo" }
    ],
    embed: {
      type: "iframe",
      url: "retail-loss-sales-demo.html"
    },
    role: [
      "Led the analysis: SQL modeling/ETL, anomaly detection, and reporting."
    ],
    notes: "Store, state, and employee identifiers are anonymized in the case study.",
    problem : "We didn't have a single view of security incidents, theft hotspots, and boycott-driven sales swings.",
    actions : [
      "Joined incident, sales, and HR tables in SQL and automated KPIs with views and stored procedures.",
      "Built Python dashboards to compare theft vs. sales by format, state, and time.",
      "Used anomaly detection to flag outlier stores and associates."
    ],
    results : [
      "Found a cluster of outlier stores averaging 14 incidents per store (about 4–5× peers).",
      "Flagged high-theft regions (up to about $991 per store per day).",
      "Quantified year-over-year boycott drops: –28.7% (May ’23), –11.6% (Jun ’23), –60.2% (Jul ’23).",
      "Flagged a small set of high-risk associates (anonymized); one outlier averaged $249 per item on two items."
    ],
    caseStudy : [
      {
        title: "Data Modeling and ETL",
        lead: "I built a SQL layer that joins incidents, theft, sales, and HR attributes so the team can slice risk by store, region, and time.",
        bullets: [
          "Joined incident, sales, and employee tables and created reusable KPI views.",
          "Standardized date keys and dimensions (store format, region/state) so filters stay consistent.",
          "Anonymized identifiers so insights can be shared safely."
        ]
      },
      {
        title: "Analysis Workflow",
        bullets: [
          "Ran targeted queries to find high-incident formats, top-theft states, and high-risk associates.",
          "Measured boycott impact with year-over-year comparisons around the timeline.",
          "Used Python charts to highlight hotspots, trends, and outliers."
        ]
      },
      {
        title: "Anomaly Detection",
        bullets: [
          "Flagged outlier stores and associates by comparing against peer baselines, not raw totals.",
          "Separated 'high frequency' (incidents per store) from 'high severity' (value per item).",
          "Kept the output focused on a short list people can investigate."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add automated weekly anomaly alerts and an audit trail for investigations.",
          "Normalize by traffic or shipments to separate volume from risk changes.",
          "Explore causal analysis for boycott drivers and mitigation tests."
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
    videoWebm: "img/projects/pizza.webm",
    videoMp4: "img/projects/pizza.mp4",
    tools: ["Excel", "Statistics", "AWS"],
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
    problem : "Tips varied a lot by neighborhood and housing type. I wanted to see what actually drives them.",
    actions : [
      "Merged 1,251 delivery tickets with NOAA weather, then cleaned the data in Power Query.",
      "Ran a multiple regression in Excel: Tip = f(cost, delivery time, rain, max/min temperature)."
    ],
    results : [
      "Order cost explains ~38% of tip variance (about +$1.10 tip per +$10 bill).",
      "Apartment customers tipped ~28% less than house residents (p < 0.001).",
      "Weather and delivery time didn’t show a meaningful effect on tip size."
    ],
    caseStudy : [
      {
        title: "Data Integration",
        lead: "I merged delivery tickets with NOAA weather to test common ideas about what drives tipping.",
        bullets: [
          "Combined 1,251 deliveries with daily weather features (rain, max/min temperature, wind).",
          "Cleaned the dataset in Power Query and derived tip percentage and delivery time (minutes).",
          "Separated housing types (apartment vs. house) to test neighborhood effects."
        ]
      },
      {
        title: "Exploratory Analysis",
        bullets: [
          "Found a strong positive correlation (0.62) between order cost and tip amount.",
          "Rainfall had only a mild relationship with delivery duration (correlation 0.14).",
          "Order counts more than doubled in summer/early fall (clear seasonality)."
        ]
      },
      {
        title: "Regression and Hypothesis Tests",
        bullets: [
          "Ran a multiple regression: Tip = f(cost, delivery time, rain, max/min temperature).",
          "Result: order cost was the main driver, explaining ~38% of tip variance (≈ +$1.10 tip per +$10 bill).",
          "Validated housing differences with a two-sample t-test: apartment customers tipped ~28% less (p < 0.001)."
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
    videoWebm: "img/projects/babynames.webm",
    videoMp4: "img/projects/babynames.mp4",
    tools: ["Python", "scikit-learn"],
    concepts: ["Machine Learning"],
    resources: [
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/baby-names-demo", label: "Live Demo" },
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/Baby-Names", label: "GitHub" },
      { icon: "img/icons/pdf-icon.png",    url: "documents/Project_2.pdf",                    label: "PDFs"  },
      { icon: "img/icons/jupyter-icon.png",url: "documents/Project_2.ipynb",                  label: "Notebook"}
    ],
    embed: {
      type: "iframe",
      url: "baby-names-demo.html"
    },
    problem : "My wife asked me to suggest baby names. I wanted something that learns her taste instead of guessing.",
    actions : [
      "Aggregated and cleaned 140+ years of SSA records and engineered trend features.",
      "Built a simple 'quiz' script to collect like/dislike labels.",
      "Trained several models and averaged their scores to produce recommendations."
    ],
    results : [
      "Generated personalized top 50 name lists for boys and girls.",
      "Helped us narrow the list when naming our child."
    ],
    caseStudy : [
      {
        title: "Data and Labeling",
        lead: "I combined Social Security Administration name data with preference labels to build a personalized recommender.",
        bullets: [
          "Aggregated 140+ years of SSA records and added recency/trend features so it doesn’t just recommend the most common names.",
          "Focused on Colorado to keep suggestions closer to what my wife actually hears day to day.",
          "Collected labels through quick quizzes so the model learns her taste over time."
        ]
      },
      {
        title: "Feature Engineering",
        bullets: [
          "Name shape features: length, vowel/consonant mix, syllable count, start/end vowel flags, and entropy.",
          "Popularity features: total count, peak year, and recent-count features to capture saturation and momentum.",
          "A rough origin signal via `langdetect` as a lightweight proxy."
        ]
      },
      {
        title: "Modeling Strategy",
        bullets: [
          "Trained multiple models (Random Forest, XGBoost, SVM, KNN, plus a deep learning baseline) with randomized hyperparameter search.",
          "Optimized for weighted F1 to handle class imbalance in 'liked' vs. 'not liked' labels.",
          "Averaged predictions across models to reduce quirks from any single model."
        ]
      },
      {
        title: "Recommendation Workflow",
        bullets: [
          "Generated a ranked Top 50 list for boys and girls and exported results for easy review.",
          "Designed the loop so new feedback becomes new training data.",
          "Kept it explainable by surfacing which features correlated with higher predicted preference."
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
      "Built the dataset and Tableau dashboard (data shaping, KPIs, and forecasting)."
    ],
    problem : "I wanted one dashboard to track earnings, compare delivery zones, and forecast tips.",
    actions : [
      "Reshaped ~12,000 rows for Tableau and built maps, histograms, and a 12-month forecast with date/zone filters.",
      "Set it up so I can compare zones quickly during a shift."
    ],
    results : [
      "Used it to compare zones and plan shifts based on tips and delivery time.",
      "Improved tip revenue per delivery by more than 10%."
    ],
    caseStudy : [
      {
        title: "Data Shaping for Tableau",
        lead: "I built a Tableau-ready dataset so the dashboard stays fast and the filters work cleanly.",
        bullets: [
          "Reshaped ~12,000 rows into a tidy table with consistent dates, zones/cities, and housing types.",
          "Defined KPIs (tip $, tip %, delivery time) that work across filters and aggregations.",
          "Kept the layout simple enough to use mid-shift."
        ]
      },
      {
        title: "Dashboard UX",
        bullets: [
          "Maps to compare zones by average tip, tip %, and delivery time.",
          "Histograms to understand distributions, not just averages.",
          "Breakdowns by housing type and city to explain why zones differ."
        ]
      },
      {
        title: "Forecasting",
        bullets: [
          "Added a 12-month tip forecast to see seasonal changes.",
          "Used date/zone filters so forecasts and comparisons stay consistent.",
          "Made it easy to compare 'tip per delivery' vs. 'tip per hour'."
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
    tools: ["Python", "PyTorch", "AWS", "Docker"],
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
    problem : "I wanted to see if an RL agent could learn to solve Nonogram puzzles.",
    actions : [
      "Trained a hybrid CNN + Transformer policy network on 25M+ generated 5×5 puzzles.",
      "Shaped rewards around unique guesses, row/column completions, and full-board solves to guide exploration."
    ],
    results : [
      "Reached 94% accuracy on unseen 5×5 boards."
    ],
    caseStudy : [
      {
        title: "Environment Design",
        lead: "I built a reinforcement learning environment for Nonograms: puzzle generation, clue computation, and reward shaping.",
        bullets: [
          "Generated large batches of unique 5×5 puzzles and computed row/column clues automatically.",
          "Represented state as board + clue embeddings so the agent can act on constraints.",
          "Rewarded progress (row/column completions) and penalized repeated guesses."
        ]
      },
      {
        title: "Policy Network",
        bullets: [
          "Combined a CNN (board) with a Transformer (clues) to capture both spatial and sequence structure.",
          "Output an action distribution over grid cells so the agent learns a strategy, not a hard-coded solver.",
          "Kept the network small enough to train over millions of puzzles."
        ]
      },
      {
        title: "Training",
        bullets: [
          "Trained with policy gradients at scale (25M+ puzzles; large batched episodes).",
          "Used reward shaping around unique guesses, row/column completions, and full-board solves.",
          "Tracked learning curves and saved checkpoints to compare policy improvements over time."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Scale beyond 5×5 with curriculum learning (5×5 → 10×10) and stronger value baselines.",
          "Add search (like MCTS) on top of the learned policy for harder puzzles.",
          "Evaluate on benchmark Nonogram sets and compare against classical solvers."
        ]
      }
    ]
  },

  {
    id: "minesweeper",
    title: "Minesweeper Solver",
    subtitle: "Reinforcement Learning (RL)",
    image: "img/projects/minesweeper.png",
    imageWidth: 1280,
    imageHeight: 720,
    tools: ["Python", "PyTorch", "AWS", "Docker"],
    concepts: ["Machine Learning", "Automation"],
    resources: [
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/minesweeper-demo.html", label: "Live Demo" },
      { icon: "img/icons/pdf-icon.png", url: "documents/Minesweeper_Reinforcement_Learning_Web_Application.pdf", label: "PDF" }
    ],
    embed: {
      type: "iframe",
      url: "minesweeper-demo.html"
    },
    problem : "I wanted to train an RL agent to play Minesweeper and ship it as a web demo.",
    actions : [
      "Built a Minesweeper environment that generates boards on demand (5×5 to 10×10), with safe first-click logic and 10–20% mines.",
      "Trained 12 variants across DQN, Double DQN, and Dueling DQN with different CNN heads and replay buffers.",
      "Deployed the best model behind an AWS Lambda container endpoint for interactive inference."
    ],
    results : [
      "Best evaluation reached ~72% success on 9x9 boards with 10 mines.",
      "The demo uses the 9x9 / 10-mine model for interactive inference."
    ],
    caseStudy : [
      {
        title: "Environment & Data Generation",
        lead: "The training data is created on the fly, so every episode is a fresh Minesweeper board.",
        bullets: [
          "Generated random boards (10–20% mine density) with a protected 3×3 first click.",
          "Normalized cell values between -0.25 and 1 to keep inputs stable.",
          "Used a debug mode to validate board rules and training inputs."
        ]
      },
      {
        title: "Model Variants",
        bullets: [
          "Evaluated DQN, Double DQN, and Dueling DQN with max-pool, adaptive-pool, and global-average CNN heads.",
          "Compared regular vs. prioritized replay buffers across each architecture (12 combinations).",
          "Found Double DQN with adaptive pooling worked best on larger grids."
        ]
      },
      {
        title: "Training & Curriculum",
        bullets: [
          "Trained each model for 10,000 episodes with 64 parallel games and a periodically updated target network.",
          "Increased grid size after hitting 50% success on a 100-game test set three times in a row.",
          "Reward shaping: safe reveals (+0.3), blind guesses (-0.3), mine hits (-1.0)."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add flagging actions and a second head for mine probability to reduce late-game guesswork.",
          "Pair the RL policy with a lightweight search layer for harder boards.",
          "Add a fixed benchmark suite to compare against classical Minesweeper solvers."
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
    videoWebm: "img/projects/website.webm",
    videoMp4: "img/projects/website.mp4",
    tools: ["HTML", "CSS", "JavaScript"],
    concepts: ["Product", "Visualization"],
    resources: [
      { icon: "img/icons/github-icon.png", url: "https://github.com/danielshort3/danielshort3.github.io", label: "GitHub" },
      { icon: "img/icons/website-icon.png", url: "https://danielshort.me/", label: "Live Site" }
    ],
    problem: "I needed a fast site to show my work, especially on mobile.",
    actions: [
      "Built a static site and rendered portfolio projects from a single data file.",
      "Added Google Analytics 4, structured data, and lazy loading for heavy assets."
    ],
    results: [
      "First Contentful Paint: 1.2s (mobile).",
      "This site helped me share my work during my job search."
    ],
    caseStudy: [
      {
        title: "Goals",
        lead: "I built this site to be fast, easy to use, and easy to update as I add projects.",
        bullets: [
          "Performance: keep pages light and avoid blocking resources.",
          "Accessibility: semantic landmarks, keyboard-friendly components, and predictable navigation.",
          "Maintainability: one project data file + build scripts to generate pages."
        ]
      },
      {
        title: "Architecture",
        bullets: [
          "Static HTML pages with shared JS for navigation, lazy loading, and UI behavior.",
          "Portfolio projects defined in one data file and rendered into modals and filters.",
          "Progressive enhancement: the portfolio page still works without JS via pre-rendered cards."
        ]
      },
      {
        title: "SEO and Sharing",
        bullets: [
          "Generated `/portfolio/<id>` pages for each project (canonical URLs, Open Graph metadata, JSON-LD).",
          "Kept `sitemap.xml` in sync via a build step.",
          "Configured Vercel rewrites so clean URLs map to generated pages under `pages/`."
        ]
      },
      {
        title: "Build and Deployment",
        bullets: [
          "CSS is bundled into a hashed file for caching while still allowing quick invalidation.",
          "Build scripts generate project pages and copy deployable artifacts into `public/`.",
          "Lightweight tests enforce markup contracts and ensure pages exist for every project ID."
        ]
      },
      {
        title: "What I'd Improve",
        bullets: [
          "Add richer case studies (diagrams, metrics, and lessons learned) for the top projects.",
          "Reduce third-party dependencies in demos and tighten CSP over time.",
          "Automate screenshots so project cards stay consistent."
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
