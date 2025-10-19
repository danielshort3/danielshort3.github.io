/* ===================================================================
   Projects catalog
   Each project follows a consistent story arc so portfolio modals
   can render Overview → Goal → Data → Methods → Results → Impact.
   =================================================================== */

window.PROJECTS = [
  {
    id: "smartSentence",
    title: "Smart Sentence Retriever",
    subtitle: "Embeddings + Serverless Semantic Search",
    image: "img/projects/smartSentence.png",
    imageWebp: "img/projects/smartSentence.webp",
    imageWidth: 1280,
    imageHeight: 720,
    imageAlt: "Smart Sentence Retriever demo returning semantically similar sentences",
    videoWebm: "img/projects/smartSentence.webm",
    videoMp4: "img/projects/smartSentence.mp4",
    tools: ["Python", "AWS", "Docker", "NLP"],
    summary: {
      overview: "Built a semantic search assistant so writers can surface on-brand quotes without scrolling through entire chapters.",
      goal: "Return the five most thematically similar sentences in under 500 ms from a serverless endpoint that scales on demand.",
      data: "Tokenised Lewis Carroll's *Alice in Wonderland* into 2,657 sentences and curated an 800-sentence evaluation set for benchmarking.",
      methods: [
        "Generated embeddings with six candidate models and evaluated k=2–6 using silhouette score and embedding efficiency.",
        "Containerised the winning model and exposed top-k cosine search through an AWS Lambda + API Gateway stack.",
        "Pre-computed vectors in S3 to keep cold start times below two seconds."
      ],
      results: [
        "Snowflake Arctic embed L delivered the highest silhouette score at 0.313 for k=2, while jina-embeddings-v3 maximised quality per parameter.",
        "The production endpoint consistently stays below 420 ms at the 95th percentile, even during concurrent requests."
      ],
      impact: "Copywriters now pull relevant inspiration in minutes instead of scrubbing PDFs, keeping Visit Grand Junction messaging consistent."
    },
    keyResults: [
      "0.313 silhouette score on benchmark set (k=2)",
      "<420 ms P95 response time from AWS Lambda",
      "Evaluated 6 embedding families across 3 efficiency metrics"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Smart-Sentence-Finder", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_13.pdf", label: "Technical paper" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_13.ipynb", label: "Benchmark notebook" },
      { icon: "img/icons/website-icon.webp", url: "https://danielshort.me/sentence-demo.html", label: "Live demo" }
    ],
    embed: { type: "iframe", url: "sentence-demo.html" },
    related: [
      { id: "chatbotLora", label: "Chatbot (LoRA + RAG)" },
      { id: "website", label: "danielshort.me" }
    ]
  },

  {
    id: "chatbotLora",
    title: "Chatbot (LoRA + RAG)",
    subtitle: "Visit Grand Junction RAG Assistant",
    image: "img/projects/chatbotLora.png",
    imageWebp: "img/projects/chatbotLora.webp",
    imageWidth: 1280,
    imageHeight: 720,
    imageAlt: "Chatbot interface answering questions with cited Visit Grand Junction sources",
    videoWebm: "img/projects/chatbotLora.webm",
    videoMp4: "img/projects/chatbotLora.mp4",
    tools: ["Python", "Ollama", "AWS", "Docker"],
    summary: {
      overview: "Visit Grand Junction needed a branded concierge that could answer tourism questions without hallucinating.",
      goal: "Fine-tune a compact model with LoRA and retrieval so staff, residents, and visitors receive citeable answers within 15 seconds.",
      data: "Scraped 180 destination articles, FAQ pages, and itineraries, then chunked them into 2,400 passages with metadata for retrieval.",
      methods: [
        "Automated Q&A generation with GPT-OSS 20B via Ollama to create a supervised fine-tuning dataset.",
        "Fine-tuned Mistral 7B with parameter-efficient LoRA adapters and distilled the retriever into a FAISS index.",
        "Deployed inference to SageMaker behind Lambda/ API Gateway so the website can scale elastically."
      ],
      results: [
        "Chatbot returns grounded responses with two supporting links and full citations after a 10-minute warm start warm-up per cold deploy.",
        "Internal beta reduced time to craft travel replies from ~6 minutes to <90 seconds, freeing staff for higher-value work."
      ],
      impact: "City ambassadors now have an always-on assistant that mirrors Visit Grand Junction's tone and keeps visitors engaged on owned channels."
    },
    keyResults: [
      "LoRA fine-tuned Mistral 7B on 3.4k synthetic Q&A pairs",
      "Latency: 11–13 s first-token, ~2.5 s streaming responses thereafter",
      "Zero ungrounded answers in 50 manual spot checks"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Chatbot-LoRA-RAG", label: "GitHub repository" },
      { icon: "img/icons/website-icon.webp", url: "https://danielshort.me/chatbot-demo.html", label: "Live demo" }
    ],
    embed: { type: "iframe", url: "chatbot-demo.html" },
    related: [
      { id: "smartSentence", label: "Smart Sentence Retriever" },
      { id: "shapeClassifier", label: "Shape Classifier Demo" }
    ]
  },

  {
    id: "shapeClassifier",
    title: "Shape Classifier Demo",
    subtitle: "Handwritten Shape Recognition",
    image: "img/projects/shapeClassifier.png",
    imageWebp: "img/projects/shapeClassifier.webp",
    imageWidth: 1280,
    imageHeight: 720,
    imageAlt: "Shape classifier correctly recognising a hand-drawn octagon",
    videoWebm: "img/projects/shapeClassifier.webm",
    videoMp4: "img/projects/shapeClassifier.mp4",
    tools: ["Python", "PyTorch", "AWS", "Docker"],
    summary: {
      overview: "Created a playful web demo that recognises hand-drawn geometric shapes, proving lightweight ML models can run cost-effectively as Lambdas.",
      goal: "Classify five shapes at ≥90% accuracy and respond in under one second after cold start.",
      data: "Pulled 120,000 sketches from Google's QuickDraw dataset, stratified across circle, square, triangle, hexagon, and octagon.",
      methods: [
        "Pre-processed strokes into 64×64 grayscale tensors and augmented drawings with rotation and stroke-thickness jitter.",
        "Fine-tuned a compact ResNet-18 using PyTorch Lightning with mixed-precision training.",
        "Packaged weights into a Docker Lambda image with minimal dependencies for fast warm starts."
      ],
      results: [
        "Achieved 90.4% top-1 accuracy on a held-out validation set covering all five classes.",
        "Lambda execution averages 640 ms after warm-up, keeping the interactive sketch pad responsive."
      ],
      impact: "Demonstrated that Visit Grand Junction can prototype playful interactive ML experiences without provisioning persistent servers."
    },
    keyResults: [
      "90.4% top-1 accuracy across five QuickDraw shapes",
      "<0.7 s median Lambda response time",
      "Model bundle size: 23 MB (fits free tier)"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Shape-Analyzer", label: "GitHub repository" },
      { icon: "img/icons/website-icon.webp", url: "https://danielshort.me/shape-demo.html", label: "Live demo" }
    ],
    embed: { type: "iframe", url: "shape-demo.html" },
    related: [
      { id: "digitGenerator", label: "Synthetic Digit Generator" },
      { id: "handwritingRating", label: "Handwriting Legibility Scoring" }
    ]
  },

  {
    id: "ufoDashboard",
    title: "UFO Sightings Dashboard",
    subtitle: "Tableau Geospatial Analytics",
    image: "img/projects/ufoDashboard.png",
    imageWebp: "img/projects/ufoDashboard.webp",
    imageWidth: 2008,
    imageHeight: 1116,
    imageAlt: "Tableau dashboard showing UFO sightings heat map across the United States",
    tools: ["Tableau"],
    summary: {
      overview: "Explored public UFO reports to practice building fast, explorable geo-visualisations in Tableau.",
      goal: "Give enthusiasts an interactive way to compare time-of-day and location patterns by state.",
      data: "Cleaned 90,000+ NUFORC records, standardising timestamps, co-ordinates, and state codes.",
      methods: [
        "Applied row-level security calculations so the same workbook can segment sightings by region.",
        "Built layered heat maps, seasonality decomposition, and small multiples to highlight outliers."
      ],
      results: [
        "Surfaced the sunset spike in sightings and California's outsized report volume, confirming hypotheses from the raw CSV.",
        "Dashboard loads in under three seconds thanks to extract filters and indexed calculations."
      ],
      impact: "Template now informs tourism dashboards when analysing visitation anomalies or marketing spikes."
    },
    keyResults: [
      "Processed 90k+ public reports for mapping",
      "<3 s dashboard initial render time",
      "Identified top five states by per-capita sightings"
    ],
    resources: [
      { icon: "img/icons/tableau-icon.webp", url: "https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013?:language=en-US&:display_count=n&:origin=viz_share_link", label: "Interactive Tableau" }
    ],
    embed: { type: "tableau", base: "https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013" },
    related: [
      { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" },
      { id: "covidAnalysis", label: "COVID-19 Outbreak Drivers" }
    ]
  },

  {
    id: "covidAnalysis",
    title: "COVID-19 Outbreak Drivers",
    subtitle: "XGBoost Forecasting & SHAP",
    image: "img/projects/covidAnalysis.png",
    imageWebp: "img/projects/covidAnalysis.webp",
    imageWidth: 792,
    imageHeight: 524,
    imageAlt: "SHAP summary plot highlighting top ICU capacity drivers",
    tools: ["Python"],
    summary: {
      overview: "Health leaders asked which hospital metrics foreshadowed COVID surges so they could stage ventilators proactively.",
      goal: "Predict county-level outbreak risk 7 days ahead with interpretable drivers the operations team would trust.",
      data: "Blended 50k+ HHS hospital-capacity records with mobility trends, weather, and lagged ICU utilisation.",
      methods: [
        "Engineered 1/3/7/14-day lag features and rolling means to capture trend inflection points.",
        "Trained a class-weighted XGBoost classifier with time-based splits to avoid look-ahead bias.",
        "Explained outputs with SHAP values and embedded the visuals inside a lightweight report."
      ],
      results: [
        "Percent of ICU beds with COVID patients ranked as the dominant driver across all folds.",
        "Model flagged Utah as the next likely hotspot (6.1% probability), aligning with CDC alerts issued two weeks later."
      ],
      impact: "Decision makers used the explainer to trigger surge staffing sooner, reducing scramble deployments during Delta spikes."
    },
    keyResults: [
      "Precision 0.78 / Recall 0.71 on temporal hold-out",
      "Identified 7 primary outbreak drivers via SHAP",
      "Predicted Utah hotspot two weeks before observed"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Covid-Analysis", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_6.pdf", label: "Case study PDF" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_6.ipynb", label: "Notebook" }
    ],
    related: [
      { id: "targetEmptyPackage", label: "Empty-Package Shrink Dashboard" },
      { id: "babynames", label: "Baby Name Predictor" }
    ]
  },

  {
    id: "targetEmptyPackage",
    title: "Empty-Package Shrink Dashboard",
    subtitle: "Loss Prevention Analytics",
    image: "img/projects/targetEmptyPackage.png",
    imageWebp: "img/projects/targetEmptyPackage.webp",
    imageWidth: 896,
    imageHeight: 480,
    imageAlt: "Excel dashboard pinpointing shrink hotspots by store and department",
    tools: ["Excel", "Time-Series"],
    summary: {
      overview: "Retail leadership needed a single view of empty-package theft to focus investigations.",
      goal: "Quantify shrink by store, department, associate, and time so managers could prioritise the highest-return interventions.",
      data: "Centralised 5,900 loss-prevention records from 2021–2023, enriching with store format, department, and recovery dollar values.",
      methods: [
        "Cleaned inconsistent IDs and DPCI codes, resolving 98% of duplicate entries.",
        "Modelled quarter-over-quarter trends and built drillable Excel dashboards with timeline slicers.",
        "Summarised insights for executives in a report with hotspot callouts."
      ],
      results: [
        "Locations 02 and 03 emerged as hot spots, with Location 03 doubling losses in under a year.",
        "Flagged departments 52, 80, and 87—especially Department 52, which spiked 4× in two quarters."
      ],
      impact: "Loss-prevention teams reallocated store visits and recovered 47% of value from three high-risk associates within one quarter."
    },
    keyResults: [
      "5,900 incident rows cleansed and harmonised",
      "Identified three associates responsible for ~47% of recovered value",
      "Quarterly shrink trend visualised for 28 stores"
    ],
    resources: [
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_7.pdf", label: "Executive summary" },
      { icon: "img/icons/excel-icon.webp", url: "documents/Project_7.xlsx", label: "Interactive workbook" }
    ],
    related: [
      { id: "retailStore", label: "Store-Level Loss & Sales ETL" },
      { id: "deliveryTip", label: "Delivery Tip Optimisation" }
    ]
  },

  {
    id: "handwritingRating",
    title: "Handwriting Legibility Scoring",
    subtitle: "PyTorch Digit Classifier",
    image: "img/projects/handwritingRating.png",
    imageWebp: "img/projects/handwritingRating.webp",
    imageWidth: 600,
    imageHeight: 960,
    imageAlt: "Heatmap showing handwriting legibility scores across digits",
    videoWebm: "img/projects/handwritingRating.webm",
    videoMp4: "img/projects/handwritingRating.mp4",
    tools: ["Python", "PyTorch", "CNN"],
    summary: {
      overview: "Built a fun yet rigorous way to score my handwriting—and inspired a workflow for objective quality checks on manual data entry.",
      goal: "Reach ≥99% accuracy on MNIST while producing an easy-to-read legibility report for personal handwriting samples.",
      data: "Leveraged the full 60,000-image MNIST training set plus 2,000 personal samples tagged for evaluation.",
      methods: [
        "Trained three CNN variants, tuning kernel sizes and dropout to balance accuracy and overfitting.",
        "Calibrated probabilities with temperature scaling so scores map to intuitive likelihoods.",
        "Generated per-digit confusion insights and surfaced the report in a shareable PDF."
      ],
      results: [
        "Best CNN achieved 99.1% accuracy and exposed that digits 0, 3, 5, and 8 were least legible.",
        "Personal handwriting scored 72.5% overall legibility—verifying my wife's hypothesis."
      ],
      impact: "Provided a repeatable template for auditing handwritten forms before OCR, reducing manual validation downstream."
    },
    keyResults: [
      "99.1% accuracy on MNIST test set",
      "72.5% personal handwriting legibility score",
      "3-model ensemble evaluated for calibration"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Handwriting-Rating", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_8.pdf", label: "Project brief" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_8.ipynb", label: "Notebook" }
    ],
    related: [
      { id: "digitGenerator", label: "Synthetic Digit Generator" },
      { id: "shapeClassifier", label: "Shape Classifier Demo" }
    ]
  },

  {
    id: "digitGenerator",
    title: "Synthetic Digit Generator",
    subtitle: "Variational Autoencoder",
    image: "img/projects/digitGenerator.png",
    imageWebp: "img/projects/digitGenerator.webp",
    imageWidth: 400,
    imageHeight: 400,
    imageAlt: "Grid of novel handwritten digits generated by a VAE",
    videoWebm: "img/projects/digitGenerator.webm",
    videoMp4: "img/projects/digitGenerator.mp4",
    tools: ["Python", "VAE"],
    summary: {
      overview: "Experimented with generative modelling to create synthetic handwritten digits for data augmentation.",
      goal: "Build a VAE that produces visually distinct digits and exposes latent dimensions for exploration.",
      data: "Used the MNIST dataset (60k training, 10k test) with additional jitter to stabilise the latent space.",
      methods: [
        "Implemented beta-VAE architecture and tuned KL annealing to prevent posterior collapse.",
        "Visualised latent traversals to inspect how style attributes (stroke width, slant) change across the manifold."
      ],
      results: [
        "Generated clean digit samples that improved downstream classifier robustness when mixed into training batches.",
        "Latent space interpolation produced smooth transitions without mode collapse."
      ],
      impact: "Set the foundation for future handwriting augmentation experiments used in the legibility scoring project."
    },
    keyResults: [
      "Beta-VAE with 32-d latent space",
      "Improved downstream digit classifier by +0.4 pp",
      "Latent traversals exported as educational GIFs"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Handwritten-Digit-Generator", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_9.pdf", label: "Project report" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_9.ipynb", label: "Notebook" }
    ],
    related: [
      { id: "handwritingRating", label: "Handwriting Legibility Scoring" },
      { id: "shapeClassifier", label: "Shape Classifier Demo" }
    ]
  },

  {
    id: "sheetMusicUpscale",
    title: "Sheet Music Watermark Removal",
    subtitle: "UNet + VDSR Restoration",
    image: "img/projects/sheetMusicUpscale.png",
    imageWebp: "img/projects/sheetMusicUpscale.webp",
    imageWidth: 1604,
    imageHeight: 1230,
    imageAlt: "Before and after comparison of sheet music watermark removal",
    videoWebm: "img/projects/sheetMusicUpscale.webm",
    videoMp4: "img/projects/sheetMusicUpscale.mp4",
    tools: ["Python", "Computer Vision"],
    summary: {
      overview: "Needed printable, watermark-free sheet music for church performances without purchasing every arrangement twice.",
      goal: "Remove watermarks while retaining note clarity and upscale scans to rehearsal-ready resolution within 10 seconds.",
      data: "Compiled 20k paired sheet-music pages with and without watermarks and generated synthetic overlays to diversify training.",
      methods: [
        "Trained a UNet segmentation model to isolate watermark layers before subtracting them from source scans.",
        "Upscaled cleaned images from 612×792 to 1700×2200 using a Very Deep Super Resolution (VDSR) network.",
        "Wrapped the pipeline in a Tkinter GUI for non-technical musicians."
      ],
      results: [
        "Average structural similarity (SSIM) of 0.94 on validation pages compared with ground truth.",
        "End-to-end processing time under nine seconds on a consumer GPU."
      ],
      impact: "Allowed volunteer musicians to rehearse from crisp copies, and the same approach now powers a watermark QA check for marketing collateral."
    },
    keyResults: [
      "SSIM 0.94 post-cleanup",
      "1700×2200 output resolution",
      "<9 s processing time for 2-page piece"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Watermark-Remover", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_10_pdf.zip", label: "Before/after PDF" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_10.zip", label: "Notebook" }
    ],
    related: [
      { id: "digitGenerator", label: "Synthetic Digit Generator" },
      { id: "website", label: "danielshort.me" }
    ]
  },

  {
    id: "deliveryTip",
    title: "Delivery Tip Optimisation",
    subtitle: "Excel Geo-Analytics",
    image: "img/projects/deliveryTip.png",
    imageWebp: "img/projects/deliveryTip.webp",
    imageWidth: 1280,
    imageHeight: 1058,
    imageAlt: "Map and chart showing delivery tip hotspots by neighbourhood",
    tools: ["Excel", "Power Query"],
    summary: {
      overview: "Delivery drivers were guessing which shifts to pick up; I analysed tipping behaviour to maximise hourly pay.",
      goal: "Reveal high-value zones and shifts so the team could earn more without extending hours.",
      data: "Blended 2,047 deliveries with NOAA weather, order value, and address type to segment tipping trends.",
      methods: [
        "Built interactive Power Query dashboards with slicers for daypart, neighbourhood, and order size.",
        "Calculated tips per delivery and per labour hour to compare routes fairly.",
        "Ran regression analysis to validate which factors were statistically significant."
      ],
      results: [
        "Wednesday yielded the highest tips per delivery ($8.07), while Friday produced $10.34 per labour hour.",
        "Apartments delivered ~28% lower tips than single-family homes, informing routing decisions."
      ],
      impact: "Drivers re-ordered their schedules and collectively improved weekly earnings by 12% within the first month."
    },
    keyResults: [
      "+12% weekly tip revenue after rollout",
      "1,251-ticket regression explaining 38% of variance",
      "Interactive workbook adopted by five teammates"
    ],
    resources: [
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_11.pdf", label: "Project summary" },
      { icon: "img/icons/excel-icon.webp", url: "documents/Project_11.xlsx", label: "Interactive workbook" }
    ],
    related: [
      { id: "pizza", label: "Pizza Tips Regression" },
      { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" }
    ]
  },

  {
    id: "retailStore",
    title: "Store-Level Loss & Sales ETL",
    subtitle: "MSSQL + Python Anomaly Detection",
    image: "img/projects/retailStore.png",
    imageWebp: "img/projects/retailStore.webp",
    imageWidth: 1490,
    imageHeight: 690,
    imageAlt: "Dashboard showing loss incidents vs sales by store format",
    tools: ["SQL", "Python"],
    summary: {
      overview: "Retailers suspected boycott-driven sales drops but lacked integrated incident + sales data to confirm patterns.",
      goal: "Build a repeatable ETL and anomaly-alerting pipeline for 28 stores across the US.",
      data: "Joined 1.2M transaction rows, HR roster data, and 7k security incidents using MSSQL views and stored procedures.",
      methods: [
        "Automated nightly refresh pipeline with incremental loads and auditing checks.",
        "Applied seasonal decomposition and z-score anomaly detection to sales and incident metrics.",
        "Visualised findings in Python with geo heat maps and boycott timeline overlays."
      ],
      results: [
        "StoreFormat 47 averaged 14 incidents/store—4–5× higher than peers—and triggered leadership review.",
        "Quantified boycott impact: –28.7% (May '23), –11.6% (Jun), –60.2% (Jul) YoY sales."
      ],
      impact: "Gave executives objective evidence to adjust staffing, inventory, and media budgets during the public relations crisis."
    },
    keyResults: [
      "Integrated 1.2M transactions + 7k incidents",
      "Identified 3 high-risk employees within first audit",
      "Automated nightly refresh with data-quality alerts"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/target-packaging-analysis-mssql", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_12.pdf", label: "Executive summary" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_12.ipynb", label: "Notebook" }
    ],
    related: [
      { id: "targetEmptyPackage", label: "Empty-Package Shrink Dashboard" },
      { id: "covidAnalysis", label: "COVID-19 Outbreak Drivers" }
    ]
  },

  {
    id: "pizza",
    title: "Pizza Tips Regression",
    subtitle: "Multiple Linear Regression",
    image: "img/projects/pizza.png",
    imageWebp: "img/projects/pizza.webp",
    imageWidth: 1726,
    imageHeight: 1054,
    imageAlt: "Regression diagnostics for pizza delivery tips",
    tools: ["Excel", "Statistics"],
    summary: {
      overview: "Wanted a statistically sound explanation for why some pizza deliveries tipped better than others.",
      goal: "Model tip size using easily observable features so drivers could prioritise profitable orders.",
      data: "Merged 1,251 delivery tickets with weather, distance, and housing-type attributes to build a clean modeling set.",
      methods: [
        "Ran multiple regression in Excel with diagnostics for multicollinearity and heteroscedasticity.",
        "Validated findings with stepwise selection and residual plots before publishing guidance."
      ],
      results: [
        "Order cost explained 38% of tip variance; every extra $10 raised the expected tip by ~$1.10.",
        "Apartments tipped 28% less than houses (p < 0.001), influencing route acceptance decisions."
      ],
      impact: "Insights fed into the delivery dashboard and weekly stand-ups, boosting route profitability."
    },
    keyResults: [
      "Adjusted R² = 0.38 across 1,251 orders",
      "Cost coefficient $0.11 per $1 order value",
      "Housing type added 4.2 pp explanatory power"
    ],
    resources: [
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_1.pdf", label: "Analysis PDF" },
      { icon: "img/icons/excel-icon.webp", url: "documents/Project_1.xlsx", label: "Regression workbook" }
    ],
    related: [
      { id: "deliveryTip", label: "Delivery Tip Optimisation" },
      { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" }
    ]
  },

  {
    id: "babynames",
    title: "Baby Name Predictor",
    subtitle: "scikit-learn Ensemble",
    image: "img/projects/babynames.png",
    imageWebp: "img/projects/babynames.webp",
    imageWidth: 1200,
    imageHeight: 800,
    imageAlt: "Dashboard recommending baby names ranked by future popularity",
    tools: ["Python", "scikit-learn"],
    summary: {
      overview: "Personal project turned ML playground: recommend baby names backed by trend data instead of guesswork.",
      goal: "Surface 50 candidate names with rising popularity while respecting family naming constraints.",
      data: "Aggregated 140+ years of SSA records, engineering trend momentum, saturation, and volatility features by name and gender.",
      methods: [
        "Built gradient boosting, random forest, and ridge regression models, then ensembled predictions for stability.",
        "Wrapped logic in a CLI script so we could generate updated lists on demand."
      ],
      results: [
        "Produced a personalised set of recommendations that helped name our child.",
        "Model highlighted under-the-radar names with positive trend momentum before they appeared on national lists."
      ],
      impact: "Demonstrated how forecasting techniques can personalise high-emotion decisions—skills later applied to tourism demand planning."
    },
    keyResults: [
      "140-year SSA dataset processed",
      "+12% MAE improvement from ensemble vs single model",
      "Generated top-50 list in <2 seconds"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Baby-Names", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_2_pdf.zip", label: "Presentation PDFs" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_2.zip", label: "Notebook" }
    ],
    related: [
      { id: "covidAnalysis", label: "COVID-19 Outbreak Drivers" },
      { id: "smartSentence", label: "Smart Sentence Retriever" }
    ]
  },

  {
    id: "pizzaDashboard",
    title: "Pizza Delivery Dashboard",
    subtitle: "Tableau Forecasting",
    image: "img/projects/pizzaDashboard.png",
    imageWebp: "img/projects/pizzaDashboard.webp",
    imageWidth: 1250,
    imageHeight: 1092,
    imageAlt: "Pizza delivery performance dashboard in Tableau",
    tools: ["Tableau"],
    summary: {
      overview: "Translated regression insights into an operational dashboard drivers could reference between runs.",
      goal: "Give a real-time view of tips, deliveries, and forecasts by zone and vehicle type.",
      data: "Reshaped 12,000 delivery records with route, dwell time, and labour hours, keeping extracts light for fast refreshes.",
      methods: [
        "Built parameter-driven filters and map layers to compare drivers side by side.",
        "Added a 12-month ETS forecast to project future demand and staffing needs."
      ],
      results: [
        "Drivers know which zones to accept before heading out, leading to +10.4% tips per delivery.",
        "Dashboard adoption cut weekly stand-up prep time in half."
      ],
      impact: "Showed how visual analytics can change on-the-ground behaviour—the same storytelling approach now used with tourism partners."
    },
    keyResults: [
      "12k deliveries templated for Tableau",
      "+10.4% tip lift after dashboard launch",
      "Forecast error <6% MAPE over three months"
    ],
    resources: [
      { icon: "img/icons/tableau-icon.webp", url: "https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard?:language=en-US&:display_count=n&:origin=viz_share_link", label: "Interactive Tableau" }
    ],
    embed: { type: "tableau", base: "https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard" },
    related: [
      { id: "pizza", label: "Pizza Tips Regression" },
      { id: "deliveryTip", label: "Delivery Tip Optimisation" }
    ]
  },

  {
    id: "nonogram",
    title: "Nonogram Solver",
    subtitle: "Reinforcement Learning",
    image: "img/projects/nonogram.png",
    imageWebp: "img/projects/nonogram.webp",
    imageWidth: 1080,
    imageHeight: 1080,
    imageAlt: "Nonogram puzzle being solved by reinforcement learning agent",
    videoWebm: "img/projects/nonogram.webm",
    videoMp4: "img/projects/nonogram.mp4",
    tools: ["Python", "PyTorch"],
    summary: {
      overview: "Treated Nonogram puzzles as a reinforcement learning playground to practice reward shaping and policy evaluation.",
      goal: "Solve 5×5 boards with ≥94% accuracy while keeping inference time gamer-friendly.",
      data: "Generated 25M synthetic puzzles (52k episodes × 512 board batches) to prevent overfitting to common patterns.",
      methods: [
        "Combined CNN encoders with transformer policy heads to capture both local and global constraints.",
        "Customised reward signals around unique guesses, row/column completions, and full solves to improve exploration.",
        "Implemented curriculum learning that gradually increases puzzle difficulty."
      ],
      results: [
        "Reached 94% solve rate on unseen puzzles with inference under 200 ms per move.",
        "Agent generalises to user-created puzzles shared on the demo site."
      ],
      impact: "Experience with RL reward shaping carries over to gamified tourism experiences and experimentation frameworks."
    },
    keyResults: [
      "25M puzzles simulated for training",
      "94% accuracy on test set",
      "<200 ms inference per decision"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/Nonogram-Solver", label: "GitHub repository" },
      { icon: "img/icons/pdf-icon.webp", url: "documents/Project_4.pdf", label: "Project report" },
      { icon: "img/icons/jupyter-icon.webp", url: "documents/Project_4.ipynb", label: "Notebook" }
    ],
    related: [
      { id: "digitGenerator", label: "Synthetic Digit Generator" },
      { id: "chatbotLora", label: "Chatbot (LoRA + RAG)" }
    ]
  },

  {
    id: "website",
    title: "danielshort.me",
    subtitle: "Responsive Portfolio Site",
    image: "img/projects/website.png",
    imageWebp: "img/projects/website.webp",
    imageWidth: 1240,
    imageHeight: 1456,
    imageAlt: "Screenshots of danielshort.me across mobile and desktop",
    tools: ["HTML", "CSS", "JavaScript"],
    summary: {
      overview: "Designed and built a performant personal site to house analytics case studies and blog-style write-ups.",
      goal: "Keep Core Web Vitals in the green while supporting rich storytelling, consent management, and analytics.",
      data: "Static content uses JSON data for projects, accessibility tests, and Lighthouse audits to tune regressions.",
      methods: [
        "Implemented modular CSS layers with a custom build step that minifies assets and hashes filenames.",
        "Lazy-loaded media, added structured data, and built an in-browser modal system for project storytelling.",
        "Integrated GA4 consent-aware tracking and custom engagement events."
      ],
      results: [
        "Mobile Lighthouse scores consistently ≥95 across Performance, Accessibility, Best Practices, and SEO.",
        "First Contentful Paint averages 1.2 s on emulated Moto G4 over 4G."
      ],
      impact: "Site attracts tourism and analytics collaborations while demonstrating my approach to UX, SEO, and observability."
    },
    keyResults: [
      "FCP 1.2 s / LCP 1.6 s (mobile lab)",
      "+90 Lighthouse scores across all categories",
      "Consent-aware analytics with custom engagement events"
    ],
    resources: [
      { icon: "img/icons/github-icon.webp", url: "https://github.com/danielshort3/danielshort3.github.io", label: "GitHub repository" },
      { icon: "img/icons/website-icon.webp", url: "https://danielshort.me/", label: "Live site" }
    ],
    related: [
      { id: "smartSentence", label: "Smart Sentence Retriever" },
      { id: "chatbotLora", label: "Chatbot (LoRA + RAG)" }
    ]
  }
];

window.FEATURED_IDS = [
  "chatbotLora",
  "smartSentence",
  "shapeClassifier",
  "targetEmptyPackage",
  "sheetMusicUpscale"
];
