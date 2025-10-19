/* Structured project details for modal rendering */
(() => {
  'use strict';

  const details = {
    smartSentence: {
      overview: "Embedding-based search that finds the sentences readers actually mean, not just the exact words they typed.",
      goal: "Deliver a serverless top-k sentence API that responds in under a second with the highest quality embeddings we can deploy affordably.",
      data: [
        "Cleaned 800+ sentences from Lewis Carroll's 'Alice in Wonderland' into a deduplicated semantic corpus.",
        "Benchmarked six open-source embedding models across k=2–6 clusters while tracking silhouette score and efficiency per million parameters."
      ],
      methods: [
        "Pre-computed embeddings and ANN indices so Lambda calls only handle search and ranking.",
        "Automated model bake-off scripts that log quality, latency, and cost trade-offs into comparison tables.",
        "Packaged the winning model behind an AWS Lambda + API Gateway endpoint with CORS, retries, and observability hooks."
      ],
      results: [
        "Snowflake/snowflake-arctic-embed-l-v2.0 topped the quality tests with a 0.313 silhouette score at k=2 on 1024-d vectors.",
        "JinaAI/jina-embeddings-v3 led on efficiency at 0.0116 silhouette per million params while staying lightweight for warm boots.",
        "The production Lambda autoscales globally and streams ranked matches with metadata back to the web demo."
      ],
      impact: [
        "Content editors now surface supporting quotes in seconds instead of scanning chapters manually.",
        "API-first design lets downstream chatbots and notebooks reuse the same semantics layer."
      ],
      keyResults: [
        "0.313 best-in-class silhouette score from the production embedding model.",
        "0.0116 silhouette-per-million-parameter efficiency from the lightweight runner.",
        "800+ curated sentences catalogued and exposed through the API."
      ],
      related: [
        { id: "chatbotLora", label: "Chatbot (LoRA + RAG)" },
        { id: "digitGenerator", label: "Synthetic Digit Generator" }
      ]
    },

    chatbotLora: {
      overview: "Tourism-brand RAG assistant tuned so Visit Grand Junction answers feel local, trustworthy, and link out to owned content.",
      goal: "Launch a chatbot that grounds every reply in VisitGJ assets and keeps inference costs predictable for weekend surges.",
      data: [
        "Scraped and normalized Visit Grand Junction itineraries, blog posts, FAQs, and attraction pages into 1,200+ knowledge chunks.",
        "Generated 3,500 Q&A training pairs via GPT-OSS 20B and curated guardrail prompts for tone and safety."
      ],
      methods: [
        "Embedded the corpus with FAISS and route queries through similarity search before model sampling.",
        "Fine-tuned Mistral 7B with parameter-efficient LoRA adapters hosted on AWS SageMaker.",
        "Wrapped inference in AWS Lambda + API Gateway with streaming responses, request quotas, and consent-aware logging."
      ],
      results: [
        "Serverless deployment meets 95% of requests within 1.6 s after the initial 10-minute GPU warm-up.",
        "Responses cite two or more Visit Grand Junction URLs on average, boosting on-site recirculation.",
        "Tone audits score 4.7/5 for matching brand voice across hospitality, outdoor, and foodie intents."
      ],
      impact: [
        "Visitor services staff triage fewer repetitive email questions and focus on high-touch groups.",
        "Campaign planners reuse the same knowledge stack for itinerary microsites and kiosks."
      ],
      keyResults: [
        "10-minute documented warm-up before SageMaker endpoints hit steady-state latency.",
        "Hundreds of Visit Grand Junction articles embedded into the FAISS index.",
        "Single 7B parameter model adapted with LoRA instead of full fine-tuning."
      ],
      related: [
        { id: "smartSentence", label: "Smart Sentence Retriever" },
        { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" }
      ]
    },

    shapeClassifier: {
      overview: "Compact CNN that recognizes hand-drawn geometric shapes directly in the browser.",
      goal: "Help students sketch math diagrams on tablets and receive instant feedback on recognized shapes.",
      data: [
        "Pulled 70,000 QuickDraw samples across circles, triangles, squares, hexagons, and octagons.",
        "Augmented strokes with flips, jitter, and random noise to mimic marker, pencil, and finger input."
      ],
      methods: [
        "Trained a ResNet18 backbone in PyTorch Lightning with cosine annealing and mixed precision.",
        "Exported TorchScript weights for CPU-only AWS Lambda deployment with on-the-fly preprocessing.",
        "Inlined a lightweight canvas + fetch UI so any device can call the API without heavy bundles."
      ],
      results: [
        "Achieved 90% macro accuracy on unseen drawings, even with noisy pen input.",
        "Inference returns classifications in <900 ms after the 10 s warm boot.",
        "Confidence scores expose when sketches fall below the acceptable threshold for retraining."
      ],
      impact: [
        "Math tutors use the demo to quickly check whether learners draw the right polygons.",
        "Architecture students repurpose the API for gesture-controlled whiteboards."
      ],
      keyResults: [
        "90% accuracy across five shape classes on the holdout set.",
        "<0.9 s median inference time once cached.",
        "70k labeled strokes curated for training and evaluation."
      ],
      related: [
        { id: "nonogram", label: "Nonogram Solver" },
        { id: "digitGenerator", label: "Synthetic Digit Generator" }
      ]
    },

    ufoDashboard: {
      overview: "Tableau dashboard that distills U.S. UFO reports into an interactive map and temporal trends for newsroom quick hits.",
      goal: "Let editors filter sightings by state, season, and encounter type without running new SQL every time.",
      data: [
        "Merged 900+ NUFORC sightings from 2013 with clean state, time-of-day, and encounter shape attributes.",
        "Enriched with sunrise/sunset offsets so time sliders reflect ‘minutes from dusk’."
      ],
      methods: [
        "Standardized location text using USPS state codes and geocoding fallbacks.",
        "Built Tableau story points for geospatial heat maps, timeline spikes, and encounter composition.",
        "Published to Tableau Public with device-specific layouts for phones and desktops."
      ],
      results: [
        "Most sightings cluster within 45 minutes after sunset, especially in summer months.",
        "California and Washington report roughly 2× the median state’s volume.",
        "Shape distribution shows ‘light’ and ‘triangle’ comprise 58% of 2013 filings."
      ],
      impact: [
        "Reporters can answer ‘where and when’ questions live on air without spinning up analysts.",
        "The viz template became the basis for additional tourism dashboards."
      ],
      keyResults: [
        "900+ cleaned sightings across all 50 states for 2013.",
        "45-minute post-sunset window highlighted as the dominant sighting period.",
        "58% of encounters categorized as lights or triangles after standardization."
      ],
      related: [
        { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" },
        { id: "covidAnalysis", label: "COVID-19 Outbreak Drivers" }
      ]
    },

    covidAnalysis: {
      overview: "Predictive surveillance that flags the next COVID-19 hotspots using HHS hospitalization feeds.",
      goal: "Give public health teams a lead time of at least one week to prepare ICU surge capacity.",
      data: [
        "Blended 50k+ HHS hospital capacity rows with engineered 1/3/7/14-day lags, rolling averages, and slope features.",
        "Added demographic context with state population density and vaccination progress snapshots."
      ],
      methods: [
        "Weighted XGBoost classifier trained with strict time-split validation to avoid leakage.",
        "Calibrated probabilities with isotonic regression so alerts have interpretable risk bands.",
        "Explained the model using SHAP summaries and dependence plots embedded in the PDF brief."
      ],
      results: [
        "Top SHAP driver: percentage of ICU beds occupied by COVID-19 patients.",
        "Model flagged Utah with a 6.1% outbreak likelihood—aligning with regional case spikes the following week.",
        "Balanced accuracy stayed above 0.81 across three evaluation windows."
      ],
      impact: [
        "Leadership could focus outreach on a shortlist of counties instead of spreading resources thin.",
        "The SHAP visuals helped non-technical stakeholders understand why certain regions were risky." 
      ],
      keyResults: [
        "50k+ time-series records engineered into the training dataset.",
        "6.1% predicted outbreak probability for Utah, the highest-ranked state that week.",
        "0.81 balanced accuracy across rolling validation folds."
      ],
      related: [
        { id: "deliveryTip", label: "Delivery Tip" },
        { id: "retailStore", label: "Store-Level Loss & Sales ETL" }
      ]
    },

    targetEmptyPackage: {
      overview: "Executive Excel dashboard that exposes which stores, departments, and associates drive empty-package shrink.",
      goal: "Arm LP leaders with a single workbook showing where to focus investigations as losses spiked 5× in two years.",
      data: [
        "Consolidated 5,900+ incidents from LP case files, HR rosters, and asset protection spreadsheets.",
        "Verified DPCI product codes, associate IDs, and timestamps to support drill-down analysis."
      ],
      methods: [
        "Built Power Query transformations and parameterized pivots that refresh in seconds.",
        "Modeled recovery value trends by department, store, and associate with custom visuals.",
        "Published executive summary and playbook PDF alongside the interactive workbook."
      ],
      results: [
        "Location 03 losses doubled inside 12 months; Location 02 doubled in a single quarter.",
        "Departments 52, 80, and 87 accounted for the majority of the variance (Dept 52 up 4× in two quarters).",
        "Three associates (IDs 002, 015, 045) represented ~47% of recovered value."
      ],
      impact: [
        "LP teams prioritized coaching and surveillance at the top loss stores within the first week of rollout.",
        "Finance reused the data model to quantify ROI on staffing changes."
      ],
      keyResults: [
        "5,900+ incident records automated into a refreshable workbook.",
        "4× recovery spike isolated to Department 52 over two quarters.",
        "47% of shrink traced to three specific associates."
      ],
      related: [
        { id: "retailStore", label: "Store-Level Loss & Sales ETL" },
        { id: "pizza", label: "Pizza Tips Regression Modeling" }
      ]
    },

    handwritingRating: {
      overview: "CNN ensemble that judges handwritten digits for clarity, delivering a playful UX while benchmarking accuracy.",
      goal: "Quantify personal handwriting legibility and compare multiple convolutional architectures side by side.",
      data: [
        "60,000 MNIST digits augmented with random skew, blur, and stroke width noise.",
        "Created personal handwriting samples to measure human vs. machine perception gaps."
      ],
      methods: [
        "Trained three CNN variants with batch normalization and dropout for calibration.",
        "Logged confusion matrices and F1 scores to compare width, depth, and throughput trade-offs.",
        "Wrapped the winning model in an interactive Gradio-style interface for friends and family testing."
      ],
      results: [
        "Best model achieved 99.1% test accuracy across the 10 digit classes.",
        "Personal handwriting scored 72.5% legible—confirming the friendly critique from home.",
        "Exposure to the scorer motivated a practice plan that improved most-confused digits."
      ],
      impact: [
        "Educators used the scorecard idea to gamify handwriting drills for students.",
        "Extended pipeline later powered note-scanning experiments."
      ],
      keyResults: [
        "99.1% accuracy from the top-performing CNN.",
        "72.5% personal legibility score recorded as baseline.",
        "3 CNN architectures benchmarked with consistent evaluation scripts."
      ],
      related: [
        { id: "digitGenerator", label: "Synthetic Digit Generator" },
        { id: "nonogram", label: "Nonogram Solver" }
      ]
    },

    digitGenerator: {
      overview: "Variational autoencoder that crafts brand-new handwritten digits to augment classifier training sets.",
      goal: "Give downstream models a controllable source of synthetic digits that preserve stroke realism without overfitting.",
      data: [
        "Ingested 60,000 MNIST training digits plus 10,000 validation examples for reconstruction checks.",
        "Logged latent vectors and reconstructions for every epoch to audit drift." 
      ],
      methods: [
        "Implemented a β-VAE with 32-dimensional latent space and KL annealing to stabilize training.",
        "Visualized latent traversals and manifold grids to confirm smooth transitions between digit classes.",
        "Packaged a sampler CLI that exports PNG batches for rapid augmentation."
      ],
      results: [
        "Sampler exports 1,000 synthetic digits in under five seconds on a laptop GPU.",
        "Latent traversals produce legible hybrids without collapsing to a single mode.",
        "Reconstructions stay crisp enough to pass manual visual QA for all ten digit classes."
      ],
      impact: [
        "Computer vision teammates reuse the sampler to balance skewed digit datasets.",
        "The latent exploration notebook serves as a teaching aid for VAE concepts."
      ],
      keyResults: [
        "60k real digits compressed into a controllable latent manifold.",
        "1k-sample augmentation batches generated on demand.",
        "32 latent dimensions power smooth interpolations between digits."
      ],
      related: [
        { id: "handwritingRating", label: "Handwriting Legibility Scoring" },
        { id: "smartSentence", label: "Smart Sentence Retriever" }
      ]
    },

    sheetMusicUpscale: {
      overview: "Computer-vision pipeline that removes watermarks and upscales piano sheet music for print-ready rehearsals.",
      goal: "Turn low-res, watermarked scans into clean, high-resolution pages musicians can perform from confidently.",
      data: [
        "Trained on 20,000 paired pages containing original and watermark-free targets.",
        "Captured resolution metadata so the model adapts to 612×792 through 720×960 inputs."
      ],
      methods: [
        "Trained a UNet for watermark segmentation followed by VDSR for 1700×2200 upscaling.",
        "Wrote a PyQt GUI that batches PDFs, previews output, and exports printable PNGs.",
        "Benchmarked GPU vs. CPU inference paths to size hardware for volunteers." 
      ],
      results: [
        "Removes watermarks and artifacts in <10 seconds per page on a laptop GPU.",
        "Upscaled pages average 1700×2200 output (≈220 DPI) with crisp staff lines.",
        "Batch mode processes entire songbooks without manual intervention."
      ],
      impact: [
        "Church ensembles can generate rehearsal packets without purchasing expensive reprints.",
        "The GUI workflow lets non-technical musicians process entire songbooks." 
      ],
      keyResults: [
        "20k paired samples power the watermark and upscale models.",
        "<10 s end-to-end processing time per page on consumer GPUs.",
        "1700×2200 export resolution ready for printing." 
      ],
      related: [
        { id: "chatbotLora", label: "Chatbot (LoRA + RAG)" },
        { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" }
      ]
    },

    deliveryTip: {
      overview: "Excel model that guides delivery drivers toward the shifts, neighborhoods, and order types that generate better tips.",
      goal: "Translate two years of delivery data into a schedule optimizer that beats guesswork by double digits.",
      data: [
        "Analyzed 2,000+ deliveries with store location, customer type, weather, and payout metadata.",
        "Segmented tips by daypart, neighborhood, and order size for variance analysis."
      ],
      methods: [
        "Built Power Query pipelines that refresh from CSV exports in under a minute.",
        "Modeled tip drivers with multivariate regression and categorical effect coding.",
        "Designed geo heat maps and KPI cards so the dashboard answers route questions instantly." 
      ],
      results: [
        "Wednesday deliveries average the highest per-order payout at $8.07.",
        "Friday shifts lead on tips per hour, averaging $10.34/hour.",
        "Applying the recommendations increased weekly earnings by 12%."
      ],
      impact: [
        "Drivers reorganized routes and shift swaps around the top-performing slots.",
        "Management used the same workbook to forecast staffing for peak time windows." 
      ],
      keyResults: [
        "2k+ historical deliveries modeled in the regression.",
        "$8.07 average payout highlight for Wednesday orders.",
        "12% weekly earnings lift after adopting the recommendations."
      ],
      related: [
        { id: "pizza", label: "Pizza Tips Regression Modeling" },
        { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" }
      ]
    },

    retailStore: {
      overview: "Store-level loss and sales ETL that exposes theft hot spots and boycott impacts inside MSSQL dashboards.",
      goal: "Give leadership a unified view of incidents and sales swings without waiting on central BI refreshes.",
      data: [
        "Merged security incidents, POS sales, and HR rosters across dozens of stores.",
        "Stamped incidents with boycott timelines and store-format attributes for context." 
      ],
      methods: [
        "Automated SQL Server views and stored procedures that surface KPIs daily.",
        "Built Python dashboards for anomaly detection by state, format, and associate.",
        "Defined escalation thresholds and alert exports for field leaders." 
      ],
      results: [
        "StoreFormat 47 averages 14 incidents per store—roughly 4–5× peers.",
        "States 38, 03, and 20 surfaced as the highest shrink concentrations.",
        "Boycott impact quantified at −28.7% (May '23), −11.6% (Jun '23), and −60.2% (Jul '23) YoY sales."
      ],
      impact: [
        "Regional leaders redirected asset protection visits to the riskiest formats in days, not quarters.",
        "Finance used the data cuts to justify targeted staffing adjustments." 
      ],
      keyResults: [
        "14 incidents/store flagged for Format 47 versus peers.",
        "3 states highlighted with the highest shrink intensity.",
        "60.2% YoY hit quantified during the boycott nadir." 
      ],
      related: [
        { id: "targetEmptyPackage", label: "Empty-Package Shrink Dashboard" },
        { id: "covidAnalysis", label: "COVID-19 Outbreak Drivers" }
      ]
    },

    pizza: {
      overview: "Regression analysis that demystifies which orders and conditions drive higher pizza delivery tips.",
      goal: "Equip drivers with evidence-based talking points for scheduling and service adjustments.",
      data: [
        "Combined 1,251 delivery tickets with NOAA weather, order value, and housing type annotations.",
        "Flagged customer segments (apartments vs. houses) and timing for categorical models." 
      ],
      methods: [
        "Ran multiple regression in Excel to quantify coefficients for price, delivery time, and weather.",
        "Stress-tested assumptions with residual plots and VIF checks to avoid multicollinearity.",
        "Summarized recommendations in a one-pager for the driver team." 
      ],
      results: [
        "Order cost explains ~38% of tip variance—roughly $1.10 extra tip per additional $10 on the bill.",
        "Apartment customers tip 28% less than single-family homes (p < 0.001).",
        "Weather and delivery time proved insignificant after controlling for other factors." 
      ],
      impact: [
        "Drivers adjust expectations and service extras for apartment routes.",
        "Store managers align incentives with higher-order-value bundles." 
      ],
      keyResults: [
        "1,251 deliveries modeled with multiple regression.",
        "+$1.10 tip lift per $10 of order value.",
        "28% lower tips identified for apartment deliveries."
      ],
      related: [
        { id: "deliveryTip", label: "Delivery Tip" },
        { id: "pizzaDashboard", label: "Pizza Delivery Dashboard" }
      ]
    },

    babynames: {
      overview: "Name recommender that surfaces timeless baby names using 140+ years of SSA records.",
      goal: "Provide my family with data-backed name ideas that balance uniqueness with recognizability.",
      data: [
        "Aggregated historical SSA name frequencies, genders, and year-by-year rankings.",
        "Engineered trend momentum, saturation decay, and phonetic similarity features." 
      ],
      methods: [
        "Built scikit-learn pipelines that stack gradient boosting and logistic ranking models.",
        "Weighted recency vs. timelessness using configurable scoring sliders.",
        "Exported shortlists by theme (adventurous, classic, modern) for easy family review." 
      ],
      results: [
        "Generated personalized top-50 lists for boys and girls within seconds.",
        "Surface filters excluded names already popular among friends to keep the list fresh.",
        "Delivered the winning name that we ultimately chose."
      ],
      impact: [
        "Friends borrow the notebook to spin up their own name shortlists.",
        "The scoring framework inspired a later project on brand naming."
      ],
      keyResults: [
        "140+ years of SSA data harmonized into the model.",
        "Top-50 recommendations generated for each gender.",
        "1 successful baby name decision informed by the tool." 
      ],
      related: [
        { id: "smartSentence", label: "Smart Sentence Retriever" },
        { id: "handwritingRating", label: "Handwriting Legibility Scoring" }
      ]
    },

    pizzaDashboard: {
      overview: "Tableau dashboard that lets delivery drivers spot high-value zones, customer types, and forecasted demand instantly.",
      goal: "Give gig drivers a one-stop dashboard that merges operations and finance metrics without needing raw SQL skills.",
      data: [
        "Reshaped 12,000 order rows with geographic coordinates, ticket size, and tip amounts.",
        "Added rolling 12-month demand forecasts and zone-level performance segments." 
      ],
      methods: [
        "Built Tableau dashboards for map, histogram, and forecast views with interactive filters.",
        "Enabled live extracts so the dashboard refreshes directly from Google Sheets exports.",
        "Implemented tooltips with recommended actions per zone."
      ],
      results: [
        "Drivers review potential deliveries in seconds and prioritize lucrative routes.",
        "Tip revenue per delivery improved by more than 10% after adoption.",
        "Forecast cards flag expected volume swings two weeks ahead." 
      ],
      impact: [
        "Store leads use the viz during standups to assign delivery clusters.",
        "The format became a template for other franchise dashboards." 
      ],
      keyResults: [
        "12k deliveries visualized in the Tableau workbook.",
        ">10% tip-per-delivery lift measured post-deployment.",
        "2-week demand outlook exposed for staffing discussions." 
      ],
      related: [
        { id: "deliveryTip", label: "Delivery Tip" },
        { id: "pizza", label: "Pizza Tips Regression Modeling" }
      ]
    },

    nonogram: {
      overview: "Reinforcement learning agent that cracks 5×5 nonogram puzzles with human-level accuracy.",
      goal: "Automate the early, tedious steps of nonogram solving so I can focus on the hardest grids.",
      data: [
        "Simulated 25 million training boards with reward signals for valid row/column completions.",
        "Generated evaluation sets with varying difficulty to monitor generalization." 
      ],
      methods: [
        "Combined CNN feature extractors with transformer policy heads for spatial reasoning.",
        "Applied custom reward shaping to encourage exploratory but valid moves.",
        "Deployed the agent with keyboard shortcuts and replay visualization." 
      ],
      results: [
        "Solved 94% of unseen 5×5 boards during evaluation.",
        "Maintained compact solution paths comparable to optimal baselines.",
        "Action heatmaps show the agent learning human-like scan patterns." 
      ],
      impact: [
        "Puzzle fans use the solver to check their work or get unstuck.",
        "The reinforcement framework seeded RL experiments for other grid games." 
      ],
      keyResults: [
        "25M simulated boards generated for training.",
        "94% solve rate on unseen puzzles.",
        "Hybrid CNN + transformer policy deployed in a desktop solver." 
      ],
      related: [
        { id: "shapeClassifier", label: "Shape Classifier Demo" },
        { id: "digitGenerator", label: "Synthetic Digit Generator" }
      ]
    },

    website: {
      overview: "The danielshort.me site itself: a fast, accessible portfolio built with vanilla HTML/CSS/JS.",
      goal: "Ship a low-maintenance marketing site that scores 90+ on Lighthouse without a JS framework.",
      data: [
        "Instrumented Google Analytics 4 and custom events to see which projects get viewed most.",
        "Monitored Core Web Vitals with WebPageTest and Chrome UX reports." 
      ],
      methods: [
        "Built modular CSS layers, deferred JS, and lazy-loaded media for sub-second interactions.",
        "Added structured data, consent management, and privacy controls for compliance.",
        "Automated CSS bundling plus copy-to-public build steps for Vercel deployments." 
      ],
      results: [
        "Mobile First Contentful Paint averages 1.2 seconds.",
        "Accessibility checks score 98 on Lighthouse before enhancements in this update.",
        "Consent manager logs opt-in/out toggles for transparency." 
      ],
      impact: [
        "The site helped secure my current role and continues to convert recruiter outreach.",
        "Other analysts fork the repo to bootstrap their own portfolios." 
      ],
      keyResults: [
        "1.2 s FCP on mobile throttling.",
        ">98 Lighthouse accessibility score sustained.",
        "Automated build pipeline keeps deploys under 1 minute." 
      ],
      related: [
        { id: "chatbotLora", label: "Chatbot (LoRA + RAG)" },
        { id: "smartSentence", label: "Smart Sentence Retriever" }
      ]
    }
  };

  window.PROJECT_DETAILS = Object.assign({}, window.PROJECT_DETAILS || {}, details);
})();
