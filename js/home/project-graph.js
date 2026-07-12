(() => {
  'use strict';

  const CATEGORIES = {
    projects: {
      label: 'Projects',
      singular: 'project',
      href: 'portfolio',
      accent: '#155dfc',
      icon: 'folder',
      summary: 'Models, dashboards, and data systems built around practical analysis and usable interfaces.',
      items: [
        {
          id: 'chatbotLora',
          title: 'RAG Chatbot',
          fullTitle: 'RAG Chatbot Fine-Tuned with LoRA',
          href: 'portfolio/chatbotLora',
          snippet: 'Conversational QA over documents using retrieval augmented generation.',
          summary: 'A source-grounded chatbot demo comparing managed RAG responses with a custom fine-tuned path.',
          tags: ['AI / NLP', 'RAG', 'LoRA', 'AWS']
        },
        {
          id: 'retailStore',
          title: 'Store Loss ETL',
          fullTitle: 'Store-Level Loss & Sales ETL',
          href: 'portfolio/retailStore',
          snippet: 'SQL pipeline and anomaly views for comparing loss signals against sales context.',
          summary: 'A SQL ETL and anomaly-detection workflow for comparing store-level loss signals with sales context.',
          tags: ['SQL', 'Analytics', 'Anomaly Detection']
        },
        {
          id: 'digitGenerator',
          title: 'Synthetic Digits',
          fullTitle: 'Synthetic Digit Generator',
          href: 'portfolio/digitGenerator',
          snippet: 'VAE sampling interface for generating handwritten-style digits.',
          summary: 'A variational autoencoder trained on MNIST to generate new handwritten-style digit samples.',
          tags: ['VAE', 'Python', 'Machine Learning']
        },
        {
          id: 'smartSentence',
          title: 'Smart Retriever',
          fullTitle: 'Smart Sentence Retriever',
          href: 'portfolio/smartSentence',
          snippet: 'Embedding-based search for surfacing semantically similar sentences.',
          summary: 'A serverless retrieval demo that searches a fixed corpus by meaning rather than exact wording.',
          tags: ['NLP', 'Embeddings', 'AWS']
        },
        {
          id: 'pizzaDashboard',
          title: 'Pizza Dashboard',
          fullTitle: 'Pizza Delivery Dashboard',
          href: 'portfolio/pizzaDashboard',
          snippet: 'Tableau dashboard for comparing delivery zones, tips, and timing.',
          summary: 'A Tableau dashboard built around delivery-zone planning, tip patterns, and forecastable trends.',
          tags: ['Tableau', 'Forecasting', 'Analytics']
        },
        {
          id: 'ufoDashboard',
          title: 'UFO Dashboard',
          fullTitle: 'UFO Sightings Dashboard',
          href: 'portfolio/ufoDashboard',
          snippet: 'Geospatial Tableau analysis of public sighting reports.',
          summary: 'A geospatial Tableau dashboard for exploring UFO sighting reports by location, time, and shape.',
          tags: ['Tableau', 'Geospatial', 'Visualization']
        },
        {
          id: 'sheetMusicUpscale',
          title: 'Sheet Music Upscale',
          fullTitle: 'Sheet Music Watermark Removal & Upscale',
          href: 'portfolio/sheetMusicUpscale',
          snippet: 'Computer-vision pipeline for cleaner, higher-resolution sheet music.',
          summary: 'A UNet and VDSR image pipeline for removing watermarks and upscaling low-resolution sheet music.',
          tags: ['Computer Vision', 'UNet', 'VDSR']
        },
        {
          id: 'deliveryTip',
          title: 'Delivery Tip',
          fullTitle: 'Delivery Tip',
          href: 'portfolio/deliveryTip',
          snippet: 'Excel geo-analysis for comparing shift and neighborhood tip patterns.',
          summary: 'An Excel and Power Query workflow for finding tip patterns across shifts, areas, and order context.',
          tags: ['Excel', 'Power Query', 'Geo-Analytics']
        },
        {
          id: 'targetEmptyPackage',
          title: 'Empty Package Shrink',
          fullTitle: 'Empty-Package Shrink Dashboard',
          href: 'portfolio/targetEmptyPackage',
          snippet: 'Forecasting and BI workflow for tracking theft trends and hotspots.',
          summary: 'An Excel forecasting and BI dashboard for surfacing shrink trends, hotspots, and acceleration.',
          tags: ['Forecasting', 'BI', 'Excel']
        },
        {
          id: 'shapeClassifier',
          title: 'Shape Classifier',
          fullTitle: 'Shape Classifier Demo',
          href: 'portfolio/shapeClassifier',
          snippet: 'QuickDraw-based model that classifies hand-drawn shapes.',
          summary: 'A PyTorch shape classifier deployed behind a browser drawing demo for supported shape classes.',
          tags: ['PyTorch', 'AWS', 'Computer Vision']
        },
        {
          id: 'handwritingRating',
          title: 'Legibility Scoring',
          fullTitle: 'Handwriting Legibility Scoring',
          href: 'portfolio/handwritingRating',
          snippet: 'CNN digit recognizer used to score handwritten digit legibility.',
          summary: 'A CNN-based handwriting demo that classifies drawn digits and scores how legible they appear.',
          tags: ['CNN', 'PyTorch', 'AWS']
        },
        {
          id: 'covidAnalysis',
          title: 'Outbreak Drivers',
          fullTitle: 'COVID-19 Outbreak Drivers',
          href: 'portfolio/covidAnalysis',
          snippet: 'XGBoost and SHAP workflow for ICU breach-risk analysis.',
          summary: 'A Python model and explanation workflow for identifying drivers behind state-level ICU breach risk.',
          tags: ['XGBoost', 'SHAP', 'Python']
        },
        {
          id: 'babynames',
          title: 'Name Predictor',
          fullTitle: 'Baby Name Predictor',
          href: 'portfolio/babynames',
          snippet: 'Preference model trained from ratings and SSA baby-name records.',
          summary: 'A personal recommendation model that learns name preferences from labels and historical name trends.',
          tags: ['scikit-learn', 'Python', 'Personal ML']
        },
        {
          id: 'pizza',
          title: 'Tips Regression',
          fullTitle: 'Pizza Tips Regression Modeling',
          href: 'portfolio/pizza',
          snippet: 'Regression model for comparing order, weather, and housing effects on tips.',
          summary: 'An Excel regression and analysis workflow built from delivery records, weather, and order context.',
          tags: ['Regression', 'Excel', 'Statistics']
        },
        {
          id: 'nonogram',
          title: 'Nonogram Solver',
          fullTitle: 'Nonogram Solver',
          href: 'portfolio/nonogram',
          snippet: 'Reinforcement-learning agent for generated 5x5 Nonogram puzzles.',
          summary: 'A reinforcement-learning puzzle solver with a live demo that steps through generated Nonogram boards.',
          tags: ['RL', 'PyTorch', 'Puzzle']
        },
        {
          id: 'website',
          title: 'Website',
          fullTitle: 'danielshort.me',
          href: 'portfolio/website',
          snippet: 'Responsive static site for projects, tools, games, and shareable pages.',
          summary: 'The static portfolio system behind this site, with generated pages, hashed assets, and searchable content.',
          tags: ['HTML', 'CSS', 'JavaScript']
        }
      ]
    },
    tools: {
      label: 'Tools',
      singular: 'tool',
      href: 'tools',
      accent: '#0798a6',
      icon: 'wrench',
      summary: 'Small browser utilities for repeated text, link, media, and workflow tasks.',
      items: [
        {
          id: 'text-compare',
          title: 'Text Compare',
          href: 'tools/text-compare',
          snippet: 'Compare drafts with inline insertions, deletions, and replacements.',
          summary: 'A local browser tool for comparing two drafts and reviewing changes without sending text away.',
          tags: ['Local', 'Text', 'Diff']
        },
        {
          id: 'nbsp-cleaner',
          title: 'NBSP Cleaner',
          fullTitle: 'Non-breaking Space Cleaner',
          href: 'tools/nbsp-cleaner',
          snippet: 'Find and replace hard spaces after confirming counts.',
          summary: 'A small cleanup tool that detects non-breaking spaces and replaces them with regular spaces.',
          tags: ['Text', 'Cleanup', 'Local']
        },
        {
          id: 'oxford-comma-checker',
          title: 'Oxford Comma',
          fullTitle: 'Oxford Comma Checker',
          href: 'tools/oxford-comma-checker',
          snippet: 'Scan drafts for possible missing Oxford commas in serial lists.',
          summary: 'A writing helper that flags likely serial-list patterns where an Oxford comma may be missing.',
          tags: ['Writing', 'Text', 'Review']
        },
        {
          id: 'point-of-view-checker',
          title: 'POV Checker',
          fullTitle: 'Point of View Checker',
          href: 'tools/point-of-view-checker',
          snippet: 'Spot first-, second-, and third-person pronoun mixing.',
          summary: 'A browser tool for scanning text and highlighting point-of-view shifts across pronoun groups.',
          tags: ['Writing', 'Pronouns', 'Local']
        },
        {
          id: 'word-frequency',
          title: 'Word Frequency',
          fullTitle: 'Word Frequency Analyzer',
          href: 'tools/word-frequency',
          snippet: 'Strip stopwords and review frequent terms locally in the browser.',
          summary: 'A local text-analysis utility that counts frequent words after optional stopword removal.',
          tags: ['Text', 'Frequency', 'Local']
        },
        {
          id: 'utm-batch-builder',
          title: 'UTM Builder',
          fullTitle: 'UTM Batch Builder',
          href: 'tools/utm-batch-builder',
          snippet: 'Generate normalized campaign URLs in large batches.',
          summary: 'A campaign-link builder with normalization, combination modes, copy helpers, and CSV export.',
          tags: ['Marketing', 'URLs', 'CSV']
        },
        {
          id: 'qr-code-generator',
          title: 'QR Generator',
          fullTitle: 'QR Code Generator',
          href: 'tools/qr-code-generator',
          snippet: 'Create high-resolution QR codes with templates and logo options.',
          summary: 'A QR code generator with templates, logo embedding, customization controls, and exports.',
          tags: ['QR', 'Export', 'Design']
        },
        {
          id: 'image-optimizer',
          title: 'Image Optimizer',
          href: 'tools/image-optimizer',
          snippet: 'Batch resize, compress, and convert images locally.',
          summary: 'A local media utility for resizing, compressing, and converting PNG, JPEG, WebP, and AVIF images.',
          tags: ['Images', 'Local', 'Batch']
        },
        {
          id: 'background-remover',
          title: 'Background Remover',
          href: 'tools/background-remover',
          snippet: 'Remove backgrounds with AI matting and edge refinement.',
          summary: 'An image tool for AI background removal, edge refinement, and transparent or solid exports.',
          tags: ['Images', 'AI Matting', 'Export']
        },
        {
          id: 'screen-recorder',
          title: 'Screen Recorder',
          href: 'tools/screen-recorder',
          snippet: 'Record the screen and download clips in browser-supported formats.',
          summary: 'A browser capture utility for recording screen clips with optional system audio when supported.',
          tags: ['Media', 'Capture', 'Local']
        },
      ]
    },
    games: {
      label: 'Games',
      singular: 'game',
      href: 'games',
      accent: '#f97316',
      icon: 'gamepad',
      summary: 'Browser games and simulations where systems, probability, balance, and feedback loops are the point.',
      items: [
        {
          id: 'project-starfall',
          title: 'Project Starfall',
          href: 'games/project-starfall',
          snippet: 'Side-scrolling RPG prototype with classes, loot, and progression systems.',
          summary: 'Choose a class, fight through side-scrolling maps, collect gear, and test RPG progression systems.',
          tags: ['RPG systems', 'Loot economy', 'Progression']
        },
        {
          id: 'stellar-dogfight',
          title: 'Stellar Dogfight',
          href: 'games/stellar-dogfight',
          snippet: 'Pilot a fighter, duel adaptive AI, and stack upgrades between waves.',
          summary: 'A browser dogfight game focused on adaptive opponents, upgrade loops, and combat tuning.',
          tags: ['Adaptive AI', 'Upgrade loops', 'Combat tuning']
        },
        {
          id: 'roulette',
          title: 'Double-Zero Roulette',
          href: 'games/roulette',
          snippet: 'Run a double-zero table and track recent-spin distributions.',
          summary: 'A roulette table experiment for inspecting bet layouts, state, and probability distributions.',
          tags: ['Probability', 'Distribution tracking', 'Table state']
        },
        {
          id: 'probability-engine',
          title: 'Probability Engine',
          href: 'games/probability-engine',
          snippet: 'Construct slot reels, chain synergies, and test long-run odds.',
          summary: 'A systems sandbox for slot reels, synergies, automation, and long-run probability tuning.',
          tags: ['Simulation', 'Synergies', 'Prestige loops']
        },
        {
          id: 'ocean-wave-simulation',
          title: 'Ocean Wave',
          fullTitle: 'Ocean Wave Simulation',
          href: 'games/ocean-wave-simulation',
          snippet: 'Adjust wave, light, and wind parameters in a realtime canvas sandbox.',
          summary: 'A realtime canvas simulation for tuning wave, lighting, and wind parameters interactively.',
          tags: ['Parameter sandbox', 'Realtime canvas', 'Visual systems']
        }
      ]
    }
  };

  const CATEGORY_ORDER = ['projects', 'tools', 'games'];

  const CATEGORY_GROUPS = Object.freeze({
    projects: [
      {
        label: 'AI / NLP',
        itemIds: ['chatbotLora', 'smartSentence']
      },
      {
        label: 'Data / Analytics',
        itemIds: ['retailStore', 'pizzaDashboard', 'ufoDashboard', 'deliveryTip', 'targetEmptyPackage', 'pizza']
      },
      {
        label: 'Computer Vision',
        itemIds: ['digitGenerator', 'sheetMusicUpscale', 'shapeClassifier', 'handwritingRating']
      },
      {
        label: 'Modeling / Systems',
        itemIds: ['covidAnalysis', 'babynames', 'nonogram', 'website']
      }
    ],
    tools: [
      {
        label: 'Text / Writing',
        itemIds: ['text-compare', 'nbsp-cleaner', 'oxford-comma-checker', 'point-of-view-checker', 'word-frequency']
      },
      {
        label: 'Campaign / Link',
        itemIds: ['utm-batch-builder', 'qr-code-generator']
      },
      {
        label: 'Media',
        itemIds: ['image-optimizer', 'background-remover', 'screen-recorder']
      }
    ],
    games: [
      {
        label: 'Playable Systems',
        itemIds: ['project-starfall', 'stellar-dogfight', 'roulette']
      },
      {
        label: 'Simulations',
        itemIds: ['probability-engine', 'ocean-wave-simulation']
      }
    ]
  });

  const MOBILE_DEPTH_TOPICS = Object.freeze({
    projects: {
      'ai-nlp': [
        { label: 'RAG Systems', itemIds: ['chatbotLora'] },
        { label: 'Retrieval', itemIds: ['smartSentence'] }
      ],
      'data-analytics': [
        { label: 'ETL', itemIds: ['retailStore'] },
        { label: 'Dashboards', itemIds: ['pizzaDashboard', 'ufoDashboard', 'targetEmptyPackage'] },
        { label: 'Field Models', itemIds: ['deliveryTip', 'pizza'] }
      ],
      'computer-vision': [
        { label: 'Generation', itemIds: ['digitGenerator'] },
        { label: 'Image Cleanup', itemIds: ['sheetMusicUpscale'] },
        { label: 'Classifiers', itemIds: ['shapeClassifier', 'handwritingRating'] }
      ],
      'modeling-systems': [
        { label: 'Drivers', itemIds: ['covidAnalysis'] },
        { label: 'Recommendations', itemIds: ['babynames'] },
        { label: 'Solvers', itemIds: ['nonogram'] },
        { label: 'Site System', itemIds: ['website'] }
      ]
    },
    tools: {
      'text-writing': [
        { label: 'Compare', itemIds: ['text-compare'] },
        { label: 'Cleanup', itemIds: ['nbsp-cleaner', 'oxford-comma-checker', 'point-of-view-checker'] },
        { label: 'Analysis', itemIds: ['word-frequency'] }
      ],
      'campaign-link': [
        { label: 'Campaign URLs', itemIds: ['utm-batch-builder'] },
        { label: 'Share Codes', itemIds: ['qr-code-generator'] }
      ],
      media: [
        { label: 'Image Flow', itemIds: ['image-optimizer', 'background-remover'] },
        { label: 'Capture', itemIds: ['screen-recorder'] }
      ]
    },
    games: {
      'playable-systems': [
        { label: 'RPG', itemIds: ['project-starfall'] },
        { label: 'Arcade', itemIds: ['stellar-dogfight', 'roulette'] }
      ],
      simulations: [
        { label: 'Probability', itemIds: ['probability-engine'] },
        { label: 'Physics', itemIds: ['ocean-wave-simulation'] }
      ]
    }
  });

  const BRANCH_LAYOUT = Object.freeze({
    projects: { angle: -155 },
    tools: { angle: -25 },
    games: { angle: 68 }
  });

  const ITEM_EXPANSION_LAYOUT = Object.freeze({
    projects: { angle: 180 },
    tools: { angle: 8 },
    games: { angle: -14 }
  });

  const ICONS = {
    folder: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 7.5h6l1.8 2h9.2v8.5a2 2 0 0 1-2 2h-15a2 2 0 0 1-2-2V9.5a2 2 0 0 1 2-2z"></path><path d="M3.5 7.5V6a2 2 0 0 1 2-2h4.1l1.9 2h7a2 2 0 0 1 2 2v1.5"></path></svg>',
    wrench: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.3a4.2 4.2 0 0 0-5.1 5.1L4 17v3h3l5.6-5.6a4.2 4.2 0 0 0 5.1-5.1l-2.9 2.9-3-3 2.9-2.9z"></path></svg>',
    gamepad: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 15h8l1.8 2.2a2.2 2.2 0 0 0 3.9-1.8l-1-6.2A4 4 0 0 0 16.8 6H7.2a4 4 0 0 0-3.9 3.2l-1 6.2a2.2 2.2 0 0 0 3.9 1.8L8 15z"></path><path d="M7 10v3M5.5 11.5h3M16 11h.01M18.5 12.5h.01"></path></svg>',
    chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 6.8A3.8 3.8 0 0 1 8.8 3h6.4A3.8 3.8 0 0 1 19 6.8v4.4a3.8 3.8 0 0 1-3.8 3.8h-3.7L7 19v-4.1A3.8 3.8 0 0 1 5 11.2V6.8z"></path><path d="M8.8 8.6h6.4M8.8 11.4h3.8"></path></svg>',
    database: '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="6.5" ry="2.8"></ellipse><path d="M5.5 5.5v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8v-6"></path><path d="M5.5 11.5v6c0 1.5 2.9 2.8 6.5 2.8s6.5-1.3 6.5-2.8v-6"></path></svg>',
    digits: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5.5h5v13H6.5"></path><path d="M14 6.5a3 3 0 0 1 3-1.2 3.2 3.2 0 0 1 2.4 2.6c.4 2.8-4.6 5-5.5 8.6h6.2"></path></svg>',
    search: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="5.8"></circle><path d="M15 15l4.5 4.5M8 10.5h5"></path></svg>',
    chart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19.5h16"></path><path d="M6.5 16V9.5M12 16V5M17.5 16v-8"></path><path d="M6.5 9.5l5.5-4.5 5.5 3"></path></svg>',
    globe: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"></circle><path d="M3.8 12h16.4M12 3.5c2.2 2.4 3.2 5.2 3.2 8.5s-1 6.1-3.2 8.5M12 3.5C9.8 5.9 8.8 8.7 8.8 12s1 6.1 3.2 8.5"></path></svg>',
    music: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18.2a2.8 2.8 0 1 1-1.2-2.3V5.5l9-2v10.7"></path><path d="M16.8 16.2a2.8 2.8 0 1 1-1.2-2.3"></path><path d="M7.8 8.5l9-2"></path></svg>',
    map: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 6.5l4.7-2 4.6 2 4.7-2v13l-4.7 2-4.6-2-4.7 2v-13z"></path><path d="M9.7 4.5v13M14.3 6.5v13"></path></svg>',
    package: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 8.2L12 4l7.5 4.2v8.6L12 21l-7.5-4.2V8.2z"></path><path d="M4.8 8.4L12 12.5l7.2-4.1M12 12.5V21"></path><path d="M8.2 6.2l7.5 4.3"></path></svg>',
    shapes: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="8" cy="8" r="3.5"></circle><path d="M14.5 4.5h5v5h-5zM5 16.5h6l-3 4.5-3-4.5z"></path><path d="M15 16h4.5v4.5H15z"></path></svg>',
    pencil: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 18.5l1-4.2L15.8 4a2.1 2.1 0 0 1 3 3L8.5 17.3l-4 1.2z"></path><path d="M14.2 5.6l4.2 4.2"></path></svg>',
    virus: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5"></circle><path d="M12 3.5v3M12 17.5v3M3.5 12h3M17.5 12h3M5.9 5.9l2.2 2.2M15.9 15.9l2.2 2.2M18.1 5.9l-2.2 2.2M8.1 15.9l-2.2 2.2"></path></svg>',
    name: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 18.5V6.5h4.2a3.2 3.2 0 0 1 0 6.4H5"></path><path d="M13.5 18.5v-7.2a3.3 3.3 0 0 1 6.5 0v7.2M13.5 14.5H20"></path></svg>',
    regression: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18.5c4.3-7.9 9.4-11.7 16-12"></path><circle cx="7.5" cy="14.5" r="1.3"></circle><circle cx="11" cy="11" r="1.3"></circle><circle cx="15.5" cy="9" r="1.3"></circle></svg>',
    grid: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h5v5H5zM14 5h5v5h-5zM5 14h5v5H5zM14 14h5v5h-5z"></path></svg>',
    code: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 8L4.5 12l4 4M15.5 8l4 4-4 4M13.5 5.5l-3 13"></path></svg>',
    diff: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 7h8M6 12h12M6 17h7"></path><path d="M17.5 5.5v5M15 8h5"></path></svg>',
    cleanup: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 16.5l8.8-8.8a2.6 2.6 0 0 1 3.7 3.7l-7.1 7.1H5v-2z"></path><path d="M13.2 8.3l3.5 3.5M5 19h14"></path></svg>',
    comma: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 10.5h3v3.4c0 2.1-1 3.7-3 4.7"></path><path d="M14.5 10.5h3v3.4c0 2.1-1 3.7-3 4.7"></path></svg>',
    eye: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 12s3-5.5 8.5-5.5 8.5 5.5 8.5 5.5-3 5.5-8.5 5.5S3.5 12 3.5 12z"></path><circle cx="12" cy="12" r="2.5"></circle></svg>',
    hash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9.5 4.5l-2 15M16.5 4.5l-2 15M5 9h15M4 15h15"></path></svg>',
    link: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9.5 14.5l5-5"></path><path d="M10.5 7.5l1.2-1.2a4 4 0 0 1 5.7 5.7l-1.2 1.2"></path><path d="M13.5 16.5l-1.2 1.2a4 4 0 1 1-5.7-5.7l1.2-1.2"></path></svg>',
    qr: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h5v5H5zM14 5h5v5h-5zM5 14h5v5H5z"></path><path d="M14 14h2v2h-2zM18 14h1v5h-5v-1M14 18h2"></path></svg>',
    image: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 6.5a2 2 0 0 1 2-2h11a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-11z"></path><circle cx="9" cy="9" r="1.4"></circle><path d="M5 16l4-4 3 3 2-2 5 5"></path></svg>',
    eraser: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 14.5l7.8-7.8a2.5 2.5 0 0 1 3.5 0l3.5 3.5a2.5 2.5 0 0 1 0 3.5l-5.8 5.8H8.5l-4-4z"></path><path d="M9.5 9.5l5 5M13.5 19.5H20"></path></svg>',
    recorder: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7.5h10a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2z"></path><path d="M17 10.2l4-2.2v8l-4-2.2"></path></svg>',
    rocket: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 14.8l-2 4.7 4.7-2M9.2 14.8L5.5 11l4.1-1.1c1.3-2.7 3.8-5.1 8.5-5.8 0 4.7-2.9 7.9-5.8 9.4l-1.1 4-2-2.7z"></path><circle cx="15" cy="8.7" r="1.5"></circle></svg>',
    jet: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l4 8 4 2-4 1.7-1.8 5.3L12 16l-2.2 4.5L8 15.2 4 13.5l4-2 4-8z"></path></svg>',
    roulette: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="2.4"></circle><path d="M12 4v5.6M12 14.4V20M4 12h5.6M14.4 12H20M6.3 6.3l4 4M13.7 13.7l4 4M17.7 6.3l-4 4M10.3 13.7l-4 4"></path></svg>',
    cube: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.8l7 4v8.4l-7 4-7-4V7.8l7-4z"></path><path d="M5.3 8.1L12 12l6.7-3.9M12 12v8"></path></svg>',
    wave: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 16.5c2.4 0 2.4-2 4.8-2s2.4 2 4.8 2 2.4-2 4.8-2 2.4 2 4.8 2"></path><path d="M4.5 11.5c3.5-5.2 9.3-6 14.8-2.2"></path></svg>'
  };

  const ITEM_ICON_BY_ID = {
    chatbotLora: 'chat',
    retailStore: 'database',
    digitGenerator: 'digits',
    smartSentence: 'search',
    pizzaDashboard: 'chart',
    ufoDashboard: 'globe',
    sheetMusicUpscale: 'music',
    deliveryTip: 'map',
    targetEmptyPackage: 'package',
    shapeClassifier: 'shapes',
    handwritingRating: 'pencil',
    covidAnalysis: 'virus',
    babynames: 'name',
    pizza: 'regression',
    nonogram: 'grid',
    website: 'code',
    'text-compare': 'diff',
    'nbsp-cleaner': 'cleanup',
    'oxford-comma-checker': 'comma',
    'point-of-view-checker': 'eye',
    'word-frequency': 'hash',
    'utm-batch-builder': 'link',
    'qr-code-generator': 'qr',
    'image-optimizer': 'image',
    'background-remover': 'eraser',
    'screen-recorder': 'recorder',
    'project-starfall': 'rocket',
    'stellar-dogfight': 'jet',
    roulette: 'roulette',
    'probability-engine': 'cube',
    'ocean-wave-simulation': 'wave'
  };

  const ITEM_VISUAL_BY_ID = Object.freeze({
    chatbotLora: '/img/projects/chatbotLora-640.webp',
    retailStore: '/img/projects/retailStore-640.webp',
    digitGenerator: '/img/projects/digitGenerator-640.webp',
    smartSentence: '/img/projects/smartSentence-640.webp',
    pizzaDashboard: '/img/projects/pizzaDashboard-640.webp',
    ufoDashboard: '/img/projects/ufoDashboard-640.webp',
    sheetMusicUpscale: '/img/projects/sheetMusicUpscale-640.webp',
    deliveryTip: '/img/projects/deliveryTip-640.webp',
    targetEmptyPackage: '/img/projects/targetEmptyPackage-640.webp',
    shapeClassifier: '/img/projects/shapeClassifier-640.webp',
    handwritingRating: '/img/projects/handwritingRating-640.webp',
    covidAnalysis: '/img/projects/covidAnalysis-640.webp',
    babynames: '/img/projects/babynames-640.webp',
    pizza: '/img/projects/pizza-640.webp',
    nonogram: '/img/projects/nonogram-640.webp',
    website: '/img/projects/website-640.webp',
    'project-starfall': '/img/project-starfall/ui/start-screen.png',
    'stellar-dogfight': '/img/games/stellar-dogfight/raster/background-nebula.png'
  });

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const toRadians = (degrees) => (degrees * Math.PI) / 180;

  const normalizeDegrees = (degrees) => {
    let value = degrees;
    while (value <= -180) value += 360;
    while (value > 180) value -= 360;
    return value;
  };

  const getIcon = (type) => ICONS[type] || ICONS.folder;

  const getCategoryIcon = (type) => getIcon(type);

  const getLinkIcon = () => (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7"></path><path d="M9 7h8v8"></path></svg>'
  );

  const getToggleIcon = (expanded) => expanded
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 12h12"></path></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14"></path><path d="M5 12h14"></path></svg>';

  const getItemIcon = (item, categoryId) => {
    const category = CATEGORIES[categoryId] || CATEGORIES.projects;
    return getIcon(ITEM_ICON_BY_ID[item?.id] || category.icon);
  };

  const getItemKey = (categoryId, itemId) => `${categoryId}:${itemId}`;
  const getGroupId = (label) => String(label || 'group')
    .toLowerCase()
    .replace(/&/g, 'and')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'group';
  const getGroupKey = (categoryId, groupId) => `group:${categoryId}:${groupId}`;
  const parseGroupKey = (key) => {
    const parts = String(key || '').split(':');
    if (parts.length !== 3 || parts[0] !== 'group') return null;
    return {
      categoryId: parts[1],
      groupId: parts[2]
    };
  };

  const parseItemKey = (key) => {
    const parts = String(key || '').split(':');
    if (parts.length !== 2) return null;
    return {
      categoryId: parts[0],
      itemId: parts[1]
    };
  };

  const getCategoryGroups = (categoryId) => {
    const category = CATEGORIES[categoryId];
    if (!category) return [];

    const itemsById = new Map(category.items.map((item) => [item.id, item]));
    const seen = new Set();
    const groups = (CATEGORY_GROUPS[categoryId] || []).map((group) => {
      const items = group.itemIds
        .map((itemId) => itemsById.get(itemId))
        .filter(Boolean);
      items.forEach((item) => seen.add(item.id));
      return {
        id: group.id || getGroupId(group.label),
        label: group.label,
        items
      };
    }).filter((group) => group.items.length);

    const remaining = category.items.filter((item) => !seen.has(item.id));
    if (remaining.length) {
      groups.push({
        id: 'more',
        label: 'More',
        items: remaining
      });
    }

    return groups;
  };

  const getCategoryGroupById = (categoryId, groupId) => getCategoryGroups(categoryId)
    .find((group) => group.id === groupId);

  const getMobileTopicId = (label) => getGroupId(label);

  const getMobileDepthTopics = (categoryId, group) => {
    if (!group) return [];
    const itemsById = new Map(group.items.map((item) => [item.id, item]));
    const seen = new Set();
    const configured = MOBILE_DEPTH_TOPICS[categoryId]?.[group.id] || [];
    const topics = configured.map((topic) => {
      const items = topic.itemIds
        .map((itemId) => itemsById.get(itemId))
        .filter(Boolean);
      items.forEach((item) => seen.add(item.id));
      return {
        id: topic.id || getMobileTopicId(topic.label),
        label: topic.label,
        items
      };
    }).filter((topic) => topic.items.length);

    const remaining = group.items.filter((item) => !seen.has(item.id));
    if (remaining.length) {
      topics.push({
        id: 'more',
        label: group.label,
        items: remaining
      });
    }

    if (!topics.length && group.items.length) {
      topics.push({
        id: group.id,
        label: group.label,
        items: group.items
      });
    }

    return topics;
  };

  const getPreferredMobileGroup = (categoryId) => getCategoryGroups(categoryId)[0] || null;

  const getPreferredMobileTopic = (topics = []) => topics[0] || null;

  const getMobileItemDepth = (categoryId, itemId) => {
    const group = getCategoryGroups(categoryId)
      .find((entry) => entry.items.some((item) => item.id === itemId));
    if (!group) return null;
    const topic = getMobileDepthTopics(categoryId, group)
      .find((entry) => entry.items.some((item) => item.id === itemId));
    return {
      group,
      topic: topic || getMobileDepthTopics(categoryId, group)[0] || null
    };
  };

  const getItemGroupLabels = (categoryId, itemId) => getCategoryGroups(categoryId)
    .filter((group) => group.items.some((item) => item.id === itemId))
    .map((group) => group.label);

  const getItemSearchTokens = (item) => [
    item.title,
    item.fullTitle,
    item.snippet,
    item.summary,
    ...(item.tags || [])
  ]
    .join(' ')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 2);

  const buildCategoryDots = (categoryId) => {
    const category = CATEGORIES[categoryId];
    const items = category?.items || [];
    const count = items.length;
    if (!count) return '';

    return items.map((item, index) => {
      const angle = toRadians(-90 + ((360 / count) * index));
      const x = 50 + (Math.cos(angle) * 50);
      const y = 50 + (Math.sin(angle) * 50);
      const title = item.fullTitle || item.title;
      return `<button type="button" class="home-graph__halo-dot" data-graph-dot="${escapeHtml(categoryId)}-${index}" data-graph-dot-category="${escapeHtml(categoryId)}" data-graph-dot-item="${escapeHtml(item.id)}" aria-label="${escapeHtml(title)}" title="${escapeHtml(title)}" style="--dot-x: ${x.toFixed(2)}%; --dot-y: ${y.toFixed(2)}%; --dot-order: ${index};"></button>`;
    }).join('');
  };

  const makeRect = (left, top, right, bottom) => ({
    left,
    top,
    right: Math.max(left, right),
    bottom: Math.max(top, bottom),
    get width() {
      return Math.max(0, this.right - this.left);
    },
    get height() {
      return Math.max(0, this.bottom - this.top);
    }
  });

  const clampPoint = (point, rect, halfWidth = 0, halfHeight = 0) => ({
    x: clamp(point.x, rect.left + halfWidth, rect.right - halfWidth),
    y: clamp(point.y, rect.top + halfHeight, rect.bottom - halfHeight)
  });

  const getGraphMetrics = (map) => {
    const rect = map?.getBoundingClientRect?.();
    const width = Math.max(280, Math.round(rect?.width || map?.clientWidth || window.innerWidth || 1200));
    const height = Math.max(300, Math.round(rect?.height || map?.clientHeight || window.innerHeight * .58 || 620));
    const isCompact = width <= 768 || window.innerWidth <= 768;
    const isNarrow = width <= 940 || window.innerWidth <= 940;
    const safeX = isCompact ? 10 : isNarrow ? 18 : 34;
    const safeY = isCompact ? 10 : isNarrow ? 18 : 28;
    const categoryHalfWidth = isCompact ? 54 : isNarrow ? 84 : 126;
    const categoryHalfHeight = isCompact ? 28 : isNarrow ? 34 : 50;
    const safe = makeRect(safeX, safeY, width - safeX, height - safeY);
    const showItemLabels = !isNarrow && width >= 1080 && height >= 520;

    return {
      width,
      height,
      isCompact,
      isNarrow,
      showItemLabels,
      safe,
      categoryHalfWidth,
      categoryHalfHeight,
      centerHalfSize: isCompact ? 33 : isNarrow ? 47 : 59,
      layoutGap: isCompact ? 12 : isNarrow ? 18 : 28
    };
  };

  const makeBox = (point, halfWidth, halfHeight) => makeRect(
    point.x - halfWidth,
    point.y - halfHeight,
    point.x + halfWidth,
    point.y + halfHeight
  );

  const boxesOverlap = (a, b, gap = 0) => (
    a.left - gap < b.right
    && a.right + gap > b.left
    && a.top - gap < b.bottom
    && a.bottom + gap > b.top
  );

  const getBoxOverlapArea = (box, fixedBox, gap = 0) => {
    if (!boxesOverlap(box, fixedBox, gap)) return 0;
    const overlapX = Math.max(0, Math.min(box.right + gap, fixedBox.right) - Math.max(box.left - gap, fixedBox.left));
    const overlapY = Math.max(0, Math.min(box.bottom + gap, fixedBox.bottom) - Math.max(box.top - gap, fixedBox.top));
    return overlapX * overlapY;
  };

  const pushPointAway = (point, fixedPoint, halfWidth, halfHeight, fixedHalfWidth, fixedHalfHeight, metrics) => {
    const box = makeBox(point, halfWidth, halfHeight);
    const fixedBox = makeBox(fixedPoint, fixedHalfWidth, fixedHalfHeight);
    if (!boxesOverlap(box, fixedBox, metrics.layoutGap)) return point;

    const dx = point.x - fixedPoint.x || 1;
    const dy = point.y - fixedPoint.y || 1;
    const overlapX = halfWidth + fixedHalfWidth + metrics.layoutGap - Math.abs(dx);
    const overlapY = halfHeight + fixedHalfHeight + metrics.layoutGap - Math.abs(dy);
    const next = { ...point };

    if (overlapX <= overlapY) {
      next.x += dx > 0 ? overlapX : -overlapX;
    } else {
      next.y += dy > 0 ? overlapY : -overlapY;
    }

    return clampPoint(next, metrics.safe, halfWidth, halfHeight);
  };

  const getCenterPosition = (metrics) => clampPoint({
    x: metrics.safe.left + (metrics.safe.width / 2),
    y: metrics.safe.top + (metrics.safe.height * (metrics.isCompact ? .38 : .5))
  }, metrics.safe, metrics.centerHalfSize, metrics.centerHalfSize);

  const getCategoryRadius = (metrics, activeItemCount = 0) => {
    const pressure = clamp((activeItemCount - 6) / 12, 0, 1);
    if (metrics.isCompact) {
      return clamp(Math.min(metrics.width * .38, metrics.height * .3) - (pressure * 4), 106, 126);
    }

    if (metrics.isNarrow) {
      return clamp(Math.min(metrics.width * .35, metrics.height * .32) - (pressure * 8), 124, 164);
    }

    return clamp(Math.min(metrics.width * .22, metrics.height * .38) - (pressure * 12), 176, 230);
  };

  const getBranchAngle = (categoryId, activeCategoryId, activeItemCount, metrics) => {
    const baseAngle = BRANCH_LAYOUT[categoryId]?.angle || 0;
    if (categoryId === activeCategoryId || metrics.isCompact) return baseAngle;

    const activeAngle = BRANCH_LAYOUT[activeCategoryId]?.angle || 0;
    const delta = normalizeDegrees(baseAngle - activeAngle);
    const pressure = clamp((activeItemCount - 5) / 12, 0, 1);
    if (Math.abs(delta) > 128) return baseAngle;
    return baseAngle + (Math.sign(delta || 1) * pressure * 8);
  };

  const getDesktopGraphFractions = (activeCategoryId) => {
    const overview = {
      center: { x: .5, y: .5 },
      categories: {
        projects: { x: .28, y: .34 },
        tools: { x: .72, y: .34 },
        games: { x: .5, y: .79 }
      }
    };

    if (!activeCategoryId || !CATEGORIES[activeCategoryId]) return overview;

    const supporting = CATEGORY_ORDER.filter((categoryId) => categoryId !== activeCategoryId);
    return {
      center: { x: .1, y: .52 },
      categories: {
        [activeCategoryId]: { x: .28, y: .52 },
        [supporting[0]]: { x: .22, y: .24 },
        [supporting[1]]: { x: .22, y: .8 }
      }
    };
  };

  const getDesktopGraphLayout = (metrics, activeCategoryId) => {
    const fractions = getDesktopGraphFractions(activeCategoryId);
    const center = clampPoint({
      x: metrics.safe.left + (metrics.safe.width * fractions.center.x),
      y: metrics.safe.top + (metrics.safe.height * fractions.center.y)
    }, metrics.safe, metrics.centerHalfSize, metrics.centerHalfSize);
    const categories = {};

    CATEGORY_ORDER.forEach((categoryId) => {
      const fraction = fractions.categories[categoryId] || getDesktopGraphFractions(null).categories[categoryId];
      categories[categoryId] = clampPoint({
        x: metrics.safe.left + (metrics.safe.width * fraction.x),
        y: metrics.safe.top + (metrics.safe.height * fraction.y)
      }, metrics.safe, metrics.categoryHalfWidth, metrics.categoryHalfHeight);
    });

    return {
      center,
      categories,
      categoryRadius: 0
    };
  };

  const getRadialGraphLayout = (metrics, activeCategoryId) => {
    const center = getCenterPosition(metrics);
    const activeItemCount = CATEGORIES[activeCategoryId]?.items?.length || 0;
    const categoryRadius = getCategoryRadius(metrics, activeItemCount);
    const categories = {};

    CATEGORY_ORDER.forEach((categoryId) => {
      const angle = toRadians(getBranchAngle(categoryId, activeCategoryId, activeItemCount, metrics));
      categories[categoryId] = clampPoint({
        x: center.x + (Math.cos(angle) * categoryRadius),
        y: center.y + (Math.sin(angle) * categoryRadius)
      }, metrics.safe, metrics.categoryHalfWidth, metrics.categoryHalfHeight);
    });

    return {
      center,
      categories,
      categoryRadius
    };
  };

  const resolveGraphCollisions = (layout, metrics) => {
    const categories = { ...layout.categories };
    const center = layout.center;

    for (let iteration = 0; iteration < 4; iteration += 1) {
      CATEGORY_ORDER.forEach((categoryId) => {
        categories[categoryId] = pushPointAway(
          categories[categoryId],
          center,
          metrics.categoryHalfWidth,
          metrics.categoryHalfHeight,
          metrics.centerHalfSize,
          metrics.centerHalfSize,
          metrics
        );
      });

      CATEGORY_ORDER.forEach((categoryId, index) => {
        CATEGORY_ORDER.slice(index + 1).forEach((otherCategoryId) => {
          categories[otherCategoryId] = pushPointAway(
            categories[otherCategoryId],
            categories[categoryId],
            metrics.categoryHalfWidth,
            metrics.categoryHalfHeight,
            metrics.categoryHalfWidth,
            metrics.categoryHalfHeight,
            metrics
          );
        });
      });
    }

    return {
      center,
      categories,
      categoryRadius: layout.categoryRadius
    };
  };

  const getGraphLayout = (metrics, activeCategoryId) => {
    if (!metrics.isNarrow) return getDesktopGraphLayout(metrics, activeCategoryId);
    const layout = getRadialGraphLayout(metrics, activeCategoryId);
    if (metrics.isCompact) return layout;
    return resolveGraphCollisions(layout, metrics);
  };

  const getItemNodeSize = (metrics) => metrics.isCompact ? 28 : metrics.isNarrow ? 34 : 42;

  const getItemNodeDimensions = (metrics) => {
    const size = getItemNodeSize(metrics);
    const showLabels = Boolean(metrics.showItemLabels);
    return {
      size,
      width: showLabels ? clamp(Math.round(metrics.width * .12), 132, 154) : size,
      height: showLabels ? 44 : size,
      showLabels
    };
  };

  const splitGroupsIntoColumns = (categoryId, groups, columnCount) => {
    if (columnCount <= 1) return [groups];
    if (categoryId === 'projects') {
      return [
        groups.slice(0, 2),
        groups.slice(2)
      ].filter((column) => column.length);
    }

    const columns = Array.from({ length: columnCount }, () => []);
    const heights = Array.from({ length: columnCount }, () => 0);
    groups.forEach((group) => {
      const target = heights.indexOf(Math.min(...heights));
      columns[target].push(group);
      heights[target] += group.items.length + 1.4;
    });
    return columns.filter((column) => column.length);
  };

  const getColumnHeight = (groups, nodeHeight, rowGap, groupLabelHeight, groupGap) => groups.reduce((height, group, index) => {
    const itemHeight = group.items.length * nodeHeight;
    const itemGaps = Math.max(0, group.items.length - 1) * rowGap;
    return height + groupLabelHeight + 8 + itemHeight + itemGaps + (index === groups.length - 1 ? 0 : groupGap);
  }, 0);

  const getGroupedColumnHalfWidth = (nodeWidth) => Math.max(nodeWidth / 2, 86);

  const getGroupedColumnObstacles = (categoryId, metrics, graphLayout) => [
    {
      point: graphLayout.center,
      halfWidth: metrics.centerHalfSize + 12,
      halfHeight: metrics.centerHalfSize + 12
    },
    ...CATEGORY_ORDER.filter(entry => entry !== categoryId).map((entry) => ({
      point: graphLayout.categories[entry],
      halfWidth: metrics.categoryHalfWidth + 26,
      halfHeight: metrics.categoryHalfHeight + 20
    }))
  ];

  const getSafeGroupedColumnStart = (x, preferredY, columnHeight, columnHalfWidth, metrics, obstacles) => {
    const topLimit = metrics.safe.top + 18;
    const bottomLimit = Math.max(topLimit, metrics.safe.bottom - columnHeight + 18);
    let yStart = clamp(preferredY, topLimit, bottomLimit);

    for (let iteration = 0; iteration < 4; iteration += 1) {
      const columnBox = makeRect(
        x - columnHalfWidth,
        yStart - 18,
        x + columnHalfWidth,
        yStart + columnHeight + 12
      );
      const obstacle = obstacles.find((entry) => boxesOverlap(
        columnBox,
        makeBox(entry.point, entry.halfWidth, entry.halfHeight),
        10
      ));

      if (!obstacle) break;
      yStart = Math.min(
        bottomLimit,
        Math.max(
          yStart,
          obstacle.point.y + obstacle.halfHeight + 28
        )
      );
    }

    return yStart;
  };

  const getColumnXPositions = (categoryId, columnCount, metrics, origin, nodeWidth) => {
    const halfWidth = nodeWidth / 2;
    const sideGap = 42;
    if (categoryId === 'projects') {
      const left = metrics.safe.left + halfWidth;
      const right = origin.x - metrics.categoryHalfWidth - sideGap - halfWidth;
      if (columnCount <= 1) return [clamp((left + right) / 2, left, Math.max(left, right))];
      return [
        clamp(left, left, Math.max(left, right)),
        clamp(right, left, Math.max(left, right))
      ];
    }

    const left = origin.x + metrics.categoryHalfWidth + sideGap + halfWidth;
    const right = metrics.safe.right - halfWidth;
    if (columnCount <= 1) return [clamp(left, left, Math.max(left, right))];
    const minGap = nodeWidth + 18;
    const minLeft = origin.x + metrics.categoryHalfWidth + 16 + halfWidth;
    const spacedLeft = Math.min(left, right - minGap);
    return [
      clamp(spacedLeft, minLeft, Math.max(minLeft, right)),
      clamp(right, left, Math.max(left, right))
    ];
  };

  const getAngleBetweenPoints = (from, to) => Math.atan2(to.y - from.y, to.x - from.x) * (180 / Math.PI);

  const getGroupedFanAngles = (categoryId, groupCount, graphLayout) => {
    const origin = graphLayout.categories[categoryId] || graphLayout.center;
    const center = graphLayout.center;
    if (categoryId === 'projects' && groupCount === 4) {
      return [92, 170, -145, -78];
    }

    const awayAngle = getAngleBetweenPoints(center, origin);
    const spread = categoryId === 'projects'
      ? 210
      : groupCount <= 2
        ? 118
        : 168;

    if (groupCount <= 1) return [awayAngle];

    return Array.from({ length: groupCount }, (_, index) => {
      const offset = ((index / (groupCount - 1)) - .5) * spread;
      return normalizeDegrees(awayAngle + offset);
    });
  };

  const getGroupedFanSlot = (categoryId, groupIndex, origin, metrics) => {
    const xScale = clamp(metrics.width / 1120, .84, 1.08);
    const yScale = clamp(metrics.height / 680, .82, 1.16);
    const slots = {
      projects: [
        { x: 0, y: 295 },
        { x: -315, y: 170 },
        { x: -305, y: -230 },
        { x: 315, y: -325 }
      ],
      tools: [
        { x: 305, y: 44 },
        { x: -224, y: -260 },
        { x: 88, y: 292 }
      ],
      games: [
        { x: -235, y: 210 },
        { x: 235, y: 210 }
      ]
    };
    const slot = slots[categoryId]?.[groupIndex];
    if (!slot) return null;

    return {
      x: origin.x + (slot.x * xScale),
      y: origin.y + (slot.y * yScale)
    };
  };

  const getBranchingItemSlot = (index, count) => {
    const patterns = {
      1: [{ r: 0, t: 0 }],
      2: [{ r: 0, t: -.86 }, { r: 0, t: .86 }],
      3: [{ r: .24, t: -1.12 }, { r: 0, t: 0 }, { r: .24, t: 1.12 }],
      4: [{ r: 0, t: -1.42 }, { r: .28, t: -.48 }, { r: .28, t: .48 }, { r: 0, t: 1.42 }],
      5: [{ r: 0, t: -1.55 }, { r: .26, t: -.76 }, { r: .52, t: 0 }, { r: .26, t: .76 }, { r: 0, t: 1.55 }],
      6: [{ r: 0, t: -1.78 }, { r: .24, t: -1.08 }, { r: .52, t: -.38 }, { r: .52, t: .38 }, { r: .24, t: 1.08 }, { r: 0, t: 1.78 }]
    };
    if (patterns[count]) return patterns[count][index];

    const ring = Math.floor(index / 3);
    const position = index % 3;
    return {
      r: ring,
      t: position === 0 ? -1 : position === 1 ? 1 : 0
    };
  };

  const getBranchingItemPoint = (groupPoint, origin, index, count, metrics, dimensions, branchAngle = null) => {
    const dx = groupPoint.x - origin.x;
    const dy = groupPoint.y - origin.y;
    const distance = Math.hypot(dx, dy) || 1;
    const radial = Number.isFinite(branchAngle)
      ? {
          x: Math.cos(toRadians(branchAngle)),
          y: Math.sin(toRadians(branchAngle))
        }
      : {
          x: dx / distance,
          y: dy / distance
        };
    const tangent = {
      x: -radial.y,
      y: radial.x
    };
    const compact = metrics.isCompact;
    const nodeWidth = dimensions.width || dimensions.size;
    const nodeHeight = dimensions.height || dimensions.size;
    const projectedRadialSize = (Math.abs(radial.x) * nodeWidth) + (Math.abs(radial.y) * nodeHeight);
    const projectedTangentSize = (Math.abs(tangent.x) * nodeWidth) + (Math.abs(tangent.y) * nodeHeight);
    const labelNode = Boolean(dimensions.showLabels);
    const firstRadius = compact
      ? projectedRadialSize + 52
      : labelNode
        ? Math.max(projectedRadialSize + 18, 164)
        : projectedRadialSize + 72;
    const radiusStep = compact ? 42 : labelNode ? 44 : 64;
    const tangentGap = compact
      ? projectedTangentSize + 20
      : labelNode
        ? Math.max(projectedTangentSize + 28, nodeHeight + 42)
        : projectedTangentSize + 40;
    const slot = getBranchingItemSlot(index, count);

    return {
      x: groupPoint.x + (radial.x * (firstRadius + (slot.r * radiusStep))) + (tangent.x * slot.t * tangentGap),
      y: groupPoint.y + (radial.y * (firstRadius + (slot.r * radiusStep))) + (tangent.y * slot.t * tangentGap)
    };
  };

  const getBalancedClusterOffsets = (count) => {
    const patterns = {
      1: [0],
      2: [-58, 58],
      3: [-82, 0, 82],
      4: [-108, -36, 36, 108],
      5: [-122, -61, 0, 61, 122],
      6: [-126, -76, -25, 25, 76, 126]
    };
    if (patterns[count]) return patterns[count];

    const spread = Math.min(252, 128 + (count * 12));
    return Array.from({ length: count }, (_, offsetIndex) => (
      count <= 1 ? 0 : ((offsetIndex / (count - 1)) - .5) * spread
    ));
  };

  const getClusteredLabelItemSlot = (index, count) => {
    const offsets = getBalancedClusterOffsets(count);
    const offset = offsets[index] ?? 0;
    const absOffset = Math.abs(offset);
    return {
      a: offset,
      r: absOffset > 104 || (count >= 5 && absOffset < 8) ? .7 : 0,
      rx: 1,
      ry: 1
    };
  };

  const getClusteredLabelItemPoint = (
    groupPoint,
    index,
    count,
    dimensions,
    branchAngle,
    categoryId = '',
    groupId = '',
    itemId = ''
  ) => {
    const slot = getClusteredLabelItemSlot(index, count, categoryId, groupId, itemId);
    const angle = toRadians(branchAngle + slot.a);

    if (dimensions.showLabels) {
      const radiusX = (Math.max(136, dimensions.width + 42) + (slot.r * 20)) * (slot.rx || 1);
      const radiusY = (Math.max(78, dimensions.height + 38) + (slot.r * 16)) * (slot.ry || 1);

      return {
        x: groupPoint.x + (Math.cos(angle) * radiusX),
        y: groupPoint.y + (Math.sin(angle) * radiusY)
      };
    }

    const radius = Math.max(124, dimensions.width + 12) + (slot.r * 40);

    return {
      x: groupPoint.x + (Math.cos(angle) * radius),
      y: groupPoint.y + (Math.sin(angle) * radius)
    };
  };

  const keepEntryNearCluster = (entry, metrics) => {
    if (!Number.isFinite(entry.clusterX) || !Number.isFinite(entry.clusterY) || !Number.isFinite(entry.clusterRadius)) return;

    const dx = entry.x - entry.clusterX;
    const dy = entry.y - entry.clusterY;
    const distance = Math.hypot(dx, dy);
    if (!distance || distance <= entry.clusterRadius) return;

    entry.x = entry.clusterX + ((dx / distance) * entry.clusterRadius);
    entry.y = entry.clusterY + ((dy / distance) * entry.clusterRadius);
    const point = clampPoint(entry, metrics.safe, entry.halfWidth, entry.halfHeight);
    entry.x = point.x;
    entry.y = point.y;
  };

  const pushEntryAwayFromObstacle = (entry, obstacle, metrics, gap) => {
    const box = makeBox(entry, entry.halfWidth, entry.halfHeight);
    const obstacleBox = makeBox(obstacle, obstacle.halfWidth, obstacle.halfHeight);
    if (!boxesOverlap(box, obstacleBox, gap)) return;

    const dx = entry.x - obstacle.x || 1;
    const dy = entry.y - obstacle.y || 1;
    const overlapX = entry.halfWidth + obstacle.halfWidth + gap - Math.abs(dx);
    const overlapY = entry.halfHeight + obstacle.halfHeight + gap - Math.abs(dy);
    if (overlapX <= 0 || overlapY <= 0) return;

    const original = {
      x: entry.x,
      y: entry.y
    };
    const makeCandidate = (axis) => {
      const candidate = {
        ...entry,
        x: original.x,
        y: original.y
      };
      if (axis === 'x') {
        candidate.x += dx > 0 ? overlapX : -overlapX;
      } else {
        candidate.y += dy > 0 ? overlapY : -overlapY;
      }
      const point = clampPoint(candidate, metrics.safe, candidate.halfWidth, candidate.halfHeight);
      candidate.x = point.x;
      candidate.y = point.y;
      keepEntryNearCluster(candidate, metrics);
      return candidate;
    };
    const preferredAxis = overlapX <= overlapY ? 'x' : 'y';
    const candidates = [
      makeCandidate(preferredAxis),
      makeCandidate(preferredAxis === 'x' ? 'y' : 'x')
    ];
    const bestCandidate = candidates.find((candidate) => (
      !boxesOverlap(makeBox(candidate, candidate.halfWidth, candidate.halfHeight), obstacleBox, gap)
    )) || candidates.reduce((best, candidate) => {
      const bestArea = getBoxOverlapArea(makeBox(best, best.halfWidth, best.halfHeight), obstacleBox, gap);
      const candidateArea = getBoxOverlapArea(makeBox(candidate, candidate.halfWidth, candidate.halfHeight), obstacleBox, gap);
      return candidateArea < bestArea ? candidate : best;
    });

    entry.x = bestCandidate.x;
    entry.y = bestCandidate.y;
  };

  const separateEntryPair = (first, second, metrics, gap) => {
    const firstBox = makeBox(first, first.halfWidth, first.halfHeight);
    const secondBox = makeBox(second, second.halfWidth, second.halfHeight);
    if (!boxesOverlap(firstBox, secondBox, gap)) return;

    const dx = second.x - first.x || 1;
    const dy = second.y - first.y || 1;
    const overlapX = first.halfWidth + second.halfWidth + gap - Math.abs(dx);
    const overlapY = first.halfHeight + second.halfHeight + gap - Math.abs(dy);
    if (overlapX <= 0 || overlapY <= 0) return;

    const originalFirst = {
      x: first.x,
      y: first.y
    };
    const originalSecond = {
      x: second.x,
      y: second.y
    };
    const makeCandidate = (axis) => {
      const firstCandidate = {
        ...first,
        x: originalFirst.x,
        y: originalFirst.y
      };
      const secondCandidate = {
        ...second,
        x: originalSecond.x,
        y: originalSecond.y
      };
      if (axis === 'x') {
        const shift = overlapX / 2;
        firstCandidate.x -= dx > 0 ? shift : -shift;
        secondCandidate.x += dx > 0 ? shift : -shift;
      } else {
        const shift = overlapY / 2;
        firstCandidate.y -= dy > 0 ? shift : -shift;
        secondCandidate.y += dy > 0 ? shift : -shift;
      }

      const firstPoint = clampPoint(firstCandidate, metrics.safe, firstCandidate.halfWidth, firstCandidate.halfHeight);
      const secondPoint = clampPoint(secondCandidate, metrics.safe, secondCandidate.halfWidth, secondCandidate.halfHeight);
      firstCandidate.x = firstPoint.x;
      firstCandidate.y = firstPoint.y;
      secondCandidate.x = secondPoint.x;
      secondCandidate.y = secondPoint.y;
      keepEntryNearCluster(firstCandidate, metrics);
      keepEntryNearCluster(secondCandidate, metrics);
      return {
        first: firstCandidate,
        second: secondCandidate
      };
    };
    const preferredAxis = overlapX <= overlapY ? 'x' : 'y';
    const candidates = [
      makeCandidate(preferredAxis),
      makeCandidate(preferredAxis === 'x' ? 'y' : 'x')
    ];
    const bestCandidate = candidates.find((candidate) => (
      !boxesOverlap(
        makeBox(candidate.first, candidate.first.halfWidth, candidate.first.halfHeight),
        makeBox(candidate.second, candidate.second.halfWidth, candidate.second.halfHeight),
        gap
      )
    )) || candidates.reduce((best, candidate) => {
      const bestArea = getBoxOverlapArea(
        makeBox(best.first, best.first.halfWidth, best.first.halfHeight),
        makeBox(best.second, best.second.halfWidth, best.second.halfHeight),
        gap
      );
      const candidateArea = getBoxOverlapArea(
        makeBox(candidate.first, candidate.first.halfWidth, candidate.first.halfHeight),
        makeBox(candidate.second, candidate.second.halfWidth, candidate.second.halfHeight),
        gap
      );
      return candidateArea < bestArea ? candidate : best;
    });

    first.x = bestCandidate.first.x;
    first.y = bestCandidate.first.y;
    second.x = bestCandidate.second.x;
    second.y = bestCandidate.second.y;
  };

  const getGroupedFanCategoryObstacle = (entryId, activeCategoryId, metrics, graphLayout) => {
    const active = entryId === activeCategoryId;
    const supportingHaloScale = metrics.showItemLabels && !metrics.isNarrow
      ? {
          x: .72,
          y: 1.3
        }
      : {
          x: .42,
          y: .72
        };

    return {
      x: graphLayout.categories[entryId].x,
      y: graphLayout.categories[entryId].y,
      halfWidth: active ? metrics.categoryHalfWidth : metrics.categoryHalfWidth * supportingHaloScale.x,
      halfHeight: active ? metrics.categoryHalfHeight : metrics.categoryHalfHeight * supportingHaloScale.y
    };
  };

  const getConnectorCorridorObstacles = (fromPoint, toPoint, metrics) => {
    if (!fromPoint || !toPoint) return [];
    const sampleCount = metrics.showItemLabels ? 7 : 5;
    const corridorHalfSize = metrics.showItemLabels ? 12 : 8;
    return Array.from({ length: sampleCount }, (_, index) => {
      const t = sampleCount <= 1 ? .5 : .22 + ((index / (sampleCount - 1)) * .58);
      return {
        x: fromPoint.x + ((toPoint.x - fromPoint.x) * t),
        y: fromPoint.y + ((toPoint.y - fromPoint.y) * t),
        halfWidth: corridorHalfSize,
        halfHeight: corridorHalfSize
      };
    });
  };

  const resolveGroupedFanEntries = (entries, categoryId, metrics, graphLayout) => {
    const resolved = entries.map((entry) => ({ ...entry }));
    const itemEntries = resolved.filter((entry) => entry.type === 'item');
    const groupEntries = resolved.filter((entry) => entry.type === 'group');
    const activeCategoryPoint = graphLayout.categories[categoryId];
    const gap = metrics.isCompact ? 4 : metrics.showItemLabels ? 16 : 12;
    const obstacles = [
      {
        x: graphLayout.center.x,
        y: graphLayout.center.y,
        halfWidth: metrics.centerHalfSize,
        halfHeight: metrics.centerHalfSize
      },
      ...CATEGORY_ORDER.map((entryId) => getGroupedFanCategoryObstacle(entryId, categoryId, metrics, graphLayout)),
      ...groupEntries.flatMap((entry) => getConnectorCorridorObstacles(activeCategoryPoint, entry, metrics)),
      ...groupEntries.map((entry) => ({
        x: entry.x,
        y: entry.y,
        halfWidth: entry.halfWidth,
        halfHeight: entry.halfHeight
      }))
    ];

    const avoidObstacles = () => {
      itemEntries.forEach((entry) => {
        obstacles.forEach((obstacle) => pushEntryAwayFromObstacle(entry, obstacle, metrics, gap));
      });
    };

    const separateItems = () => {
      itemEntries.forEach((entry, index) => {
        itemEntries.slice(index + 1)
          .forEach((otherEntry) => {
            const separationGap = otherEntry.groupId === entry.groupId
              ? gap
              : gap + (metrics.showItemLabels ? 12 : 4);
            separateEntryPair(entry, otherEntry, metrics, separationGap);
          });
      });
    };

    for (let iteration = 0; iteration < 28; iteration += 1) {
      avoidObstacles();
      separateItems();
      avoidObstacles();
    }

    return resolved;
  };

  const getGroupLabelHalfWidth = (label, dimensions) => {
    if (!dimensions.showLabels) {
      return Math.min(
        Math.max((String(label || '').length * 3.6) + 40, (dimensions.width / 2) + 24),
        106
      );
    }
    return Math.min(
      Math.max((String(label || '').length * 3.8) + 46, (dimensions.width / 2) + 30),
      112
    );
  };

  const getGroupClusterHaloSize = (categoryId, groupId, count, dimensions) => {
    const fallback = dimensions.showLabels
      ? {
          width: count >= 6 ? 360 : count >= 4 ? 336 : 272,
          height: count >= 6 ? 238 : count >= 4 ? 218 : 172
        }
      : {
          width: 276,
          height: 204
        };
    const customSizes = {
      projects: {
        'ai-nlp': { width: 274, height: 176 },
        'data-analytics': { width: 362, height: 242 },
        'computer-vision': { width: 346, height: 222 },
        'modeling-systems': { width: 350, height: 224 }
      }
    };

    return customSizes[categoryId]?.[groupId] || fallback;
  };

  const getGroupedFanItemLayout = (categoryId, metrics, graphLayout, dimensions) => {
    const category = CATEGORIES[categoryId];
    const origin = graphLayout.categories[categoryId] || graphLayout.center;
    const groups = getCategoryGroups(categoryId);
    const itemRadius = metrics.categoryHalfWidth + (dimensions.width / 2) + 116;
    const itemOrder = new Map(category.items.map((item, index) => [item.id, index]));
    const angles = getGroupedFanAngles(categoryId, groups.length, graphLayout);
    const entries = [];

    groups.forEach((group, groupIndex) => {
      const angle = toRadians(angles[groupIndex] || 0);
      const radial = {
        x: Math.cos(angle),
        y: Math.sin(angle)
      };
      const slottedCenter = getGroupedFanSlot(categoryId, groupIndex, origin, metrics);
      const rawCenter = slottedCenter || {
        x: origin.x + (radial.x * itemRadius),
        y: origin.y + (radial.y * itemRadius)
      };
      const hubScale = categoryId === 'projects' ? .76 : categoryId === 'tools' ? 1 : .76;
      const groupBranchMarginX = dimensions.showLabels ? dimensions.width + 92 : 92;
      const groupBranchMarginY = dimensions.showLabels ? dimensions.height + 96 : 18;
      const groupPoint = clampPoint({
        x: origin.x + ((rawCenter.x - origin.x) * hubScale),
        y: origin.y + ((rawCenter.y - origin.y) * hubScale)
      }, metrics.safe, groupBranchMarginX, groupBranchMarginY);
      const branchAngle = getAngleBetweenPoints(origin, groupPoint);
      const clusterHaloSize = getGroupClusterHaloSize(categoryId, group.id, group.items.length, dimensions);

      entries.push({
        type: 'group',
        id: getGroupKey(categoryId, group.id),
        groupId: group.id,
        label: group.label,
        items: group.items,
        x: groupPoint.x,
        y: groupPoint.y,
        halfWidth: getGroupLabelHalfWidth(group.label, dimensions),
        halfHeight: dimensions.showLabels ? 21 : 20,
        haloWidth: clusterHaloSize.width,
        haloHeight: clusterHaloSize.height,
        index: entries.length
      });

      group.items.forEach((item, itemIndex) => {
        const clusteredPoint = getClusteredLabelItemPoint(groupPoint, itemIndex, group.items.length, dimensions, branchAngle, categoryId, group.id, item.id);
        const point = clampPoint(
          clusteredPoint,
          metrics.safe,
          dimensions.width / 2,
          dimensions.height / 2
        );
        entries.push({
          type: 'item',
          item,
          groupId: group.id,
          x: point.x,
          y: point.y,
          halfWidth: dimensions.width / 2,
          halfHeight: dimensions.height / 2,
          clusterX: groupPoint.x,
          clusterY: groupPoint.y,
          clusterRadius: dimensions.showLabels ? 232 : 182,
          index: itemOrder.get(item.id) || 0
        });
      });
    });

    const resolvedEntries = resolveGroupedFanEntries(entries, categoryId, metrics, graphLayout);

    return {
      entries: resolvedEntries,
      positions: resolvedEntries.filter((entry) => entry.type === 'item'),
      nodeSize: dimensions.size,
      nodeWidth: dimensions.width,
      nodeHeight: dimensions.height,
      showLabels: true
    };
  };

  const getNarrowGroupClusterSlot = (categoryId, groupId, groupIndex, groupCount, origin, metrics) => {
    const xOffset = metrics.isNarrow ? 154 : 206;
    const yOffset = metrics.isNarrow ? 132 : 164;
    const projectSlots = {
      'computer-vision': { x: -xOffset, y: -yOffset, branchAngle: -148 },
      'modeling-systems': { x: xOffset, y: -yOffset, branchAngle: -34 },
      'data-analytics': { x: -xOffset, y: yOffset, branchAngle: 142 },
      'ai-nlp': { x: xOffset, y: yOffset, branchAngle: 48 }
    };
    const genericSlots = {
      1: [{ x: 0, y: yOffset, branchAngle: 90 }],
      2: [
        { x: -xOffset, y: 0, branchAngle: 180 },
        { x: xOffset, y: 0, branchAngle: 0 }
      ],
      3: [
        { x: -xOffset, y: -yOffset * .7, branchAngle: -150 },
        { x: xOffset, y: -yOffset * .7, branchAngle: -30 },
        { x: 0, y: yOffset, branchAngle: 90 }
      ],
      4: [
        { x: -xOffset, y: -yOffset, branchAngle: -148 },
        { x: xOffset, y: -yOffset, branchAngle: -34 },
        { x: -xOffset, y: yOffset, branchAngle: 142 },
        { x: xOffset, y: yOffset, branchAngle: 48 }
      ]
    };
    const slot = categoryId === 'projects'
      ? projectSlots[groupId] || genericSlots[groupCount]?.[groupIndex]
      : genericSlots[groupCount]?.[groupIndex];
    const fallbackAngle = groupCount <= 1 ? 90 : -130 + ((260 / Math.max(1, groupCount - 1)) * groupIndex);
    const fallback = {
      x: Math.cos(toRadians(fallbackAngle)) * xOffset,
      y: Math.sin(toRadians(fallbackAngle)) * yOffset,
      branchAngle: fallbackAngle
    };
    const resolvedSlot = slot || fallback;

    return {
      point: clampPoint({
        x: origin.x + resolvedSlot.x,
        y: origin.y + resolvedSlot.y
      }, metrics.safe, 96, 30),
      branchAngle: resolvedSlot.branchAngle ?? fallbackAngle
    };
  };

  const getConstrainedClusterBranchAngle = (groupPoint, origin, branchAngle, metrics, dimensions) => {
    if (dimensions.showLabels || metrics.isCompact) return branchAngle;
    const edgePressure = Math.max(dimensions.size + 86, 124);
    const pointsLeft = groupPoint.x < origin.x;
    if (groupPoint.y - metrics.safe.top < edgePressure && Math.sin(toRadians(branchAngle)) < .2) {
      return pointsLeft ? 135 : 45;
    }
    if (metrics.safe.bottom - groupPoint.y < edgePressure && Math.sin(toRadians(branchAngle)) > -.2) {
      return pointsLeft ? -135 : -45;
    }
    return branchAngle;
  };

  const getNarrowGroupedClusterLayout = (categoryId, metrics, graphLayout, dimensions) => {
    const category = CATEGORIES[categoryId];
    const origin = graphLayout.categories[categoryId] || graphLayout.center;
    const groups = getCategoryGroups(categoryId);
    const itemOrder = new Map(category.items.map((item, index) => [item.id, index]));
    const entries = [];

    groups.forEach((group, groupIndex) => {
      const slot = getNarrowGroupClusterSlot(categoryId, group.id, groupIndex, groups.length, origin, metrics);
      const groupPoint = slot.point;
      const clusterHaloSize = getGroupClusterHaloSize(categoryId, group.id, group.items.length, dimensions);
      const branchAngle = getConstrainedClusterBranchAngle(groupPoint, origin, slot.branchAngle, metrics, dimensions);

      entries.push({
        type: 'group',
        id: getGroupKey(categoryId, group.id),
        groupId: group.id,
        label: group.label,
        items: group.items,
        x: groupPoint.x,
        y: groupPoint.y,
        halfWidth: getGroupLabelHalfWidth(group.label, dimensions),
        halfHeight: 20,
        haloWidth: Math.min(clusterHaloSize.width, metrics.isNarrow ? 196 : 244),
        haloHeight: Math.min(clusterHaloSize.height, metrics.isNarrow ? 146 : 178),
        index: entries.length
      });

      group.items.forEach((item, itemIndex) => {
        const clusteredPoint = getBranchingItemPoint(
          groupPoint,
          origin,
          itemIndex,
          group.items.length,
          metrics,
          dimensions,
          branchAngle
        );
        const point = clampPoint(
          clusteredPoint,
          metrics.safe,
          dimensions.size / 2,
          dimensions.size / 2
        );
        entries.push({
          type: 'item',
          item,
          groupId: group.id,
          x: point.x,
          y: point.y,
          halfWidth: dimensions.size / 2,
          halfHeight: dimensions.size / 2,
          clusterX: groupPoint.x,
          clusterY: groupPoint.y,
          clusterRadius: metrics.isNarrow ? 138 : 178,
          index: itemOrder.get(item.id) || 0
        });
      });
    });

    const resolvedEntries = resolveGroupedFanEntries(entries, categoryId, metrics, graphLayout);

    return {
      entries: resolvedEntries,
      positions: resolvedEntries.filter((entry) => entry.type === 'item'),
      nodeSize: dimensions.size,
      nodeWidth: dimensions.width,
      nodeHeight: dimensions.height,
      showLabels: false
    };
  };

  const getGroupedItemLayout = (categoryId, metrics, graphLayout, dimensions) => {
    const category = CATEGORIES[categoryId];
    if (!metrics.isNarrow) {
      return getGroupedFanItemLayout(categoryId, metrics, graphLayout, dimensions);
    }

    const origin = graphLayout.categories[categoryId] || graphLayout.center;
    const groups = getCategoryGroups(categoryId);
    const rowGap = metrics.height < 620 ? 7 : 10;
    const groupGap = metrics.height < 620 ? 12 : 18;
    const groupLabelHeight = 18;
    const maxColumnHeight = Math.max(260, metrics.safe.height - 24);
    const singleColumnHeight = getColumnHeight(groups, dimensions.height, rowGap, groupLabelHeight, groupGap);
    const columnCount = categoryId === 'projects' || singleColumnHeight > maxColumnHeight ? 2 : 1;
    const columns = splitGroupsIntoColumns(categoryId, groups, columnCount);
    const xPositions = getColumnXPositions(categoryId, columns.length, metrics, origin, dimensions.width);
    const entries = [];
    const itemOrder = new Map(category.items.map((item, index) => [item.id, index]));
    const obstacles = getGroupedColumnObstacles(categoryId, metrics, graphLayout);
    const columnHalfWidth = getGroupedColumnHalfWidth(dimensions.width);

    columns.forEach((columnGroups, columnIndex) => {
      const columnHeight = getColumnHeight(columnGroups, dimensions.height, rowGap, groupLabelHeight, groupGap);
      const x = xPositions[columnIndex] || origin.x;
      const yStart = getSafeGroupedColumnStart(
        x,
        origin.y - (columnHeight / 2),
        columnHeight,
        columnHalfWidth,
        metrics,
        obstacles
      );
      let cursorY = yStart;

      columnGroups.forEach((group, groupIndex) => {
        entries.push({
          type: 'group',
          id: getGroupKey(categoryId, group.id),
          groupId: group.id,
          label: group.label,
          items: group.items,
          x,
          y: cursorY,
          halfWidth: getGroupedColumnHalfWidth(dimensions.width),
          halfHeight: (groupLabelHeight / 2) + 4,
          index: entries.length
        });
        cursorY += groupLabelHeight + 16;

        group.items.forEach((item) => {
          const point = clampPoint({
            x,
            y: cursorY + (dimensions.height / 2)
          }, metrics.safe, dimensions.width / 2, dimensions.height / 2);
          entries.push({
            type: 'item',
            item,
            groupId: group.id,
            x: point.x,
            y: point.y,
            halfWidth: dimensions.width / 2,
            halfHeight: dimensions.height / 2,
            index: itemOrder.get(item.id) || 0
          });
          cursorY += dimensions.height + rowGap;
        });

        cursorY += groupGap;
      });
    });

    const resolvedEntries = resolveGroupedFanEntries(entries, categoryId, metrics, graphLayout);

    return {
      entries: resolvedEntries,
      positions: resolvedEntries.filter((entry) => entry.type === 'item'),
      nodeSize: dimensions.size,
      nodeWidth: dimensions.width,
      nodeHeight: dimensions.height,
      showLabels: true
    };
  };

  const getSelectedGroupIdForLayout = (categoryId, selectedKey) => {
    const groupSelection = parseGroupKey(selectedKey);
    if (groupSelection?.categoryId === categoryId) return groupSelection.groupId;

    const itemSelection = parseItemKey(selectedKey);
    if (itemSelection?.categoryId !== categoryId) return '';
    const group = getCategoryGroups(categoryId)
      .find((entry) => entry.items.some((item) => item.id === itemSelection.itemId));
    return group?.id || '';
  };

  const getDrawerNodeSlots = (count) => {
    const layouts = {
      1: [[50, 52]],
      2: [[30, 52], [70, 52]],
      3: [[18, 52], [50, 52], [82, 52]],
      4: [[29, 28], [71, 28], [29, 72], [71, 72]],
      5: [[25, 27], [75, 27], [16, 72], [50, 72], [84, 72]],
      6: [[16, 27], [50, 27], [84, 27], [16, 72], [50, 72], [84, 72]]
    };
    const slots = layouts[Math.min(6, Math.max(1, count))] || layouts[6];
    return Array.from({ length: count }, (_, index) => {
      const slot = slots[index] || [16 + ((index % 3) * 34), 27 + (Math.floor(index / 3) * 45)];
      return { x: slot[0], y: slot[1] };
    });
  };

  const getDesktopGroupDrawerLayout = (categoryId, metrics, graphLayout, dimensions, selectedKey) => {
    const groups = getCategoryGroups(categoryId);
    const selectedGroupId = getSelectedGroupIdForLayout(categoryId, selectedKey) || groups[0]?.id || '';
    const selectedGroup = groups.find((group) => group.id === selectedGroupId) || groups[0] || null;
    const selectedItemKey = parseItemKey(selectedKey);
    const selectedItemId = selectedItemKey?.categoryId === categoryId
      && selectedGroup?.items.some((item) => item.id === selectedItemKey.itemId)
      ? selectedItemKey.itemId
      : '';
    const drawerWidth = clamp(Math.round(metrics.safe.width * .37), 340, 420);
    const drawerHeight = clamp(Math.round(metrics.safe.height * .72), 330, 390);
    const drawerLeft = metrics.safe.right - drawerWidth - 18;
    const groupHalfWidth = clamp(Math.round(metrics.safe.width * .1), 92, 112);
    const groupHalfHeight = 29;
    const groupX = clamp(
      drawerLeft - groupHalfWidth - 52,
      metrics.safe.left + groupHalfWidth,
      metrics.safe.right - drawerWidth - groupHalfWidth - 32
    );
    const rowGap = groups.length >= 4 ? 72 : groups.length === 3 ? 92 : 116;
    const groupCenterY = metrics.safe.top + (metrics.safe.height / 2);
    const firstGroupY = groupCenterY - ((Math.max(0, groups.length - 1) * rowGap) / 2);
    const entries = groups.map((group, groupIndex) => {
      const point = clampPoint({
        x: groupX,
        y: firstGroupY + (groupIndex * rowGap)
      }, metrics.safe, groupHalfWidth, groupHalfHeight);
      return {
        type: 'group',
        id: getGroupKey(categoryId, group.id),
        groupId: group.id,
        label: group.label,
        items: group.items,
        x: point.x,
        y: point.y,
        halfWidth: groupHalfWidth,
        halfHeight: groupHalfHeight,
        index: groupIndex
      };
    });
    const selectedGroupEntry = entries.find((entry) => entry.groupId === selectedGroupId) || entries[0] || null;
    const drawerY = clamp(
      selectedGroupEntry?.y || groupCenterY,
      metrics.safe.top + (drawerHeight / 2),
      metrics.safe.bottom - (drawerHeight / 2)
    );
    const slots = getDrawerNodeSlots(selectedGroup?.items.length || 0);

    return {
      entries,
      positions: [],
      nodeSize: dimensions.size,
      nodeWidth: dimensions.width,
      nodeHeight: dimensions.height,
      showLabels: false,
      selectedGroupId,
      drawer: selectedGroup ? {
        id: `home-graph-drawer-${categoryId}`,
        categoryId,
        groupId: selectedGroup.id,
        label: selectedGroup.label,
        left: drawerLeft,
        top: drawerY - (drawerHeight / 2),
        x: drawerLeft + (drawerWidth / 2),
        y: drawerY,
        width: drawerWidth,
        height: drawerHeight,
        selectedItemId,
        items: selectedGroup.items.map((item, index) => ({
          item,
          index,
          x: slots[index]?.x || 50,
          y: slots[index]?.y || 50
        }))
      } : null
    };
  };

  const getCompactItemPositions = (selectedGroup, groupPoint, origin, metrics, dimensions, itemOrder) => {
    if (!selectedGroup) return [];

    const itemCount = selectedGroup.items.length;
    const nodeHalf = dimensions.size / 2;

    return selectedGroup.items.map((item, itemIndex) => {
      const point = clampPoint(
        getBranchingItemPoint(groupPoint, origin, itemIndex, itemCount, metrics, dimensions, 90),
        metrics.safe,
        nodeHalf,
        nodeHalf
      );

      return {
        type: 'item',
        item,
        groupId: selectedGroup.id,
        x: point.x,
        y: point.y,
        halfWidth: nodeHalf,
        halfHeight: nodeHalf,
        index: itemOrder.get(item.id) || 0
      };
    });
  };

  const getCompactGroupedItemLayout = (categoryId, metrics, graphLayout, dimensions, selectedKey) => {
    const category = CATEGORIES[categoryId];
    const origin = graphLayout.categories[categoryId] || graphLayout.center;
    const groups = getCategoryGroups(categoryId);
    const selectedGroupId = getSelectedGroupIdForLayout(categoryId, selectedKey);
    const selectedGroup = selectedGroupId
      ? groups.find((group) => group.id === selectedGroupId)
      : null;
    const itemOrder = new Map(category.items.map((item, index) => [item.id, index]));
    const groupHeight = 28;
    const groupHalfWidth = Math.min(86, Math.max(58, (metrics.safe.width / 2) - 16));
    const count = groups.length;
    const columnCount = Math.min(2, Math.max(1, Math.ceil(Math.sqrt(count * 1.15))));
    const leftX = metrics.safe.left + groupHalfWidth;
    const rightX = metrics.safe.right - groupHalfWidth;
    const centerX = metrics.safe.left + (metrics.safe.width / 2);
    const categoryBottom = Math.max(
      ...CATEGORY_ORDER.map((entry) => (graphLayout.categories[entry]?.y || 0) + metrics.categoryHalfHeight)
    );
    const firstGroupY = clamp(
      Math.max(
        categoryBottom + 42,
        graphLayout.center.y + metrics.centerHalfSize + 46,
        origin.y + metrics.categoryHalfHeight + 40
      ),
      metrics.safe.top + (groupHeight / 2),
      metrics.safe.bottom - (groupHeight / 2) - (selectedGroup ? 148 : 18)
    );
    const entries = [];
    const groupPoints = new Map();

    groups.forEach((group, groupIndex) => {
      const row = Math.floor(groupIndex / columnCount);
      const column = groupIndex % columnCount;
      const x = columnCount === 1 ? centerX : (column === 0 ? leftX : rightX);
      const y = clamp(
        firstGroupY + (row * 48),
        metrics.safe.top + (groupHeight / 2),
        metrics.safe.bottom - (groupHeight / 2) - (selectedGroup ? 128 : 0)
      );
      const point = clampPoint({ x, y }, metrics.safe, groupHalfWidth, groupHeight / 2);
      groupPoints.set(group.id, point);
      entries.push({
        type: 'group',
        id: getGroupKey(categoryId, group.id),
        groupId: group.id,
        label: group.label,
        items: group.items,
        x: point.x,
        y: point.y,
        halfWidth: groupHalfWidth,
        halfHeight: groupHeight / 2,
        index: entries.length
      });
    });

    if (selectedGroup) {
      const groupPoint = groupPoints.get(selectedGroup.id) || {
        x: centerX,
        y: firstGroupY
      };
      entries.push(...getCompactItemPositions(selectedGroup, groupPoint, origin, metrics, dimensions, itemOrder));
    }

    const resolvedEntries = resolveGroupedFanEntries(entries, categoryId, metrics, graphLayout);

    return {
      entries: resolvedEntries,
      positions: resolvedEntries.filter((entry) => entry.type === 'item'),
      nodeSize: dimensions.size,
      nodeWidth: dimensions.width,
      nodeHeight: dimensions.height,
      showLabels: false,
      selectedGroupId
    };
  };

  const getItemLayout = (categoryId, count, metrics, graphLayout, selectedKey = '') => {
    const dimensions = getItemNodeDimensions(metrics);
    const nodeSize = dimensions.size;
    const category = CATEGORIES[categoryId];
    if (!categoryId || !category || !count) {
      return {
        entries: [],
        positions: [],
        nodeSize,
        nodeWidth: dimensions.width,
        nodeHeight: dimensions.height,
        showLabels: false
      };
    }

    if (!metrics.isNarrow) {
      return getDesktopGroupDrawerLayout(categoryId, metrics, graphLayout, dimensions, selectedKey);
    }

    if (metrics.isCompact) {
      return getCompactGroupedItemLayout(categoryId, metrics, graphLayout, dimensions, selectedKey);
    }

    return getNarrowGroupedClusterLayout(categoryId, metrics, graphLayout, dimensions);
  };

  const computeGraphLayout = (map, activeCategoryId, selectedKey = '') => {
    const metrics = getGraphMetrics(map);
    const graphLayout = getGraphLayout(metrics, activeCategoryId);
    const activeItems = CATEGORIES[activeCategoryId]?.items || [];
    const itemLayout = getItemLayout(activeCategoryId, activeItems.length, metrics, graphLayout, selectedKey);
    return {
      metrics,
      center: graphLayout.center,
      categories: graphLayout.categories,
      items: itemLayout.entries,
      nodeSize: itemLayout.nodeSize,
      nodeWidth: itemLayout.nodeWidth,
      nodeHeight: itemLayout.nodeHeight,
      showLabels: itemLayout.showLabels,
      selectedGroupId: itemLayout.selectedGroupId || '',
      drawer: itemLayout.drawer || null
    };
  };

  const getRelatedItems = (categoryId, selectedId) => {
    const category = CATEGORIES[categoryId];
    const selected = category?.items.find(item => item.id === selectedId);
    if (!category || !selected) return [];

    const selectedTags = new Set((selected.tags || []).map(tag => tag.toLowerCase()));
    const selectedGroups = new Set(getItemGroupLabels(categoryId, selectedId));
    const selectedTokens = new Set(getItemSearchTokens(selected));

    return category.items
      .filter(item => item.id !== selectedId)
      .map((item, index) => {
        const itemTags = (item.tags || []).map(tag => tag.toLowerCase());
        const sharedTags = itemTags.filter(tag => selectedTags.has(tag)).length;
        const sharedGroups = getItemGroupLabels(categoryId, item.id)
          .filter(label => selectedGroups.has(label)).length;
        const sharedTokens = getItemSearchTokens(item)
          .filter(token => selectedTokens.has(token)).length;
        const score = (sharedGroups * 8) + (sharedTags * 5) + sharedTokens;
        return { item, index, score };
      })
      .sort((a, b) => (b.score - a.score) || (a.index - b.index))
      .slice(0, 3)
      .map(entry => entry.item);
  };

  const createDesktopGroupDrawerHtml = ({ categoryId, drawer, selectedKey }) => {
    if (!drawer?.items?.length) return '';
    const category = CATEGORIES[categoryId] || CATEGORIES.projects;
    const titleId = `${drawer.id}-title`;
    const synapses = drawer.items.map((entry) => {
      const selected = selectedKey === getItemKey(categoryId, entry.item.id);
      const controlX = Math.max(14, entry.x - 18);
      return `<path class="home-graph__drawer-synapse${selected ? ' is-active' : ''}" d="M 2 50 C 18 50, ${controlX} ${entry.y}, ${entry.x} ${entry.y}"></path>`;
    }).join('');
    const selectedEntry = drawer.items.find((entry) => selectedKey === getItemKey(categoryId, entry.item.id));
    const outputSynapse = selectedEntry
      ? `<path class="home-graph__drawer-synapse home-graph__drawer-synapse--output is-active" d="M ${selectedEntry.x} ${selectedEntry.y} C ${Math.min(88, selectedEntry.x + 18)} ${selectedEntry.y}, 88 50, 100 50"></path>`
      : '';
    const nodes = drawer.items.map((entry) => {
      const item = entry.item;
      const selected = selectedKey === getItemKey(categoryId, item.id);
      const title = item.fullTitle || item.title;
      return `
        <button type="button" class="home-graph__layer-node${selected ? ' is-selected' : ''}" data-graph-item="${escapeHtml(getItemKey(categoryId, item.id))}" data-category-id="${escapeHtml(categoryId)}" data-item-id="${escapeHtml(item.id)}" data-group-id="${escapeHtml(drawer.groupId)}" aria-label="${escapeHtml(title)}" aria-pressed="${selected ? 'true' : 'false'}" title="${escapeHtml(title)}" style="--node-x: ${entry.x}%; --node-y: ${entry.y}%; --accent: ${escapeHtml(category.accent)}; --node-index: ${entry.index};">
          <span class="home-graph__layer-soma" aria-hidden="true">${getItemIcon(item, categoryId)}</span>
          <span class="home-graph__layer-label">${escapeHtml(item.title)}</span>
        </button>
      `;
    }).join('');

    return `
      <section class="home-graph__group-drawer is-open" id="${escapeHtml(drawer.id)}" data-graph-drawer data-category-id="${escapeHtml(categoryId)}" data-group-id="${escapeHtml(drawer.groupId)}" aria-labelledby="${escapeHtml(titleId)}" style="--drawer-left: ${drawer.left}px; --drawer-top: ${drawer.top}px; --drawer-width: ${drawer.width}px; --drawer-height: ${drawer.height}px; --accent: ${escapeHtml(category.accent)};">
        <h3 class="home-graph__drawer-title" id="${escapeHtml(titleId)}">${escapeHtml(drawer.label)} layer</h3>
        <div class="home-graph__drawer-network">
          <svg class="home-graph__drawer-synapses" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            ${synapses}
            ${outputSynapse}
            <circle class="home-graph__drawer-input" cx="2" cy="50" r="1.5"></circle>
            <circle class="home-graph__drawer-output" cx="99" cy="50" r="1.5"></circle>
          </svg>
          ${nodes}
        </div>
      </section>
    `;
  };

  const createInspectorHtml = ({ categoryId, item, isOverview = false }) => {
    const category = CATEGORIES[categoryId] || CATEGORIES.projects;
    const contentType = categoryId === 'projects' ? 'project' : categoryId === 'tools' ? 'tool' : 'game';
    const title = isOverview ? category.label : (item.fullTitle || item.title);
    const summary = isOverview ? category.summary : item.summary;
    const tags = isOverview
      ? [`${category.items.length} items`, 'Interactive map', 'Library']
      : (item.tags || []);
    const href = isOverview ? category.href : item.href;
    const cta = isOverview
      ? `View all ${category.label.toLowerCase()}`
      : `Open ${category.singular}`;
    const related = isOverview ? category.items.slice(0, 3) : getRelatedItems(categoryId, item.id);
    const contentId = isOverview ? categoryId : item.id;
    const resourceType = isOverview ? 'library' : (contentType === 'project' ? 'case_study' : contentType);
    const closeButton = isOverview
      ? ''
      : '<button type="button" class="home-graph__inspector-close" data-graph-close aria-label="Show overview">x</button>';

    return `
      <div class="home-graph__inspector-kicker">
        <span>Preview</span>
        ${closeButton}
      </div>
      <div class="home-graph__inspector-head">
        <span class="home-graph__inspector-icon" style="--accent: ${escapeHtml(category.accent)}">${isOverview ? getCategoryIcon(category.icon) : getItemIcon(item, categoryId)}</span>
        <div>
          <h2 class="home-graph__inspector-title">${escapeHtml(title)}</h2>
          <div class="home-graph__inspector-type">
            <span class="home-graph__pill" style="--accent: ${escapeHtml(category.accent)}">${escapeHtml(category.label.slice(0, -1) || category.label)}</span>
            <span class="home-graph__pill" style="--accent: ${escapeHtml(category.accent)}">Connected</span>
          </div>
        </div>
      </div>
      <p class="home-graph__summary">${escapeHtml(summary)}</p>
      <div class="home-graph__inspector-section">
        <h3>Tags</h3>
        <ul class="home-graph__tag-list">
          ${tags.map(tag => `<li>${escapeHtml(tag)}</li>`).join('')}
        </ul>
      </div>
      <div class="home-graph__inspector-section home-graph__inspector-section--related">
        <h3>Related ${escapeHtml(category.label.toLowerCase())}</h3>
        <ul class="home-graph__related-list">
          ${related.map(entry => `<li><a href="${escapeHtml(entry.href)}" data-content-open="true" data-content-id="${escapeHtml(entry.id)}" data-content-type="${escapeHtml(contentType)}" data-resource-type="${escapeHtml(contentType === 'project' ? 'case_study' : contentType)}" data-source-surface="home_inspector_related">${escapeHtml(entry.title)} <span aria-hidden="true">></span></a></li>`).join('')}
        </ul>
      </div>
      <div class="home-graph__inspector-section">
        <a class="home-graph__cta" href="${escapeHtml(href)}" data-content-open="true" data-content-id="${escapeHtml(contentId)}" data-content-type="${escapeHtml(isOverview ? 'directory' : contentType)}" data-resource-type="${escapeHtml(resourceType)}" data-source-surface="home_inspector">${escapeHtml(cta)} ${getLinkIcon()}</a>
        <a class="home-graph__library-link" href="${escapeHtml(category.href)}">View in library</a>
      </div>
    `;
  };

  const createGroupInspectorHtml = ({ categoryId, group }) => {
    const category = CATEGORIES[categoryId] || CATEGORIES.projects;
    const items = group?.items || [];
    const tags = Array.from(new Set(items.flatMap((item) => item.tags || []))).slice(0, 6);
    const summary = `${group.label} groups ${items.length} ${category.singular}${items.length === 1 ? '' : 's'} within the ${category.label.toLowerCase()} branch, keeping the map organized before opening individual nodes.`;

    return `
      <div class="home-graph__inspector-kicker">
        <span>Subcategory</span>
        <button type="button" class="home-graph__inspector-close" data-graph-close aria-label="Show ${escapeHtml(category.label)} overview">x</button>
      </div>
      <div class="home-graph__inspector-head">
        <span class="home-graph__inspector-icon" style="--accent: ${escapeHtml(category.accent)}">${getCategoryIcon(category.icon)}</span>
        <div>
          <h2 class="home-graph__inspector-title">${escapeHtml(group.label)}</h2>
          <div class="home-graph__inspector-type">
            <span class="home-graph__pill" style="--accent: ${escapeHtml(category.accent)}">${items.length} nodes</span>
            <span class="home-graph__pill" style="--accent: ${escapeHtml(category.accent)}">${escapeHtml(category.label)}</span>
          </div>
        </div>
      </div>
      <p class="home-graph__summary">${escapeHtml(summary)}</p>
      <div class="home-graph__inspector-section">
        <h3>Signals</h3>
        <ul class="home-graph__tag-list">
          ${(tags.length ? tags : [group.label, category.label, 'Subcategory']).map(tag => `<li>${escapeHtml(tag)}</li>`).join('')}
        </ul>
      </div>
      <div class="home-graph__inspector-section home-graph__inspector-section--related">
        <h3>Nodes in this group</h3>
        <ul class="home-graph__related-list">
          ${items.map(entry => `<li><a href="${escapeHtml(entry.href)}">${escapeHtml(entry.title)} <span aria-hidden="true">></span></a></li>`).join('')}
        </ul>
      </div>
      <div class="home-graph__inspector-section">
        <a class="home-graph__cta" href="${escapeHtml(category.href)}" data-content-open="true" data-content-id="${escapeHtml(categoryId)}" data-content-type="directory" data-resource-type="library" data-source-surface="home_group_inspector">Open ${escapeHtml(category.label.toLowerCase())} library ${getLinkIcon()}</a>
      </div>
    `;
  };

  const createSiteOverviewHtml = () => {
    const totals = CATEGORY_ORDER.map((categoryId) => CATEGORIES[categoryId].items.length);
    const totalItems = totals.reduce((sum, count) => sum + count, 0);

    return `
      <div class="home-graph__inspector-kicker">
        <span>Preview</span>
      </div>
      <div class="home-graph__inspector-head">
        <span class="home-graph__inspector-icon" style="--accent: var(--home-graph-blue)">${getCategoryIcon('folder')}</span>
        <div>
          <h2 class="home-graph__inspector-title">Projects, Tools & Games</h2>
          <div class="home-graph__inspector-type">
            <span class="home-graph__pill" style="--accent: var(--home-graph-blue)">ML</span>
            <span class="home-graph__pill" style="--accent: var(--home-graph-blue)">Data</span>
          </div>
        </div>
      </div>
      <p class="home-graph__summary">Explore machine learning projects, practical browser tools, lightweight games, and experiments through an interactive graph. Select a branch or open a library directly.</p>
      <div class="home-graph__inspector-section">
        <h3>Library</h3>
        <ul class="home-graph__tag-list">
          <li>${totalItems} total nodes</li>
          <li>${CATEGORIES.projects.items.length} projects</li>
          <li>${CATEGORIES.tools.items.length} tools</li>
          <li>${CATEGORIES.games.items.length} games</li>
        </ul>
      </div>
      <div class="home-graph__inspector-section home-graph__inspector-section--related">
        <h3>Branches</h3>
        <ul class="home-graph__related-list">
          ${CATEGORY_ORDER.map((categoryId) => `<li><a href="${escapeHtml(CATEGORIES[categoryId].href)}">${escapeHtml(CATEGORIES[categoryId].label)} <span aria-hidden="true">></span></a></li>`).join('')}
        </ul>
      </div>
      <div class="home-graph__inspector-section">
        <a class="home-graph__cta" href="portfolio" data-content-open="true" data-content-id="projects" data-content-type="directory" data-resource-type="library" data-source-surface="home_overview">Open project library ${getLinkIcon()}</a>
        <a class="home-graph__library-link" href="tools">View tools</a>
        <a class="home-graph__library-link" href="games">Play games</a>
      </div>
    `;
  };

  const getMobileClusterSlot = (index, count) => {
    const patterns = {
      1: [{ x: 50, y: 22 }],
      2: [{ x: 26, y: 38 }, { x: 74, y: 62 }],
      3: [{ x: 50, y: 14 }, { x: 24, y: 70 }, { x: 76, y: 70 }],
      4: [{ x: 50, y: 13 }, { x: 24, y: 48 }, { x: 76, y: 48 }, { x: 50, y: 87 }],
      5: [{ x: 50, y: 13 }, { x: 24, y: 36 }, { x: 76, y: 36 }, { x: 30, y: 84 }, { x: 70, y: 84 }],
      6: [{ x: 50, y: 12 }, { x: 24, y: 34 }, { x: 76, y: 34 }, { x: 24, y: 70 }, { x: 76, y: 70 }, { x: 50, y: 88 }]
    };
    if (patterns[count]) return patterns[count][index];

    const angle = toRadians(-90 + ((360 / count) * index));
    return {
      x: 50 + (Math.cos(angle) * 36),
      y: 50 + (Math.sin(angle) * 34)
    };
  };

  const getMobileLinePath = (slot) => {
    const controlX = 50 + ((slot.x - 50) * .46);
    const controlY = 50 + ((slot.y - 50) * .18);
    return `M 50 50 C ${controlX.toFixed(1)} ${controlY.toFixed(1)}, ${(slot.x - ((slot.x - 50) * .22)).toFixed(1)} ${(slot.y - ((slot.y - 50) * .12)).toFixed(1)}, ${slot.x.toFixed(1)} ${slot.y.toFixed(1)}`;
  };

  const getMobileGroupIcon = (categoryId, groupId) => {
    const icons = {
      projects: {
        'ai-nlp': 'chat',
        'data-analytics': 'database',
        'computer-vision': 'shapes',
        'modeling-systems': 'cube'
      },
      tools: {
        'text-writing': 'diff',
        'campaign-link': 'link',
        media: 'image'
      },
      games: {
        'playable-systems': 'gamepad',
        simulations: 'wave'
      }
    };

    return getIcon(icons[categoryId]?.[groupId] || CATEGORIES[categoryId]?.icon || 'folder');
  };

  const getMobileMiniNetworkHtml = () => `
    <svg viewBox="0 0 88 64" aria-hidden="true">
      <path d="M17 44 37 20 56 42 72 17"></path>
      <path d="M17 44 56 42M37 20 72 17M37 20 49 32"></path>
      <circle cx="17" cy="44" r="4.8"></circle>
      <circle cx="37" cy="20" r="4.8"></circle>
      <circle cx="49" cy="32" r="4.8"></circle>
      <circle cx="56" cy="42" r="4.8"></circle>
      <circle cx="72" cy="17" r="4.8"></circle>
    </svg>
  `;

  const createMobileClassicDeckHtml = () => {
    const createMobileSection = (categoryId, index) => {
      const category = CATEGORIES[categoryId];
      const preferredGroup = getPreferredMobileGroup(categoryId);
      const items = (preferredGroup?.items?.length ? preferredGroup.items : category.items).slice(0, 3);
      const featured = items[0] || category.items[0];
      const sectionTitle = {
        projects: 'Project Examples',
        tools: 'Tools I Keep Refining',
        games: 'Games & Simulations'
      }[categoryId] || category.label;
      const sectionKicker = {
        projects: 'Featured work',
        tools: 'Browser utilities',
        games: 'Playable systems'
      }[categoryId] || category.singular;
      const sectionSummary = {
        projects: 'A quick look at machine learning, analytics, and interface-driven project work.',
        tools: 'Small utilities for text, campaign links, media, and repeated workflow cleanup.',
        games: 'Lightweight games and simulations built around systems, feedback loops, and interaction design.'
      }[categoryId] || category.summary;
      const sectionVisual = featured ? ITEM_VISUAL_BY_ID[featured.id] : '';
      const sectionStyle = [
        `--accent: ${escapeHtml(category.accent)}`,
        sectionVisual ? `--section-image: url('${escapeHtml(sectionVisual)}')` : ''
      ].filter(Boolean).join('; ');
      const sectionCta = {
        projects: 'View portfolio',
        tools: 'Open tools',
        games: 'Open games'
      }[categoryId] || `Browse ${category.label.toLowerCase()}`;
      const itemContentType = categoryId === 'projects' ? 'project' : categoryId === 'tools' ? 'tool' : 'game';

      return `
        <section class="home-graph__mobile-section home-graph__mobile-section--${escapeHtml(categoryId)}" style="${sectionStyle}" aria-labelledby="home-mobile-${escapeHtml(categoryId)}-title">
          <div class="home-graph__mobile-section-head">
            <span class="home-graph__mobile-section-icon" aria-hidden="true">${getCategoryIcon(category.icon)}</span>
            <div>
              <p>${escapeHtml(sectionKicker)}</p>
              <h2 id="home-mobile-${escapeHtml(categoryId)}-title">${escapeHtml(sectionTitle)}</h2>
            </div>
          </div>
          <p class="home-graph__mobile-section-summary">${escapeHtml(sectionSummary)}</p>
          <div class="home-graph__mobile-section-list">
            ${items.map((item, itemIndex) => {
              const title = item.fullTitle || item.title;
              const itemVisual = ITEM_VISUAL_BY_ID[item.id];
              const itemStyle = [
                `--card-index: ${index + itemIndex}`,
                itemVisual ? `--card-image: url('${escapeHtml(itemVisual)}')` : ''
              ].filter(Boolean).join('; ');
              return `
                <a class="home-graph__mobile-section-card" href="${escapeHtml(item.href)}" style="${itemStyle}" data-content-open="true" data-content-id="${escapeHtml(item.id)}" data-content-type="${escapeHtml(itemContentType)}" data-resource-type="${escapeHtml(itemContentType === 'project' ? 'case_study' : itemContentType)}" data-source-surface="home_mobile_card">
                  <span class="home-graph__mobile-section-art${itemVisual ? ' has-visual' : ''}" aria-hidden="true">${itemVisual ? '' : getItemIcon(item, categoryId)}</span>
                  <span class="home-graph__mobile-section-copy">
                    <strong>${escapeHtml(title)}</strong>
                    <span>${escapeHtml(item.snippet || item.summary || '')}</span>
                  </span>
                </a>
              `;
            }).join('')}
          </div>
          <a class="home-graph__mobile-section-cta" href="${escapeHtml(category.href)}" data-content-open="true" data-content-id="${escapeHtml(categoryId)}" data-content-type="directory" data-resource-type="library" data-source-surface="home_mobile_section">${escapeHtml(sectionCta)} ${getLinkIcon()}</a>
        </section>
      `;
    };

    return `
      <div class="home-graph__mobile-classic">
        <section class="home-graph__mobile-hero" aria-labelledby="home-mobile-hero-title">
          <div class="home-graph__mobile-hero-copy">
            <h2 id="home-mobile-hero-title">Daniel Short</h2>
            <div class="home-graph__mobile-hero-identity">
              <img src="img/hero/head-avatar-192.jpg" srcset="img/hero/head-avatar-192.jpg 192w, img/hero/head-avatar-384.jpg 384w" sizes="112px" alt="Portrait of Daniel Short" width="112" height="112" decoding="async" fetchpriority="high">
              <p>Builder of useful interfaces</p>
            </div>
            <p>I build machine learning projects, practical browser tools, and playful experiments that make complex ideas easier to use.</p>
            <div class="home-graph__mobile-hero-actions">
              <a href="portfolio">Projects</a>
              <a href="tools">Tools</a>
              <a href="games">Games</a>
            </div>
            <a class="home-graph__mobile-hero-scroll" href="#home-mobile-projects-title">Browse featured work ${getLinkIcon()}</a>
          </div>
        </section>
        ${CATEGORY_ORDER.map(createMobileSection).join('')}
      </div>
    `;
  };

  const createMobileDepthDeckHtml = ({
    categoryId = 'projects',
    group = null,
    groups = [],
    topic = null,
    topics = [],
    selectedItem = null
  } = {}) => {
    const category = CATEGORIES[categoryId] || CATEGORIES.projects;
    const resolvedGroups = groups.length ? groups : getCategoryGroups(categoryId);
    const resolvedGroup = group || getPreferredMobileGroup(categoryId) || resolvedGroups[0] || null;
    const resolvedTopics = topics.length ? topics : getMobileDepthTopics(categoryId, resolvedGroup);
    const resolvedTopic = topic || getPreferredMobileTopic(resolvedTopics);
    const clusterItems = (resolvedTopic?.items?.length
      ? resolvedTopic.items
      : resolvedGroup?.items?.length
        ? resolvedGroup.items
        : category.items
    ).filter(Boolean);
    const previewItem = selectedItem && clusterItems.some((item) => item.id === selectedItem.id)
      ? selectedItem
      : clusterItems[0] || category.items[0] || null;
    const previewTitle = previewItem ? previewItem.fullTitle || previewItem.title : category.label;
    const previewSummary = previewItem ? previewItem.summary || previewItem.snippet || '' : category.summary;
    const previewTags = (previewItem?.tags || [category.label, 'Interactive map']).slice(0, 4);
    const activeGroupLabel = resolvedGroup?.label || category.label;
    const activeTopicLabel = resolvedTopic?.label || activeGroupLabel;

    const categoryChips = CATEGORY_ORDER.map((entryId) => {
      const entry = CATEGORIES[entryId];
      const active = entryId === categoryId;
      return `
        <button type="button" class="home-graph__mobile-category-chip${active ? ' is-active' : ''}" data-mobile-category="${escapeHtml(entryId)}" aria-pressed="${active ? 'true' : 'false'}">
          ${getCategoryIcon(entry.icon)}
          <span>${escapeHtml(entry.label)}</span>
        </button>
      `;
    }).join('');

    const groupChips = resolvedGroups.map((entry) => {
      const active = resolvedGroup?.id === entry.id;
      return `
        <button type="button" class="home-graph__mobile-group-chip${active ? ' is-active' : ''}" data-mobile-group="${escapeHtml(entry.id)}" aria-pressed="${active ? 'true' : 'false'}">
          <span class="home-graph__mobile-group-icon" aria-hidden="true">${getMobileGroupIcon(categoryId, entry.id)}</span>
          <span class="home-graph__mobile-group-text">
            <strong>${escapeHtml(entry.label)}</strong>
            <small>${entry.items.length} ${escapeHtml(category.singular)}${entry.items.length === 1 ? '' : 's'}</small>
          </span>
        </button>
      `;
    }).join('');

    const topicChips = resolvedTopics.map((entry) => {
      const active = resolvedTopic?.id === entry.id;
      return `
        <button type="button" class="home-graph__mobile-topic-chip${active ? ' is-active' : ''}" data-mobile-topic="${escapeHtml(entry.id)}" aria-pressed="${active ? 'true' : 'false'}">
          <span>${entry.items.length}</span>
          ${escapeHtml(entry.label)}
        </button>
      `;
    }).join('');

    const clusterLines = clusterItems.map((item, itemIndex) => {
      const slot = getMobileClusterSlot(itemIndex, clusterItems.length);
      return `<path d="${escapeHtml(getMobileLinePath(slot))}"></path>`;
    }).join('');

    const clusterNodes = clusterItems.map((item, itemIndex) => {
      const slot = getMobileClusterSlot(itemIndex, clusterItems.length);
      const title = item.fullTitle || item.title;
      const selected = previewItem?.id === item.id;
      return `
        <button type="button" class="home-graph__mobile-project-node${selected ? ' is-selected' : ''}" data-mobile-item="${escapeHtml(item.id)}" aria-label="${escapeHtml(title)}" title="${escapeHtml(title)}" style="--node-x: ${slot.x}%; --node-y: ${slot.y}%; --node-index: ${itemIndex};">
          <span class="home-graph__mobile-project-icon" aria-hidden="true">${getItemIcon(item, categoryId)}</span>
          <span class="home-graph__mobile-project-title">${escapeHtml(item.title)}</span>
        </button>
      `;
    }).join('');

    return `
      <div class="home-graph__mobile-depth">
        <section class="home-graph__mobile-depth-card home-graph__mobile-depth-card--branch" aria-label="Select graph branch">
          <div class="home-graph__mobile-layer-head">
            <span class="home-graph__mobile-layer-icon" aria-hidden="true">${getCategoryIcon(category.icon)}</span>
            <span class="home-graph__mobile-depth-copy">
              <span>Branch</span>
              <strong>${escapeHtml(category.label)}</strong>
              <small>${escapeHtml(category.summary)}</small>
            </span>
          </div>
          <span class="home-graph__mobile-network-mini" aria-hidden="true">${getMobileMiniNetworkHtml()}</span>
          <div class="home-graph__mobile-category-switch" aria-label="Branches">${categoryChips}</div>
        </section>

        <section class="home-graph__mobile-depth-card home-graph__mobile-depth-card--group" aria-label="Select subcategory">
          <div class="home-graph__mobile-layer-head">
            <span class="home-graph__mobile-layer-icon" aria-hidden="true">${getMobileGroupIcon(categoryId, resolvedGroup?.id)}</span>
            <span class="home-graph__mobile-depth-copy">
              <span>Subcategory</span>
              <strong>${escapeHtml(activeGroupLabel)}</strong>
              <small>${resolvedGroup?.items?.length || 0} connected ${escapeHtml(category.singular)}${resolvedGroup?.items?.length === 1 ? '' : 's'}</small>
            </span>
          </div>
          <div class="home-graph__mobile-rail" aria-label="${escapeHtml(category.label)} subcategories">${groupChips}</div>
        </section>

        <section class="home-graph__mobile-depth-card home-graph__mobile-depth-card--final" aria-label="Explore selected topic">
          <div class="home-graph__mobile-final-head">
            <div class="home-graph__mobile-chain" aria-label="Selected path">
              <span>${escapeHtml(category.label)}</span>
              <b aria-hidden="true">/</b>
              <span>${escapeHtml(activeGroupLabel)}</span>
              <b aria-hidden="true">/</b>
              <span aria-current="step">${escapeHtml(activeTopicLabel)}</span>
            </div>
            <div class="home-graph__mobile-topic-list" aria-label="Topics">${topicChips}</div>
          </div>

          <div class="home-graph__mobile-cluster-card" aria-label="${escapeHtml(activeTopicLabel)} cluster">
            <svg class="home-graph__mobile-cluster-lines" viewBox="0 0 100 100" aria-hidden="true">${clusterLines}</svg>
            <button type="button" class="home-graph__mobile-hub" data-mobile-group="${escapeHtml(resolvedGroup?.id || '')}" aria-label="${escapeHtml(activeGroupLabel)} hub">
              <span class="home-graph__mobile-hub-icon" aria-hidden="true">${getMobileGroupIcon(categoryId, resolvedGroup?.id)}</span>
              <span>${escapeHtml(activeGroupLabel)}</span>
            </button>
            ${clusterNodes}
          </div>

          ${previewItem ? `
            <div class="home-graph__mobile-preview">
              <span class="home-graph__mobile-preview-kicker">Preview</span>
              <strong>${escapeHtml(previewTitle)}</strong>
              <p>${escapeHtml(previewSummary)}</p>
              <ul>
                ${previewTags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join('')}
              </ul>
              <a href="${escapeHtml(previewItem.href)}" data-content-open="true" data-content-id="${escapeHtml(previewItem.id)}" data-content-type="${escapeHtml(categoryId === 'projects' ? 'project' : categoryId === 'tools' ? 'tool' : 'game')}" data-resource-type="${escapeHtml(categoryId === 'projects' ? 'case_study' : categoryId === 'tools' ? 'tool' : 'game')}" data-source-surface="home_mobile_preview">Open ${escapeHtml(category.singular)} ${getLinkIcon()}</a>
            </div>
          ` : ''}
        </section>
      </div>
    `;
  };

  const initHomeGraph = () => {
    const root = $('[data-home-graph]');
    if (!root) return;

    const map = $('[data-graph-map]', root);
    const svg = $('[data-graph-svg]', root);
    const categoryHost = $('[data-graph-category-host]', root);
    const itemHost = $('[data-graph-item-host]', root);
    const tooltip = $('[data-graph-tooltip]', root);
    const inspector = $('[data-graph-inspector]', root);
    const status = $('[data-graph-status]', root);
    const center = $('[data-graph-center]', root);

    if (!map || !svg || !categoryHost || !itemHost || !inspector) return;

    const mobileDeck = document.createElement('div');
    mobileDeck.className = 'home-graph__mobile-deck';
    mobileDeck.dataset.graphMobileDeck = '';
    map.appendChild(mobileDeck);

    const state = {
      active: null,
      selectedKey: '',
      mobileGroupId: '',
      mobileTopicId: '',
      transitionTimer: null,
      lineTimer: null,
      latestLayout: null,
      mobileRenderKey: ''
    };

    const usesMobileDeck = () => window.matchMedia?.('(max-width: 768px)').matches || window.innerWidth <= 768;
    const usesDesktopDrawer = () => !usesMobileDeck() && !getGraphMetrics(map).isNarrow;
    const getCurrentMobileCategoryId = () => (state.active && CATEGORIES[state.active] ? state.active : 'projects');

    const getMobileDepth = (categoryId = state.active, itemId = '') => {
      const category = CATEGORIES[categoryId] || CATEGORIES.projects;
      const groups = getCategoryGroups(categoryId);
      const itemDepth = itemId ? getMobileItemDepth(categoryId, itemId) : null;
      const group = itemDepth?.group
        || groups.find((entry) => entry.id === state.mobileGroupId)
        || getPreferredMobileGroup(categoryId)
        || groups[0]
        || null;
      const topics = getMobileDepthTopics(categoryId, group);
      const topic = itemDepth?.topic
        || topics.find((entry) => entry.id === state.mobileTopicId)
        || getPreferredMobileTopic(topics)
        || null;
      const selectedItem = itemId
        ? category.items.find((entry) => entry.id === itemId) || null
        : null;

      return {
        category,
        groups,
        group,
        topics,
        topic,
        selectedItem
      };
    };

    const syncMobileDepth = (categoryId = state.active, itemId = '') => {
      const depth = getMobileDepth(categoryId, itemId);
      state.mobileGroupId = depth.group?.id || '';
      state.mobileTopicId = depth.topic?.id || '';
      return depth;
    };

    const setActiveAccent = () => {
      const category = CATEGORIES[state.active] || CATEGORIES.projects;
      root.style.setProperty('--active-accent', category.accent);
    };

    const updateOverviewState = () => {
      const overviewActive = !state.active;
      if (status) {
        status.textContent = overviewActive
          ? 'Daniel Short overview active'
          : `${CATEGORIES[state.active]?.label || 'Section'} active`;
      }
      if (!center) return;
      center.classList.toggle('is-active', overviewActive);
      center.setAttribute('aria-pressed', overviewActive ? 'true' : 'false');
      center.setAttribute(
        'aria-label',
        overviewActive ? 'Daniel Short overview active' : 'Show Daniel Short overview'
      );
    };

    const showTooltip = (item, node, options = {}) => {
      if (!tooltip || !node) return;
      if (window.matchMedia?.('(hover: hover)').matches && !options.force) return;
      const mapRect = map.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      const left = nodeRect.left + (nodeRect.width / 2) - mapRect.left;
      const top = nodeRect.top - mapRect.top - 66;
      tooltip.innerHTML = `<strong>${escapeHtml(item.fullTitle || item.title)}</strong>${escapeHtml(item.snippet || item.summary || '')}`;
      tooltip.style.left = `${clamp(left, 132, mapRect.width - 132)}px`;
      tooltip.style.top = `${Math.max(12, top)}px`;
      tooltip.classList.add('is-visible');
    };

    const hideTooltip = () => {
      if (tooltip) tooltip.classList.remove('is-visible');
    };

    const getItemFromNode = (node) => {
      const category = CATEGORIES[node.dataset.categoryId];
      const item = category?.items.find(entry => entry.id === node.dataset.itemId);
      return item ? { category, item } : null;
    };

    const getItemFromDot = (dot) => {
      const categoryId = dot?.dataset?.graphDotCategory;
      const itemId = dot?.dataset?.graphDotItem;
      const category = CATEGORIES[categoryId];
      const item = category?.items.find(entry => entry.id === itemId);
      return item ? { categoryId, category, item } : null;
    };

    const restoreInspectorPreview = () => {
      const groupSelection = parseGroupKey(state.selectedKey);
      if (groupSelection) {
        const group = getCategoryGroupById(groupSelection.categoryId, groupSelection.groupId);
        const category = CATEGORIES[groupSelection.categoryId];
        if (group && category) {
          inspector.style.setProperty('--accent', category.accent);
          inspector.innerHTML = createGroupInspectorHtml({ categoryId: groupSelection.categoryId, group });
        }
        return;
      }

      if (state.selectedKey) {
        const [categoryId, itemId] = state.selectedKey.split(':');
        const category = CATEGORIES[categoryId];
        const item = category?.items.find(entry => entry.id === itemId);
        if (item) {
          inspector.style.setProperty('--accent', category.accent);
          inspector.innerHTML = createInspectorHtml({ categoryId, item });
        }
        return;
      }

      selectOverview(state.active);
    };

    const selectInspectorGroup = (categoryId, groupId, options = {}) => {
      const category = CATEGORIES[categoryId] || CATEGORIES.projects;
      const group = getCategoryGroupById(categoryId, groupId) || getCategoryGroups(categoryId)[0];
      if (!group) return;
      const nextSelectedKey = getGroupKey(categoryId, group.id);
      const shouldUpdateLayout = Boolean(
        options.updateLayout
        && state.latestLayout?.metrics?.isCompact
        && state.selectedKey !== nextSelectedKey
      );
      state.active = categoryId;
      state.selectedKey = nextSelectedKey;
      state.mobileGroupId = group.id;
      state.mobileTopicId = getMobileDepthTopics(categoryId, group)[0]?.id || '';
      if (shouldUpdateLayout) {
        render();
        if (options.focusGroup) {
          requestAnimationFrame(() => {
            const groupButton = $(`[data-graph-group][data-group-id="${group.id}"]`, itemHost);
            groupButton?.focus?.({ preventScroll: true });
          });
        } else if (options.focusInspector) {
          const heading = $('.home-graph__inspector-title', inspector);
          if (heading) heading.setAttribute('tabindex', '-1');
          heading?.focus?.({ preventScroll: true });
        }
        return;
      }
      inspector.classList.remove('is-closed');
      inspector.classList.remove('is-overview');
      inspector.style.setProperty('--accent', category.accent);
      inspector.innerHTML = createGroupInspectorHtml({ categoryId, group });
      $$('[data-graph-item]', root).forEach((node) => {
        node.classList.remove('is-selected');
      });
      $$('[data-graph-group]', root).forEach((node) => {
        node.classList.toggle('is-selected', node.dataset.graphGroup === state.selectedKey);
      });
      if (options.focusGroup) {
        const groupButton = $(`[data-graph-group][data-group-id="${group.id}"]`, itemHost);
        groupButton?.focus?.({ preventScroll: true });
      } else if (options.focusInspector) {
        const heading = $('.home-graph__inspector-title', inspector);
        if (heading) heading.setAttribute('tabindex', '-1');
        heading?.focus?.({ preventScroll: true });
      }
    };

    const selectDesktopDrawerGroup = (categoryId, groupId, options = {}) => {
      const group = getCategoryGroupById(categoryId, groupId) || getCategoryGroups(categoryId)[0];
      const item = group?.items?.[0];
      if (!group || !item) return;
      state.active = categoryId;
      state.selectedKey = getItemKey(categoryId, item.id);
      syncMobileDepth(categoryId, item.id);
      render();
      if (options.focusDrawer) {
        requestAnimationFrame(() => {
          const node = $(`[data-graph-drawer] [data-item-id="${item.id}"]`, itemHost);
          node?.focus?.({ preventScroll: true });
        });
      }
    };

    const selectInspectorItem = (categoryId, itemId, options = {}) => {
      const category = CATEGORIES[categoryId] || CATEGORIES.projects;
      const item = category.items.find(entry => entry.id === itemId) || category.items[0];
      state.active = categoryId;
      state.selectedKey = getItemKey(categoryId, item.id);
      syncMobileDepth(categoryId, item.id);
      if (options.updateLayout) {
        render();
        if (options.focusItem) {
          requestAnimationFrame(() => {
            const node = $(`[data-graph-drawer] [data-item-id="${item.id}"]`, itemHost);
            node?.focus?.({ preventScroll: true });
          });
        }
        return;
      }
      inspector.classList.remove('is-closed');
      inspector.classList.remove('is-overview');
      inspector.style.setProperty('--accent', category.accent);
      inspector.innerHTML = createInspectorHtml({ categoryId, item });
      $$('[data-graph-item]', root).forEach((node) => {
        node.classList.toggle('is-selected', node.dataset.graphItem === state.selectedKey);
      });
      $$('[data-graph-group]', root).forEach((node) => {
        node.classList.toggle(
          'is-selected',
          node.dataset.groupId === getSelectedGroupIdForLayout(categoryId, state.selectedKey)
        );
      });
      if (options.focusInspector) {
        const heading = $('.home-graph__inspector-title', inspector);
        if (heading) heading.setAttribute('tabindex', '-1');
        heading?.focus?.({ preventScroll: true });
      }
    };

    const selectOverview = (categoryId = state.active) => {
      if (!categoryId || !CATEGORIES[categoryId]) {
        state.selectedKey = '';
        inspector.classList.remove('is-closed');
        inspector.classList.add('is-overview');
        inspector.style.setProperty('--accent', 'var(--home-graph-blue)');
        inspector.innerHTML = createSiteOverviewHtml();
        return;
      }
      const category = CATEGORIES[categoryId] || CATEGORIES.projects;
      state.selectedKey = '';
      $$('[data-graph-item]', root).forEach((node) => {
        node.classList.remove('is-selected');
      });
      $$('[data-graph-group]', root).forEach((node) => {
        node.classList.remove('is-selected');
      });
      inspector.classList.remove('is-closed');
      inspector.classList.add('is-overview');
      inspector.style.setProperty('--accent', category.accent);
      inspector.innerHTML = createInspectorHtml({ categoryId, item: category.items[0], isOverview: true });
    };

    const drawLines = () => {
      const layout = state.latestLayout;
      const mapRect = map.getBoundingClientRect();
      if (!mapRect.width || !mapRect.height || !layout) return;
      svg.setAttribute('viewBox', `0 0 ${mapRect.width} ${mapRect.height}`);

      const makeConnectorPath = (from, to, tension = .42) => {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        return `M ${from.x} ${from.y} C ${from.x + (dx * tension)} ${from.y + (dy * .08)}, ${to.x - (dx * tension)} ${to.y - (dy * .08)}, ${to.x} ${to.y}`;
      };

      const getEdgeAnchorPoint = (from, to, halfWidth, halfHeight, gap = 3) => {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        if (!dx && !dy) return from;
        const scaleX = dx ? (halfWidth + gap) / Math.abs(dx) : Infinity;
        const scaleY = dy ? (halfHeight + gap) / Math.abs(dy) : Infinity;
        const scale = Math.min(scaleX, scaleY, 1);
        return {
          x: from.x + (dx * scale),
          y: from.y + (dy * scale)
        };
      };

      const makeEdgeConnectorPath = (from, to, fromBox, toBox, tension = .42) => makeConnectorPath(
        getEdgeAnchorPoint(from, to, fromBox.halfWidth, fromBox.halfHeight),
        getEdgeAnchorPoint(to, from, toBox.halfWidth, toBox.halfHeight),
        tension
      );

      const getCategoryLineBox = (categoryId) => {
        const supporting = Boolean(state.active) && categoryId !== state.active;
        return {
          halfWidth: layout.metrics.categoryHalfWidth * (supporting ? .42 : 1),
          halfHeight: layout.metrics.categoryHalfHeight * (supporting ? .72 : 1)
        };
      };

      const centerBox = {
        halfWidth: layout.metrics.centerHalfSize,
        halfHeight: layout.metrics.centerHalfSize
      };

      const centerPoint = layout.center;
      if (!centerPoint) return;

      const lineSpecs = [];
      Object.keys(CATEGORIES).forEach((categoryId) => {
        const category = CATEGORIES[categoryId];
        const categoryPoint = layout.categories[categoryId];
        if (!categoryPoint) return;
        lineSpecs.push({
          key: `category:${categoryId}`,
          className: 'home-graph__line',
          d: makeEdgeConnectorPath(centerPoint, categoryPoint, centerBox, getCategoryLineBox(categoryId), .42),
          color: category.accent,
          opacity: state.active && categoryId !== state.active ? '.12' : '.5'
        });
      });

      const activeCategoryPoint = layout.categories[state.active];
      if (state.active && activeCategoryPoint) {
        const groupPoints = new Map();
        layout.items.filter(entry => entry.type === 'group').forEach((entry) => {
          const groupPoint = { x: entry.x, y: entry.y };
          const selected = entry.groupId === layout.selectedGroupId;
          const groupBox = {
            halfWidth: entry.halfWidth || 86,
            halfHeight: entry.halfHeight || 16
          };
          groupPoints.set(entry.groupId, groupPoint);
          lineSpecs.push({
            key: `group:${state.active}:${entry.groupId}`,
            className: `home-graph__line home-graph__line--group${selected ? ' is-active' : ''}`,
            d: makeEdgeConnectorPath(activeCategoryPoint, groupPoint, getCategoryLineBox(state.active), groupBox, .48),
            color: CATEGORIES[state.active]?.accent || 'var(--home-graph-blue)',
            opacity: selected ? '.78' : '.28'
          });
        });

        if (layout.drawer) {
          const drawerGroup = layout.items.find((entry) => (
            entry.type === 'group' && entry.groupId === layout.drawer.groupId
          ));
          const drawerGroupPoint = groupPoints.get(layout.drawer.groupId);
          if (drawerGroup && drawerGroupPoint) {
            lineSpecs.push({
              key: `drawer:${state.active}:${layout.drawer.groupId}`,
              className: 'home-graph__line home-graph__line--drawer is-active',
              d: makeEdgeConnectorPath(
                drawerGroupPoint,
                { x: layout.drawer.left, y: layout.drawer.y },
                { halfWidth: drawerGroup.halfWidth || 106, halfHeight: drawerGroup.halfHeight || 29 },
                { halfWidth: 0, halfHeight: 0 },
                .5
              ),
              color: CATEGORIES[state.active]?.accent || 'var(--home-graph-blue)',
              opacity: '.86'
            });
          }
        }

        layout.items.filter(entry => entry.type === 'item').forEach((entry) => {
          const fromPoint = groupPoints.get(entry.groupId) || activeCategoryPoint;
          const itemPoint = { x: entry.x, y: entry.y };
          const fromBox = groupPoints.has(entry.groupId)
            ? {
              halfWidth: (layout.items.find(candidate => candidate.type === 'group' && candidate.groupId === entry.groupId)?.halfWidth) || 86,
              halfHeight: (layout.items.find(candidate => candidate.type === 'group' && candidate.groupId === entry.groupId)?.halfHeight) || 16
            }
            : getCategoryLineBox(state.active);
          const itemBox = {
            halfWidth: entry.halfWidth || (layout.nodeWidth / 2),
            halfHeight: entry.halfHeight || (layout.nodeHeight / 2)
          };
          lineSpecs.push({
            key: `item:${state.active}:${entry.item.id}`,
            className: 'home-graph__line home-graph__line--item',
            d: makeEdgeConnectorPath(fromPoint, itemPoint, fromBox, itemBox, .5),
            color: CATEGORIES[state.active]?.accent || 'var(--home-graph-blue)',
            opacity: '.38'
          });
        });
      }

      const existing = new Map(
        $$('[data-graph-line-key]', svg).map(path => [path.dataset.graphLineKey, path])
      );
      const used = new Set();

      lineSpecs.forEach((spec, index) => {
        let path = existing.get(spec.key);
        if (!path) {
          path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path.dataset.graphLineKey = spec.key;
        }
        used.add(spec.key);
        path.setAttribute('class', spec.className);
        if (path.getAttribute('d') !== spec.d) path.setAttribute('d', spec.d);
        path.style.setProperty('--line-color', spec.color);
        path.style.setProperty('--line-opacity', spec.opacity);
        if (svg.children[index] !== path) {
          svg.insertBefore(path, svg.children[index] || null);
        }
      });

      existing.forEach((path, key) => {
        if (!used.has(key)) path.remove();
      });
    };

    const scheduleLineDraw = () => {
      clearTimeout(state.lineTimer);
      requestAnimationFrame(drawLines);
    };

    const renderCategories = (layout) => {
      Object.entries(CATEGORIES).forEach(([categoryId, category]) => {
        const pos = layout.categories[categoryId];
        const active = categoryId === state.active;
        const supporting = Boolean(state.active) && !active;
        let categoryEl = $(`[data-graph-category="${categoryId}"]`, categoryHost);
        if (!categoryEl) {
          categoryEl = document.createElement('div');
          categoryEl.dataset.graphCategory = categoryId;
          categoryHost.appendChild(categoryEl);
        }

        categoryEl.className = `home-graph__category${active ? ' is-active' : supporting ? ' is-supporting' : ' is-idle'}`;
        categoryEl.style.setProperty('--x', `${pos.x}px`);
        categoryEl.style.setProperty('--y', `${pos.y}px`);
        categoryEl.style.setProperty('--accent', category.accent);
        categoryEl.innerHTML = `
            <div class="home-graph__halo" aria-hidden="true">${buildCategoryDots(categoryId)}</div>
            <div class="home-graph__category-card">
              <button type="button" class="home-graph__category-button" data-graph-category-button="${escapeHtml(categoryId)}" aria-label="${active ? 'Collapse' : 'Expand'} ${escapeHtml(category.label)}" aria-expanded="${active}" aria-controls="home-graph-layer-host" title="${escapeHtml(category.label)}"></button>
              <span class="home-graph__category-icon" aria-hidden="true">${getCategoryIcon(category.icon)}</span>
              <span class="home-graph__category-copy">
                <span class="home-graph__category-title">${escapeHtml(category.label)}</span>
                <span class="home-graph__category-count">${category.items.length} items</span>
              </span>
              <a class="home-graph__category-link" href="${escapeHtml(category.href)}" aria-label="Open ${escapeHtml(category.label)} page" title="Open ${escapeHtml(category.label)} page">${getLinkIcon()}</a>
              <span class="home-graph__category-toggle" aria-hidden="true">${getToggleIcon(active)}</span>
            </div>
        `;
      });
    };

    const renderItems = (layout) => {
      const category = CATEGORIES[state.active];
      if (!category) {
        itemHost.innerHTML = '';
        return;
      }

      const entriesHtml = layout.items.map((pos) => {
        const entry = pos;
        if (entry.type === 'group') {
          const selected = entry.groupId === layout.selectedGroupId;
          const groupTitle = `${entry.label}: ${entry.items.length} ${category.singular}${entry.items.length === 1 ? '' : 's'}`;
          const drawerId = selected && layout.drawer ? layout.drawer.id : '';
          const disclosureAttribute = layout.drawer || layout.metrics.isCompact
            ? ` aria-expanded="${selected}"`
            : '';
          const groupIcon = layout.drawer
            ? `<span class="home-graph__group-icon" aria-hidden="true">${getMobileGroupIcon(state.active, entry.groupId)}</span>`
            : '';
          return `
            <button type="button" class="home-graph__group-label${selected ? ' is-selected' : ''}" data-graph-group="${escapeHtml(entry.id)}" data-category-id="${escapeHtml(state.active)}" data-group-id="${escapeHtml(entry.groupId)}" aria-label="${escapeHtml(groupTitle)}"${disclosureAttribute}${drawerId ? ` aria-controls="${escapeHtml(drawerId)}"` : ''} title="${escapeHtml(groupTitle)}" style="--x: ${pos.x}px; --y: ${pos.y}px; --accent: ${escapeHtml(category.accent)}; --node-index: ${entry.index}; --cluster-width: ${entry.haloWidth || 276}px; --cluster-height: ${entry.haloHeight || 204}px;">
              ${groupIcon}
              <span>${escapeHtml(entry.label)}</span>
              <span class="home-graph__group-count">${entry.items.length}</span>
            </button>
          `;
        }

        const item = entry.item;
        const selected = state.selectedKey === getItemKey(state.active, item.id);
        const title = item.fullTitle || item.title;
        const snippet = item.snippet || item.summary || '';
        return `
          <button type="button" class="home-graph__node${selected ? ' is-selected' : ''}${layout.showLabels ? ' has-label' : ''}" data-graph-item="${escapeHtml(getItemKey(state.active, item.id))}" data-category-id="${escapeHtml(state.active)}" data-item-id="${escapeHtml(item.id)}" data-group-id="${escapeHtml(entry.groupId || '')}" aria-label="${escapeHtml(title)}" title="${escapeHtml(title)}" style="--x: ${pos.x}px; --y: ${pos.y}px; --accent: ${escapeHtml(category.accent)}; --node-index: ${entry.index}; --node-size: ${layout.nodeSize}px; --node-width: ${layout.nodeWidth}px; --node-height: ${layout.nodeHeight}px;">
            <span class="home-graph__node-title">${escapeHtml(item.title)}</span>
            <span class="home-graph__node-icon" aria-hidden="true">${getItemIcon(item, state.active)}</span>
            <span class="home-graph__node-preview" aria-hidden="true"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(snippet)}</span></span>
          </button>
        `;
      }).join('');
      const drawerHtml = layout.drawer
        ? createDesktopGroupDrawerHtml({ categoryId: state.active, drawer: layout.drawer, selectedKey: state.selectedKey })
        : '';
      itemHost.innerHTML = `${entriesHtml}${drawerHtml}`;
    };

    const renderMobileDeck = (options = {}) => {
      const renderKey = 'classic';
      if (!options.force && state.mobileRenderKey === renderKey) return;
      state.mobileRenderKey = renderKey;
      mobileDeck.style.removeProperty('--accent');
      mobileDeck.innerHTML = createMobileClassicDeckHtml();
    };

    const render = () => {
      setActiveAccent();
      if (state.active) {
        root.dataset.graphActive = state.active;
      } else {
        delete root.dataset.graphActive;
      }
      updateOverviewState();
      const layout = computeGraphLayout(map, state.active, state.selectedKey);
      state.latestLayout = layout;
      if (layout.drawer) {
        root.dataset.graphLayout = 'drawer';
      } else {
        delete root.dataset.graphLayout;
      }
      if (center) {
        center.style.setProperty('--center-x', `${layout.center.x}px`);
        center.style.setProperty('--center-y', `${layout.center.y}px`);
      }
      renderCategories(layout);
      renderItems(layout);
      renderMobileDeck();
      const category = CATEGORIES[state.active];
      if (!category) {
        state.selectedKey = '';
        selectOverview(null);
        scheduleLineDraw();
        return;
      }
      const groupSelection = parseGroupKey(state.selectedKey);
      if (groupSelection && groupSelection.categoryId === state.active) {
        selectInspectorGroup(groupSelection.categoryId, groupSelection.groupId);
      } else if (state.selectedKey.startsWith(`${state.active}:`)) {
        const selectedId = state.selectedKey.split(':')[1];
        selectInspectorItem(state.active, selectedId);
      } else {
        selectOverview(state.active);
      }
      scheduleLineDraw();
    };

    const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    const focusCategoryControl = (categoryId) => {
      if (!categoryId) return;
      requestAnimationFrame(() => {
        const button = $(`[data-graph-category-button="${categoryId}"]`, categoryHost);
        button?.focus?.({ preventScroll: true });
      });
    };

    const renderAfterCondense = (callback, options = {}) => {
      hideTooltip();
      clearTimeout(state.transitionTimer);
      itemHost.classList.add('is-condensing');
      svg.classList.add('is-condensing');
      state.transitionTimer = window.setTimeout(() => {
        callback();
        itemHost.classList.remove('is-condensing');
        svg.classList.remove('is-condensing');
        render();
        focusCategoryControl(options.focusCategoryId);
      }, prefersReducedMotion() ? 0 : 150);
    };

    const collapseGraph = (options = {}) => {
      if (!state.active) {
        selectOverview(null);
        focusCategoryControl(options.focusCategoryId);
        return;
      }

      renderAfterCondense(() => {
        state.active = null;
        state.selectedKey = '';
      }, options);
    };

    const selectCategory = (categoryId, itemId = null, options = {}) => {
      if (!CATEGORIES[categoryId]) return;
      if (categoryId === state.active) {
        if (itemId) {
          state.selectedKey = getItemKey(categoryId, itemId);
          syncMobileDepth(categoryId, itemId);
          render();
        } else if (usesMobileDeck()) {
          state.selectedKey = '';
          syncMobileDepth(categoryId);
          render();
        } else {
          collapseGraph(options);
        }
        return;
      }

      const preferredItemId = itemId || (
        usesDesktopDrawer()
          ? getCategoryGroups(categoryId)[0]?.items?.[0]?.id || null
          : null
      );
      if (usesMobileDeck()) {
        state.active = categoryId;
        state.selectedKey = preferredItemId ? getItemKey(categoryId, preferredItemId) : '';
        syncMobileDepth(categoryId, preferredItemId || '');
        render();
        return;
      }

      renderAfterCondense(() => {
        state.active = categoryId;
        state.selectedKey = preferredItemId ? getItemKey(categoryId, preferredItemId) : '';
        syncMobileDepth(categoryId, preferredItemId || '');
      }, options);
    };

    const selectMobileGroup = (groupId) => {
      const categoryId = getCurrentMobileCategoryId();
      const group = getCategoryGroupById(categoryId, groupId);
      if (!group) return;
      state.active = categoryId;
      state.mobileGroupId = group.id;
      state.mobileTopicId = getMobileDepthTopics(categoryId, group)[0]?.id || '';
      state.selectedKey = getGroupKey(categoryId, group.id);
      render();
    };

    const selectMobileTopic = (topicId) => {
      const categoryId = getCurrentMobileCategoryId();
      const group = getCategoryGroupById(categoryId, state.mobileGroupId) || getPreferredMobileGroup(categoryId);
      const topic = getMobileDepthTopics(categoryId, group)
        .find((entry) => entry.id === topicId);
      if (!group || !topic) return;
      state.active = categoryId;
      state.mobileGroupId = group.id;
      state.mobileTopicId = topic.id;
      state.selectedKey = getGroupKey(categoryId, group.id);
      render();
    };

    categoryHost.addEventListener('click', (event) => {
      const link = event.target.closest('.home-graph__category-link');
      if (link) return;
      const dot = event.target.closest('[data-graph-dot-item]');
      if (dot) {
        const entry = getItemFromDot(dot);
        if (entry) selectCategory(entry.categoryId, entry.item.id);
        return;
      }
      const button = event.target.closest('[data-graph-category-button]');
      if (button) {
        const categoryId = button.dataset.graphCategoryButton;
        selectCategory(categoryId, null, {
          focusCategoryId: event.detail === 0 ? categoryId : ''
        });
        return;
      }
      const categoryEl = event.target.closest('[data-graph-category]');
      if (!categoryEl) return;
      selectCategory(categoryEl.dataset.graphCategory);
    });

    const handleDotPreviewEnter = (event) => {
      const dot = event.target.closest('[data-graph-dot-item]');
      if (!dot) return;
      const entry = getItemFromDot(dot);
      if (!entry) return;
      showTooltip(entry.item, dot, { force: true });
    };

    const handleDotPreviewLeave = (event) => {
      if (event.relatedTarget && event.currentTarget.contains(event.relatedTarget)) return;
      hideTooltip();
      restoreInspectorPreview();
    };

    categoryHost.addEventListener('pointerover', handleDotPreviewEnter);
    categoryHost.addEventListener('mouseover', handleDotPreviewEnter);
    categoryHost.addEventListener('pointermove', handleDotPreviewEnter);
    categoryHost.addEventListener('mousemove', handleDotPreviewEnter);
    categoryHost.addEventListener('pointerout', handleDotPreviewLeave);
    categoryHost.addEventListener('mouseout', handleDotPreviewLeave);

    categoryHost.addEventListener('focusin', (event) => {
      const dot = event.target.closest('[data-graph-dot-item]');
      if (!dot) return;
      const entry = getItemFromDot(dot);
      if (!entry) return;
      showTooltip(entry.item, dot, { force: true });
    });

    categoryHost.addEventListener('focusout', () => {
      hideTooltip();
      restoreInspectorPreview();
    });

    itemHost.addEventListener('click', (event) => {
      const group = event.target.closest('[data-graph-group]');
      if (group) {
        if (usesDesktopDrawer()) {
          selectDesktopDrawerGroup(group.dataset.categoryId, group.dataset.groupId, {
            focusDrawer: event.detail === 0
          });
          return;
        }
        selectInspectorGroup(group.dataset.categoryId, group.dataset.groupId, {
          focusInspector: window.innerWidth <= 940,
          focusGroup: event.detail === 0,
          updateLayout: true
        });
        return;
      }
      const node = event.target.closest('[data-graph-item]');
      if (!node) return;
      selectInspectorItem(node.dataset.categoryId, node.dataset.itemId, {
        focusInspector: window.innerWidth <= 940,
        updateLayout: usesDesktopDrawer(),
        focusItem: event.detail === 0
      });
    });

    mobileDeck.addEventListener('click', (event) => {
      event.stopPropagation();
      const categoryButton = event.target.closest('[data-mobile-category]');
      if (categoryButton) {
        selectCategory(categoryButton.dataset.mobileCategory);
        return;
      }

      const groupButton = event.target.closest('[data-mobile-group]');
      if (groupButton) {
        selectMobileGroup(groupButton.dataset.mobileGroup);
        return;
      }

      const topicButton = event.target.closest('[data-mobile-topic]');
      if (topicButton) {
        selectMobileTopic(topicButton.dataset.mobileTopic);
        return;
      }

      const itemButton = event.target.closest('[data-mobile-item]');
      if (itemButton) {
        selectInspectorItem(getCurrentMobileCategoryId(), itemButton.dataset.mobileItem);
        renderMobileDeck({ force: true });
      }
    });

    const handleItemPreviewEnter = (event) => {
      const node = event.target.closest('[data-graph-item]');
      if (!node) return;
      const entry = getItemFromNode(node);
      if (entry) {
        showTooltip(entry.item, node);
      }
    };

    const handleItemPreviewLeave = (event) => {
      if (event.relatedTarget && event.currentTarget.contains(event.relatedTarget)) return;
      hideTooltip();
      restoreInspectorPreview();
    };

    itemHost.addEventListener('pointerover', handleItemPreviewEnter);
    itemHost.addEventListener('mouseover', handleItemPreviewEnter);
    itemHost.addEventListener('pointerout', handleItemPreviewLeave);
    itemHost.addEventListener('mouseout', handleItemPreviewLeave);

    itemHost.addEventListener('focusin', (event) => {
      const node = event.target.closest('[data-graph-item]');
      if (!node) return;
      const entry = getItemFromNode(node);
      if (entry) {
        showTooltip(entry.item, node);
      }
    });

    itemHost.addEventListener('focusout', () => {
      hideTooltip();
      restoreInspectorPreview();
    });

    inspector.addEventListener('click', (event) => {
      const close = event.target.closest('[data-graph-close]');
      if (!close) return;
      const returnCategoryId = state.active;
      selectOverview(state.active);
      if (state.latestLayout?.drawer) render();
      focusCategoryControl(returnCategoryId);
    });

    map.addEventListener('click', (event) => {
      const interactive = event.target.closest?.('[data-graph-category], [data-graph-group], [data-graph-item], [data-graph-center], [data-graph-tooltip], [data-graph-mobile-deck], [data-graph-drawer]');
      if (interactive) return;
      collapseGraph();
    });

    center?.addEventListener('click', () => collapseGraph());
    center?.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      collapseGraph();
    });

    window.addEventListener('resize', () => requestAnimationFrame(render));
    if ('ResizeObserver' in window) {
      const observer = new ResizeObserver(() => requestAnimationFrame(render));
      observer.observe(map);
    }

    render();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHomeGraph, { once: true });
  } else {
    initHomeGraph();
  }
})();
