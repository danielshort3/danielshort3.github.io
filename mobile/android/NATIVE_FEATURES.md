# Native feature coverage

The app uses the website for the published case studies and most interactive experiences. All 16 published project cards open their first-party case studies in an in-app WebView, with bookmark/share controls and a retained native offline summary. Eleven first-party demos and five games open their website routes in the app, with the existing native adaptations available as alternatives. Nine public tools open the website in the Android browser so imports, downloads, and website account flows work there; their Android versions remain available. Screen Recorder stays native. Probability Engine opens in the browser with a native alternative because its website file import/export is not supported by the in-app WebView. The alternatives below use Compose, Kotlin, Canvas, Android media APIs, and JSON data; changing website JavaScript does not change them.

## Project case studies

Opening a published project card shows its canonical website case study in the app. The app retains its own saved-project bookmark and Android share action on that screen. The **Offline summary** action opens the native text, preview, resources, and demo entry where available. The native summary comes from the cached public catalog; the website case study needs connectivity for a reliable first load. Their content can differ until both the website page and app feed are deployed and refreshed.

## Games

Project Starfall, Stellar Dogfight, Double-Zero Roulette, Stormbreak, and Ocean Wave Simulation use their deployed website games in-app by default. The table describes their retained native alternatives. Probability Engine opens its website game in the Android browser by default and offers its native adaptation.

| Game | Included native behavior | Deliberate differences from the website |
| --- | --- | --- |
| Double-Zero Roulette | Local play chips, bets, payout and recent results | No real money; no website account or shared history |
| Stellar Dogfight | Drag movement, auto targeting, three enemy movement roles, ten waves, Overdrive, persistent weapon and hull upgrades | One ship and compact campaign; no complete web campaign, ship roster, or effects pipeline |
| Ocean Wave Simulation | Native animated wave surface, drag view, wind/wave/light controls, quiet/storm presets | Canvas rendering with six wave components; no WebGL, FFT ocean, Ultra graphics, or website camera/environment parity |
| Project Starfall | The retained offline native expedition has three authored classes, platforms/gravity/jump, directional attacks and skill cooldown, enemy combat, XP/levels, gear drops/equipment, health recovery and a local save | The native expedition is one field with repeated increasing waves; it does not have the website's complete world, dialogue, quest tree, item catalog, or combat/animation system. Its checkpoint is separate from the web game's save |
| Probability Engine | 3×3 grid, 15 authored symbols, neighboring synergies/consumption, deck editing, packs, luck/motor upgrades, jackpot recovery, optional auto spins | Compact deck/economy adaptation; not the complete web progression or presentation |
| Stormbreak | Three zones, boss waves, Zeus attacks/abilities, upgrades, XP, gold, ambrosia and capped offline gold | Core combat progression with a 200-wave bound; no full web skill/tree/event systems or ambrosia spending; simplified native scene |

Native game frames stop when the screen leaves the foreground or the user navigates away. Their checkpoints are local and independent of website saves. Opening a website game does not convert or erase native progress. In-app website game saves use app WebView storage; Probability Engine's website save uses the Android browser's storage. Both are separate from native checkpoints and a regular browser on another device. Stormbreak offline gold is an estimate capped at four hours, granted when loading a saved native game with auto attack enabled; an explicitly paused checkpoint does not earn it. It does not simulate combat or grant offline XP. Native Probability auto spins restart off after loading. Reduced motion suppresses extra flashes and starts the native ocean paused; core gameplay movement remains available.

## Project demos

Handwriting Rating, Shape Classifier, Synthetic Digit Generator, Smart Sentence Retriever, Grand Junction Travel Chat, Nonogram Solver, Baby Name Explorer, COVID-19 Outbreak Drivers, Retail Sales & Loss, Empty-Package Dashboard, and Pizza Tips Regression Modeling open their website demos in-app by default. The table describes retained native alternatives and the two native Tableau dashboard adaptations.

| Project | Retained native behavior and data source | Network / limitations |
| --- | --- | --- |
| Handwriting Rating | Black drawing canvas, ten selectable handwriting samples, submission to the existing handwriting model, all returned scores | AWS inference requires connectivity; drawings are sent only on Rate digit |
| Shape Classifier | Matching black drawing canvas and existing five-shape classifier | AWS inference requires connectivity; no shape autofill |
| Synthetic Digit Generator | Default 6×6 image grid, digit selector/Regenerate, advanced seed/grid/latent settings beneath output | Existing AWS VAE generates actual images; unavailable backend is reported, with no fabricated fallback |
| Smart Sentence Retriever | Query/top-count interface and ranked corpus sentences | Website and native search both require connectivity and the existing AWS ranking service |
| Grand Junction Travel Chat | Question input, asynchronous request/polling, and selectable answer/source URLs | Existing Bedrock service requires connectivity; the native fallback is a single-question interface rather than the full website chat UI |
| Nonogram Solver | Trained-agent steps, replay/pause, solution, and prediction accuracy | Existing AWS service requires connectivity; native replay pauses in background and is an agent demonstration rather than a separate native puzzle game |
| Baby Name Explorer | Search, gender, rating/recommendation and minimum-score filtering | Published historical JSON, cached after loading; personal preference scores, not population forecasts |
| COVID-19 Outbreak Drivers | Date/state selection, model-risk/driver values and historical risk chart | Published historical JSON, cached after loading; not live surveillance or medical guidance |
| Retail Sales & Loss | Sales/incidents period filters, metric charts, inventory and package summaries | Published anonymized historical JSON, cached after loading; region totals are shown only for all years |
| Empty-Package Dashboard | Year/location filters and department/location/condition/month aggregation | Published anonymized historical JSON, cached after loading; estimated package value is not proven theft |
| Pizza Tips Regression Modeling | Cost/city/housing/hour inputs, tip and independent tip-rate estimates, and selectable intervals | The website and map tiles need connectivity for a reliable first load. The native model works fully offline with saved version 4 coefficients/transforms/error widths; native city selection replaces polygon lookup and has no map/heatmap |
| Pizza Delivery Dashboard | City/housing/month filters, delivery/average-tip/order/duration metrics and native charts | Bundled snapshot of all 1,251 records from the published Tableau workbook; not a Tableau WebView |
| UFO Sightings Dashboard | Year/state/shape filters and count charts; 2013 default matches the published 6,334 reports | Bundled grouped contiguous-US reports through May 2014. Counts preserve underlying report totals and zero-count periods; no Tableau map or web tooltip implementation |

The standalone Pizza Tips Regression Modeling demo is active in the website catalog and is implemented. The temporarily hidden **Tip Prediction tab inside the Tableau dashboard stays hidden**. The Delivery Tip case study has no separate interactive demo; Minesweeper remains outside the published native catalog. The 11 website demos use app WebView storage, separate from their native adaptations and a regular browser's website storage; no demo-state or account transfer between those stores is implemented.

## Tools

Image Optimizer, Text Compare, UTM Batch Builder, Background Remover, Non-breaking Space Cleaner, QR Code Generator, Oxford Comma Checker, Point of View Checker, and Word Frequency Analyzer open their canonical website pages in the Android browser by default. Each catalog card also offers **Open Android version**. The website has import, export, settings, and account workflows that the compact native adaptations do not all share. Browser downloads and website sign-in stay in the browser; native tool inputs and output do not transfer into it. Screen Recorder opens the native Android recorder because Android screen capture uses MediaProjection consent. Its clips remain in app-private storage until exported or deleted.

## Update boundaries

Public catalog text, images and links refresh through the deployed website feed. After an app version with these routes is installed, the 16 website case studies, 11 in-app demos, five in-app games, nine browser tools, and browser Probability Engine load deployed website HTML, CSS, JavaScript, and assets. Website changes reach those routes after website deployment; routing changes require a new APK. Their first load needs connectivity, and website caching does not guarantee complete offline behavior. The four JSON-backed native analytical demos fetch the website's published dataset files when opened and can reuse their cached copies offline. Other native model-backed demos call the current website services. Native behavior, bundled game assets, the saved tip-model coefficients, and the two extracted Tableau snapshots require a new APK when changed. A website JavaScript/CSS change does not automatically update a native adaptation.

No website account/session or game-save synchronization is implemented. Browser and in-app WebView storage are separate, as are native bookmarks and checkpoints. Public resource files and credentials still open the appropriate Android browser or document app. The app never labels those external resources as native experiences.

## Review scope

The source review checked native routing eligibility, authored data/formulas, persistence/lifecycle, bounds, tooltips/labels and basic responsive structure. It added regression coverage for model parity, replay correctness, filtered aggregations, adjacency, checkpoint motion/cooldowns and finite progression. Device results and the final APK build are recorded separately in `QA.md`; source review alone is not a claim that every remote AWS request or every hardware configuration was tested.
