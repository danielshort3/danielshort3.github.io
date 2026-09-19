# Native feature coverage

The Android screens use Compose, Kotlin, Canvas, Android media APIs, and JSON data. They do not embed the website or execute downloaded JavaScript. These are native adaptations; the larger games intentionally start with core gameplay, as requested.

## Games

| Game | Included native behavior | Deliberate differences from the website |
| --- | --- | --- |
| Double-Zero Roulette | Local play chips, bets, payout and recent results | No real money; no website account or shared history |
| Stellar Dogfight | Drag movement, auto targeting, three enemy movement roles, ten waves, Overdrive, persistent weapon and hull upgrades | One ship and compact campaign; no complete web campaign, ship roster, or effects pipeline |
| Ocean Wave Simulation | Native animated wave surface, drag view, wind/wave/light controls, quiet/storm presets | Canvas rendering with six wave components; no WebGL, FFT ocean, Ultra graphics, or website camera/environment parity |
| Project Starfall | Three authored classes, platforms/gravity/jump, directional attacks and skill cooldown, enemy combat, XP/levels, gear drops/equipment, health recovery and local save | One expedition field with repeated increasing waves; no complete world, dialogue, full quest tree, full item catalog, or web combat/animation system |
| Probability Engine | 3×3 grid, 15 authored symbols, neighboring synergies/consumption, deck editing, packs, luck/motor upgrades, jackpot recovery, optional auto spins | Compact deck/economy adaptation; not the complete web progression or presentation |
| Stormbreak | Three zones, boss waves, Zeus attacks/abilities, upgrades, XP, gold, ambrosia and capped offline gold | Core combat progression with a 200-wave bound; no full web skill/tree/event systems or ambrosia spending; simplified native scene |

Game frames stop when the screen leaves the foreground or the user navigates away. Checkpoints are local and independent of website saves. Stormbreak offline gold is an estimate capped at four hours, granted when loading a saved game with auto attack enabled; an explicitly paused checkpoint does not earn it. It does not simulate combat or grant offline XP. Probability auto spins restart off after loading. Reduced motion suppresses extra flashes and starts the ocean paused; core gameplay movement remains available.

## Project demos

| Project | Native behavior and data source | Network / limitations |
| --- | --- | --- |
| Handwriting Rating | Black drawing canvas, ten selectable handwriting samples, submission to the existing handwriting model, all returned scores | AWS inference requires connectivity; drawings are sent only on Rate digit |
| Shape Classifier | Matching black drawing canvas and existing five-shape classifier | AWS inference requires connectivity; no shape autofill |
| Synthetic Digit Generator | Default 6×6 image grid, digit selector/Regenerate, advanced seed/grid/latent settings beneath output | Existing AWS VAE generates actual images; unavailable backend is reported, with no fabricated fallback |
| Smart Sentence Retriever | Query/top-count controls and ranked corpus sentences | Existing AWS ranking service |
| Grand Junction Travel Chat | Native question input, asynchronous request/polling, selectable answer and source URLs | Existing Bedrock service; a single-question interface rather than the full website chat UI |
| Nonogram Solver | Actual trained-agent steps, replay/pause and solution view with prediction accuracy | Existing AWS service; replay pauses in background; this is an agent demonstration rather than a separate native puzzle game |
| Baby Name Explorer | Search, gender, rating/recommendation and minimum-score filtering | Published historical JSON, cached after loading; personal preference scores, not population forecasts |
| COVID-19 Outbreak Drivers | Date/state selection, model-risk/driver values and historical risk chart | Published historical JSON, cached after loading; not live surveillance or medical guidance |
| Retail Sales & Loss | Sales/incidents period filters, metric charts, inventory and package summaries | Published anonymized historical JSON, cached after loading; region totals are shown only for all years |
| Empty-Package Dashboard | Year/location filters and department/location/condition/month aggregation | Published anonymized historical JSON, cached after loading; estimated package value is not proven theft |
| Pizza Tips Regression Modeling | Native cost/city/housing/hour inputs, tip and independent tip-rate estimates with selectable intervals | Fully offline; exact saved version 4 model coefficients/transforms/error widths. Native city selection replaces polygon lookup. No map/heatmap |
| Pizza Delivery Dashboard | City/housing/month filters, delivery/average-tip/order/duration metrics and native charts | Bundled snapshot of all 1,251 records from the published Tableau workbook; not a Tableau WebView |
| UFO Sightings Dashboard | Year/state/shape filters and count charts; 2013 default matches the published 6,334 reports | Bundled grouped contiguous-US reports through May 2014. Counts preserve underlying report totals and zero-count periods; no Tableau map or web tooltip implementation |

The standalone Pizza Tips Regression Modeling demo is active in the website catalog and is implemented. The temporarily hidden **Tip Prediction tab inside the Tableau dashboard stays hidden**. The Delivery Tip case study has no separate interactive demo; Minesweeper remains outside the published native catalog.

## Update boundaries

Public catalog text, images and links refresh through the deployed website feed. The four JSON-backed analytical demos fetch the website's published dataset files when opened and can reuse their cached copies offline. Model-backed demos call the current website services. Native behavior, bundled game assets, the saved tip-model coefficients and the two extracted Tableau snapshots require a new APK when changed. A website JavaScript/CSS change does not automatically update a native implementation.

No website account/session or game-save synchronization is implemented. Public resource files and credentials still open the appropriate Android browser or document app. The app never labels those external resources as native experiences.

## Review scope

The source review checked native routing eligibility, authored data/formulas, persistence/lifecycle, bounds, tooltips/labels and basic responsive structure. It added regression coverage for model parity, replay correctness, filtered aggregations, adjacency, checkpoint motion/cooldowns and finite progression. Device results and the final APK build are recorded separately in `QA.md`; source review alone is not a claim that every remote AWS request or every hardware configuration was tested.
