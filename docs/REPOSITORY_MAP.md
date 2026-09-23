# Repository map

Use this as a task index: find the feature, open its source, and choose the relevant checks. It is deliberately not a complete file inventory. Paths are relative to this file; directory links identify where to find the matching feature file.

## Start here

1. Read [AGENTS.md](../AGENTS.md) and check `git status --short` before editing; preserve unrelated work.
2. Find the task below. For an unfamiliar URL, look up its rewrite in [vercel.json](../vercel.json), then check whether its destination is generated before changing it.
3. Edit the authoritative content, renderer, or runtime; rebuild generated output rather than patching it.
4. Run the relevant checks and inspect the actual desktop/mobile flow for UI changes. A local build does not prove a deployment succeeded.

## Website: where to make changes

| Task | Authoritative entry points | Related behavior / styling |
| --- | --- | --- |
| About copy, brand tagline, interests, experience, education, credentials | [personal audience content](../content/audiences/personal.json) | `brandTagline` flows through the generated audience config into the closed-home welcome; [CMS widgets](../api/_lib/cms-widgets.js): `renderHomeBackground` renders the resume-style timeline; [About CSS](../css/components/home-about.css), [background CSS](../css/components/home-background.css) |
| Homepage tabs, fully closed state, transitions | [category-accordion.js](../js/home/category-accordion.js) | [accordion CSS](../css/components/home-category-accordion.css); shared navigation/frame sources below |
| Shared frame size, scrolling, mobile rails, navigation | [site-frame.js](../js/navigation/site-frame.js), [site-frame-policy.js](../js/navigation/site-frame-policy.js), [site-route-runtime.js](../js/navigation/site-route-runtime.js) | [page-transitions.js](../js/navigation/page-transitions.js), [frame CSS](../css/components/site-frame.css), [mobile dock CSS](../css/components/mobile-site-dock.css) |
| DS header, navigation, footer, breadcrumbs | [site content](../content/site/), [CMS renderers](../build/lib/cms-renderers.js): `renderHeader` / `renderFooter` | [header-breadcrumbs.js](../js/navigation/header-breadcrumbs.js); generated partials are consumed by [inject-header.js](../build/inject-header.js) and [inject-footer.js](../build/inject-footer.js) |
| Personal mobile header and bottom section navigation | [navigation.js](../js/navigation/navigation.js): `setupMobileSectionNavigation` | [mobile shell CSS](../css/components/mobile-site-dock.css), [frame layout](../js/navigation/site-frame.js). Full bars hide on downward page scrolling and return on upward scrolling; nested tool scrolling and protected interactions do not hide them. [Browser checks](../tests/site/mobile-scroll-chrome.browser.cjs) |
| Consistent project/tool/game headings, buttons, divider lines | [personal-accordion-shell.js](../build/lib/personal-accordion-shell.js), [project page generator](../build/generate-project-pages.js) | [masthead CSS](../css/components/page-masthead.css), [shell CSS](../css/components/personal-accordion-shell.css), [project CSS](../css/components/project-page.css) |
| Featured homepage cards and library copy/order | [personal audience content](../content/audiences/personal.json), [tool content](../content/tools/), [project content](../content/projects/), [games catalog](../content/pages/games.json) | [CMS widgets](../api/_lib/cms-widgets.js), [CMS renderers](../build/lib/cms-renderers.js), [home library CSS](../css/components/home-library.css); catalog JavaScript is generated |
| Project copy, preview image, links, visibility, embed URL | Matching JSON in [content/projects](../content/projects/) | [generate-project-pages.js](../build/generate-project-pages.js) owns demo-first case-study markup, next-project links, and the project question action; [site-frame.js](../js/navigation/site-frame.js) places that action outside the desktop scrolling viewport; [project CSS](../css/components/project-page.css) |
| Tool inputs, controls, results | Authored tool body in [pages](../pages/), corresponding module in [js/tools](../js/tools/), tool metadata in [content/tools](../content/tools/) | Matching [component CSS](../css/components/); preserve generated header/shell regions in tool HTML |
| Text Compare | [text-compare.html](../pages/text-compare.html), [text-compare.js](../js/tools/text-compare.js) | [diff core](../js/tools/text-compare-core.js), [worker](../js/tools/text-compare-worker.js), [text-compare.css](../css/components/text-compare.css) |
| UTM Batch Builder | [src/utm-batch-builder](../src/utm-batch-builder/) | [build-utm-batch-builder.js](../build/build-utm-batch-builder.js) bundles this React tool; its output under `js/tools/` is not the source |
| Digit generator, handwriting, shape classifier, other project demos | Authored [demos](../demos/) HTML and [js/demos](../js/demos/) clients | [demo theme](../css/components/project-demo-theme.css), [demo layout](../css/components/project-demo-layout.css), [compact layout](../css/components/project-demo-compact-layout.css); isolated wrappers are generated separately |
| Isolated demo URL / wrapper | [project-demo-routes.js](../build/lib/project-demo-routes.js), [generate-project-demo-wrappers.js](../build/generate-project-demo-wrappers.js) | [project-demo-wrapper.js](../js/navigation/project-demo-wrapper.js), [vercel.json](../vercel.json); generated `pages/demos/` wraps the authored raw demo |
| Tableau desktop/mobile embeds | [Pizza project](../content/projects/pizzaDashboard.json), [UFO project](../content/projects/ufoDashboard.json) | [common.js](../js/common/common.js) owns embed loading/device selection; [project CSS](../css/components/project-page.css) owns geometry. Workbook authoring: [Tableau guide](../design/tableau/README.md) |
| Games and simulation behavior | [pages/games](../pages/games/), [js/games](../js/games/), [games catalog](../content/pages/games.json) | Ocean is an exception: [ocean-wave-simulation.html](../pages/ocean-wave-simulation.html) and `ocean-wave-*.js` in [js/tools](../js/tools/). Starfall: [design/implementation docs](project-starfall/README.md) |
| Contact cards, map, message form | [contact content](../content/pages/contact.json), [personal audience content](../content/audiences/personal.json), [CMS widgets](../api/_lib/cms-widgets.js) | [contact-map.js](../js/common/contact-map.js) owns lazy loading/preservation; [contact.js](../js/forms/contact.js) owns the form; [contact API](../api/contact.js) |
| Account buttons, login, autosave, history | [tools-account-ui.js](../js/accounts/tools-account-ui.js), [tools-auth.js](../js/accounts/tools-auth.js), [tools-state.js](../js/accounts/tools-state.js), [tools-config.js](../js/accounts/tools-config.js) | [tools API router](../api/tools/), [endpoint handlers](../api/_lib/tools-endpoints/); [local sign-in guide](tools-local-sign-in.md). Preserve per-tool capture/restore hooks and account isolation |
| Model inference / AWS demo connection | [aws-client.js](../js/demos/aws-client.js), [demo API router](../api/demos/), [demo-proxy.js](../api/_lib/demo-proxy.js) | Service implementations live under [aws](../aws/); consult the service's own README. Some demos instead use bundled browser data/models |
| Privacy, consent, analytics | Authored [privacy page body](../pages/privacy.html), [js/privacy](../js/privacy/), [js/analytics](../js/analytics/) | Root `privacy.html` is a mirror. Keep consent behavior and actual data handling aligned with the copy |
| Session-only guest/contact draft recovery | [session-drafts.js](../js/common/session-drafts.js), [tools-account-ui.js](../js/accounts/tools-account-ui.js), [contact.js](../js/forms/contact.js) | Shared bounded session store; tools serialize only eligible fields, account drafts remain isolated. [Draft notice CSS](../css/components/draft-recovery.css) |
| Contact delivery deadlines and retry | [contact API](../api/contact.js), [contact form](../js/forms/contact.js) | 20s upstream/25s browser full-response deadline; explicit retry, confirmed-success-only draft clearing |
| Offline resource limits | [sw.js](../sw.js) | Owned document/core/media caches plus metadata, expiry/LRU/byte limits; API exclusions and natural worker upgrades |
| Visitor performance measurement | [web-vitals.js](../js/analytics/web-vitals.js), [ga4-events.js](../js/analytics/ga4-events.js), [GTM export generator](../build/gtm/generate-activity-container.js) | Consent-gated, document-entry route attribution; [reporting instructions](../build/gtm/README.md). Publication is separate |
| Release accessibility, screenshots and performance | [Playwright config](../tests/release/playwright.config.cjs), [deterministic fixtures](../tests/release/fixtures.cjs), [Lighthouse runner](../build/measure-lighthouse.cjs) | [Release checks and manual screen-reader checklist](RELEASE_QUALITY.md); reviewed baselines under `tests/release/baselines/` and `tests/site/baselines/` |
| Colors, typography, icons, illustrations | [variables.css](../css/variables.css), [component CSS](../css/components/), [catalog-icons.js](../js/common/catalog-icons.js), [img](../img/) | [visual style guide](visual-style.md). Choose the relevant `styles-*.css` entry and check [route-component-styles.json](../build/route-component-styles.json); not every component belongs in the global bundle |
| Brand reference PDF | [guide copy](brand-guide.json), [PDF renderer](../build/generate-brand-guide.py) | [maintenance instructions](brand-guide.md); [capture script](../build/capture-brand-guide.cjs) refreshes real examples in `docs/brand-guide-assets/`; output is `documents/brand_guide.pdf`. This is separate from the website build. |
| Logo variants, favicon, personal social artwork | [brand asset index](../img/brand/README.md), [brand asset authoring](../build/generate-brand-assets.py), [favicon generator](../build/resize_logo.js) | The approved master stays in `img/brand/00-ds-logo-master-full-color.svg`. `npm run build:icons` derives the small optical SVG and raster/ICO exports. Brand authoring is an optional separate step; the website build publishes checked-in outputs. Default share-image metadata lives in `content/site/settings.json`. |
| Other audience sites and resumes | [content/audiences](../content/audiences/), [content/resumes](../content/resumes/) | [CMS renderer](../build/lib/cms-renderers.js), [CMS generator](../build/generate-cms-artifacts.js); shared renderer changes can affect more than the personal homepage |
| Search, metadata, social previews, clean URLs, CSP | [generate-search-index.js](../build/generate-search-index.js), [inject-head-metadata.js](../build/inject-head-metadata.js), [seo-routing.js](../build/lib/seo-routing.js), [vercel.json](../vercel.json) | [site settings](../content/site/settings.json), [generate-ai-digests.js](../build/generate-ai-digests.js), [SEO validator](../build/validate-seo.js) |
| Stale styles after an update / offline cache | [sw.js](../sw.js), [service-worker-register.js](../js/common/service-worker-register.js), [inject-head-metadata.js](../build/inject-head-metadata.js) | Route component CSS uses content-versioned URLs to avoid mixing fresh HTML with old cached styles; route manifests must retain the version query |

### Project Starfall asset ownership

Start with the [asset guide](project-starfall/ASSET_GENERATION_GUIDE.md), [prompt templates](../asset-sources/project-starfall/prompts/README.md), and [illustrated-v1 migration report](project-starfall/ASSET_OVERHAUL_V1.md). The [manifest](../asset-sources/project-starfall/asset-generation-manifest.json) records export contracts; [animation data](../js/games/project-starfall/data/animations.js) owns playback timing and holds.

Production masters and provenance live under [overhaul-v1](../asset-sources/project-starfall/overhaul-v1/): `players`, `enemies`, `fx`, `scenery`, and `icons`. Corresponding writers are [player importer](../build/process-project-starfall-overhaul-players.js), [enemy importer](../build/process-project-starfall-overhaul-enemies.js), [FX importer](../build/process-project-starfall-overhaul-fx.js) plus [native combat FX generator](../build/generate-project-starfall-combat-fx.js), [scenery importer](../build/process-project-starfall-overhaul-scenery.js), and [icon/UI importer](../build/project-starfall-overhaul-icons.js). [Equipment illustration masters](../build/project-starfall-equipment-illustrations.js) feed the existing [equipment generator](../build/generate-project-starfall-equipment-atlases.js). Legacy commands defer to these owners; older source atlases are retained history.

Outputs remain under `img/project-starfall/`, with the enemy inventory explicitly mapping retired compact sheets to their illustrated replacements. Normal website builds copy outputs into `public/` and derive optimized screen alternatives. [Coverage auditing](../build/audit-project-starfall-overhaul.js) compares the immutable active-asset baseline and protected session images. Focused checks: `npm.cmd run validate:project-starfall-assets`, `npm.cmd run test:starfall:full`, `node build/verify-project-starfall-overhaul-icons.js`, and `node build/audit-project-starfall-overhaul.js --complete`.

Local animation comparisons are generated by [the art review builder](../build/generate-project-starfall-overhaul-review.js) and [the alignment review builder](../build/generate-project-starfall-alignment-review.js), using the shared [review template](../build/templates/starfall-overhaul-review.template.html). Their output under `output/` is review evidence, outside deployment. After reviewed enemy imports, [refresh the integrity report](../build/refresh-project-starfall-enemy-validation.js) to reconcile production hashes and packing metrics. `npm.cmd run test:starfall:assets` includes image-based registration regression checks for explicitly reviewed body landmarks; it does not certify every pose in the collection.

Enemy combat occupancy is generated from production actor PNGs by [generate-project-starfall-enemy-hurtboxes.js](../build/generate-project-starfall-enemy-hurtboxes.js), which owns [data/enemy-hurtboxes.js](../js/games/project-starfall/data/enemy-hurtboxes.js). [engine/enemy-hurtboxes.js](../js/games/project-starfall/engine/enemy-hurtboxes.js) applies the shared sprite draw transform and performs solid-pixel rectangle/circle queries; [the engine](../js/games/project-starfall/project-starfall-engine.js) routes combat through those queries while retaining terrain bodies for movement. [Enemy activation](../build/integrate-project-starfall-overhaul-enemies.js) regenerates masks after accepted imports; [build-js.js](../build/build-js.js) requires their `--check` before bundling. The focused command is `npm.cmd run test:starfall:hitboxes`, also included in asset checks. [The hitbox review builder](../build/generate-project-starfall-hitbox-review.js) creates local overlays under `output/`; [the hitbox audit](project-starfall/ENEMY_HITBOX_AUDIT.md) explains the method and limits.

The collision table is published as a separate content-hashed `project-starfall-hurtboxes` chunk by [its bundle entry](../build/entries/project-starfall-hurtboxes.entry.js). [Script injection](../build/inject-script-bundles.js) supplies its URL on the game root; [game startup](../js/games/project-starfall/project-starfall-main.js) requests it only after Start and blocks character selection until it is ready, with explicit retry after failure. The existing script cache policy applies. Node consumers load the same source synchronously. [Start-gate tests](../tests/project-starfall/project-starfall-start-data.test.js) and [bundle tests](../tests/project-starfall/project-starfall-bundle.test.js) cover request timing and both initial and combined transfer limits.

### Project Starfall enemy behavior

[The engine](../js/games/project-starfall/project-starfall-engine.js) owns initial/return spawns, attack commitment, aggro and locomotion. [Map runtime](../js/games/project-starfall/engine/map-runtime.js) supplies platform graphs and authored ramp connections; [enemy definitions](../js/games/project-starfall/data/enemies.js) supply behavior and movement rates. Enemy ground contact must preserve continuous surface travel across ramp seams, while terrain bodies remain separate from combat masks. `npm.cmd run test:starfall:enemies` covers map populations and safe entry placement, real ramp traversal, attack cancellation, return-home motion and flyers; it also runs with the systems suite.

### Project Starfall maps and training

[Map publication](../js/games/project-starfall/data/map-publication.js) owns authored encounter groups and public-field pacing; [map builders](../js/games/project-starfall/data/map-builders.js) own generated connectors. [Shared scenery placement](../js/games/project-starfall/engine/scenery-placement.js) supplies surface anchors and clearance to Canvas and Pixi. [The map guide](project-starfall/MAP_AND_LEVEL_DESIGN_GUIDE.md) defines loop, pursuit, scenery and measured review requirements. [The training CLI](../build/analyze-project-starfall-training.js) runs the [real-engine harness](../tests/project-starfall/project-starfall-training-harness.js); formula balance output is explicitly separate. Focused checks: `test:starfall:maps`, `test:starfall:map-visuals`, `validate:starfall:maps`, and `test:starfall:enemies`. [The map audit](project-starfall/MAP_CONSISTENCY_AUDIT.md) records scope, evidence and measured limitations; `compare:starfall:training` compares compatible runs.

## Android: separate native implementation

The website feed supplies supported content and image references. Compose screens, native logic, permissions, and bundled assets require an APK rebuild; editing the corresponding website JavaScript does not change native behavior. Start with the [Android build guide](../mobile/android/README.md) and [native feature guide](../mobile/android/NATIVE_FEATURES.md).

All Kotlin paths below are under [`mobile/android/app/src/main/java/me/danielshort/app/`](../mobile/android/app/src/main/java/me/danielshort/app/).

| Task | Start here |
| --- | --- |
| Navigation, section branding, project details, native feature eligibility | [ui/DanielShortApp.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/DanielShortApp.kt); [ui/ScrollChrome.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/ScrollChrome.kt) owns the scroll-aware header/navigation; `NATIVE_TOOL_IDS`, `NATIVE_GAME_IDS`, and `NATIVE_DEMO_IDS` route eligible content |
| Settings screens and preference choices | [ui/SettingsScreen.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/SettingsScreen.kt) owns the overview, Updates, Storage and App information; [ui/SettingsControls.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/SettingsControls.kt) owns rows/dialogs; [data/SettingsChoices.kt](../mobile/android/app/src/main/java/me/danielshort/app/data/SettingsChoices.kt) maps modes onto the existing [AppSettings.kt](../mobile/android/app/src/main/java/me/danielshort/app/data/AppSettings.kt) keys. [Settings guide](../mobile/android/SETTINGS.md) documents defaults, independence and validation |
| Reduced motion and contextual maintenance actions | [ui/SystemMotion.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/SystemMotion.kt) observes Android animation preferences; [ui/BrowseMenus.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/BrowseMenus.kt) owns content refresh and confirmed saved-project removal within Projects |
| Automatic content refresh scheduling | [SiteApplication.kt](../mobile/android/app/src/main/java/me/danielshort/app/SiteApplication.kt) for background scheduling; [MainActivity.kt](../mobile/android/app/src/main/java/me/danielshort/app/MainActivity.kt) for foreground entry |
| Tablet and resizable-window layouts | [ui/AdaptiveSiteLayout.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/AdaptiveSiteLayout.kt) owns window-width policy, vertical section tabs and bounded content; [ui/DanielShortApp.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/DanielShortApp.kt) preserves navigation across size changes |
| Launch checks and automatic installation | [updates/AppUpdateCoordinator.kt](../mobile/android/app/src/main/java/me/danielshort/app/updates/AppUpdateCoordinator.kt) coordinates cold-launch/network checks and safe background updates; [updates/AutomaticAppInstaller.kt](../mobile/android/app/src/main/java/me/danielshort/app/updates/AutomaticAppInstaller.kt) owns Android installer sessions, verified staging and recovery |
| Verified native app updates and public binary patches | [ui/AppUpdateSection.kt](../mobile/android/app/src/main/java/me/danielshort/app/ui/AppUpdateSection.kt), [updates directory](../mobile/android/app/src/main/java/me/danielshort/app/updates/); [prepare-app-update.cjs](../mobile/android/scripts/prepare-app-update.cjs) generates immutable release artifacts; [UPDATES.md](../mobile/android/UPDATES.md) describes signing, patch validation, and publication. Only explicitly staged `mobile/android/releases/{review,stable}/latest.json` manifests are copied by [app-update-feeds.cjs](../build/lib/app-update-feeds.cjs) into `/app-updates/`; APKs remain separate release assets |
| Feed schema, offline cache, refresh, bookmarks | Website [generate-mobile-content.js](../build/generate-mobile-content.js); Android [data directory](../mobile/android/app/src/main/java/me/danielshort/app/data/): `SiteContent.kt` parser and `ContentRepository.kt` storage/network behavior |
| Text Compare and native utilities | [NativeTextCompareScreen.kt](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/NativeTextCompareScreen.kt), [TextDiff.kt](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/TextDiff.kt); [tools directory](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/tools/) for text, QR, image UI and processing |
| Screen recording, clip persistence, sharing | [recording directory](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/recording/): `NativeScreenRecorder.kt`, `ScreenRecordingService.kt`, `RecordingFiles.kt`; [AndroidManifest.xml](../mobile/android/app/src/main/AndroidManifest.xml) and [resource XML](../mobile/android/app/src/main/res/xml/) for permissions/provider configuration |
| Demos, inference, native dashboards | [demos directory](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/demos/): `NativeProjectDemos.kt`, `DemoApi.kt`, `NativeDashboardDemos.kt`, `TableauNativeViews.kt`, `PizzaTipsDemo.kt` |
| Games and saved progress | [games directory](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/games/): `NativeGamesScreen.kt`, game `*Model.kt` files, `GameStorage.kt`; roulette lives separately in [NativeGameScreen.kt](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/NativeGameScreen.kt) and [RouletteGame.kt](../mobile/android/app/src/main/java/me/danielshort/app/nativefeatures/RouletteGame.kt) |
| Bundled artwork, launcher branding, Tableau data snapshots | [assets](../mobile/android/app/src/main/assets/), [resources](../mobile/android/app/src/main/res/), [extract-tableau-data.py](../mobile/android/scripts/extract-tableau-data.py) |
| App version, SDK/dependencies, feed endpoint, signing, CI | [app/build.gradle.kts](../mobile/android/app/build.gradle.kts), root files in [mobile/android](../mobile/android/), [Android CI](../.github/workflows/android.yml). Keep signing material and `local.properties` untracked |

Tests live in the [JVM test tree](../mobile/android/app/src/test/) and [device test tree](../mobile/android/app/src/androidTest/). The [focused settings emulator script](../mobile/android/scripts/test-settings-device.sh) runs navigation/update UI regressions and collects test screenshots; it also runs in Android CI. Existing QA reports are dated evidence, not a source for current deployment or release status.

## Generated output: trace back before editing

The complete build order lives in [build-site.js](../build/build-site.js). Some generated HTML and catalogs are tracked; that does not make them authoritative source.

| Output / managed region | Source and generator |
| --- | --- |
| Managed homepage, library, audience and resume pages; catalog JS in `js/portfolio/`, `js/home/`; audience config; header/footer partials | [content](../content/) → [content-loader.js](../build/lib/content-loader.js) / [cms-content-model.js](../api/_lib/cms-content-model.js) → [generate-cms-artifacts.js](../build/generate-cms-artifacts.js) with CMS renderers/widgets |
| `pages/portfolio/` case studies and audience project variants | `content/projects/` + [generate-project-pages.js](../build/generate-project-pages.js) |
| Accordion shell / tool heading regions inside page HTML | [generate-personal-accordion-pages.js](../build/generate-personal-accordion-pages.js) + [personal-accordion-shell.js](../build/lib/personal-accordion-shell.js). Preserve authored tool bodies outside generated regions |
| `pages/demos/` isolated wrappers | [generate-project-demo-wrappers.js](../build/generate-project-demo-wrappers.js) + [project-demo-routes.js](../build/lib/project-demo-routes.js); raw demo content remains in `demos/` |
| Root `contact.html`, `privacy.html`, `sitemap.html` mirrors | Corresponding `pages/` file → [sync-root-pages.js](../build/sync-root-pages.js); check whether the `pages/` file is itself CMS-managed |
| `dist/` CSS/JS and derived assets | [build-css.js](../build/build-css.js), [build-js.js](../build/build-js.js), [JS entries](../build/entries/), [image optimizer](../build/optimize-site-images.js) |
| `js/tools/utm-batch-builder.js` and its worker bundle | [src/utm-batch-builder](../src/utm-batch-builder/) → [build-utm-batch-builder.js](../build/build-utm-batch-builder.js) |
| `dist/app-content/v1/catalog.json`, deployed `/app-content/v1/catalog.json`, Android offline catalog | [generate-mobile-content.js](../build/generate-mobile-content.js); website copy step publishes the feed; Android Gradle generates/copies its bundled fallback |
| `public/` deployment tree | [copy-to-public.js](../build/copy-to-public.js). New shipped asset directories must be included here; `docs/` and Android sources are not published |
| Android `build/` directories, APKs, reports | Gradle. Debug APK: `mobile/android/app/build/outputs/apk/debug/app-debug.apk` |

## Validation and local preview

Run from the repository root. [package.json](../package.json) and [tests/README.md](../tests/README.md) are the command reference; focused tests below supplement the normal checks.

```powershell
npm.cmd run dev -- --port 4173
```

This uses [build/dev.js](../build/dev.js) for clean routes, APIs, and rebuilds. For website implementation changes, build and run the applicable tests; before pushing, follow the repository's full build/test requirement:

```powershell
npm.cmd run build
npm.cmd test
git diff --check
```

| Changed area | Useful focused checks after building |
| --- | --- |
| Homepage accordion / remembered scroll / closed layout | `node tests/site/home-category-accordion.test.js`, `node tests/site/home-tab-transitions.browser.cjs`, `node tests/site/home-closed.browser.cjs` |
| Shared frame / header | `node tests/site/frame-compact-motion.test.js`, `node tests/site/frame-scroll.browser.cjs`, `node tests/site/frame-geometry.browser.cjs`, `node tests/site/header-breadcrumbs.test.js` |
| Text Compare | `node tests/site/text-compare-simple.browser.cjs`, `node tests/tools/text-compare-continuation.test.js` |
| Project / demo structure and controls | `node tests/site/project-content-clarity.browser.cjs`, `node tests/site/project-demo-consistency.browser.cjs`, `node tests/site/digit-generator-simple.browser.cjs` |
| Project question button / returning visitors with cached styles | `node tests/site/project-question-cache.browser.cjs` tests the real service worker with stale CSS, mobile/desktop sizing, popup behavior, and navigation |
| Tableau embeds | `node tests/site/tableau-project-integration.browser.cjs`; also check live desktop and narrow-screen interactions |
| Contact form / map | `node tests/site/contact-map.test.js`, `node tests/site/contact-loader.test.js`, `node tests/site/contact-form-layout.browser.cjs` |
| Drafts, contact recovery, visitor metrics | `npm run test:recovery`, `node tests/site/session-drafts.test.js`, `node tests/site/contact-delivery.test.js`, `npm run test:analytics` |
| Offline caches and upgrades | `node tests/site/service-worker-cache.test.js`, `node tests/site/service-worker-cache.browser.cjs` |
| Cross-browser / WCAG / screenshots / mobile performance | `npm run test:release`, `npm run test:geometry`, `npm run performance:lab`; baseline updates require explicit review |
| Android catalog / website feed | `node tests/site/mobile-content.test.js`; inspect the generated feed and native refresh/offline behavior |
| Broad browser smoke / SEO | `npm.cmd run test:browser`, `npm.cmd run test:seo`; the browser script is a smoke suite, not every `*.browser.cjs` test |

For native changes, configure the JDK/SDK per the Android guide, then run:

```powershell
.\mobile\android\gradlew.bat -p mobile/android testDebugUnitTest lintDebug assembleDebug
# Requires a running emulator or connected device:
.\mobile\android\gradlew.bat -p mobile/android connectedDebugAndroidTest
```

Device checks remain necessary for recording consent/capture, file import/export, lifecycle recovery, and changed native interactions. For documentation-only edits, check referenced paths and `git diff --check`; rebuilding the application is unnecessary.

## Other entry points and upkeep

- New feature scaffolding: [template guide](../build/templates/README.md); register routes in [vercel.json](../vercel.json) and content in the appropriate catalog.
- Job Tracker backend: [aws/job-application-tracker](../aws/job-application-tracker/README.md), `npm.cmd run test:tracker` / `build:tracker`.
- Browser extension: [job-application-copilot](../browser-extension/job-application-copilot/README.md), `npm.cmd run test:ext` / `build:ext`.
- Website CI: [ci.yml](../.github/workflows/ci.yml). A website build does not deploy AWS services or distribute an APK.
- Update the relevant row when moving an entry point, changing source ownership, or changing build/test responsibilities. Prefer links to existing guides over duplicated instructions.
- Keep this map about code ownership. Check Git, CI, deployment, account configuration, and release availability live; do not store credentials, temporary URLs, current branch claims, or task history here.
