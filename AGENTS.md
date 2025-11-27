# Repository Guidelines

## Project Structure & Module Organization
Source HTML lives at the root (`index.html`, `portfolio.html`, `contact.html`), while assets stay under `css/`, `js/` (grouped by area like `js/navigation/`), `img/`, and `documents/`. Build helpers sit in `build/` (`build-css.js`, `copy-to-public.js`, `resize_logo.py`). The CSS bundle lands in `dist/styles.[hash].css`, and deployable artifacts are mirrored into `public/`. Add new asset folders only after updating `build/copy-to-public.js` so they are included in the deployment.

## Build, Test, and Development Commands
Run `npm run build:css` to compose all `css/styles.css` imports into the hashed bundle. `npm run build:icons` regenerates favicons from `img/ui/logo.png`. `npm run build` runs the CSS build and copies site assets into `public/`. Lightweight project checks live in `npm test`. After any build, open `public/index.html` (or the root files) in a browser or serve `public/` through any static server for validation.

## Coding Style & Naming Conventions
Use 2-space indentation everywhere; HTML attributes stay in double quotes. JavaScript is ES2015+, with camelCase variables/functions, kebab-case filenames, and uppercase constants. End every JS statement with semicolons and keep concise semantic selectors; behavior hooks should rely on data attributes. CSS honors the existing `@layer` order (`tokens` → `base` → `layout` → `components` → `utilities` → `overrides`), so add new component styles under `css/components/` and import them from `css/styles.css`.

## Testing Guidelines
`npm test` runs `test.js`, which enforces markup contracts (e.g., `.skip-link`, `<main id="main">`), checks data arrays, and timing expectations such as `START_TIMEOUT_SEC ≥ 600` in `chatbot-demo.html`. Keep additions fast, dependency-free, and expand the tests whenever new pages or requirements arise.

## Commit & Pull Request Guidelines
Follow clear imperative subjects (e.g., “Update hero copy”); prefixes like `feat:` or `fix:` are welcome but optional. Before pushing, run `npm run build && npm test` and describe any visual changes or build-script updates. Pull requests should link issues, summarize scope, and attach screenshots for UI tweaks. Keep diffs focused and mention if build tooling or deployment assets changed.

## Security & Configuration Tips
Never commit secrets or API keys; analytics helpers belong in `js/analytics/`. Deployment reads from `public/` via `vercel.json`, so ensure `robots.txt` and `sitemap.xml` remain current, especially when adding pages or blocking crawlers.

## Slot Machine Demo Upgrade Plan
- Port the Slot-Machine-v4 spin engine: deterministic RNG seeded spins, v4 line generation tiers (straight/diagonal/zigzag/all-touching), bet-multiplier scaling per machine tier, spinTiming from JSON, and jackpot/hot-streak/perk hooks while keeping the AWS-auth shell.
- Add progression/state parity: per-machine coins/VIP stored alongside seeds and bet pct, XP/levels gating upgrades (autoplay, rows/reels/lines), energy cost per spin, and machine unlock/drawer flow similar to v4.
- Bring over drop/perk systems: v4 DropSystem multipliers (bet-based, gear/VIP/totem/skill), spin boosters and machine mods, worker automation collecting coins/drops, and hot-streak/premium upgrades feeding both payouts and drops with a detail overlay.
- Refresh the UI to match v4 controls: replace the bet slider with the bet-button list, add skill buttons (drop/payout/wild) with cooldowns, auto-spin toggle/worker status, level/XP bar, and machine theming pulled from slot assets.
- Sync configs/tests: align slot configs/upgrade definitions with v4 JSON (symbols, payouts, spinTiming, costs), and extend tests to cover seeded spins, line generation, drop multiplier calculations, and auto-spin/visibility handling.

## Slot Demo Sync & Export Plan
- Introduce a `/sync` endpoint (Lambda/API GW) that accepts a signed session snapshot (balance, bet, upgrades, drops, gear/cards, idle, seed/version) and returns the canonical copy plus a revision. Keep `/auth/*` and `/daily` as-is; spins stay client-side.
- Add a client snapshot builder that serializes deterministic state, stamps `rev`, `updatedAt`, and HMAC (session token + server secret), and queues outbound syncs (debounced + periodic). Apply optimistic updates locally and reconcile on server response.
- Implement conflict handling: server wins if `rev` mismatches—client rewinds to payload, reapplies queued local deltas (spins/drops) then resends. Persist a small outbound queue in localStorage so offline play resends on next login.
- Export/import key: serialize minimal state JSON, compress (e.g., deflate), encrypt with AES-GCM using a key derived from the session token + server-provided salt, then Base64-url encode `salt.iv.ciphertext`. Import path validates HMAC, decrypts, merges, and bumps `rev`. No plaintext should be human-readable.
- UX: add “Export key” and “Import key” fields in the account sheet with status toasts; include a “Sync now” button that forces a `/sync` call and shows revision/time.
- Tests: add unit tests for snapshot round-trip, HMAC rejection, AES-GCM encrypt/decrypt, Base64-url encode/decode, and conflict resolution (outdated rev).
