# Daniel Short's Website

**Solving everyday problems with data and thoughtful tools.**

My personal website brings together data and machine learning projects, useful browser tools, and playable experiments. It also shares the interests behind my work: practical uses for AI, family, and 20 years of playing French horn.

[Visit the website](https://www.danielshort.me/) · [Projects](https://www.danielshort.me/portfolio) · [Tools](https://www.danielshort.me/tools) · [Games](https://www.danielshort.me/games) · [Contact](https://www.danielshort.me/contact)

Making a change? Start with the [Repository map](docs/REPOSITORY_MAP.md) for authoritative source files, generated-output boundaries, Android entry points, and focused checks.

## Explore the website

| Area | What you will find |
| --- | --- |
| About | Personal interests, experience, education, and linked credentials in a resume-style layout |
| Projects | Case studies with STAR Summary narratives, interactive demos, historical dashboards, code, and supporting resources |
| Tools | Utilities for comparing and reviewing text, creating campaign links and QR codes, processing images, and recording the screen |
| Games | Browser games and simulations exploring probability, progression, feedback loops, and real-time interaction |
| Contact | A message form, direct email and GitHub links, and a location map loaded when the Contact section is opened |

The homepage opens with all five sections condensed. Colored vertical tabs define the desktop layout, with responsive navigation on smaller screens. Shared headers, breadcrumbs, search, and page transitions connect the homepage, libraries, and individual experiences.

### Selected projects

- [Sheet Music Watermark Removal & Upscale](https://www.danielshort.me/portfolio/sheetMusicUpscale): compare original, restored, and upscaled sheet-music scans.
- [Handwriting Legibility Scoring](https://www.danielshort.me/portfolio/handwritingRating): draw a digit and explore how confidently a model recognizes it.
- [Baby Name Predictor](https://www.danielshort.me/portfolio/babynames): explore recommendations informed by family preferences and historical naming trends.
- [Synthetic Digit Generator](https://www.danielshort.me/portfolio/digitGenerator) and [Shape Classifier](https://www.danielshort.me/portfolio/shapeClassifier): interact with generative and classification models.
- [Pizza Delivery Dashboard](https://www.danielshort.me/portfolio/pizzaDashboard) and [UFO Sightings Dashboard](https://www.danielshort.me/portfolio/ufoDashboard): explore historical records through embedded Tableau dashboards.

The [project library](https://www.danielshort.me/portfolio) also includes semantic search, a travel chatbot, retail analytics, regression modeling, and a reinforcement-learning puzzle solver. Availability of cloud demos depends on their backing services; historical dashboards are not live operational data.

### Tools and games

Start with [Text Compare](https://www.danielshort.me/tools/text-compare), [QR Code Generator](https://www.danielshort.me/tools/qr-code-generator), [Image Optimizer](https://www.danielshort.me/tools/image-optimizer), or [Screen Recorder](https://www.danielshort.me/tools/screen-recorder). The tool library also includes writing checks, word-frequency analysis, background removal, and UTM batch creation. Job Application Tracker and File Transcriber require an account; campaign administration tools require admin access.

The games collection includes **Project Starfall** (a work-in-progress action RPG), **Stellar Dogfight**, **Ocean Wave Simulator**, **Double-Zero Roulette**, **Probability Engine**, and **Stormbreak: Idle Olympus**.

Supported tools can recover eligible signed-out drafts within the browser session. Signed-in accounts provide saved sessions and history. Processing and storage vary by tool; see the [privacy page](https://www.danielshort.me/privacy) for the current explanation.

## How it is built

| Layer | Implementation |
| --- | --- |
| Website | HTML, CSS, and JavaScript with shared navigation and feature modules; React is used for UTM Batch Builder |
| Content and build | Managed JSON under `content/`, Node.js generators, esbuild bundles, and Sharp image optimization |
| Visual system | DS branding, a light navy-and-white shell, section accent colors, local Inter fonts, SVG interface icons, and illustrated catalog artwork |
| APIs and accounts | Vercel API handlers, Amazon Cognito, DynamoDB, and AWS services for cloud-backed tools and model inference |
| Dashboards | Tableau embeds and browser-based visualizations over published historical datasets |
| Verification | Node checks, Playwright browser tests, axe accessibility checks, reviewed screenshots, and Lighthouse measurements |

See the [visual style guide](docs/visual-style.md) for shared design rules and the [release-quality guide](docs/RELEASE_QUALITY.md) for verification procedures.

The build also generates [AI-readable summaries](https://www.danielshort.me/llms.txt) under `/ai/` from public content. These optional summaries link back to canonical pages; the main URLs serve the same website to people and agents. Account and admin workspaces are excluded from the summary catalog.

### Android companion

The [Android app](mobile/android/README.md) uses Kotlin and Jetpack Compose for navigation, settings, bookmarks, screen recording, and native adaptations. It combines those features with website case studies, demos, and games in an in-app WebView, while selected tools open in the Android browser. Native alternatives remain available for supported experiences.

The website build generates a [versioned content feed](https://www.danielshort.me/app-content/v1/catalog.json). Content and website experiences can refresh after deployment; changes to native code require a new app release. See the [Android update guide](mobile/android/UPDATES.md) for verified APK and patch delivery.

## Setup and installation

Use **Node.js 22.x** and npm. From a fresh clone:

```bash
git clone https://github.com/danielshort3/danielshort3.github.io.git
cd danielshort3.github.io
npm ci
npm --prefix aws/job-application-tracker ci
npm --prefix browser-extension/job-application-copilot ci
npm run dev
```

The two subproject installs are needed by the full repository test suite. On Windows PowerShell, use `npm.cmd` if script-execution policy prevents `npm` from running.

## Local development

What `npm run dev` does:

- Runs a full `build/build-site.js` pass first.
- Watches key source folders/files and reruns that build whenever you change code/content.
- Starts a repo-native Node server at `http://localhost:3000` for clean URLs, static output, and the local CMS file API. If the starting port is busy, it automatically tries the next open port and prints the final URL.

Use a different port:

```bash
npm run dev -- --port 4173
```

The local server loads `.env.local`, then `.env`, from the repository root. Existing shell variables take precedence, and `.env.local` overrides `.env`. Restart the server after changing these files. The same loading applies to `createLocalServer()` when starting a QA server directly.

Tools sign-in also requires an exact Cognito callback for the local hostname and port. See [Tools sign-in on localhost](docs/tools-local-sign-in.md) for the callback setup helper and local backend configuration.

Use this server for clean URLs and API routes rather than opening HTML files directly. Public static pages and local browser tools can be previewed without configuring cloud services.

Empty-Package Shrink, Store-Level Loss & Sales, COVID Outbreak Drivers, Pizza Tips, and Baby Names run in the browser using bundled data or calculations. Sheet Music's image comparison and Delivery Tip's project presentation also use site assets. Their supporting downloads are tracked under an explicit `documents/` allowlist and copied into `public/`; no AWS credentials are needed for these project experiences. The underlying Sheet Music desktop workflow and Excel analysis remain local authoring workflows.

### Optional cloud services

The remaining AWS-backed model demos need their `DEMO_*_FUNCTION_ARN` configured locally. For example, add the following to the ignored `.env.local`, replacing the ARN with the demo's qualified Lambda alias:

```dotenv
AWS_AUTH_MODE=auto
DEMO_AWS_REGION=us-east-2
DEMO_REQUIRE_DDB_RATE_LIMIT=false
DEMO_SHAPE_FUNCTION_ARN=arn:aws:lambda:us-east-2:<account-id>:function:shape-classifier:live
```

Use your local AWS profile with permission to invoke that alias (`aws sts get-caller-identity` checks the login). Leave `DEMO_INVOKE_AWS_ROLE_ARN` unset locally so the AWS SDK uses your local credentials; that role is for Vercel OIDC. The server does not load `.vercel/.env.production.local` or other deployment environment files.

The travel chatbot uses `/api/chatbot-demo/bedrock/status` and the corresponding `/api/chatbot-demo/qwen/*` routes on local hosts. The development server forwards only the chatbot's supported status, warmup, submit, and result requests to its fixed API Gateway, so changing the local port does not require an AWS CORS update. Bedrock answers stream through `/api/chatbot-stream`; set `CHATBOT_STREAM_FUNCTION_ARN` in `.env.local` to the qualified `VGJBedrockStream:live` alias, set `CHATBOT_STREAM_AWS_REGION` to the alias region (`us-east-2`), and use a local AWS profile permitted to invoke it.

### Google Maps embed

The contact page location map is upgraded during `npm run build` with the Google Maps Embed API. For local builds, put the key in `google_maps_api_key.txt` at the repo root or set `GOOGLE_MAPS_API_KEY`; for Vercel, set `GOOGLE_MAPS_API_KEY` as an environment variable. Do not commit the key file.

## Editing the website

- **Content:** edit the matching JSON under `content/`. Project pages, libraries, homepage content, and catalog scripts are generated from these sources.
- **Tool and demo interfaces:** edit the authoritative body in `pages/` or `demos/` and its feature module. Preserve generated shell regions.
- **Styles:** use shared tokens in `css/variables.css` and the relevant component stylesheet. Follow the route-specific stylesheet entry points in the repository map.
- **New routes:** use the [scaffold templates](build/templates/) and add clean-URL rewrites in [vercel.json](vercel.json).
- **Generated output:** rebuild instead of editing `dist/`, `public/`, or generated page regions directly. A tracked generated file is still generated output.

## Checks and publishing

Build before testing changes to the website:

```bash
npm run build
npm test
git diff --check
```

`npm test` includes site contracts, account and recovery checks, subproject suites, SEO, performance budgets, and source-to-public parity. For rendered changes, also run the applicable browser checks:

```bash
npx playwright install chromium firefox webkit
npm run test:browser
npm run test:release
```

Use `npx playwright install --with-deps chromium firefox webkit` when Linux browser dependencies are missing. Reviewed screenshot comparisons run on Linux/WSL. The [release-quality guide](docs/RELEASE_QUALITY.md) covers recovery checks, mobile Lighthouse measurements, explicit screenshot-baseline review, and the separate manual screen-reader pass. Automated checks do not establish complete accessibility.

Vercel is configured to run `npm run build` and serve `public/`, with API handlers under `api/`. [vercel.json](vercel.json) owns clean URLs, response headers, and external-service policy. Verify the deployment and live routes after publishing; a local build alone does not publish the site. Website builds do not deploy AWS services or distribute an Android APK.

For documentation-only edits, verify referenced paths and run `git diff --check`; an application rebuild is unnecessary.

## Admin link management

The restricted **Links & QR codes** workspace manages short links, saved QR designs, and link activity.

- **Redirects:** `https://<your-domain>/go/<slug>` returns a `301` or `302` to the stored destination.
- **Dashboard:** `/tools/short-links`, authored in `pages/short-links.html`.
- **Admin auth:** use a recognized tools-admin account or a workspace access key. The key is configured with `SHORTLINKS_ADMIN_TOKEN` in the server environment; do not commit it.
- **Storage:** AWS DynamoDB. In Vercel, set `SHORTLINKS_DDB_TABLE`, `AWS_REGION`, `SHORTLINKS_AWS_ROLE_ARN`, `AWS_AUTH_MODE=oidc`, and `AWS_OIDC_AUDIENCE=sts.amazonaws.com`. Static keys are supported only as a temporary/local fallback.

Generate an admin token locally:

- `openssl rand -hex 32`
- `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`

## Tools accounts (shared login + saved sessions)

This repo includes an optional account layer for tools under `/tools`:

- **Account entry:** `/tools/dashboard` handles sign-in and the authentication callback. Supported tools expose account and saved-session controls in their own interfaces.
- **Auth:** Amazon Cognito Hosted UI (PKCE) configured in `js/accounts/tools-config.js`; sign-in requests `select_account` by default so Google prompts for the intended account. In the default `dual` migration mode, a verified ID token is also exchanged for an AES-256-GCM authenticated, Secure, HttpOnly, `SameSite=Lax`, same-origin `__Host-` session cookie.
- **Storage:** AWS DynamoDB (see `.env.example`).

Signed-in tools automatically save supported inputs, settings, and results to the account as work changes. Browser tools remain local while signed out unless a user shares their work or invokes a cloud feature. Account snapshots exclude password and file-upload fields; tool-specific capture hooks can include file names, result summaries, or processed previews as disclosed on each tool and the privacy page. File downloads and cloud-tool record submissions keep their own controls.

Required configuration:

- Update your Cognito app client **Allowed callback URLs** to include `https://www.danielshort.me/tools/dashboard`.
- Set `TOOLS_COGNITO_ISSUER` + `TOOLS_COGNITO_CLIENT_ID` in your Vercel environment (used for server-side JWT verification).
- Set `TOOLS_SESSION_SECRETS` to one or more comma-separated 32-byte base64url (or 64-character hex) keys. The first encrypts new cookies; all keys decrypt during rotation. Keep `TOOLS_AUTH_BEARER_FALLBACK=true` during the staged rollout, and retain the previous key after the new key until old cookies expire.
- `TOOLS_SESSION_TTL_SECONDS` is optional (default `28800`; clamped to 900–86400 seconds). `TOOLS_AUTH_BEARER_FALLBACK` is optional and defaults to `true` when omitted.
- Create a DynamoDB table with partition key `pk` (string) and sort key `sk` (string), then set `TOOLS_DDB_TABLE` and the environment-specific `TOOLS_AWS_ROLE_ARN` for Vercel OIDC session/activity storage.

Session rollout notes:

- The server cookie contains only bounded subject, email, display-name, group, issued-at, and expiry claims. Cognito ID/access/refresh tokens are never written to server storage or placed in the cookie.
- `sessionMode: 'dual'` preserves the current Job Tracker flow while same-origin tools APIs authenticate the cookie first. During migration, the browser also sends the existing ID-token header so the configured Bearer fallback still works if the cookie is missing or expired.
- Do not change the client to `sessionMode: 'cookie'` or disable Bearer fallback until Job Tracker has a same-origin backend-for-frontend route and Preview/Production auth canaries pass. Cookie-only mode intentionally rejects cross-origin authenticated API requests.
- Logout clears the browser state immediately and expires the cookie. Sessions are stateless and cannot be individually revoked server-side, so keep `TOOLS_SESSION_TTL_SECONDS` bounded (the implementation caps it at 24 hours).

Activation checklist (must be verified after deployment):

1. Add the Cognito variables and `TOOLS_SESSION_SECRETS` to Preview first; keep the checked-in client in `dual` mode and Bearer fallback enabled.
2. Sign in, confirm `/api/tools/auth/exchange` sets `__Host-tools_session` with `Secure`, `HttpOnly`, `SameSite=Lax`, and `Path=/`, then verify `/api/tools/auth/session`, a saved-session mutation, and Transcribe authentication.
3. Sign out and verify `/api/tools/auth/logout` sends `Max-Age=0`; repeat after manually clearing the cookie to confirm the dual-mode Bearer path still works.
4. Repeat the canary in Production before considering `sessionMode: 'cookie'` or `TOOLS_AUTH_BEARER_FALLBACK=false`.

The CSP keeps `object-src 'none'`; resume PDFs therefore use titled same-origin iframes with visible download fallbacks. `frame-ancestors 'self'` is intentional so the portfolio can embed its own demo pages while external sites remain unable to frame them. Inline script/style allowances remain temporarily for compatibility and should be removed only through a separately tested nonce/hash migration.

## Local CMS

The custom `/admin` editor is a local-only content editor for managed JSON files under `content/`.

- **Admin:** `http://localhost:3000/admin` while `npm run dev` is running.
- **Windows launcher:** run `start-local-cms-wsl.bat` from Windows to start the WSL dev server and open `/admin`.
- **WSL access:** the Windows launcher binds the WSL server to `0.0.0.0`; it opens `localhost` when Windows can reach it, otherwise it falls back to the current WSL IP.
- **Storage:** local files in `content/site`, `content/pages`, `content/audiences`, `content/resumes`, `content/projects`, and `content/tools`.
- **Safety:** write APIs only accept localhost requests; `/admin` is not copied into `public/`.
- **Publishing:** after saving content, run `npm run build && npm test`, review `git diff`, then deploy manually.

## File structure

| Directory | Purpose |
| --- | --- |
| `content/` | Managed site, audience, project, tool, and resume content |
| `pages/`, `demos/` | Authored tool/demo pages alongside generated pages and wrappers; check ownership before editing |
| `css/`, `js/`, `src/` | Website styles, feature modules, and bundled application sources |
| `api/`, `aws/` | Site API handlers and backend service subprojects |
| `img/`, `documents/` | Visual assets and downloadable resources |
| `build/` | Renderers, generators, bundlers, route-aware development server |
| `mobile/android/` | Android companion, website integration, native features, Gradle build, and tests |
| `browser-extension/` | Browser extension subproject |
| `tests/`, `docs/` | Verification suites and development documentation |
| `dist/`, `public/` | Generated bundles and deployment output |

See the [Repository map](docs/REPOSITORY_MAP.md) for task-to-file lookup and build dependencies.
