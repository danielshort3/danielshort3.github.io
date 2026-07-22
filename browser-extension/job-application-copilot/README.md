# Job Application Copilot browser extension

Job Application Copilot is a complete local-first Manifest V3 workflow for preparing job-application answers with Ollama, encrypted candidate sources, exact evidence citations, controlled field filling, and an explicit handoff to the website's application tracker.

The extension never submits an application. CAPTCHA, consent, signature, demographic, identity, security-code, hidden, and other excluded fields stay manual. Consequential answers require review, and exact profile fields come only from user-verified facts.

## Requirements

- Chrome 120 or newer.
- Node.js 22.13 or newer for development and packaging (required by the bundled PDF.js release).
- Ollama 0.32.0 or newer, listening on `http://127.0.0.1:11434`.
- The configured local models. Defaults are `qwen3.5:27b`, fallback `qwen3:8b`, and `nomic-embed-text`.

Pull the defaults if they are not already installed:

```text
ollama pull qwen3.5:27b
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Open **Settings** and copy the exact origin rendered from `chrome.runtime.id`. The checked-in public key is aligned with the Chrome Web Store item and produces `jigajpmnbiofgmgcnmdeechgibpjlfop`; Store packaging removes the key, while the published item retains that same ID.

```text
OLLAMA_ORIGINS=chrome-extension://<extension-id>
OLLAMA_NO_CLOUD=1
```

Keep Ollama on loopback and disable Ollama cloud. The manifest grants only `http://127.0.0.1:11434/*`; the client rejects `localhost`, alternate ports, paths, remote hosts, explicit cloud model names, and `/api/show` responses containing remote model metadata.

## Build and install locally

From this package:

```text
npm ci
npm run verify
npm run package
```

Or from the website repository root:

```text
npm run test:extension
npm run build:extension
npm run package:extension
```

Load the unpacked build:

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select `browser-extension/job-application-copilot/dist`.
5. Pin **Job Application Copilot**, then click its toolbar action to open the side panel.

`npm run package` (or `package:dev`) rebuilds and writes `artifacts/job-application-copilot-0.1.0.zip` with the stable development key. `npm run package:store` writes the separately validated `artifacts/job-application-copilot-0.1.0-chrome-web-store.zip` with that key removed, as required for the Store-assigned item ID. Both include runtime icons and generated third-party notices. `dist/` and `artifacts/` are ignored build outputs and should not be committed.

## V1 workflow

Before any vault, Ollama, or application controller starts, first use presents a blocking, versioned privacy notice. Acceptance is stored locally. Settings lets the user review or withdraw consent; withdrawal locks and retains the encrypted vault, clears trusted session state, and blocks all Copilot features. Synthetic `preview=1` mode bypasses the gate without reading or writing Chrome storage.

### 1. Create and unlock the local vault

Open **Sources**, create a passphrase of at least eight characters, and keep it somewhere safe. The passphrase is never stored. Losing it requires resetting the local vault.

Documents, pasted notes, verified facts, Website and custom links, custom thoughts or preferences, retained originals, retrieval chunks/embeddings, citations, application packets, approved answers, clarifications, and exact page-revision generation caches are encrypted in IndexedDB with AES-GCM. The derived key exists only in trusted `chrome.storage.session`; **Lock vault**, browser-session end, and reset clear it.

### 2. Add verified facts and sources

Verified facts use fixed keys such as name, email, phone, address, LinkedIn, Website, portfolio, and GitHub. They are conservative exact mappings: an applicant email cannot fill a supervisor or reference email field.
Custom links pair a user-defined field label with an HTTP(S) URL. They match only that exact or safely qualified label, including link/profile fields implemented as ordinary text inputs; they never use fuzzy model inference.
Custom thoughts provide citation-backed context to local Ollama, such as a salary expectation or an explanation of transferable experience. They are never deterministic fills, and protected-status, employment-eligibility, identity, attestation, and signature content is rejected. Salary-history language is not fillable even if a thought contains it.
Optional U.S. employment-eligibility facts use three explicit Yes/No selectors: authorization to work, sponsorship now, and sponsorship in the future. They are encrypted with the rest of the vault but excluded from retrieval and every Ollama prompt; deterministic rules can answer only clearly U.S.-scoped authorization/sponsorship questions with exact Yes/No options.
Citizenship, nationality, immigration or visa status/type/class, permanent-resident or green-card status, I-9/E-Verify, export-control, government-ID, and protected demographic self-identification remain manual and outside model context. Non-U.S. or country-ambiguous eligibility questions fail closed.

The parser worker accepts PDF, DOCX, TXT, Markdown, and HTML up to 10 MiB each. It rejects legacy DOC, empty documents, and image-only PDFs without extractable text. Original bytes are encrypted for every supported import; only retained PDF/DOCX records appear in tracker attachment selectors. Legacy PDF records that show `0 B` are repaired locally from their retained encrypted bytes on vault refresh. If retained bytes are unavailable, the UI shows **Size unavailable** and the file must be re-imported.

Choose a source role in user terms:

- Resume / candidate evidence
- Cover letter / style evidence
- Position or job description / requirement
- Company notes / context
- Verified material

Pasted notes are manual input only. A source URL is attribution; the extension never fetches it. **Source library** is a reusable catalogue with safe defaults: the newest resume, newest cover-letter/style source, reusable verified materials, and context saved for the current application are recommended and selected automatically. Importing a newer resume or cover letter replaces the older default for that role. Moving to another application carries reusable candidate/style material but drops job and company context from the previous application. The catalogue shows **in use / saved** counts, marks recommended and current-application records, disables records scoped to another application, and still lets the user change every available selection. Verified profile and custom links remain an exact-match path; relevant custom thoughts enter local retrieval and appear as canonical source citations.

### 3. Analyze and review an application page

Open an HTTP(S) job posting or application form and select **Analyze current page**. The side panel injects the isolated page runtime into the active tab, scans structural metadata without reading field values, captures a bounded visible job-description section when available, and excludes unsupported or sensitive fields. A posting can be captured before you navigate to its separate ATS form.

Dedicated application-format adapters are included for ApplicantPro, Greenhouse, Lever, Ashby, SmartRecruiters, and Workday. Bounded platform profiles add iCIMS, Oracle Recruiting/Taleo, BambooHR, ADP Recruiting, UKG, Jobvite, JazzHR, Recruitee, Pinpoint, and SAP SuccessFactors without treating a host name alone as proof that a form is safe. Other sites use a conservative **free-format** mode that scores likely application forms, ignores sign-in/search/newsletter/referral surfaces, uses bounded structured metadata such as JSON-LD when available, and discovers supported controls in the page, accessible same-origin nested frames, and bounded open Shadow DOM. Workday account-creation gates stay manual and are not treated as applications.

Free-format mode can propose cited answers for ordinary open-ended application questions. Native controls remain eligible for reviewed filling; bounded custom Yes/No groups, other custom ARIA widgets, and contenteditable controls that cannot be safely driven with native setters are explicitly **copy only**. Likely application frames with a cross-origin URL are reported as unsupported even when the outer page has no recognizable application form. Closed Shadow DOM cannot be inspected and generally cannot be detected from outside the component; an adapter reports it only when the site structure provides an independent, reliable signal.

Retrieval is application-scoped. Encrypted application packets preserve the captured job context, approved cited answers, and verified clarifications as you move from a posting to a separate ATS form. Context can be refreshed from the current page, attached from another saved application, replaced with explicitly selected page text, or corrected in the collapsed manual editor. Retrieval uses local embeddings when available and visibly falls back to lexical retrieval if the embedding model is unavailable. F1 exact fields never enter model generation: they become either a cited deterministic verified fact or `ask_user`. F2 consequential fields are generated in bounded, field-scoped local micro-batches with targeted retry and safe per-field recovery.

Each proposal card shows:

- semantic confidence (`Verified`, `Strong evidence`, `Partial evidence`, `Low confidence`, or `Needs input`);
- a concise review or copy-only status when it applies;
- the reviewed answer or abstention reason;
- canonical citation cards resolved from the frozen stored evidence;
- compact **Fill**, **Copy**, and F2-only **Approve & fill** / **Revise** controls.

The compact application preflight reduces the page to **Ready** and **Needs attention** counts. Advanced controls stay collapsed: one clarification panel can strengthen multiple affected answers, and **Review issues** filters the list only when something needs attention. **Revise** opens optional feedback and regeneration using the same frozen evidence snapshot. A DOM revision, field fingerprint change, panel cancellation, lock, reset, or unload prevents stale generation from committing.

### 4. Fill safely

F1 values can be filled only from exact user-verified facts. For one F2 field, clicking **Approve & fill** is the explicit approval for that exact displayed draft; there is no second per-field checkbox. For bulk filling, one acknowledgement covers the currently displayed drafted answers and sources for that analysis. You can then fill every ready reviewed answer or choose **Verified only**; the extension never fills needs-input, manual, stale, or blocked controls. Before every fill, the panel rescans and compares the page ID, URL hash, DOM revision, field ID, and fingerprint. The runtime uses native setters plus `input`, `change`, and `blur`, then verifies the resulting field value. Failed verification becomes copy-only.

If a rendered interactive CAPTCHA control or challenge is present, the extension displays: **Interactive CAPTCHA challenge detected—filling paused. Complete it, then rescan.** Generated answers remain copyable, but mutation is blocked. Passive provider scripts, invisible integrations, hidden response fields, and badges do not block reviewed ordinary-field filling. The runtime never clicks buttons, interacts with CAPTCHA, agrees to terms, signs, consents, or submits.

### 5. Send reviewed metadata to the website tracker

After analysis, review the company, title, URL, location, source, dates, status, tags, and optional resume/cover-letter selections, then click **Review in tracker**. That deliberate click is the explicit extension-side handoff approval; there is no redundant confirmation checkbox.

The extension registers a short-lived session capability through `tracker-capture-begin`. The website first validates only manifest metadata. Original file chunks are requested only after the user selects **Review in application form** on the signed-in tracker, then sizes and full-file SHA-256 digests are verified before prefill. The tracker still requires the user's final Save action.

A handoff can contain at most one `resume` and one `cover-letter`, each a retained PDF or DOCX no larger than 10 MiB. The website protocol carries only the fixed job allowlist and selected file descriptors; it does not carry application packets, research, questions/answers, approved-answer snapshots, clarifications, custom links, custom thoughts, compensation preferences, notes, batch metadata, follow-up fields, or unrelated vault records.

## Ollama and diagnostics

Open **Settings** and select **Check Ollama**. The status surface reports the loopback version, locally verified primary/embedding models, blocked remote models, and an optional fallback warning. Every configured model is checked through `/api/show` before `/api/embed` or `/api/chat`; cloud/remote metadata fails closed. An untagged local model name is satisfied by its installed `:latest` tag. Clearly older Ollama versions show an update instruction; unknown future or prerelease formats are not rejected solely by syntax.

The side panel contains no cloud-model credential UI. Model-name preferences are the only panel settings stored in trusted `chrome.storage.local`.

For ATS debugging, analyze the page and select **Export sanitized fixture**. The JSON contains only schema version, adapter, CAPTCHA flag, actionable field labels/types/options/risk classes/fill modes, categorical discovery counts and limitations, and an opaque structural signature. It excludes page URL, entered values, answers, hidden fields, excluded sensitive fields, arbitrary page text, secrets, and source contents.

## Manual ApplicantPro QA

Use a non-production test application when possible. Do not submit during QA.

1. Load the unpacked extension and unlock a test vault.
2. Add a uniquely identifiable verified email and one small resume source; confirm the newest resume is marked **Recommended for this application**.
3. Open an ApplicantPro form and select **Analyze current page**.
4. Confirm the adapter badge reads `applicantpro` and the job metadata is bounded to company/title/location/source.
5. Verify CAPTCHA, file upload, consent/signature, EEO/demographic, security, hidden, and submit controls do not appear as actionable cards.
6. Confirm an exact email field uses only the verified fact and its canonical citation. A supervisor/reference email must remain `Needs input`.
7. Review one F2 text proposal. Confirm it shows its source, **Copy**, **Revise**, and **Approve & fill**; clicking **Approve & fill** is the explicit field approval and no separate checkbox is shown.
8. Fill a harmless test field and verify the page reports a controlled, post-update verified fill; confirm no navigation or submission occurs.
9. Change the form structure after analysis and verify the stale warning blocks fill until **Refresh answers**.
10. If the page has a rendered interactive CAPTCHA control or challenge, verify all fill actions pause and copy remains available. Verify a passive invisible-CAPTCHA badge does not pause ordinary-field filling.
11. Select a field from the in-page overlay and verify the side panel focuses that exact answer card. Use the panel's own **Copy** control.
12. Export the sanitized fixture and inspect it for structural fields only. Attach that fixture—not a screenshot with personal values—to an adapter bug report.
13. Exercise **Review in tracker** with zero files, resume only, cover letter only, and both. Verify the website asks for sign-in/review and never saves until the final explicit action.

## Verification commands

```text
npm test
npm run build
npm run package:dev
npm run verify:store
```

The test suite covers adapters, page scanning/filling, CAPTCHA/stale safety, parser formats, encrypted raw IndexedDB records, source/application isolation, two-page cache scoping, strict verified-fact mapping, F1 model exclusion, RAG/grounding, retry/fallback/cancellation, Ollama origin/version/model handling, tracker routing, and package asset integrity.

For a visual-only synthetic state after building, open `dist/sidepanel/sidepanel.html?preview=1` in a local browser. The query must be exactly `preview=1`; production operation never creates or loads the synthetic records.

## Security boundaries and known limits

- No automatic application submission, CAPTCHA solving, consent, signature, or demographic answering.
- No cloud model, cloud research fetch, remote Ollama host, or arbitrary network permission.
- No field-value scraping; the runtime reads structural labels/options/nearby text and writes only an explicitly reviewed proposal.
- Free-format discovery is deliberately bounded. Likely cross-origin application frames are diagnosed from visible frame structure, while closed Shadow DOM cannot be inferred merely because a component exposes no open root. Unsupported custom widgets, account gates, bot traps, and forms that cannot be confidently identified remain manual.
- No model-generated F1 identity/contact fill. Missing verified facts become `ask_user`.
- No legacy DOC parsing or OCR. Convert DOC to DOCX/PDF; run OCR on scanned PDFs or paste reviewed text.
- Source defaults are bounded by role and application: only the newest reusable resume/style records, reusable verified material, and current-application context are recommended automatically. Previous job/company context is dropped, unavailable application-scoped records stay disabled, and retrieval/cache keys include the application page scope.
- No plaintext source, original file, citation, embedding, or draft cache in `chrome.storage.local` or `chrome.storage.session`.
- The browser extension is a review assistant, not a representation that a job site's terms permit automation. Follow the employer and ATS rules.

## Main implementation entry points

- `src/sidepanel/sidepanel.js`: long-lived UI controller, encrypted source selection, retrieval/generation, fill safety, tracker handoff, and diagnostics.
- `src/vault/indexeddb-vault.js`: encrypted record vault and lock/reset lifecycle.
- `src/workers/document-parser.worker.js`: isolated PDF/DOCX/TXT/HTML parser entry.
- `src/rag/chunker.js`, `src/rag/retrieval.js`: deterministic chunks and hybrid retrieval.
- `src/ollama/client.js`: bounded loopback Ollama client.
- `src/ollama/generation-orchestrator.js`: retry, local fallback, cancellation, and lexical degradation.
- `src/grounding/postprocessor.js`: conservative answer acceptance and canonical citation-card resolution.
- `src/content/page-runtime.js`: structural scanning, overlay, controlled fill, post-fill verification, CAPTCHA and stale-page enforcement.
- `src/background/tracker-capture-router.js`, `src/shared/tracker-protocol.js`: short-lived reviewed website handoff.
