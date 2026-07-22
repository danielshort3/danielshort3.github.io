# Chrome Web Store privacy declarations

Use this document as the source of truth when completing the dashboard Privacy practices form. Re-verify the wording against the current package before every release.

## Single purpose

**Paste-ready statement:**

Job Application Copilot helps a user review and prepare evidence-grounded answers for the job application in the active tab by using role-scoped local sources the user can review or change and a local Ollama model, with source citations and an explicit approval action before any consequential fill.

## First-use disclosure acknowledgement

Before application analysis or any Ollama request is enabled, the extension presents a versioned disclosure describing: active-page structure inspection; encrypted local-vault storage; loopback model processing; the optional, separately confirmed tracker handoff; excluded sensitive fields; and the prohibition on CAPTCHA solving, navigation, consent, signature, employer-file upload, and submission. The user must affirmatively acknowledge the disclosure. Dismissal leaves analysis and model actions disabled. A policy-text version change requires acknowledgement again.
The current disclosure separately identifies optional U.S. employment-eligibility selections. These encrypted Yes/No facts are used only through deterministic mappings, never sent to Ollama, and do not authorize citizenship, immigration-status, I-9/E-Verify, export-control, or protected demographic filling.

## Permission justifications

### `activeTab`

Used only after the user clicks **Analyze current page**, **Refresh answers**, or a field action. It limits scanning and controlled field filling to the currently active application tab rather than granting persistent access to every visited site.

### `scripting`

Used to inject the packaged, isolated page scanner and field-control runtime into the active tab after a user action. The runtime reads supported field structure, excludes current field values and prohibited fields, and fills only deterministic verified proposals or drafts approved by the user's individual or bulk review action.

### `storage`

Used for trusted extension settings and session state. Model-name preferences are stored in restricted `chrome.storage.local`; the temporary unlocked vault key is held in restricted `chrome.storage.session` and cleared on lock or browser-session end. Documents, extracted text, chunks, citations, embeddings, and draft caches are encrypted before IndexedDB storage.

### `sidePanel`

Used for the extension's primary review interface: vault controls, source selection, local-model status, proposal review, citations, regeneration feedback, and explicit fill/copy actions.

### Host access: `http://127.0.0.1:11434/*`

Used only to communicate with the user's loopback-bound Ollama service for model availability, embeddings, and structured answer generation. The client rejects remote hosts, `localhost`, alternate ports, and arbitrary paths. Responses are treated as data and never executed as code. Release setup requires Ollama local-only mode with OLLAMA_NO_CLOUD=1, and configured model names must resolve to installed local models rather than cloud-tagged models.

### Tracker content script match

The packaged tracker bridge runs only on `https://www.danielshort.me/tools/job-application-tracker*`. It supports a short-lived handoff of allowlisted job metadata and up to two explicitly selected original files after the user reviews the form and clicks **Review in tracker**. It does not receive Cognito or AWS credentials and does not transfer generated answers, application questions, research, citations, or unrelated vault contents.

## Remote code declaration

**Answer:** No, this extension does not use remote code.

All executable JavaScript, parser libraries, and workers are bundled in the submitted package. Ollama model output and website messages are parsed as constrained data; neither is executed, imported, or evaluated as code. The package uses no CDN scripts, `eval`, dynamic remote imports, WebAssembly fetched at runtime, or remotely hosted executable logic.

## User-data handling

Disclose the following categories if the dashboard presents them:

- **Personally identifiable information:** user-entered verified profile facts such as name, email, phone, address, the built-in Website URL, and custom profile links; processed locally and encrypted at rest.
- **Employment eligibility:** optional explicit U.S. work-authorization and current/future sponsorship Yes/No selections; encrypted locally, excluded from Ollama and tracker transfer, and used only for deterministic compatible questions.
- **User-provided content:** resumes, cover letters, job descriptions, company notes, custom thoughts or preferences, extracted text, selected original files, and user feedback; processed locally and encrypted at rest.
- **Financial and payment information:** user-entered compensation or salary preferences can be stored encrypted locally and sent only to the loopback Ollama service when relevant to a field. The extension does not process payment credentials or transactions, publicly disclose these preferences, or include them in tracker transfers.
- **Website content:** visible job metadata and actionable form labels, types, options, required state, and nearby instructional text from the active application page. Existing user-entered field values, hidden fields, and prohibited sensitive fields are not read into the model context.
- **Generated content:** proposed answers, abstention reasons, confidence/risk labels, and citations; processed locally and encrypted when cached.
- **Optional tracker-transfer data:** company, position, job URL, location, source, dates, tags, status, and up to one selected resume and one selected cover letter. Transfer begins only when the user clicks **Review in tracker** after reviewing those inputs, followed by a second website review.

Select the dashboard's financial/payment-information category if it is presented, because compensation preferences are handled even though processing is local-only. Do **not** declare browsing history, device location tracking, health data, authentication credentials, or payment credentials unless a future code change actually introduces such handling. A job's textual location is application metadata, not device geolocation.

## Data-use attestations

- Data is used only to provide and improve the user-facing extension workflow and the optional user-requested tracker handoff.
- Data is not sold.
- Data is not used for advertising, creditworthiness, lending, or personalized ads.
- Data is not used for analytics or behavioral profiling.
- Data is not provided to human reviewers by the extension.
- Local Ollama processing stays on the user's device. Release setup requires OLLAMA_NO_CLOUD=1, loopback binding, and verification that every configured model is installed locally and is not cloud-tagged.
- Citizenship, immigration/visa status, I-9/E-Verify, export-control, government identifiers, and protected demographic self-identification stay manual and out of model context.
- The optional tracker handoff sends only the allowlisted fields and selected files described above to the website's authenticated storage flow.
- The extension complies with the Chrome Web Store Limited Use requirements.

## Privacy-policy consistency gate

Before submission, confirm the hosted policy states all of the following:

1. what the encrypted local vault contains;
2. that the passphrase is not stored and reset deletes local vault data;
3. that Ollama requests use loopback only, OLLAMA_NO_CLOUD=1 is required, and no cloud-tagged model is accepted;
4. what page structure is scanned and which values/fields are excluded;
5. all CAPTCHA, submission, consent, signature, file-upload, and sensitive-field limits;
6. the exact optional tracker-transfer allowlist and two-stage review;
7. retention, lock/reset, tracker deletion, security, contact, and policy-change details.
8. the first-use disclosure, affirmative acknowledgement gate, and re-acknowledgement after material disclosure changes.

