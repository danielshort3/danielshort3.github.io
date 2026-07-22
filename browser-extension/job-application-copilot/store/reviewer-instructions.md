# Chrome Web Store reviewer instructions

These instructions are finalized for Store item `jigajpmnbiofgmgcnmdeechgibpjlfop` and submitted version `0.1.0`. No website account or reviewer credential is required for the core extension workflow.

## Environment

- Chrome 120 or newer on Windows
- Ollama running only on `http://127.0.0.1:11434`
- Store item ID: `jigajpmnbiofgmgcnmdeechgibpjlfop`
- Submitted version: `0.1.0`

Install the local models before testing:

```text
ollama pull qwen3.5:27b
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Allow only the submitted extension origin, then restart Ollama:

```text
OLLAMA_ORIGINS=chrome-extension://jigajpmnbiofgmgcnmdeechgibpjlfop
```

Set OLLAMA_NO_CLOUD=1 in the Ollama service environment as well as the exact OLLAMA_ORIGINS value above, then fully quit and restart Ollama. Keep the service loopback-bound.

Before review, verify that Ollama logs report cloud disabled as true and that ollama list shows the requested qwen and nomic models as installed local models with no cloud tag. The extension Settings check must list the same installed local model names. No OpenAI key, cloud-model account, paid service, or application-site login is needed to exercise the primary workflow.

## Core review path

Use a clean Chrome profile for the first-use check. The first panel open must present the versioned data-and-safety disclosure before application analysis or model use is available. Confirm that dismissing it leaves those actions disabled, then acknowledge it and continue. The acknowledgement must not authorize filling, tracker transfer, or submission; those remain separate actions.

1. Click the toolbar action to open the Chrome side panel.
2. In **Sources**, create a test vault using any passphrase of at least eight characters. The passphrase is not stored.
3. Add a test verified email such as `reviewer@example.com`.
4. Optionally set the three U.S. employment-eligibility selectors. Confirm the disclosure says these encrypted selections use deterministic rules and never enter Ollama prompts.
5. Add a synthetic custom link such as `Tableau Public profile` and a synthetic custom thought such as `Salary expectation: I am targeting the posted range.` Confirm both appear as encrypted local records and the thought is identified as local Ollama context.
6. Import a small text or PDF resume containing a non-sensitive sentence such as: `Built transparent analytics tools and presented findings to cross-functional stakeholders.` Confirm it is automatically marked **Recommended for this application**. The catalogue should show an **in use / saved** count; a newer resume replaces the older resume default without selecting old job/company context.
7. Open a non-production HTTP(S) form with a visible email input and a prose textarea. Do not submit the form.
8. Select **Analyze current page**. Confirm the email proposal comes only from the verified fact and shows its canonical citation.
9. Confirm the prose proposal shows **Review**, a source excerpt, **Copy**, **Revise**, and **Approve & fill**. There is no separate per-field confirmation checkbox; clicking **Approve & fill** explicitly approves that exact displayed draft.
10. Use **Approve & fill** only on a harmless test field. The extension uses a native setter and verifies the result; it never navigates or submits. For bulk filling, confirm one review acknowledgement covers the currently displayed drafted answers for that analysis.
11. If the synthetic form includes the matching link label or a salary-expectation question, confirm the custom link matches exactly and the thought appears only as a cited, review-required proposal.
12. Change the form structure, then try the old proposal. Confirm stale-page protection blocks filling until **Refresh answers**.
13. On a page with a rendered interactive CAPTCHA control or challenge, confirm all DOM filling pauses while proposals remain reviewable and copyable. On a page with only an invisible-CAPTCHA badge or passive provider integration, confirm reviewed ordinary-field filling remains available and the CAPTCHA itself stays untouched.
14. Open **Settings** and use **Check Ollama**. Confirm the status reports loopback availability and the same installed, non-cloud model names shown by ollama list. Confirm the Ollama service log reports cloud disabled as true.
15. Use **Export sanitized fixture** after a scan. Confirm the JSON contains only adapter, CAPTCHA state, field labels/types/options/risk classes, and no URL, entered values, answers, secrets, hidden fields, or source content.

## Compatibility review

Deep adapters cover ApplicantPro, Greenhouse, Lever, Ashby, SmartRecruiters, and Workday. Bounded structural profiles cover iCIMS, Oracle Recruiting/Taleo, BambooHR, ADP Recruiting, UKG, Jobvite, JazzHR, Recruitee, Pinpoint, and SAP SuccessFactors. Other pages use conservative free-format discovery with bounded JSON-LD metadata, accessible same-origin nested frames, and open Shadow DOM. A host match alone is not enough to expose controls.

Cross-origin application frames and closed Shadow DOM remain unsupported/manual. Custom Yes/No button groups and other non-native widgets can produce reviewable copy-only proposals, but the extension does not click them.

## Safety cases to verify

- Password, hidden, file-upload, submit, navigation, signature, consent, attestation, CAPTCHA, demographic, disability, medical, veteran, criminal-history, government-ID, and security-code fields do not become fillable proposals.
- Citizenship, immigration or visa status/type, permanent-resident/green-card status, I-9/E-Verify, export-control, and protected demographic self-identification remain manual; non-U.S. or ambiguous eligibility wording fails closed.
- A supervisor or reference email field does not reuse the applicant's verified email.
- Consequential prose requires a deliberate **Approve & fill** action for one field or one bulk-review acknowledgement for the current analysis.
- The extension never clicks Continue, Next, or Submit.
- The extension never uploads a document into an employer form.
- The extension does not access a cloud AI or search provider.
- A configured model name with a cloud tag is rejected or left unavailable; only locally installed model names may be used.

## Optional tracker handoff

The Daniel Short job tracker is not required to review the extension's primary purpose. If exercised, review the displayed metadata and selected files, then click **Review in tracker**. That deliberate click is the extension-side confirmation; no extra checkbox is shown. The tracker opens with a short-lived capture ID and presents a second review before its own Save action. A signed-out reviewer can inspect the pending-sign-in state. Do not save test data to the production tracker.

The transfer allowlist is company, position, job URL, location, source, dates, tags, status, and at most one selected resume plus one selected cover letter. Generated answers, questions, research, citations, custom links, custom thoughts, compensation preferences, and other vault contents are not sent.

## Visual review without personal data

The source repository contains an exact synthetic preview mode at `sidepanel/sidepanel.html?preview=1`. It displays clearly synthetic data and makes no Ollama or storage request. Store screenshots are captured from the real built side-panel renderer using that mode; they are not mockups of unsupported behavior.
