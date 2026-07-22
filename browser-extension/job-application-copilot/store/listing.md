# Chrome Web Store listing

## Product details

- **Name:** Job Application Copilot
- **Summary:** Draft evidence-grounded job application answers with local Ollama, source citations, and review-before-fill controls.
- **Category:** Productivity
- **Language:** English (United States)
- **Visibility:** Unlisted
- **Homepage:** https://www.danielshort.me/job-application-copilot
- **Support:** https://www.danielshort.me/job-application-copilot#support
- **Privacy policy:** https://www.danielshort.me/job-application-copilot/privacy

## Detailed description

Job Application Copilot helps you prepare job-application answers from sources you control. Import a resume, cover letter, job description, and reviewed research; save exact Website or custom-link values; and add custom thoughts such as a compensation preference in an encrypted local vault. Its source catalogue safely recommends the newest reusable resume and cover-letter material plus context for the current application, while keeping every available selection reviewable and adjustable. The extension retrieves relevant evidence, asks an Ollama model running on your own computer to draft grounded responses, and shows the exact source excerpts beside every supported proposal.

On first use, a concise disclosure explains the page structure the extension can inspect, the material stored in the local vault, the loopback Ollama request, the optional tracker handoff, and the actions the extension will never take. You must acknowledge that disclosure before application analysis or model use is enabled.

You remain in control of every field:

- Review one application step at a time in Chrome's side panel.
- See confidence and risk labels before using an answer.
- Open canonical source excerpts for every supported factual claim.
- Fill an exact verified value directly, use **Approve & fill** for one reviewed draft, or acknowledge the displayed drafts once before bulk filling.
- Open **Revise** to regenerate a prose response with optional feedback while keeping the same evidence boundary.
- Use dedicated or bounded profiles across major ATS platforms, with conservative free-format discovery for custom applications.
- Use verified profile facts, including Website, for conservative exact contact-field mappings.
- Add custom link labels for exact matching to URL or link/profile text fields.
- Add custom thoughts as citation-backed context for local generation; consequential answers still require review.
- Export a sanitized ATS diagnostic containing structure only, never entered answers or source contents.

The extension is intentionally limited. It never clicks Continue, Next, or Submit; never accepts an attestation; never signs or consents for you; never uploads a file into an employer form; and never solves or bypasses a CAPTCHA. When a rendered interactive CAPTCHA control or challenge is detected, DOM filling pauses until you complete it and refresh the analysis. Passive invisible-CAPTCHA badges do not block reviewed ordinary-field filling, and the CAPTCHA itself always remains untouched. Cross-origin application frames, closed Shadow DOM, and unsupported custom widgets remain manual. Demographic, disability, medical, veteran, criminal-history, government-ID, signature, attestation, and consent fields remain manual and are excluded from model context.

Your source documents, verified profile and custom links, custom thoughts or compensation preferences, extracted text, retrieval data, citations, and drafts stay encrypted in the browser profile. Relevant custom thoughts can enter source-cited model requests, which go only to Ollama at `127.0.0.1:11434`; there is no cloud AI provider, automated web research, analytics, advertising, or remote executable code in this version.

An optional handoff begins only when you review the displayed metadata and files and click **Review in tracker**. It can open Daniel Short's job tracker with reviewed job metadata and up to one selected resume and one selected cover letter. The tracker presents another review step and requires its own final Save action. Application questions, generated answers, research, citations, custom links, custom thoughts, compensation preferences, and the rest of the vault are never transferred.

Requirements: Chrome 120 or newer on Windows, a loopback-bound Ollama installation, and locally installed compatible chat and embedding models. See the support page for setup and troubleshooting.

## Asset assignments

- **Store icon:** `assets/icon-128.png` (128 x 128)
- **Small promotional tile:** `assets/small-promo-440x280.png` (440 x 280)
- **Screenshots:** capture the real built extension at 1280 x 800; do not upload a composited or fabricated UI image
- **Marquee promotional tile:** not supplied for v1
- **YouTube video:** not supplied for v1

## Localization note

The extension UI and listing are English-only in v1. Do not select additional languages until localized runtime strings, support content, and screenshots exist.
