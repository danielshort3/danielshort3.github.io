# Unlisted release checklist

## 1. Account and policy readiness

- [ ] Developer account registration is complete and the one-time registration fee is paid.
- [ ] The developer agreement and required account disclosures are accepted by the account owner.
- [ ] The developer email is verified.
- [ ] Two-step verification is enabled on the publishing Google account.
- [ ] The account owner has made the required trader/non-trader declaration based on their own legal situation.
- [ ] The hosted homepage, support section, and nested privacy policy at https://www.danielshort.me/job-application-copilot/privacy are live over HTTPS and match the submitted package.

## 2. Create the item and lock the production ID

- [ ] Build and validate an initial ZIP without publishing it.
- [ ] Upload it only far enough for the Developer Dashboard to create the item and assign a Web Store item ID/public key.
- [ ] Copy the dashboard's **public** key into the manifest `key` field. Never store a private signing key in the repository.
- [ ] Rebuild and confirm the unpacked extension ID equals the assigned Web Store item ID.
- [x] Reviewer instructions use assigned Store item ID `jigajpmnbiofgmgcnmdeechgibpjlfop`.
- [ ] Before Store review, configure Ollama with exactly `chrome-extension://jigajpmnbiofgmgcnmdeechgibpjlfop`; do not use a wildcard origin.
- [ ] Set OLLAMA_NO_CLOUD=1 for the Ollama service, restart it, confirm its log reports cloud disabled as true, and confirm every configured model appears locally in ollama list without a cloud tag.
- [ ] If the ID/key changed after the initial upload, increment the extension version, rebuild, and upload the corrected ZIP.

## 3. Package validation

- [ ] Run the extension test, build, and package workflows from a clean dependency install.
- [ ] Confirm the ZIP opens at its root with `manifest.json` there, not inside an extra directory.
- [ ] Confirm every manifest-referenced script, worker, HTML, CSS, and icon exists in the ZIP.
- [ ] Confirm no `node_modules`, source maps, test fixtures, real resumes, application data, secrets, private keys, store-working files, or unbundled source materials are present.
- [ ] Confirm the manifest is MV3 and contains no remote executable code path.
- [ ] Confirm permissions remain limited to `activeTab`, `scripting`, `storage`, `sidePanel`, loopback Ollama, and the production tracker bridge.
- [ ] Confirm the version in the ZIP matches `0.1.0` in reviewer instructions.

## 4. Store listing

- [ ] Paste the name, summary, detailed description, category, and English language from `listing.md`.
- [ ] Upload `assets/icon-128.png` and verify it is exactly 128 x 128.
- [ ] Upload `assets/small-promo-440x280.png` and verify it is exactly 440 x 280.
- [ ] Upload at least one genuine 1280 x 800 screenshot from the built extension; review every pixel for personal data, employer data, secrets, browser-profile details, and misleading functionality.
- [ ] Add the live homepage, support, and privacy-policy URLs.
- [ ] Keep the listing visibility **Unlisted**. Do not choose Public or domain-restricted Private distribution.

## 5. Privacy practices

- [ ] Paste the single-purpose statement from `privacy-dashboard-disclosures.md`.
- [ ] Provide each permission and host-access justification.
- [ ] Confirm a clean profile shows the versioned first-use disclosure and requires affirmative acknowledgement before analysis or any Ollama request.
- [ ] Declare **No remote code**.
- [ ] Disclose the locally handled profile, source, website-structure, generated, and optional tracker-transfer data categories conservatively.
- [ ] Certify Limited Use only after confirming the hosted policy and package agree.
- [ ] Recheck that no analytics, advertising, automated research, cloud AI, or hidden credential flow was added.
- [ ] Verify the listing, dashboard declarations, nested privacy policy, first-use disclosure, and reviewer notes describe the same local-only model and data boundaries.

## 6. Manual release QA

- [ ] Test installation from the uploaded draft item in a separate Chrome profile if the dashboard permits it.
- [ ] In a clean profile, verify dismissal of the first-use disclosure leaves application analysis and model actions disabled; acknowledge it, then run the core reviewer path with synthetic sources and a non-production form.
- [ ] Verify citations open the exact canonical excerpts used by the proposal.
- [ ] Verify F1 values come only from exact user-verified facts.
- [ ] Verify F2 prose requires review and regeneration remains within the frozen evidence set.
- [ ] Verify sensitive/unsupported fields are absent or manual-only.
- [ ] Verify a rendered interactive CAPTCHA control or challenge blocks every DOM fill until the user completes it and selects **Refresh answers**, while a passive invisible-CAPTCHA badge does not block reviewed ordinary-field filling.
- [ ] Verify stale DOM blocks filling.
- [ ] Verify no action submits, navigates, signs, consents, or uploads to an employer.
- [ ] Verify Ollama unavailable, missing-model, and fallback states are understandable.
- [ ] Verify Ollama runs with OLLAMA_NO_CLOUD=1, logs cloud disabled as true, configured models are local/non-cloud, and a cloud-tagged model cannot be selected for generation.
- [ ] Verify clicking **Review in tracker** after reviewing the handoff form is the explicit extension action and that the website still presents its own review; do not expect or add a redundant checkbox, and do not save production test data.

## 7. Submission gates

- [ ] The account owner has reviewed the final listing, privacy declarations, screenshots, and package hash.
- [ ] Obtain explicit user confirmation immediately before the final ZIP upload if it has not already been uploaded.
- [ ] Obtain explicit user confirmation immediately before **Submit for review** / **Publish**.
- [ ] Submit the item as **Unlisted**.
- [ ] Record the submitted version, SHA-256, Web Store item ID, submission timestamp, and dashboard status.
- [ ] After approval, install from the unlisted link in a clean Chrome profile and repeat a non-submitting smoke test.
- [ ] Share the unlisted link only with intended users; anyone who has the link can install it.
