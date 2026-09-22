# Release and browser security

## Production approval

CI audits all three lockfiles and runs the build/test, cross-browser release,
recovery and mobile-performance jobs. `release-ready` fails if any dependency
fails, is cancelled or is skipped. The Vercel build command separately checks
that all required jobs passed in the main push workflow for the exact commit
being deployed. Missing metadata, API failures and a 20-minute deadline fail
closed. Local and preview builds do not receive production approval.

The repository owner must additionally require pull requests and `release-ready`
on `main` (including up-to-date branches), and configure Vercel Deployment Checks
to require that same GitHub check. Keep bypass privileges limited. These native
controls cannot be installed by adding a workflow file. The build guard does not
prevent an administrator from manually promoting an already-built preview.
An intentional rollback to an older deployment is an owner action, not an excuse
to disable required checks. Retry a failed production build only after its exact
main commit passes CI. GitHub API unavailability may delay a release; it must not
silently grant approval. No additional deployment credential is required for the
public repository's read-only verification.

## Content Security Policy

Public HTML disallows arbitrary inline script execution and inline event
handlers. The legacy raw demo documents use reviewed SHA-256 script hashes.
The allowlist is generated from the completed build, never from request bodies
or user content. Sensitive tracker routes retain their narrower same-origin
policies; WebAssembly and existing tool-specific evaluation permissions are not
removed indiscriminately.

After changing an authored inline demo script, run:

```sh
npm run build
node build/csp-hashes.cjs --write
npm run test:security
npm run test:csp:browser
```

Review the new hashes and source changes together. Commit the config, not
`public/` or local browser artifacts. Normal CI checks hashes without rewriting
them. Scripts added outside the owned demo directory fail the scan.

**CMS preview compatibility exception:** `/admin/` is the only surface retaining
its existing inline-script policy. Its editor builds dynamic `srcdoc` previews
(including tool highlighting and review selections). Arbitrary user-supplied
preview text must not be added to the public hash list. This exception does not
apply to visitor pages, tools, accounts or APIs. Removing it requires converting
the editor's dynamic preview markers to external scripts/data and validating its
editable-content flows; blanket removal would silently break the editor. The
policy test prevents the exception from expanding beyond this authoring route.

## Mobile lab budgets

The three-run median LCP ceiling is 3,500 ms, TBT is 200 ms and CLS is 0.1.
The existing tighter per-route regression tolerances and reviewed baseline are
also enforced; neither is reset automatically. Invalid or missing metrics fail.
The LCP ceiling is a staged laboratory constraint, not a claim of good real-user
Core Web Vitals. The field LCP goal remains 2,500 ms; review real-user p75 LCP,
INP and CLS separately, by device class and with sample sizes. TBT is not INP.
