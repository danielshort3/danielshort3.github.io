# Local status coverage

Build the checkout under test, then run the suite using the repository's installed Playwright browser.

    node tests/status-coverage/run.cjs --site-root C:/path/to/checkout --report C:/path/to/new/private-report.json

The suite reads the target checkout's catalog and built public files. It checks each route separately from the named browser action. The loopback server serves built files, and Playwright blocks outside traffic and service workers. Tool and provider fixtures live only in the test browser. A passed fixture demonstrates the local interaction. A production check requires its own evidence.

Use `--only tool:aws-status` or repeat `--only` to diagnose specific items. Reports from subset runs are for local debugging. Use a full run for the status dashboard. With `--report`, the destination must be a new absolute path in an existing folder. The JSON includes the check ID, environment, action, result, route result, method, time, duration, and a bounded reason. Keep reports private.

Exit code 0 means the report completed without a failed check. An unknown item is recorded separately. Exit code 1 means at least one route or functional check failed. Exit code 2 means the runner could not complete the report.

The managed checks reuse the site's focused browser fixtures and actual built pages. Some checks cover one local path, such as color-based background removal, simulated screen capture, media selection, or an external service response fixture. The report describes that coverage for each item.
