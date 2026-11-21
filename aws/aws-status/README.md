# System Status history

This standalone CloudFormation stack owns the private Ohio history for the [System Status dashboard](https://www.danielshort.me/tools/system-status). The website's exact production Vercel OIDC subject can invoke the existing `website-aws-status` reader. The reader queries the dedicated status table and the website still requires an authenticated tools administrator. `website-aws-status-writer` handles scheduling and `website-aws-status-report` accepts authorized local and CI results. Each function uses its own execution role and 14-day log group.

## What the checks cover

Monitoring cycles begin at 00:00, 06:00, 12:00 and 18:00 UTC. The following times are minutes after the start of each cycle.

| Schedule | Source | Evidence |
| --- | --- | --- |
| Every six hours at :10 | Fixed public homepage HEAD and Cognito OIDC discovery GET | Homepage HTTP 200 and the exact Ohio Cognito issuer and JWKS URI. Discovery does not test a user login. |
| Every six hours at :10 when configured | HEAD to one approved `https://dshort.me/<slug>` | The exact `https://www.danielshort.me/tools/aws-status` redirect without following it or recording a click. This existing short link targets the dashboard's legacy URL. The slug is empty by default. |
| Every six hours at :10 | Five named Tracker and contact metric alarms | Selected alarm states. These do not prove a Tracker save or delivery of contact email. |
| Every six hours at :10 | `VGJ-Chatbot-Async` and `VGJ-Markdown-Async` | Endpoint metadata and optional maximum processing time from the preceding measurement window. Zero active instances is expected for idle endpoints in service. |
| Every six hours at :10 | Ohio Amazon Transcribe batch job summaries | User job totals from the previous six hours and jobs still queued or running at the boundary. The scheduled test and its outcome are recorded separately. |

Each cycle also submits the [fixed speech recording](fixtures/status-probe.wav) to Amazon Transcribe in Ohio and checks the returned transcript against “the blue river flows past the garden.” Each cycle uses a distinct job name shared by its retries. Four short AWS transcription jobs are scheduled per day. User recordings are never used for this test.

The transcription test starts at :10. A terminal AWS event triggers verification of the returned text. At :35 a second check looks up the same job and records a timeout if it is still running. The private bucket expires test transcripts after seven days. The dashboard samples Tracker and contact alarms every six hours. Existing CloudWatch actions and notifications continue independently.

The collector uses bounded HTTPS reads, exact API requests, and one SDK attempt. It creates no alarms and sends no notifications, model requests, emails, or application writes. The dedicated transcription test uses private test storage. The AWS collector schedules are disabled while `ScheduleEnabled` is `false`.

## History and permissions

The on-demand DynamoDB table stores compact hourly observations per connection per Mountain day. Earlier five-minute samples and exact alarm transitions remain readable. UTC timestamps keep incident times distinct across daylight-saving changes. Missing slots remain unknown. The reader returns 72 hourly and 90 daily tiles per connection from the same bounded partition queries. A failure tile shows one representative incident with observed bounds, exact alarm transition times when available, and whether recovery was observed. Periodic samples show observed status and coverage.

History rows expire after 90 Mountain dates. TTL removes expired rows asynchronously, so the reader also enforces the date range and expiration. The table is retained on stack deletion or replacement. Only status evidence is stored, with no application records, raw alarm reasons, or response bodies. The Vercel role cannot invoke the scheduled collector or report writer. The reporter may write only the catalog partition and the collector may write only the fixed connection and homepage partitions.

The reader returns `{ok, checkedAt, region, days, connections, checks}`. For each connection, `days` contains Mountain calendar days and `hours` contains UTC buckets. The dashboard uses Mountain time for display. A UTC cycle occupies one timeline bar after the cutover, and older hourly observations keep their original timestamps. Daily and weekly views summarize the recorded checks. `checkedAt` is null until an observation is recorded. Catalog results are keyed by source so the website can keep production, browser and local evidence distinct. The homepage HEAD result is marked `method: "homepage-head"`.

All five groups publish together when the scheduled AWS, local, and browser observations are ready. Missing results show Unknown after the grace period. Refresh reads the latest stored observations.

The scheduled collector lists website batch jobs with read-only permission and records counts for the previous six hours. In the new timeline the AWS File Transcriber status comes from the test recording and transcript comparison. The Tools → File Transcriber browser check independently queues a fixture WAV. Older AWS bars retain the outcome or activity they originally measured.

### Home GPU history

Local services has a separate Home GPU timeline when `HomeGpuMonitoring=true`. The `Daniel Short Home GPU Heartbeat` Windows task checks the existing loopback worker health at :02 in each UTC cycle and publishes only `ready`, `unavailable`, or `unknown` over HTTPS to the dedicated Ohio IoT rule. The collector reads that heartbeat at :10. A received ready report is Online, an explicitly unavailable worker is Worker unavailable, and a missing report is Offline and can also indicate a connection or task interruption. No health history is backfilled. AWS Transcribe activity has its own row.

Deploy [the separate heartbeat stack](home-gpu-heartbeat.yaml) and use [the installer](install-home-gpu-heartbeat.ps1) from elevated Windows PowerShell. Run `-Phase Request -ExchangeFolder <folder>` to create a non-exportable machine key and public signing request. Use the temporary AWS administrator session to issue an Ohio IoT certificate from that request, attach only the `website-home-gpu-heartbeat` policy to it, and activate it. Put the certificate in `home-gpu-device.cer` and a JSON object with `endpoint`, `thumbprint`, and `scriptSha256` in `home-gpu-device.json` in the exchange folder. Then run `-Phase Install -ExchangeFolder <folder>`. Local Service needs private-key read access only. Confirm a report stored under `home-gpu-heartbeat` before enabling the monitor. AWS timestamps and filters the inbound state, and raw report rows expire after 48 hours. The normal dashboard retention applies to the displayed check history.

To pause monitoring, set `HomeGpuMonitoring=false` and disable the Windows task. The device certificate can be independently deactivated in AWS IoT. Keep the File Transcriber worker and its private network configuration separate from this task.

## Deploy and verify

Confirm the intended account through STS, then review an Ohio CloudFormation change set under the deliberate `website-admin` profile. Upload the template to a unique key in the private Ohio deployment-package bucket and use its S3 URL for the change set. The template requires `CAPABILITY_NAMED_IAM` and the existing team OIDC provider. Keep the initial change set at `ScheduleEnabled=false` and `ProbeShortLinkSlug=""`, and retain the configured values of `ScheduleEnabled`, `ProbeShortLinkSlug`, and `HomeGpuMonitoring` when updating a live stack. Set the production-only Vercel variable `AWS_STATUS_FUNCTION_ARN` from `AwsStatusFunctionArn` and `AWS_STATUS_AWS_ROLE_ARN` from `AwsStatusAwsRoleArn`. Reviewed reader updates preserve the unqualified function ARN.

Set `SixHourStart` to the first scheduled UTC boundary, for example `2026-11-21T06:00:00Z`. Keep that value on future updates so earlier observations continue to use their original cadence. Upload `aws/aws-status/fixtures/status-probe.wav` to `fixture/status-probe.wav` in the bucket named by the stack output `SystemStatusCanaryBucketName`. Run the heartbeat installer with `-Phase UpdateSchedule` from elevated Windows PowerShell to update an already enrolled machine.

After updating the main stack, pin its deployed report code to the existing `browser-ci` alias before publishing seven-hour browser reports. Update the Ohio scheduler with the new dispatcher package. Verify the scheduler expressions and the Windows task’s next UTC boundary.

Run the focused checks from the repository root.

```sh
python3 -B aws/aws-status/test_collector.py
```

After the initial deployment, use the admin session to invoke the reader with `{}` and the writer with `{"operation":"probes"}` and `{"operation":"hourly"}`. Verify timestamps and observations on a second reader call and confirm separate read and write permissions. Enable the rules through a separate reviewed change set, then verify a delivery and the corresponding stored observation. To pause scheduled collection set `ScheduleEnabled=false` through CloudFormation.

To verify the transcription during a cycle, invoke the writer with `{"operation":"canary"}` and then `{"operation":"canary-result"}`. Both operations use the current UTC cycle. The terminal result `passed` requires the expected transcript content. Confirm the same result in the reader and in the AWS Connections row of System Status.

## Submit local or CI results

The stack outputs `AwsStatusReportFunctionArn` and an unattached `AwsStatusReportInvokePolicyArn`. The managed policy grants only invocation of the report function. Attach it only to a separately reviewed temporary-access role authorized to submit check results. The website role is for reading. The report handler has no collection or probe operation.

Invoke the report synchronously with one to 70 unique check IDs and only the five fields shown below. IDs use `tool:<slug>`, `project:<id>` or `game:<id>`. States are `operational`, `degraded`, `outage`, and `unknown`. Sources are `production`, `browser`, and `local`. Use current ISO UTC timestamps and an expiry after the observation, up to 90 days. The website may apply shorter freshness limits. Each source keeps up to 70 unexpired latest check IDs. An older observation cannot replace newer evidence from the same source, and re-submitting an exact result is idempotent.

```json
{
  "checks": [
    {
      "id": "tool:text-compare",
      "state": "operational",
      "lastChecked": "2026-11-13T03:00:00.000Z",
      "source": "local",
      "expiresAt": "2026-11-14T03:00:00.000Z"
    }
  ]
}
```

Require `ok: true` and the expected `accepted` count before treating the report as published. The stored issue text comes from the result state, so submit only the enum and timestamp fields. A failed mixed-source invocation may have saved an earlier source. A repeat with the same input is safe and leaves newer evidence in place.

### Six-hour GitHub Actions reports

The Ohio scheduler starts the [browser workflow](../../.github/workflows/status-coverage-hourly.yml) at :47 in the opening hour of each UTC cycle. Before starting the suite, the workflow checks whether another run already started browser checks for the same cycle. This applies to dispatcher and manual starts. If a report is missing, check the [run history](https://github.com/danielshort3/danielshort3.github.io/actions/workflows/status-coverage-hourly.yml) for the browser and publish steps. Browser results expire seven hours after the check.

The workflow selects the most recent successful Vercel production revision reported in GitHub and runs the catalog browser suite on a standard public-repository runner. The browser job has no AWS credentials. The separate publish job uses a 15-minute AWS session and sends the result IDs, statuses, timestamps, and optional allowlisted browser reason codes. Completed runs with test failures still publish their outcomes. An interrupted run leaves the previous report to expire.

Vercel compares changes against its last successful deployment for the branch. A commit limited to the scheduled workflow and its listed support files can skip the website build while the new workflow still runs from `main`. Changes to the site, API, fixtures, or AWS template trigger the website build.

Deploy the report code change in `template.yaml` first. Use temporary admin access to confirm the unqualified report function's exact code and obtain `CodeSha256` from `lambda get-function-configuration --function-name website-aws-status-report --region us-east-2 --query CodeSha256 --output text`. Review a separate Ohio change set for `github-reporter.yaml` with `EnableGithubReporter=true` and that `ReportCodeSha256`. The GitHub provider must be absent before creating the new stack. Confirm the default-branch OIDC subject and reusable publisher ref against the trust policy before activation.

The separate stack publishes the reviewed code as a version and points `browser-ci` at it. When changing report code, update this stack with the new deployed hash to move the alias. Only the exact GitHub main workflow identity can assume the publisher role, which can invoke only that alias. The alias accepts browser results with an expiry no more than seven hours from observation and submission. Use `GithubReporterRoleArn` and `BrowserReportAliasArn` from that stack's outputs. The manual path above continues to invoke the unqualified function.

For Projects & Tools history, deploy the reader and report code in `template.yaml` first, preserving the existing stack parameters. In `github-reporter.yaml`, retain the original `ReportCodeSha256`, set `HistoryReportCodeSha256` to the new report code hash, and set `UseCatalogHistory=true`. Review the change set for the additional pinned version and the existing alias target update. Set `UseCatalogHistory=false` to move the alias back while retaining both versions.

The browser alias preserves the latest complete report for each elapsed UTC hour in a Mountain calendar-day row under the existing catalog partition. Archived rows expire after 90 Mountain dates, independent of live result freshness. The dashboard groups new observations into six-hour cycles and retains historical hourly, calendar-day, and weekly views. Earlier periods and missing checks appear as Unknown. History records browser fixture outcomes and uses the report's observation time. It starts with the first archived run.

Verify one default-branch publish through the alias and confirm the browser result appears in the private dashboard. Confirm the same role cannot invoke the unqualified report, reader or scheduled collector, and confirm the alias rejects a non-browser source. Removing the separate stack disables CI publishing and removes its provider, so verify no other role uses that provider before stack deletion.

### Six-hour browser scheduler

The [Ohio scheduler stack](github-fallback.yaml) dispatches the browser workflow on main at :47 of the opening UTC hour when the cycle has no published report. GitHub serializes runs and checks whether the browser suite already started in that cycle. Report timestamps reflect when the browser checks completed.

Install a private GitHub App on only this repository with Actions read and write and mandatory metadata read. Store its client ID, installation ID and GitHub-generated signing key in the Standard SecureString parameter `/website-aws-status/fallback/github-app` in the Ohio account. Keep the key out of the repository and workflow. The dispatcher uses it to request a short-lived token limited to this repository. It checks the token scope before using it.

Run `node --test aws/aws-status/fallback-dispatcher.test.cjs`. Zip only `fallback-dispatcher.cjs` at the archive root and upload the archive to a new `website-status-fallback/` object key in the Ohio deployment package bucket. Review the CloudFormation change set before activation. The schedule defaults to disabled. Set `EnableSchedule=true` to start it, or `false` to pause it without changing the website.

Verify a scheduled invocation by matching its dispatcher log to the workflow run ID, a successful publish step, and the timestamp in the live Projects & Tools tab. Check the dispatcher logs for skipped, dispatched and failed requests. Keep the latest working package for rollback.
