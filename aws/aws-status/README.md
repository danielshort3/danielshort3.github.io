# AWS Status history

This standalone CloudFormation stack owns the AWS Status dashboard's private Ohio history. The website's exact production Vercel OIDC subject can invoke the existing `website-aws-status` reader. The reader queries the dedicated status table and the website still requires an authenticated tools administrator. `website-aws-status-writer` handles scheduling and `website-aws-status-report` accepts authorized local and CI results. Each function uses its own execution role and 14-day log group.

## What the checks cover

| Schedule | Source | Evidence |
| --- | --- | --- |
| Every five minutes | Fixed public homepage HEAD and Cognito OIDC discovery GET | Homepage HTTP 200 and the exact Ohio Cognito issuer and JWKS URI. Discovery does not test a user login. |
| Every five minutes when configured | HEAD to one approved `https://dshort.me/<slug>` | The exact `https://www.danielshort.me/tools/aws-status` redirect without following it or recording a click. The slug is empty by default. |
| Hourly at minute 10 UTC | Five named Tracker and contact metric alarms | Selected alarm states. These do not prove a Tracker save or delivery of contact email. |
| Hourly at minute 10 UTC | `VGJ-Chatbot-Async` and `VGJ-Markdown-Async` | Endpoint metadata and optional maximum processing time from the previous completed hour. Zero active instances is expected for idle endpoints in service. |
| Hourly at minute 10 UTC | Existing Ohio Transcribe batch metrics | Accepted `StartTranscriptionJob` requests and batch server errors for the previous hour, at account level. No datapoints means unknown. |
| On state change | The same five Tracker and contact alarms | A recorded transition and a recovery when one is observed. Existing alarm actions and notifications continue independently. |

The collector uses bounded HTTPS reads, exact API requests, and one SDK attempt. It creates no alarms and sends no notifications, model requests, transcriptions, emails, or application writes. No schedules run while `ScheduleEnabled` is `false`.

## History and permissions

The on-demand DynamoDB table stores compact five-minute or hourly observation slots and up to 32 selected alarm transitions per connection per Mountain day. UTC timestamps keep incident times distinct across daylight-saving changes. Missing slots remain unknown. The reader returns 90 calendar tiles per connection using seven bounded partition queries. A failure tile shows one representative incident with observed bounds, exact alarm transition times when available, and whether recovery was observed. It does not infer uninterrupted monitoring or exact uptime from periodic samples.

History rows expire after 90 Mountain dates. TTL removes expired rows asynchronously, so the reader also enforces the date range and expiration. The table is retained on stack deletion or replacement. Only status evidence is stored, with no application records, raw alarm reasons, or response bodies. The Vercel role cannot invoke the scheduled collector or report writer. The reporter may write only the catalog partition and the collector may write only the fixed connection and homepage partitions.

The reader returns `{ok, checkedAt, region, days, connections, checks}`. `checkedAt` is null until an observation is recorded. Catalog results are keyed by source so the website can keep production, browser and local evidence distinct. The homepage HEAD result is marked `method: "homepage-head"`.

## Deploy and verify

Confirm the intended account through STS, then review an Ohio CloudFormation change set under the deliberate `website-admin` profile. The inline template requires `CAPABILITY_NAMED_IAM` and the existing team OIDC provider. Keep the initial change set at `ScheduleEnabled=false` and `ProbeShortLinkSlug=""`. Set the production-only Vercel variable `AWS_STATUS_FUNCTION_ARN` from `AwsStatusFunctionArn` and `AWS_STATUS_AWS_ROLE_ARN` from `AwsStatusAwsRoleArn`. Reviewed inline updates preserve the unqualified function ARN.

Run the focused checks from the repository root.

```sh
python3 -B aws/aws-status/test_collector.py
```

After the initial deployment, use the admin session to invoke the reader with `{}` and the writer with `{"operation":"probes"}` and `{"operation":"hourly"}`. Verify timestamps and observations on a second reader call and confirm separate read and write permissions. Enable the rules through a separate reviewed change set, then verify a delivery and the corresponding stored observation. To pause scheduled collection set `ScheduleEnabled=false` through CloudFormation.

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

### Hourly GitHub Actions reports

The [hourly workflow](../../.github/workflows/status-coverage-hourly.yml) is scheduled for minute 53 UTC. It selects the most recent successful Vercel production revision reported in GitHub and runs the catalog browser suite on a standard public-repository runner. The browser job has no AWS credentials. The separate publish job uses a 15-minute AWS session and sends only the result IDs, statuses and timestamps. Completed runs with test failures still publish their outcomes. An interrupted run leaves the previous report to expire.

Deploy the report code change in `template.yaml` first. Use temporary admin access to confirm the unqualified report function's exact code and obtain `CodeSha256` from `lambda get-function-configuration --function-name website-aws-status-report --region us-east-2 --query CodeSha256 --output text`. Review a separate Ohio change set for `github-reporter.yaml` with `EnableGithubReporter=true` and that `ReportCodeSha256`. The GitHub provider must be absent before creating the new stack. Confirm the default-branch OIDC subject and reusable publisher ref against the trust policy before activation.

The separate stack publishes the reviewed code as a version and points `browser-ci` at it. When changing report code, update this stack with the new deployed hash to move the alias. Only the exact GitHub main workflow identity can assume the publisher role, which can invoke only that alias. The alias accepts browser results with an expiry no more than three hours from observation and submission. Use `GithubReporterRoleArn` and `BrowserReportAliasArn` from that stack's outputs. The manual path above continues to invoke the unqualified function.

Verify one default-branch publish through the alias and confirm the browser result appears in the private dashboard. Confirm the same role cannot invoke the unqualified report, reader or scheduled collector, and confirm the alias rejects a non-browser source. Removing the separate stack disables CI publishing and removes its provider, so verify no other role uses that provider before stack deletion.
