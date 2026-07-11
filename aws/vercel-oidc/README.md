# Vercel OIDC roles for AWS

These templates replace persistent AWS access keys in Vercel with short-lived credentials from `AssumeRoleWithWebIdentity`. They intentionally bind trust to the team issuer, project, and one deployment environment:

- Issuer: `https://oidc.vercel.com/daniel-shorts-projects`
- Audience: `sts.amazonaws.com`
- Subject: `owner:daniel-shorts-projects:project:website:environment:<production|preview>`

The production and preview roles must point at different tables, buckets, and Lambda aliases. Do not reuse production resource names in the preview parameter file.

## Templates

- `provider.template.yaml` creates the one account-level, team-scoped IAM OIDC provider. It is retained if its stack is deleted.
- `environment-resources.template.yaml` creates the isolated Preview tables/bucket and one demo rate-limit table per environment. Its Production deployment creates only `website-demo-rate-limit-production`; it never adopts or replaces existing Production tables or the Production Transcribe bucket.
- `roles.template.yaml` is deployed twice, once with `Environment=preview` and once with `Environment=production`.
- The `*-resources.parameters.example.json` files drive the environment-resource stacks; the other Production/Preview parameter files drive the role stacks.
- The example parameter files are non-secret inventories reconciled against the July 11 backup manifests and live read-only AWS checks.
- `resource-resolution.json` records the evidence, every verified value, and the deployed Production and Preview stack status.

## Resource reconciliation status

Any parameter file containing a value beginning with `__UNRESOLVED_` is not deployable. These sentinels deliberately fail the cutover gate instead of granting Preview access to Production or inventing an ARN for a resource that does not exist.

Production is deployed and verified for Tools, Short Links, Chatbot DynamoDB, Transcribe S3, Bedrock, the verified `noreply@danielshort.me` SES sender, all ten immutable `:live` demo aliases, and `website-demo-rate-limit-production`. The new rate-limit table is owned by the Production deployment of `environment-resources.template.yaml`.

Preview is deployed and verified. `environment-resources.template.yaml` owns four isolated DynamoDB tables, the encrypted/private/unversioned three-day Transcribe bucket, and `website-demo-rate-limit-preview`; `aws/workloads/demo-preview.template.yaml` owns ten Preview-specific functions and immutable `:live` aliases. Preview intentionally reuses the verified SES sender through a separate Contact role. The Bedrock model resources and account-level OIDC provider are intentionally shared; the assuming roles and trust subjects remain environment-specific.

## Role outputs

Map each stack output to the same-named Vercel environment variable in only that environment:

| Stack output | Vercel variable |
| --- | --- |
| `ToolsAwsRoleArn` | `TOOLS_AWS_ROLE_ARN` |
| `ShortLinksAwsRoleArn` | `SHORTLINKS_AWS_ROLE_ARN` |
| `TranscribeAwsRoleArn` | `TRANSCRIBE_AWS_ROLE_ARN` |
| `ChatbotBedrockAwsRoleArn` | `CHATBOT_BEDROCK_AWS_ROLE_ARN` |
| `ChatbotDdbAwsRoleArn` | `CHATBOT_DDB_AWS_ROLE_ARN` |
| `ContactAwsRoleArn` | `CONTACT_AWS_ROLE_ARN` |
| `DemoInvokeAwsRoleArn` | `DEMO_INVOKE_AWS_ROLE_ARN` |

Set `AWS_OIDC_AUDIENCE=sts.amazonaws.com` and `AWS_AUTH_MODE=oidc` only after the corresponding role variables and environment-specific resources exist. Vercel supplies the OIDC token during builds and functions; do not add `VERCEL_OIDC_TOKEN` manually.

The demo role is used only by the static `/api/demos/<project>/<operation>` registry. Configure these environment-specific, `:live`-qualified aliases; the handler rejects unqualified ARNs, arbitrary function names, unknown projects, and unknown operations:

- `DEMO_SHAPE_FUNCTION_ARN`
- `DEMO_SMART_SENTENCE_FUNCTION_ARN`
- `DEMO_NONOGRAM_FUNCTION_ARN`
- `DEMO_HANDWRITING_FUNCTION_ARN`
- `DEMO_DIGIT_GENERATOR_FUNCTION_ARN`
- `DEMO_COVID_OUTBREAK_FUNCTION_ARN`
- `DEMO_PIZZA_TIPS_FUNCTION_ARN`
- `DEMO_TARGET_EMPTY_PACKAGE_FUNCTION_ARN`
- `DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN`
- `DEMO_MINESWEEPER_FUNCTION_ARN` (retained unpublished/noindex)

The legacy Whisper Function URL and Bedrock chatbot streaming Lambda are intentionally excluded; they use the Transcribe and Chatbot workload boundaries. Set `DEMO_RATE_LIMIT_TABLE` to a DynamoDB table with string partition key `pk`, string sort key `sk`, and TTL attribute `ttl`. Set `DEMO_REQUIRE_DDB_RATE_LIMIT=true` and a random `DEMO_HASH_SALT` before production Lambda-mode cutover.

For contact delivery, configure `CONTACT_SENDER_EMAIL` as a verified SES identity and `CONTACT_RECIPIENT_EMAIL`, then change `CONTACT_DELIVERY_MODE` from `legacy` to `ses`. The direct SES handler preserves the public `{ ok, error? }` response shape and does not expose SES or upstream errors.

## Staged deployment

1. Validate all three templates and create change sets. Inspect every resource name, IAM action, resource ARN, role name, trust audience, and exact subject before execution.
2. Deploy the provider stack once. Deploy `environment-resources.template.yaml` and `aws/workloads/demo-preview.template.yaml` for Preview, verify their outputs, then deploy the Preview roles.
3. Keep `AWS_AUTH_MODE=legacy`, `CONTACT_DELIVERY_MODE=legacy`, and `DEMO_PROXY_MODE=legacy` during the first code deployment. Add Preview role variables and Preview-specific `:live` demo aliases, switch only Preview to OIDC/SES/Lambda modes, and run positive and negative permission tests.
4. Deploy `environment-resources.template.yaml` for Production to create only its demo rate-limit table, then deploy the Production roles with the completed parameters. Add Production role variables, switch Production to `oidc`, and validate CloudTrail role attribution.
5. Preserve two known-good OIDC deployments before deactivating static keys. The legacy mode is a temporary rollback path, not a permanent credential strategy.
6. After the stability window, remove Vercel static-key variables and delete the legacy branches from `api/_lib/aws-credentials.js` in a follow-up change.

Before each Preview or Production cutover, run `npm run test:aws-auth` and `npm run audit:dependencies`. The repository pins the AWS SDK family above the credential/XML advisory ranges present before this migration; the audit must report zero vulnerabilities.

Local Node processes leave `AWS_AUTH_MODE` blank and use the AWS SDK default chain. Set `AWS_PROFILE=website-readonly`, `website-operator`, or `website-admin` after signing in through IAM Identity Center. A local OIDC test may instead pull a short-lived Vercel token and explicitly set `AWS_AUTH_MODE=oidc` plus one workload role.

References: [Vercel OIDC](https://vercel.com/docs/oidc), [Vercel OIDC reference](https://vercel.com/docs/oidc/reference), and [AWS IAM OIDC roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html).
