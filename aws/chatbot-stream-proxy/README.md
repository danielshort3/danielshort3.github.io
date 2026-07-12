# Private chatbot response-streaming bridge

`/chatbot-demo` sends Bedrock prompts only to the same-origin
`POST /api/chatbot-stream` Vercel Function. The function validates the browser
origin and JSON payload, applies the shared chatbot DynamoDB rate limit, assumes
a Vercel OIDC role, and calls AWS Lambda with `InvokeWithResponseStream`.

The browser must not call a Lambda Function URL directly and must never receive
AWS credentials. The Vercel function forwards bounded newline-delimited JSON
events (`meta`, `token`, `done`, and a sanitized `error`).

## Repository contract

- `CHATBOT_STREAM_FUNCTION_ARN` is mandatory and must be a qualified Lambda
  alias ARN. Production rejects anything except the `live` alias.
- `CHATBOT_STREAM_AWS_ROLE_ARN` is optional. If omitted, the function reuses
  `DEMO_INVOKE_AWS_ROLE_ARN`.
- AWS authentication uses the shared `AWS_AUTH_MODE` and
  `AWS_OIDC_AUDIENCE` settings. Production fails closed unless the resolved
  credential source is Vercel OIDC.
- Rate limiting reuses `CHATBOT_DDB_TABLE`, `CHATBOT_DDB_AWS_ROLE_ARN`,
  `CHATBOT_HASH_SALT`, and `CHATBOT_REQUIRE_DDB`. The stream proxy deliberately
  ignores the user-controlled `x-chatbot-session` header so actor limits remain
  IP-based.
- The Vercel boundary accepts only same-origin JSON `POST` requests, at most
  8 KiB per body and 1,200 characters per prompt. It stops the upstream call
  after 35 seconds by default and forwards at most 512 KiB. The existing
  `followup_context` is preserved only through its explicit bounded schema;
  unknown fields are discarded.
- Lambda response streaming begins with HTTP metadata JSON and an eight-NUL
  delimiter. The proxy requires and validates that prelude, strips it, and
  exposes only the following NDJSON body to the browser.

## Live-state snapshot before rollout

Read-only inspection on 2026-07-11 found:

- Function: `VGJBedrockStream` in `us-east-2`.
- Function URL: `AWS_IAM` with `RESPONSE_STREAM` invoke mode.
- Function versions: only `$LATEST`; no aliases existed.
- Runtime: Node.js 20, 30-second timeout, 512 MiB memory.
- Production OIDC role: `website-demo-invoke-production`.
- The role trust is restricted to the Vercel project `website`, team
  `daniel-shorts-projects`, environment `production`, and audience
  `sts.amazonaws.com`.
- Its `allowlisted-demo-functions` policy grants `lambda:InvokeFunction` only
  to the existing ten buffered demo aliases, not `VGJBedrockStream`.
- The Vercel production project already has `DEMO_INVOKE_AWS_ROLE_ARN` and all
  shared chatbot rate-limit variables. It does not yet have
  `CHATBOT_STREAM_FUNCTION_ARN`.

AWS documents that `InvokeWithResponseStream` requires
`lambda:InvokeFunction`; it does not require `lambda:InvokeFunctionUrl`.
See [InvokeWithResponseStream](https://docs.aws.amazon.com/lambda/latest/api/API_InvokeWithResponseStream.html)
and [Lambda response streaming](https://docs.aws.amazon.com/lambda/latest/dg/configuration-response-streaming.html).

## Required cloud changes

Do these in order. They are intentionally not performed by repository tests.

### 1. Publish and pin the production function

Confirm `$LATEST` is the intended build, then publish an immutable version and
point `live` at it:

```powershell
$version = aws lambda publish-version `
  --function-name VGJBedrockStream `
  --region us-east-2 `
  --description "Private Vercel stream proxy release" `
  --query Version `
  --output text

aws lambda create-alias `
  --function-name VGJBedrockStream `
  --name live `
  --function-version $version `
  --region us-east-2
```

If `live` already exists, use `update-alias` instead of `create-alias`. Never
configure production with `$LATEST` or an unqualified function ARN.

### 2. Update the CloudFormation-managed OIDC role

Update the source template for stack `website-vercel-oidc-production`. Append
this exact ARN to the existing `DemoFunctionArns` parameter/policy resource
list and deploy the stack normally:

```text
arn:aws:lambda:us-east-2:<account-id>:function:VGJBedrockStream:live
```

The resulting least-privilege statement is:

```yaml
- Effect: Allow
  Action: lambda:InvokeFunction
  Resource: arn:aws:lambda:us-east-2:<account-id>:function:VGJBedrockStream:live
```

Do not apply a one-off `put-role-policy`; that would drift from the
CloudFormation-managed `allowlisted-demo-functions` policy.

For Preview support, keep the preview OIDC role environment-specific. It may
invoke the shared immutable `live` alias when a rollout deliberately validates
the exact production candidate in both environments. Otherwise, publish or
reuse an immutable version behind a separate `preview` alias and allowlist that
qualified ARN only in `website-vercel-oidc-preview`.

### 3. Configure Vercel and redeploy

Add this encrypted Production environment variable:

```text
CHATBOT_STREAM_FUNCTION_ARN=arn:aws:lambda:us-east-2:<account-id>:function:VGJBedrockStream:live
```

The existing `DEMO_INVOKE_AWS_ROLE_ARN` is sufficient. Set
`CHATBOT_STREAM_AWS_ROLE_ARN` only if a dedicated streaming role is created.
After changing Vercel environment variables, create a new production
deployment; existing deployments do not acquire the new value.

### 4. Verify without reopening the Function URL

First confirm the OIDC role policy resolves to the alias:

```powershell
aws iam simulate-principal-policy `
  --policy-source-arn arn:aws:iam::<account-id>:role/website-demo-invoke-production `
  --action-names lambda:InvokeFunction `
  --resource-arns arn:aws:lambda:us-east-2:<account-id>:function:VGJBedrockStream:live
```

Then issue one same-origin production canary and confirm that the response is
`application/x-ndjson`, contains incremental `token` events followed by one
`done` event, includes a numeric `X-Chatbot-Lambda-Version`, and never exposes
AWS errors or credentials. Also verify that a cross-origin request is rejected,
an oversized prompt returns `413`, and an immediate repeat returns `429`.

## Function URL cleanup

The SDK bridge does not use the Function URL. During rollout it can remain
`AWS_IAM`; after the same-origin canary passes, delete the unused URL or retain
it only for an explicitly documented signed caller. Remove the stale resource
policy statement `AllowPublicFunctionUrlInvoke`, whose condition still refers
to `AuthType: NONE`, to keep the function policy unambiguous.

Do not restore anonymous Function URL access as a rollback. Roll back the site
deployment and alias/IAM changes together instead.
