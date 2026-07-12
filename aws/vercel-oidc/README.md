# Vercel OIDC roles

`template.yaml` is the source of truth for the environment-scoped AWS roles used by Vercel. Deploy it once as `website-vercel-oidc-production` and once as `website-vercel-oidc-preview`.

Each role trusts only the exact Vercel team, project, environment subject and the `sts.amazonaws.com` audience. Keep production and preview resources separate except for explicitly shared immutable Lambda aliases such as the chatbot stream `:live` alias.

## Update workflow

1. Validate the template:

   ```powershell
   aws cloudformation validate-template --template-body file://aws/vercel-oidc/template.yaml --region us-east-2
   ```

2. Create a CloudFormation change set with the stack's existing parameter values. Update only the intended allowlists or resource names.
3. Review every resource action and require `Replacement: False` for IAM-only changes.
4. Execute the change set and wait for `UPDATE_COMPLETE`.
5. Use `aws iam simulate-principal-policy` to verify intended resources are allowed and unqualified or cross-environment resources remain denied.
6. Deploy a Vercel preview and exercise each OIDC-backed API before deleting static-key rollback variables.

The `DemoFunctionArns` parameter must contain qualified alias ARNs, never mutable unqualified functions. `/api/chatbot-stream` additionally requires `CHATBOT_STREAM_FUNCTION_ARN` in Vercel and invokes the alias through `DEMO_INVOKE_AWS_ROLE_ARN` unless a dedicated `CHATBOT_STREAM_AWS_ROLE_ARN` is configured.

The Tools role needs `dynamodb:ConditionCheckItem` in addition to the normal
read/write actions because its transactional session and activity writes use an
explicit DynamoDB `ConditionCheck` deletion guard.

Do not commit account IDs, tokens, access keys, session tokens, or environment-specific parameter files.
