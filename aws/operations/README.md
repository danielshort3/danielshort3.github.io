# AWS operations artifacts

`MIGRATION-STATUS-2026-07-11.md` is the secret-free live-state, rollback, and timed-follow-up handoff for the migration.

## ECR retention

`ecr-lifecycle-policy.json` is the canonical policy for the eight site image repositories. It retains the ten newest `release-*` images and expires untagged images 30 days after their original push time.

The July 11 migration checkpoint added 80 `backup-20260711-*` tags to older images. ECR lifecycle age is based on `imagePushedAt`, not the time a tag was attached. All 80 images were already 59–335 days old on July 11, so adding a lifecycle rule such as `tagPrefixList: ["backup-20260711-"]` with `sinceImagePushed: 30 days` would immediately select every checkpoint image and destroy the intended recovery window.

`ecr-backup-retirement.template.yaml` handles that one-time case safely. It creates eight EventBridge Scheduler universal targets for `ecr:BatchDeleteImage`, one per repository, scheduled for `2026-08-11T00:00:00` UTC. Each request removes only the exact backup tags captured on July 11. It cannot select a digest, a release tag, another repository, or a later backup series. Removing a last tag may delete that image; removing one of several tags leaves the other tags intact.

The Scheduler execution role is limited to `ecr:BatchDeleteImage` on these repositories:

- `nonogram-solver-lambda`
- `shape-analyzer`
- `digit-generator-lambda`
- `minesweeper-solver`
- `whisper-transcribe`
- `vgj-chatbot-inference`
- `smart-sentence-finder-lambda`
- `handwriting-rating`

Validate before deployment:

```powershell
Get-Content aws/operations/ecr-lifecycle-policy.json -Raw | ConvertFrom-Json | Out-Null
cfn-lint aws/operations/ecr-backup-retirement.template.yaml
aws cloudformation validate-template --region us-east-2 --template-body file://aws/operations/ecr-backup-retirement.template.yaml
```

After August 11, verify the schedule invocations in CloudTrail and confirm no `backup-20260711-*` tags remain. Delete the CloudFormation stack only after that evidence is recorded. Do not manually delete image digests.
