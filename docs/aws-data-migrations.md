# AWS data migrations

These commands are intentionally dry-run by default. They scan one explicitly selected `production` or `preview` table, print counts and opaque record hashes only, and do not print credentials, destinations, snapshots, notes, or other item payloads.

## Prerequisites

1. Use Node 22 and an authenticated AWS SDK credential chain (for example, an AWS SSO profile). Do not pass access keys as command arguments.
2. Verify a current DynamoDB backup, point-in-time recovery, or export for the target table.
3. Pause Short Links admin writes while its reservation migration is being applied.
4. Review the dry-run summary before adding `--apply --backup-confirmed`.
5. Run Preview first, verify it, and then repeat for Production.

You may pass `--table` and `--region` explicitly or configure:

- Short Links: `SHORTLINKS_DDB_TABLE_PREVIEW`, `SHORTLINKS_DDB_TABLE_PRODUCTION`, or `SHORTLINKS_DDB_TABLE`
- Tools: `TOOLS_DDB_TABLE_PREVIEW`, `TOOLS_DDB_TABLE_PRODUCTION`, or `TOOLS_DDB_TABLE`
- Region: the service-specific region variable, `AWS_REGION`, or `AWS_DEFAULT_REGION`

The AWS identity running the Short Links command needs `dynamodb:DescribeTable`, `dynamodb:Scan`, and `dynamodb:TransactWriteItems`. The Tools command needs `dynamodb:DescribeTable`, `dynamodb:DescribeTimeToLive`, `dynamodb:Scan`, and `dynamodb:UpdateItem`.

## Short Links lowercase reservations

Dry run:

```powershell
npm run migrate:short-links-reservations -- --environment preview --table <preview-table> --region us-east-2
npm run migrate:short-links-reservations -- --environment production --table <production-table> --region us-east-2
```

The preflight groups live, non-internal links by normalized lowercase slug. Any case-insensitive collision, malformed reservation, invalid live slug, or reservation pointing to a different canonical slug aborts before writes. Collision identifiers are SHA-256 prefixes rather than raw slugs.

Apply only after the corresponding dry run and backup review:

```powershell
npm run migrate:short-links-reservations -- --environment preview --table <preview-table> --region us-east-2 --apply --backup-confirmed
npm run migrate:short-links-reservations -- --environment production --table <production-table> --region us-east-2 --apply --backup-confirmed
```

Each reservation is created in a conditional DynamoDB transaction that also verifies the canonical live-link record still exists. An existing reservation is accepted only when it points to the same canonical slug. The operation is idempotent; rerun the dry run and require `missingReservations: 0` after applying.

## Tools-account TTL backfill

The fixed policy is:

- session and session-index records: 365 days from their last update (or creation/encoded legacy timestamp)
- activity records: 90 days from their event timestamp
- user/tool metadata: durable, with the configured TTL attribute removed
- unrelated records, including Transcribe ledger entities in a shared table: unchanged

Enable DynamoDB TTL on `TOOLS_DDB_TTL_ATTRIBUTE` (default `ttl`) before applying. A dry run reports mismatched TTL configuration but never writes; apply mode aborts unless the correct attribute is `ENABLED` or `ENABLING`.

```powershell
npm run migrate:tools-ttl -- --environment preview --table <preview-table> --region us-east-2
npm run migrate:tools-ttl -- --environment production --table <production-table> --region us-east-2

npm run migrate:tools-ttl -- --environment preview --table <preview-table> --region us-east-2 --apply --backup-confirmed
npm run migrate:tools-ttl -- --environment production --table <production-table> --region us-east-2 --apply --backup-confirmed
```

Every update conditionally verifies the scanned TTL, entity type, and timestamp state before changing the item. Ambiguous managed records abort before writes. If a record changes during apply, the conditional update fails instead of overwriting the newer state. Rerun the dry run after applying and require both `ttlSetsPlanned: 0` and `ttlRemovalsPlanned: 0`.

Both migrations can be interrupted and safely rerun. A nonzero exit means the target should be rescanned and reviewed before continuing.
