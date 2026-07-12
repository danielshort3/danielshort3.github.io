# AWS data migrations

These commands are intentionally dry-run by default. They scan one explicitly selected `production` or `preview` table, print counts and opaque record hashes only, and do not print credentials, destinations, snapshots, notes, or other item payloads.

## Prerequisites

1. Use Node 22 and an authenticated AWS SDK credential chain (for example, an AWS SSO profile). Do not pass access keys as command arguments.
2. Verify a current DynamoDB backup, point-in-time recovery, or export for the target table.
3. Pause Short Links admin writes while its reservation migration is being applied. Pause redirect/click writes for the click reconciliation migration.
4. Review the dry-run summary before adding `--apply --backup-confirmed`.
5. Run Preview first, verify it, and then repeat for Production.

You may pass `--table` and `--region` explicitly or configure:

- Short Links: `SHORTLINKS_DDB_TABLE_PREVIEW`, `SHORTLINKS_DDB_TABLE_PRODUCTION`, or `SHORTLINKS_DDB_TABLE`
- Short Links clicks: `SHORTLINKS_DDB_CLICKS_TABLE_PREVIEW`, `SHORTLINKS_DDB_CLICKS_TABLE_PRODUCTION`, or `SHORTLINKS_DDB_CLICKS_TABLE`
- Tools: `TOOLS_DDB_TABLE_PREVIEW`, `TOOLS_DDB_TABLE_PRODUCTION`, or `TOOLS_DDB_TABLE`
- Region: the service-specific region variable, `AWS_REGION`, or `AWS_DEFAULT_REGION`

The AWS identity running the Short Links reservation command needs `dynamodb:DescribeTable`, `dynamodb:Scan`, and `dynamodb:TransactWriteItems`. Click reconciliation additionally needs `dynamodb:DescribeTimeToLive` and `dynamodb:UpdateItem` on the clicks table; the optional TTL-disable step needs `dynamodb:UpdateTimeToLive`. The Tools command needs `dynamodb:DescribeTable`, `dynamodb:DescribeTimeToLive`, `dynamodb:Scan`, and `dynamodb:UpdateItem`.

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

## Short Links click reconciliation and durable history

The link table's `clicks` value is an all-time aggregate, while the separate click table contains individual events. Older deployments put an `expiresAt` attribute on each event and configured DynamoDB TTL, so the aggregate can legitimately be larger than the remaining event count. This migration preserves that distinction without manufacturing event rows:

- every real event row keeps its original `clickId`, `clickedAt`, and captured metadata, but loses `expiresAt`
- every live link gets one reserved `__historical_baseline__` row with `entityType: clickBaseline`
- `recordedEventCount` is the number of real event rows found for the slug
- `historicalClicks` is exactly `link.clicks - recordedEventCount`
- `aggregateClicks` is the canonical total at reconciliation and must equal the other two counts combined
- the baseline has no `clickedAt`, destination, referrer, user-agent, or location fields, so it cannot be mistaken for an observed click

An event count greater than the link aggregate, a malformed baseline, or another invalid managed record fails preflight. Deleted-link event partitions are reported as orphans and retained; the migration does not invent a live-link aggregate for them. Printed identifiers are opaque hashes.

First run both environments without `--apply`:

```powershell
npm run migrate:short-links-clicks -- --environment preview --links-table <preview-links-table> --clicks-table <preview-clicks-table> --region us-east-2
npm run migrate:short-links-clicks -- --environment production --links-table <production-links-table> --clicks-table <production-clicks-table> --region us-east-2
```

After reviewing the dry run, verify a current backup and pause redirect/click writes. Deploy the transactional redirect code that no longer writes `expiresAt` while traffic remains paused, then apply the reconciliation and request TTL disablement in one guarded run:

```powershell
npm run migrate:short-links-clicks -- --environment preview --links-table <preview-links-table> --clicks-table <preview-clicks-table> --region us-east-2 --apply --backup-confirmed --traffic-paused --disable-ttl
```

The command requests TTL disablement and then immediately removes `expiresAt` from every scanned click row before writing baselines. DynamoDB can continue processing TTL deletions briefly after disablement, but a row whose TTL attribute is removed before deletion is no longer eligible. If TTL is already `DISABLING` or `DISABLED`, rerun the same apply command without `--disable-ttl` to finish any interrupted row work.

Keep traffic paused until the final dry run is clean. The runtime preserves a reconciled baseline when a deleted slug is recreated, so its all-time aggregate does not reset while durable history remains. If an older orphan partition has detailed events but no trustworthy baseline, slug reuse is rejected instead of silently joining incompatible totals.

Verify Preview before repeating the apply step for Production. Baselines are written in transactions that conditionally verify both the scanned link aggregate and any prior baseline values. A click that lands after the scan therefore causes a conditional failure rather than being overwritten; keep traffic paused, rerun the dry run, and retry. TTL removals are also conditional and the whole migration is idempotent. Once DynamoDB reports `DISABLED`, the final dry run should report `ttlRemovalsPlanned: 0`, `baselineCreatesPlanned: 0`, `baselineUpdatesPlanned: 0`, and `aggregateMismatches: 0`.

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
