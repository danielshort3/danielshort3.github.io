# Website AWS workloads

## Preview demo Lambda stacks

`demo-preview.template.yaml` creates isolated Preview copies of the ten portfolio demo Lambdas. It owns ten new execution roles, ten finite-retention log groups, ten functions whose names end in `-preview`, published versions, `live` aliases, and three private/versioned Preview-only data buckets. It intentionally creates no Function URLs and gives each execution role only its own log stream permissions plus the exact Preview S3 objects that its code reads.

`demo-preview-alarms.template.yaml` is the companion monitoring stack. It creates separate error, throttle, and duration alarms for every Preview function and sends alarm transitions to the supplied security/operations SNS topic. Deploy it after the function stack. Keeping monitoring separate keeps each template below the CloudFormation `TemplateBody` validation limit.

The Shape Analyzer and Smart Sentence Finder use separate roles and log groups here. `DEBUG_TOKEN` is intentionally omitted so the old debug path has no secret in Lambda environment variables. Minesweeper is retained for validation but remains unpublished/noindex.

### Verified immutable artifacts

The image functions use these exact ECR digests from the July 11 manifest; tags are not used:

| Function | ECR digest |
| --- | --- |
| `shape-analyzer-preview` | `sha256:78174eb27e07c1c3338bf2ff8a7a8b520cde7cb7191f9dbb160ffaa400cbef38` |
| `smart-sentence-finder-preview` | `sha256:860926b8441422073686608b44fe9f443efe0be2ad8cc3e1eb55e7efe870750a` |
| `nonogram-solver-preview` | `sha256:49668d156c3794a7e533ac60f62590696bf9366e269cb4c3f6921c362c518bdc` |
| `handwriting-rating-demo-preview` | `sha256:21e441a5d329cf7b9c29bf18c031a9a378a31395bb9bf8889204f06148a1be6b` |
| `digit-generator-preview` | `sha256:b250e895c38c79b8503fafa790ee5fd5b450ad43b9af02e1b954fa2059ab84c4` |
| `minesweeper-solver-preview` | `sha256:bb6189816dd435e439702cafb475a171f2f7877d3319b498ad0c94f5c4e7d63a` |

The four ZIP artifacts are the downloaded production packages in the recovery checkpoint's `lambda-packages` directory:

| Artifact | Bytes | SHA-256 (hex) | Lambda `CodeSha256` |
| --- | ---: | --- | --- |
| `covid-outbreak-drivers.zip` | 1,739 | `ECF79D6C4C814B0C0CA266506F17F09C6E3740BD61246A1AEDA54C24701B7DBE` | `7PedbEyBSwwMomZQbxfwnG43QL1hJGoa7aVMJHAbfb4=` |
| `pizza-tips-predict.zip` | 320,125 | `0DC641542C535E2B77EE907D2390C8E00ED3C0653FB537AE4D34EA82B2D685DE` | `DcZBVCxTXit37pB9I5DI4A7TwGU/tTeuTTTqgrLWhd4=` |
| `retail-loss-sales.zip` | 1,038 | `76789DB229C073667393C0703F1726DBDEAC191EEEEC985B621A40306AC4A5CD` | `dnidsinAc2Zzk8BwPxcm296sGR7u7JhbYhpAMGrEpc0=` |
| `target-empty-package.zip` | 1,038 | `860ED543BE4FB5F18BA78890ABB8449FF40F74246DC6F3D3AC3BE9C3E1FA9977` | `hg7VQ75PtfGLp4iQq7hEn/QPdCRtxvPTrDvpw+H6mXc=` |

The deployed Preview stack reads those unchanged bytes from the private, versioned artifact bucket recorded in `demo-preview.parameters.example.json`. For future releases, upload the replacement bytes first and record each exact object key and returned `VersionId`; the template does not guess keys. The `AWS::Lambda::Version` resources reject a ZIP whose bytes do not match the captured `CodeSha256`.

Do not use a production data bucket as a function parameter. The stack creates these isolated destinations:

- `website-covid-outbreak-preview-<account>-<region>` at `covid-outbreak/v1/`
- `website-retail-loss-sales-preview-<account>-<region>` at `retail-loss-sales/v1/data.json`
- `website-target-empty-package-preview-<account>-<region>` at `target-empty-package/v1/data.json`

Seed the public analytical datasets into those destinations only after the function stack creates them. A one-time server-side copy from the corresponding production bucket is acceptable; the Preview functions themselves never receive permission to the production buckets. Record object versions and compare source/destination SHA-256 values before canary invocation.

### Validation and change-set sequence

From the repository root:

```powershell
cfn-lint aws/workloads/demo-preview.template.yaml aws/workloads/demo-preview-alarms.template.yaml
aws cloudformation validate-template --region us-east-2 --template-body file://aws/workloads/demo-preview.template.yaml
aws cloudformation validate-template --region us-east-2 --template-body file://aws/workloads/demo-preview-alarms.template.yaml
```

Create and inspect future change sets with `CAPABILITY_NAMED_IAM`; do not execute an update until its versioned artifacts are verified, the three datasets are ready, the security/operations SNS subscription is confirmed, and the Lambda regional concurrency quota can support the functions. No reserved concurrency is set in these templates; apply reservations only after the requested quota of 200 is approved.

After the function stack succeeds, use its `DemoFunctionArns` output verbatim for the `DemoFunctionArns` parameter in `aws/vercel-oidc/preview.parameters.example.json`. Map each individual alias output to its same-named `DEMO_*_FUNCTION_ARN` Vercel Preview variable. All values are `:live`-qualified; never pass `$LATEST` or an unqualified function ARN.

Deploy the alarm stack next, then invoke every `:live` alias through the same-origin Preview proxy. Confirm CloudTrail attributes invocation to the Preview demo OIDC role, each function writes only its own log group, the three data functions read only Preview buckets, and direct anonymous access is impossible because no Function URL or public Lambda permission exists.
