# Target Empty Package Demo API

This Lambda serves the anonymized empty-package recovery dataset from S3.

## Data prep

Generate the anonymized JSON from the Excel workbook:

```bash
python3 aws/target-empty-package/precompute.py
```

Outputs land in:

```
aws/target-empty-package/output/data.json
```

## S3 upload

Pick a unique bucket name and sync the output:

```bash
BUCKET=target-empty-package-data-886623862678
aws s3api create-bucket --bucket "$BUCKET" --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2
aws s3 cp aws/target-empty-package/output/data.json "s3://$BUCKET/target-empty-package/v1/data.json"
```

## IAM role

```bash
aws iam create-role \
  --role-name targetEmptyPackageLambdaRole \
  --assume-role-policy-document file://aws/target-empty-package/trust-policy.json

aws iam attach-role-policy \
  --role-name targetEmptyPackageLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy \
  --role-name targetEmptyPackageLambdaRole \
  --policy-name targetEmptyPackageS3Read \
  --policy-document file://aws/target-empty-package/s3-policy.json
```

## Deploy Lambda

```bash
python3 -m zipfile -c aws/target-empty-package/target-empty-package.zip aws/target-empty-package/app.py

aws lambda create-function \
  --function-name target-empty-package \
  --runtime python3.11 \
  --handler app.lambda_handler \
  --role arn:aws:iam::886623862678:role/targetEmptyPackageLambdaRole \
  --memory-size 256 \
  --timeout 8 \
  --zip-file fileb://aws/target-empty-package/target-empty-package.zip \
  --environment "Variables={DATA_BUCKET=$BUCKET,DATA_KEY=target-empty-package/v1/data.json}"
```

## Private browser access

Publish an immutable version, point the `live` alias at it, and add the qualified alias ARN to `DemoFunctionArns` in `aws/vercel-oidc/template.yaml`. The browser calls `/api/demos/target-empty-package/*`; it never calls Lambda directly. Do not create an anonymous Function URL or add Lambda origins to the CSP.

## Endpoints

- `GET /data`
- `GET /meta`
- `GET /health`

## Update data

Re-run precompute and upload:

```bash
python3 aws/target-empty-package/precompute.py
aws s3 cp aws/target-empty-package/output/data.json "s3://$BUCKET/target-empty-package/v1/data.json"
```
