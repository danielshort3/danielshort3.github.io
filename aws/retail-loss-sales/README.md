# Store-Level Loss & Sales ETL Demo API

This Lambda serves aggregated store-loss and sales metrics from S3.

## Data prep

Build the JSON bundle from the anonymized CSVs:

```bash
python3 aws/retail-loss-sales/precompute.py
```

Output lands in:

```
aws/retail-loss-sales/output/data.json
```

## S3 upload

Pick a bucket name and sync the output:

```bash
BUCKET=retail-loss-sales-data-886623862678
aws s3api create-bucket --bucket "$BUCKET" --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2
aws s3 cp aws/retail-loss-sales/output/data.json "s3://$BUCKET/retail-loss-sales/v1/data.json"
```

## IAM role

```bash
aws iam create-role \
  --role-name retailLossSalesLambdaRole \
  --assume-role-policy-document file://aws/retail-loss-sales/trust-policy.json

aws iam attach-role-policy \
  --role-name retailLossSalesLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy \
  --role-name retailLossSalesLambdaRole \
  --policy-name retailLossSalesS3Read \
  --policy-document file://aws/retail-loss-sales/s3-policy.json
```

## Deploy Lambda

```bash
python3 -m zipfile -c aws/retail-loss-sales/retail-loss-sales.zip aws/retail-loss-sales/app.py

aws lambda create-function \
  --function-name retail-loss-sales \
  --runtime python3.11 \
  --handler app.lambda_handler \
  --role arn:aws:iam::886623862678:role/retailLossSalesLambdaRole \
  --memory-size 256 \
  --timeout 8 \
  --zip-file fileb://aws/retail-loss-sales/retail-loss-sales.zip \
  --environment "Variables={DATA_BUCKET=$BUCKET,DATA_KEY=retail-loss-sales/v1/data.json}"
```

## Function URL

```bash
aws lambda create-function-url-config \
  --function-name retail-loss-sales \
  --auth-type NONE \
  --cors file://aws/retail-loss-sales/cors.json

aws lambda add-permission \
  --function-name retail-loss-sales \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE
```

Function URL:

```
https://oafstar74okqvfqaxayeici3ry0yikkx.lambda-url.us-east-2.on.aws/
```

Update `demos/retail-loss-sales-demo.html` and `vercel.json` connect-src if the Function URL changes.

## Endpoints

- `GET /data`
- `GET /meta`
- `GET /health`

## Update data

Re-run the precompute and upload:

```bash
python3 aws/retail-loss-sales/precompute.py
aws s3 cp aws/retail-loss-sales/output/data.json "s3://$BUCKET/retail-loss-sales/v1/data.json"
```
