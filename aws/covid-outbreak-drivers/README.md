# COVID Outbreak Drivers Lambda (Zip)

This Lambda serves precomputed outbreak summaries from S3.

## Data prep

Generate summaries from the local COVID analysis CSVs:

```bash
python3 aws/covid-outbreak-drivers/precompute.py
```

Outputs land in:

```
aws/covid-outbreak-drivers/output/
  meta.json
  by-date/2022-07-01.json
  state/CA.json
```

## S3 upload

Pick a unique bucket name and sync the output:

```bash
BUCKET=covid-outbreak-drivers-data-886623862678
aws s3api create-bucket --bucket "$BUCKET" --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2
aws s3 sync aws/covid-outbreak-drivers/output "s3://$BUCKET/covid-outbreak/v1"
```

## IAM role

```bash
aws iam create-role \
  --role-name covidOutbreakDriversLambdaRole \
  --assume-role-policy-document file://aws/covid-outbreak-drivers/trust-policy.json

aws iam attach-role-policy \
  --role-name covidOutbreakDriversLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy \
  --role-name covidOutbreakDriversLambdaRole \
  --policy-name covidOutbreakDriversS3Read \
  --policy-document file://aws/covid-outbreak-drivers/s3-policy.json
```

## Deploy Lambda

```bash
python3 -m zipfile -c aws/covid-outbreak-drivers/covid-outbreak-drivers.zip aws/covid-outbreak-drivers/app.py

aws lambda create-function \
  --function-name covid-outbreak-drivers \
  --runtime python3.11 \
  --handler app.lambda_handler \
  --role arn:aws:iam::886623862678:role/covidOutbreakDriversLambdaRole \
  --memory-size 512 \
  --timeout 10 \
  --zip-file fileb://aws/covid-outbreak-drivers/covid-outbreak-drivers.zip \
  --environment "Variables={DATA_BUCKET=$BUCKET,DATA_PREFIX=covid-outbreak/v1}"
```

## Function URL

```bash
aws lambda create-function-url-config \
  --function-name covid-outbreak-drivers \
  --auth-type NONE \
  --cors file://aws/covid-outbreak-drivers/cors.json

aws lambda add-permission \
  --function-name covid-outbreak-drivers \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE
```

Function URL:

```
https://lv4inwnj6yyo3kfdajpk2v5eda0hrtao.lambda-url.us-east-2.on.aws/
```

## Endpoints

- `GET /meta` (add `?refresh=1` to flush cache)
- `GET /states?date=YYYY-MM-DD`
- `GET /state/{id}?date=YYYY-MM-DD`
- `POST /query` with `{ "date": "YYYY-MM-DD" }` or `{ "state": "CA", "date": "YYYY-MM-DD" }`
- `GET /health`

## Update data

Re-run the precompute step and sync:

```bash
python3 aws/covid-outbreak-drivers/precompute.py
aws s3 sync aws/covid-outbreak-drivers/output "s3://$BUCKET/covid-outbreak/v1"
```
