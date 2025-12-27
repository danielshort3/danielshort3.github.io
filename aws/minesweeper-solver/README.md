# Minesweeper Solver Demo API

Serverless Minesweeper solver powered by a DQN model (PyTorch) and exposed via a Lambda Function URL.

## Docker build + push

```bash
aws ecr create-repository --repository-name minesweeper-solver --region us-east-2
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 886623862678.dkr.ecr.us-east-2.amazonaws.com

docker build -t minesweeper-solver aws/minesweeper-solver

docker tag minesweeper-solver:latest 886623862678.dkr.ecr.us-east-2.amazonaws.com/minesweeper-solver:latest

docker push 886623862678.dkr.ecr.us-east-2.amazonaws.com/minesweeper-solver:latest
```

## Deploy Lambda

```bash
aws iam create-role \
  --role-name minesweeperSolverLambdaRole \
  --assume-role-policy-document file://aws/minesweeper-solver/trust-policy.json

aws iam attach-role-policy \
  --role-name minesweeperSolverLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws lambda create-function \
  --function-name minesweeper-solver \
  --package-type Image \
  --code ImageUri=886623862678.dkr.ecr.us-east-2.amazonaws.com/minesweeper-solver:latest \
  --role arn:aws:iam::886623862678:role/minesweeperSolverLambdaRole \
  --memory-size 2048 \
  --timeout 15
```

## Function URL

```bash
aws lambda create-function-url-config \
  --function-name minesweeper-solver \
  --auth-type NONE \
  --cors file://aws/minesweeper-solver/cors.json

aws lambda add-permission \
  --function-name minesweeper-solver \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE
```

Function URL:

```
https://jnvd3mdbyb5f44yh4afzsvqlwy0mdtzy.lambda-url.us-east-2.on.aws/
```

Update `demos/minesweeper-demo.html` and `vercel.json` connect-src if the Function URL changes.

## Endpoints

- `GET /` returns model metadata
- `POST /` generates a fresh puzzle and AI solve trace
