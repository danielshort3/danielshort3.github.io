## Slot Machine Lambda

Server-side logic for the slot machine demo. The function keeps authoritative player balances in DynamoDB and is deployed behind an API Gateway HTTP API.

### AWS resources

- **DynamoDB**: `slot-machine-demo-players` (partition key `playerId`)
- **Lambda**: `slot-machine-demo` (runtime `nodejs18.x`)
- **IAM role**: `slotMachineDemoLambdaRole`
- **API Gateway**: HTTP API `slot-machine-demo-api` (`4kvebym8b3`, stage `prod`)
- **Public endpoint**: `https://4kvebym8b3.execute-api.us-east-2.amazonaws.com/prod`

### Environment variables

```
TABLE_NAME=slot-machine-demo-players
STARTING_CREDITS=1000
MAX_BET=100
ALLOWED_ORIGINS=https://danielshort.dev,https://www.danielshort.dev,https://danielshort3.github.io,https://danielshort3-github-io.vercel.app
```

### Deploy flow

```
cd aws/slot-machine-function
npm install                   # only needed after dependency changes
python3 - <<'PY'
import zipfile, pathlib
root = pathlib.Path('aws/slot-machine-function')
zip_path = pathlib.Path('aws/slot-machine-function.zip')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob('*'):
        if path.is_file():
            zf.write(path, path.relative_to(root))
PY
aws lambda update-function-code \
  --function-name slot-machine-demo \
  --zip-file fileb://aws/slot-machine-function.zip
```

For convenience, you can run the helper snippet from the repo root as-is. It rebuilds `aws/slot-machine-function.zip` with the latest source and `node_modules/`.

```
python3 - <<'PY'
import zipfile, pathlib
root = pathlib.Path('aws/slot-machine-function')
zip_path = pathlib.Path('aws/slot-machine-function.zip')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob('*'):
        if path.is_file():
            zf.write(path, path.relative_to(root))
PY
```

Update the API gateway or DynamoDB settings with the AWS CLI as needed (see provisioning history in the CLI transcript for the commands that created the stack).
