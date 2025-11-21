## Slot Machine Lambda

Server-side logic for the slot machine demo. The function keeps authoritative player balances in DynamoDB, stores hashed user credentials/sessions, and is deployed behind an API Gateway HTTP API.

### AWS resources

- **DynamoDB**: `slot-machine-demo-players` (partition key `playerId`)
- **DynamoDB**: `slot-machine-users` (partition key `username`, GSI `sessionToken-index`)
- **DynamoDB**: `slot-machine-spin-history` (partition key `playerId`, sort key `spinTime`)
- **Lambda**: `slot-machine-demo` (runtime `nodejs18.x`)
- **IAM role**: `slotMachineDemoLambdaRole`
- **API Gateway**: HTTP API `slot-machine-demo-api` (`4kvebym8b3`, stage `prod`)
- **Public endpoint**: `https://4kvebym8b3.execute-api.us-east-2.amazonaws.com/prod`

### Environment variables

```
TABLE_NAME=slot-machine-demo-players
USERS_TABLE=slot-machine-users
HISTORY_TABLE=slot-machine-spin-history
STARTING_CREDITS=1000
MAX_BET=100
ALLOWED_ORIGINS=https://danielshort.dev,https://www.danielshort.dev,https://danielshort3.github.io,https://danielshort3-github-io.vercel.app,https://danielshort.me,https://www.danielshort.me
SESSION_TTL_MINUTES=4320
PASSWORD_SALT_ROUNDS=12
```

### HTTP routes

All routes accept/return JSON:

- `POST /auth/register` → `{ username, password }` ⇒ `{ token, username, playerId, balance, spins, ... }`
- `POST /auth/login` → `{ username, password }`
- `POST /auth/logout` → `{ token }`
- `POST /auth/delete` → `{ token }` — removes the user, player record, and spin history entries
- `POST /upgrade` → `{ token, type }` (`type` ∈ `rows`,`reels`,`lines`) upgrades the machine if the player has enough credits
- `POST /session` → `{ token?, playerId? }`
- `POST /spin` → `{ token?, playerId, bet }`

Every successful spin (and rejected bet attempts when we know the player) is recorded into `slot-machine-spin-history`, so you can audit player activity or build dashboards from that table without touching the live balance records.

### API Gateway routing check

If upgrades start 404ing or missing CORS headers, the HTTP API may have lost the `/upgrade` route. Reattach it to the existing Lambda integration (currently `oxwy3f3`) and AutoDeploy on the `prod` stage will publish it immediately:

```
INTEGRATION_ID=$(aws apigatewayv2 get-integrations --api-id 4kvebym8b3 --query 'Items[0].IntegrationId' --output text)
aws apigatewayv2 create-route \
  --api-id 4kvebym8b3 \
  --route-key "POST /upgrade" \
  --target "integrations/${INTEGRATION_ID}"
```

Sanity check CORS with the live domain:

```
curl -i -X POST -H 'Origin: https://www.danielshort.me' -d '{}' https://4kvebym8b3.execute-api.us-east-2.amazonaws.com/prod/upgrade
```

### Slot engine + assets

- `slot-engine.js` shares the same JSON definition as the browser client. During packaging it attempts to load `slot-config/classic.json`; if that file is missing (e.g., in the Lambda bundle) it falls back to `classic-config.js`, which mirrors the same symbol/payout data.
- If you tweak the slot JSON under `/slot-config`, re-run the deploy zip step below so Lambda gets the updated fallback copy.
- The client now depends on `machine` metadata returned by `/session` and `/spin`, so make sure the Lambda package and the front-end are deployed together.

### Deploy flow

```
cd aws/slot-machine-function
npm install                   # only needed after dependency changes
python3 - <<'PY'
import zipfile, pathlib
root = pathlib.Path('.')
zip_path = pathlib.Path('..') / 'slot-machine-function.zip'
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
