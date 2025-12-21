## Pizza Tips Prediction Lambda

Server-side linear regression for pizza tip predictions. Keeps coefficients private and exposes a single POST `/predict`-style handler via a Lambda Function URL.

### AWS resources

- **Lambda**: `pizza-tips-predict` (runtime `nodejs18.x`)
- **Lambda Function URL**: `https://2d6lrg4ozy564xymwi2epjbqty0nvhje.lambda-url.us-east-2.on.aws/`

### Environment variables

```
CONFIDENCE_LEVEL=0.9
```

CORS is configured on the Lambda Function URL. The handler does not append CORS headers.

### Request

POST JSON payload (required numeric fields follow `model.inputFeatures`):

```
{
  "latitude": 33.11,
  "longitude": -96.82,
  "cost": 35,
  "housing": "Residential",
  "orderHour": 18,
  "confidenceLevel": 0.8
}
```

City is inferred from the latitude/longitude using city boundary polygons (lat/lon are not model inputs). Housing must match one of the training categories.
Current housing categories: Residential, Apartment, Hotel, Business.

If housing coefficients are removed during feature selection, housing becomes optional. Fields not listed in `model.inputFeatures` can be omitted.

`confidenceLevel` supports 0.8, 0.85, 0.9, or 0.95 (defaults to 0.8 when omitted).

### Target transforms

The model applies `log1p` transforms to `tip` and `tipPercent` during training and back-transforms predictions with `expm1`. Confidence intervals are computed on the transformed scale and then back-transformed for display.

### Heatmap grid

Send an optional `grid` object to receive hotspot predictions for the current scenario:

```
{
  "grid": {
    "rows": 24,
    "cols": 24,
    "bounds": { "latMin": 33.01, "latMax": 33.19, "lonMin": -96.93, "lonMax": -96.71 }
  }
}
```

When present, the response includes `heatmap` with `points` (`lat`, `lon`, `tip`) plus min/max tip values for legend scaling.

### Response

```
{
  "ok": true,
  "bucket": { "city": "Frisco", "housing": "Residential" },
  "predictions": {
    "tip": { "value": 7.2, "interval": { "level": 0.9, "low": 1.4, "high": 13.0 } },
    "tipPercent": { "value": 0.19, "interval": { "level": 0.9, "low": 0.0, "high": 0.43 } }
  },
  "breakdown": { "tip": { "items": [ ... ] }, "tipPercent": { "items": [ ... ] } }
}
```

### Deploy flow

```
python3 - <<'PY'
import zipfile, pathlib
root = pathlib.Path('aws/pizza-tips-predict')
zip_path = pathlib.Path('aws/pizza-tips-predict.zip')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob('*'):
        if path.is_file():
            zf.write(path, path.relative_to(root))
PY

aws lambda update-function-code \
  --function-name pizza-tips-predict \
  --zip-file fileb://aws/pizza-tips-predict.zip
```

After deploying, update the Function URL constant in `demos/pizza-tips-demo.html` and add the URL to the `connect-src` directive in `vercel.json`.

### City boundaries

City polygons live in `aws/pizza-tips-predict/city-boundaries.json`. To refresh them, run:

```
node build/fetch-pizza-city-boundaries.js
node build/build-pizza-tips-model.js
```
