# Lambda CORS Setup

To call the shape-classifier Lambda function from the website you must allow browsers to access it. Ensure your Lambda code returns the required CORS headers and responds to pre-flight requests. The helper that builds JSON responses should include:

```python
def _json(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body),
        "isBase64Encoded": False,
    }
```

Handle the CORS pre-flight once inside your handler:

```python
def handler(event, context):
    if event["requestContext"]["http"]["method"] == "OPTIONS":
        return _json(204, {})
    # ... existing logic ...
```

Without these headers, browsers will report `TypeError: Failed to fetch`.
