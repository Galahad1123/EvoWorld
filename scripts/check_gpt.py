"""Check authenticated model metadata; does not run generation or print credentials."""
import json
import sys
from openai import OpenAI, APIStatusError

try:
    with OpenAI(timeout=20, max_retries=0) as client:
        model = client.models.retrieve("gpt-6-astra")
    print(json.dumps({"status": "pass", "check": "authenticated_model_metadata", "model": model.id}))
except APIStatusError as exc:
    body = exc.body if isinstance(exc.body, dict) else {}
    detail = body.get("error", body)
    detail = detail if isinstance(detail, dict) else {}
    # Do not print exception messages: provider errors can contain portions of a key.
    print(json.dumps({"status": "fail", "http_status": exc.status_code, "error_code": detail.get("code"), "error_type": detail.get("type")}))
    sys.exit(1)
except Exception as exc:
    print(json.dumps({"status": "fail", "error_type": type(exc).__name__}))
    sys.exit(1)
