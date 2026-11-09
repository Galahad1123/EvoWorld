"""Load EvoWorld credentials/proxy settings without interpreting .env as shell code."""
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
if args[:1] == ["--"]:
    args = args[1:]
if not args:
    raise SystemExit("Usage: envs/sage/bin/python scripts/with_gpt_env.py -- COMMAND [ARGS...]")
path = root / ".env"
if not path.is_file():
    raise SystemExit("EvoWorld .env is missing on this host.")
settings = dotenv_values(path, interpolate=False)
env = dict(os.environ)
# Project settings deliberately override inherited shell values.
env.update({k: v for k, v in settings.items() if v is not None})
key = env.get("OPENAI_API_KEY", "")
if not key or key in {"你的实际密钥", "YOUR_API_KEY", "your-api-key"}:
    raise SystemExit("Set a real OPENAI_API_KEY in the remote EvoWorld .env.")
for upper, lower in (("HTTPS_PROXY", "https_proxy"), ("HTTP_PROXY", "http_proxy"), ("NO_PROXY", "no_proxy")):
    if upper in settings:
        env[lower] = env[upper]
proxy = env.get("HTTPS_PROXY") or env.get("https_proxy")
if proxy:
    try:
        parsed = urlsplit(proxy)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Unsupported proxy")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=3):
            pass
    except (OSError, ValueError):
        raise SystemExit("EvoWorld HTTPS proxy is unavailable. Start/check Clash on 5090-1 and its configured proxy port.")
os.chdir(root)
try:
    os.execvpe(args[0], args, env)
except FileNotFoundError:
    raise SystemExit("Requested command was not found; use a project-relative or absolute executable path.")
