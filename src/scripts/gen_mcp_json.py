"""Writes .mcp.json with this machine's real venv/server paths and current
.env values, instead of the placeholders in .mcp.json.example. Run via
`make gen-mcp-json`. Merges in -- other existing entries are left as-is.
See README for what it writes and why."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
SERVER_PY = ROOT / "src" / "server.py"
OUT_PATH = ROOT / ".mcp.json"

STDIO_KEY = "doc-rag-mcp"
HTTP_KEY = "doc-rag-mcp-http"


def main():
    if not VENV_PYTHON.exists():
        sys.exit(f"{VENV_PYTHON} not found -- run `make install` first.")

    for var in ("RAG_DATABASE_URL", "RAG_DOCS_DIR", "VOYAGE_API_KEY"):
        if not os.environ.get(var):
            print(f"warning: {var} is not set in .env -- the generated entry will have it empty.",
                  file=sys.stderr)

    stdio_entry = {
        "command": str(VENV_PYTHON),
        "args": [str(SERVER_PY)],
        "env": {
            "RAG_DATABASE_URL": os.environ.get("RAG_DATABASE_URL", ""),
            "RAG_COLLECTION": os.environ.get("RAG_COLLECTION", "project_docs"),
            # Without this, list_documents()'s staleness check falls back to
            # ./docs relative to wherever the MCP client happens to spawn
            # this process from -- usually missing, which used to crash the
            # whole server (see _compute_index_freshness() in server.py).
            "RAG_DOCS_DIR": os.environ.get("RAG_DOCS_DIR", ""),
            "VOYAGE_API_KEY": os.environ.get("VOYAGE_API_KEY", ""),
        },
    }

    config = {}
    if OUT_PATH.exists():
        try:
            config = json.loads(OUT_PATH.read_text())
        except json.JSONDecodeError:
            sys.exit(f"{OUT_PATH} exists but isn't valid JSON -- fix or remove it first.")

    config.setdefault("mcpServers", {})
    config["mcpServers"][STDIO_KEY] = stdio_entry
    print(f"Wrote \"{STDIO_KEY}\" (stdio) -- command: {stdio_entry['command']}")

    auth_token = os.environ.get("RAG_AUTH_TOKEN")
    if auth_token:
        port = os.environ.get("RAG_PORT", "8743")
        http_entry = {
            "type": "http",
            "url": f"http://localhost:{port}/mcp",
            "headers": {"Authorization": f"Bearer {auth_token}"},
        }
        config["mcpServers"][HTTP_KEY] = http_entry
        print(f"Wrote \"{HTTP_KEY}\" (http) -- url: {http_entry['url']}")
    else:
        config["mcpServers"].pop(HTTP_KEY, None)
        print(f"RAG_AUTH_TOKEN not set -- skipping \"{HTTP_KEY}\" "
              "(run `make auth-token` first if you want the HTTP entry too).")

    OUT_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {OUT_PATH}.")


if __name__ == "__main__":
    main()
