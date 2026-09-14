.PHONY: install auth-token gen-mcp-json ingest ingest-rebuild serve serve-http query clean \
        build up down restart redeploy db db-down db-reset reindex reindex-rebuild logs clear-cache

VENV        := .venv
PYTHON      := $(VENV)/bin/python
# mcp>=2.0.0 requires Python 3.10+; the system `python3` on macOS is often
# older (e.g. Apple's bundled 3.9), so prefer a newer interpreter if one is
# installed (`brew install python@3.12`) and only fall back to `python3`.
PYTHON_BASE := $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)
# Local (non-Docker) targets don't get .env for free the way docker compose
# does -- load it into the shell before running python, if it exists.
LOAD_ENV    := set -a; [ -f .env ] && . ./.env; set +a;

install: $(VENV)/bin/activate

## Generate a fresh RAG_AUTH_TOKEN and write it into .env (creating .env
## from .env.example first if it doesn't exist yet). Required before
## `make up`/`make serve-http`.
auth-token:
	@test -f .env || cp .env.example .env
	@TOKEN=$$(openssl rand -hex 24); \
	if grep -q '^RAG_AUTH_TOKEN=' .env; then \
		sed -i.bak "s/^RAG_AUTH_TOKEN=.*/RAG_AUTH_TOKEN=$$TOKEN/" .env && rm -f .env.bak; \
	else \
		echo "RAG_AUTH_TOKEN=$$TOKEN" >> .env; \
	fi; \
	echo "RAG_AUTH_TOKEN set in .env: $$TOKEN"

$(VENV)/bin/activate: requirements.txt
	$(PYTHON_BASE) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch $(VENV)/bin/activate

## Write .mcp.json using this machine's real venv/server paths and your
## actual .env values (RAG_DATABASE_URL, RAG_COLLECTION, VOYAGE_API_KEY) --
## no placeholders to hand-edit. Merges in, so any other server entries
## already in .mcp.json (e.g. local-docs-remote) are left alone.
gen-mcp-json: install
	@$(LOAD_ENV) $(PYTHON) src/scripts/gen_mcp_json.py

## Start just Postgres (pgvector), exposed on localhost:5433, so `ingest`/
## `serve` below can run locally against it instead of inside Docker.
db:
	docker compose up -d db

db-down:
	docker compose stop db

## Wipe the Postgres volume entirely (all collections lost). Confirms first.
db-reset:
	@read -p "This deletes ALL indexed data in Postgres. Continue? [y/N] " ans; \
	[ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || exit 1
	docker compose down -v

## Build/update the index. Incremental -- only new/changed docs are re-embedded.
## Requires Postgres reachable (see `make db`).
ingest: install
	@$(LOAD_ENV) $(PYTHON) src/ingest.py

## Force a full re-embed of every doc (e.g. after changing RAG_EMBED_MODEL).
ingest-rebuild: install
	@$(LOAD_ENV) $(PYTHON) src/ingest.py --rebuild

## Run the MCP server over stdio (what Claude Code launches as a subprocess).
## Requires Postgres reachable (see `make db`).
serve: install
	@$(LOAD_ENV) $(PYTHON) src/server.py

## Run the MCP server over Streamable HTTP (for a tunnel). Requires RAG_AUTH_TOKEN
## (run `make auth-token` first).
serve-http: install
	@$(LOAD_ENV) $(PYTHON) src/server.py --transport http

## Query the knowledge base straight from the terminal, no MCP client
## needed. Drops into an interactive prompt by default; pass QUERY="..."
## for a single one-off question (optionally TOP_K=N, default 5). Requires
## Postgres reachable and the index already built (see `make db`, `make ingest`).
query: install
	@$(LOAD_ENV) $(PYTHON) src/scripts/query_cli.py $(if $(QUERY),"$(QUERY)") $(if $(TOP_K),--top-k $(TOP_K))

clean:
	rm -rf $(VENV) __pycache__

## --- Docker Compose ---
## (these run the containerized version -- for local/stdio use the targets above)

build:
	docker compose build

## Build/update the index inside the container (incremental, same as `ingest`).
reindex:
	docker compose run --rm ingest

## Force a full re-embed of every doc inside the container.
reindex-rebuild:
	docker compose run --rm ingest --rebuild

up:
	docker compose up -d rag-mcp

## Restart the container as-is -- same image, same code. Useful to clear
## in-process state (e.g. the query cache) without a rebuild. Does NOT
## pick up source/Dockerfile/compose changes -- use `make redeploy` for
## that.
restart:
	docker compose restart rag-mcp

## Rebuild the image and recreate the container -- use this after changing
## src/*.py, Dockerfile, or docker-compose.yaml. Required because code is
## COPY'd into the image at build time, not live-mounted: a plain restart
## reuses the old image, so the old code would keep running otherwise.
redeploy: build up

down:
	docker compose down

logs:
	docker compose logs -f rag-mcp

clear-cache:
	docker compose exec rag-mcp python -c "import server; server._cache.clear()"
