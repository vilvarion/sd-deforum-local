#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$BACKEND/.venv"

# ── Python venv setup ────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
  echo ">>> Creating Python venv in backend/.venv ..."
  python3 -m venv "$VENV"
fi

PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
UVICORN="$VENV/bin/uvicorn"

# Install/upgrade backend deps if requirements changed
REQ_HASH_FILE="$VENV/.req_hash"
REQ_HASH=$(md5 -q "$BACKEND/requirements.txt" 2>/dev/null || md5sum "$BACKEND/requirements.txt" | cut -d' ' -f1)
if [ ! -f "$REQ_HASH_FILE" ] || [ "$(cat "$REQ_HASH_FILE")" != "$REQ_HASH" ]; then
  echo ">>> Installing backend dependencies ..."
  "$PIP" install --upgrade pip -q
  "$PIP" install -r "$BACKEND/requirements.txt"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
fi

# ── Frontend node_modules ─────────────────────────────────────────────────────
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo ">>> Installing frontend dependencies ..."
  (cd "$FRONTEND" && npm install)
fi

# ── Launch both processes ─────────────────────────────────────────────────────
echo ""
echo ">>> Starting backend  on http://localhost:8000"
echo ">>> Starting frontend on http://localhost:5173"
echo ""

(cd "$BACKEND" && "$UVICORN" main:app --reload --port 8000 --host 127.0.0.1) &
BACKEND_PID=$!

(cd "$FRONTEND" && npm run dev) &
FRONTEND_PID=$!

# Clean shutdown on Ctrl-C
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; wait" INT TERM

wait $BACKEND_PID $FRONTEND_PID
