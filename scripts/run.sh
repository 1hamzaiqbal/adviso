#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the Streamlit app
# Usage:
#   ./scripts/run.sh [BACKEND_URL]
# Example:
#   ./scripts/run.sh https://advaluate-api-uriyv6ylzq-uc.a.run.app

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

BACKEND_URL=${1:-${ADVALUATE_BACKEND_URL:-}}

if [[ -n "$BACKEND_URL" ]]; then
  export ADVALUATE_BACKEND_URL="$BACKEND_URL"
  echo "Using backend: $ADVALUATE_BACKEND_URL"
else
  echo "No backend URL provided. You can pass one as an argument or set ADVALUATE_BACKEND_URL."
fi

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

# Avoid headless font warnings
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-"$ROOT_DIR/.mplconfig"}
mkdir -p "$MPLCONFIGDIR"

echo "Starting Streamlit..."
exec streamlit run app.py

