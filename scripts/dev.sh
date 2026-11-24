#!/usr/bin/env bash
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT
uv run python main.py &
uv run --group frontend streamlit run frontend/app.py &
wait
