#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${1:-all}
PROJECT=project-bae2ca8e-7983-420c-b17
REGION=europe-west1
API_SERVICE=claims
UI_SERVICE=claims-ui
SA=359388058343-compute@developer.gserviceaccount.com

env_value() {
  grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'" | tr -d '[:space:]'
}

store_secret() {
  local name=$1 value=$2
  if gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --project "$PROJECT" --data-file=- >/dev/null
  else
    printf '%s' "$value" | gcloud secrets create "$name" --project "$PROJECT" --data-file=- >/dev/null
  fi
  gcloud secrets add-iam-policy-binding "$name" --project "$PROJECT" \
    --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor" --quiet >/dev/null
}

deploy_api() {
  local gemini_key groq_key llm_model reviews_url choices="" secrets
  gemini_key=$(env_value GEMINI_API_KEY)
  groq_key=$(env_value GROQ_API_KEY)
  llm_model=$(env_value LLM_MODEL)
  llm_model=${llm_model:-qwen/qwen3.8-27b}

  [ -n "$gemini_key" ] || { echo "GEMINI_API_KEY is empty in .env"; exit 1; }
  store_secret gemini-api-key "$gemini_key"
  secrets="GEMINI_API_KEY=gemini-api-key:latest"

  [ -n "$groq_key" ] || { echo "GROQ_API_KEY is empty in .env"; exit 1; }
  store_secret groq-api-key "$groq_key"
  secrets="$secrets,GROQ_API_KEY=groq-api-key:latest"
  choices="gemini:gemini-3.5-flash-lite"

  reviews_url=$(env_value REVIEWS_DATABASE_URL)
  if [ -n "$reviews_url" ]; then
    store_secret reviews-database-url "$reviews_url"
    secrets="$secrets,REVIEWS_DATABASE_URL=reviews-database-url:latest"
  else
    echo "REVIEWS_DATABASE_URL is empty in .env: the queue will use a local SQLite file and reset on restart."
  fi

  gcloud run services update "$API_SERVICE" --region "$REGION" --project "$PROJECT" \
    --memory 2Gi --cpu-boost --max-instances 2 \
    --update-env-vars "^|^LLM_PROVIDER=groq|LLM_MODEL=$llm_model|RATE_LIMIT=10/minute|LLM_CHOICES=$choices" \
    --update-secrets "$secrets"
}

deploy_ui() {
  local api_url
  api_url=$(gcloud run services describe "$API_SERVICE" --region "$REGION" --project "$PROJECT" --format="value(status.url)")
  gcloud run deploy "$UI_SERVICE" --source . --region "$REGION" --project "$PROJECT" \
    --port 8501 --memory 1Gi --max-instances 2 --allow-unauthenticated \
    --command streamlit \
    --args run,frontend/app.py,--server.port,8501,--server.address,0.0.0.0,--server.headless,true \
    --set-env-vars "API_URL=$api_url"
  gcloud run services describe "$UI_SERVICE" --region "$REGION" --project "$PROJECT" --format="value(status.url)"
}

case "$TARGET" in
  api) deploy_api ;;
  ui) deploy_ui ;;
  all) deploy_api; deploy_ui ;;
  *) echo "usage: $0 [api|ui|all]"; exit 1 ;;
esac
