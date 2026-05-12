#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-asia-southeast1}"
REPO_NAME="${REPO_NAME:-facemask-repo}"
SERVICE_NAME="${SERVICE_NAME:-facemask-compliance}"

if [[ -z "${TAG:-}" ]]; then
  if git rev-parse --short HEAD >/dev/null 2>&1; then
    TAG="$(git rev-parse --short HEAD)"
    if ! git diff --quiet || ! git diff --cached --quiet; then
      TAG="$TAG-$(date +%Y%m%d%H%M%S)"
    fi
  else
    TAG="$(date +%Y%m%d%H%M%S)"
  fi
fi

CPU="${CPU:-2}"
MEMORY="${MEMORY:-4Gi}"
CONCURRENCY="${CONCURRENCY:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required. Set it with: export PROJECT_ID=<your-gcp-project-id>" >&2
  exit 1
fi

if [[ ! -f models/best.pt ]]; then
  echo "Missing models/best.pt. Place the trained deploy weight before running this script." >&2
  exit 1
fi

ARTIFACT_HOST="$REGION-docker.pkg.dev"
REPO_PATH="$ARTIFACT_HOST/$PROJECT_ID/$REPO_NAME"
IMAGE="$REPO_PATH/$SERVICE_NAME:$TAG"

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Repository: $REPO_NAME"
echo "Service: $SERVICE_NAME"
echo "Image: $IMAGE"
echo "Deploy weight: $(du -h models/best.pt | awk '{print $1}') models/best.pt"

echo "Enabling required Google Cloud APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  --project "$PROJECT_ID"

echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker repository for face mask compliance demo" \
  --project "$PROJECT_ID" \
  --quiet || true

echo "Building and pushing image with Cloud Build..."
gcloud builds submit . \
  --tag "$IMAGE" \
  --project "$PROJECT_ID"

echo "Deploying Cloud Run service..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --concurrency "$CONCURRENCY" \
  --min-instances "$MIN_INSTANCES" \
  --timeout "$TIMEOUT" \
  --set-env-vars MODEL_PATH=models/best.pt,APP_CONFIG=configs/app.yaml,PRELOAD_MODEL=1,YOLO_CONFIG_DIR=/tmp/Ultralytics \
  --project "$PROJECT_ID"

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format 'value(status.url)')"

echo "Service URL: $SERVICE_URL"
echo "Health check:"
curl -fsS "$SERVICE_URL/health"
echo
