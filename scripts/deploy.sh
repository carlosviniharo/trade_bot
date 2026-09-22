#!/usr/bin/env bash
# ==============================================================================
# Trade Bot — Droplet Deployment & Service Update Script
# ==============================================================================
# Executed on the droplet during CI/CD deployment or manually by the deploy user.
# ==============================================================================

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

echo "==> [1/5] Deploying Trade Bot at ${APP_DIR}..."

# ------------------------------------------------------------------------------
# 1. Generate .env file from environment variables
# ------------------------------------------------------------------------------
echo "==> [2/5] Updating .env file..."
if [ -n "${ENV_FILE:-}" ]; then
    printf "%s\n" "$ENV_FILE" > .env
else
    cat <<EOF > .env
MONGODB_URI=${MONGODB_URI:-}
MONGODB_NAME=${MONGODB_NAME:-my_database}
WHATSAPP_TOKEN=${WHATSAPP_TOKEN:-}
PHONE_NUMBER_ID=${PHONE_NUMBER_ID:-}
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-}
ENV=${APP_ENV:-production}
EOF
fi
chmod 600 .env

# ------------------------------------------------------------------------------
# 2. Authenticate to Docker Hub if credentials provided
# ------------------------------------------------------------------------------
if [ -n "${REGISTRY_USERNAME:-}" ] && [ -n "${REGISTRY_PASSWORD:-}" ]; then
    echo "==> [3/5] Authenticating to Docker Registry..."
    echo "$REGISTRY_PASSWORD" | docker login -u "$REGISTRY_USERNAME" --password-stdin
fi

# ------------------------------------------------------------------------------
# 3. Pull latest pre-built images and recreate containers
# ------------------------------------------------------------------------------
echo "==> [4/5] Pulling images and restarting services..."
export REGISTRY_USERNAME="${REGISTRY_USERNAME:-carlosviniharo}"
export IMAGE_TAG="${IMAGE_TAG:-latest}"

docker compose pull
docker compose up -d --remove-orphans

# Clean up dangling images to keep droplet disk lean
docker image prune -f

# ------------------------------------------------------------------------------
# 4. Verify deployment health
# ------------------------------------------------------------------------------
echo "==> [5/5] Verifying service responsiveness..."
SUCCESS=0
for i in {1..20}; do
    if curl -sf http://127.0.0.1:8000/docs > /dev/null 2>&1 || curl -sf http://127.0.0.1/docs > /dev/null 2>&1; then
        echo "✅ Trade Bot container is up, healthy, and serving traffic!"
        SUCCESS=1
        break
    fi
    echo "    Waiting for services to become responsive (attempt $i/20)..."
    sleep 3
done

if [ "$SUCCESS" -ne 1 ]; then
    echo "❌ Health check timed out! Recent container logs:"
    docker compose logs --tail=50
    exit 1
fi
