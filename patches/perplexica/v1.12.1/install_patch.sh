#!/usr/bin/env sh
set -eu

CONTAINER="${PERPLEXICA_CONTAINER:-perplexica}"
RUNTIME_PATH="/home/perplexica/.next/server/chunks/641.js"
PATCH_FILE="641.js.diag"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command not found" >&2
  exit 1
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ERROR: container '$CONTAINER' does not exist" >&2
  exit 1
fi

if [ ! -f "$PATCH_FILE" ]; then
  echo "ERROR: missing $PATCH_FILE in current directory" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_path="${RUNTIME_PATH}.backup-${timestamp}"

echo "Backing up ${CONTAINER}:${RUNTIME_PATH} to ${backup_path}"
docker exec "$CONTAINER" sh -lc "test -f '$RUNTIME_PATH' && cp '$RUNTIME_PATH' '$backup_path'"

echo "Installing ${PATCH_FILE} to ${CONTAINER}:${RUNTIME_PATH}"
docker cp "$PATCH_FILE" "${CONTAINER}:${RUNTIME_PATH}"

echo "Restarting ${CONTAINER}"
docker restart "$CONTAINER" >/dev/null

echo "Recent ${CONTAINER} logs:"
docker logs --tail 80 "$CONTAINER"
