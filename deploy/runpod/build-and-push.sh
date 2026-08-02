#!/usr/bin/env bash
# Build + push the LatentSync RunPod worker image from THIS repo checkout.
# Usual build host: the prod box (docker + ghcr login). The model code ships
# from the checkout — commit before building so the image matches the repo.
#
# Usage: ./build-and-push.sh v1
set -euo pipefail

TAG=${1:?usage: build-and-push.sh vN   # versioned tags only — workers cache :latest}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
STAGE=/data/runpod-build/latentsync

mkdir -p "$STAGE"
rsync -a --delete --exclude .git --exclude __pycache__ --exclude checkpoints \
    "$REPO_ROOT/" "$STAGE/LatentSync/"
git -C "$REPO_ROOT" rev-parse HEAD > "$STAGE/LatentSync/.pinned-commit" 2>/dev/null || true
cp "$HERE/handler.py" "$STAGE/handler.py"
cp "$HERE/Dockerfile.runpod-latentsync" "$STAGE/Dockerfile"

docker build -t "ghcr.io/cvoalex/runpod-latentsync:$TAG" "$STAGE"
docker push "ghcr.io/cvoalex/runpod-latentsync:$TAG"
echo "pushed ghcr.io/cvoalex/runpod-latentsync:$TAG — update the RunPod template imageName to this tag"
