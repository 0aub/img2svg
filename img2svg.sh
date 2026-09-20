#!/usr/bin/env bash
# Thin wrapper so you can forget Docker exists:
#     ./img2svg.sh logo.png --report
# Files are read from and written to the current directory.
set -euo pipefail
IMAGE="${IMG2SVG_IMAGE:-img2svg:local}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE ..." >&2
  docker build -t "$IMAGE" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
exec docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" "$IMAGE" "$@"
