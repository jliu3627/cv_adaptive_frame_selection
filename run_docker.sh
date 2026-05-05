#!/usr/bin/env bash

set -e

echo "Building Docker image..."
docker build --platform linux/arm64 -t adaptive-frame-selection .

echo ""
echo "Running Docker container..."
docker run --rm -it adaptive-frame-selection