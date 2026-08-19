#!/bin/bash

# Override environment file if user provides one
ENV_FILE=".env.docker"
if [[ "$1" == "--env" ]]; then
    ENV_FILE="$2"
fi

# Load Docker configuration from .env.docker
set -a
source "docker/$ENV_FILE"
set +a

echo "Repository: $ECR_REPOSITORY_NAME"
echo "Tag: $IMAGE_TAG"
echo ""

# Build the image (Dockerfile is in docker/ folder)
echo "Building Docker image..."
docker build -f $DOCKERFILE_PATH --platform linux/amd64 --network sagemaker -t "$ECR_REPOSITORY_NAME:$IMAGE_TAG" .
if [ $? -eq 0 ]; then
    echo "✓ Image built successfully"
else
    echo "✗ Image build failed"
    exit 1
fi