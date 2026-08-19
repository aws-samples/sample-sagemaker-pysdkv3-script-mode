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

# If AWS_REGION is empty, resolve it from the standard AWS chain so this repo
# runs in any region without editing the env file.
if [ -z "$AWS_REGION" ]; then
    AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region)}}"
    export AWS_REGION
fi
if [ -z "$AWS_REGION" ]; then
    echo "✗ Could not determine AWS region. Set AWS_REGION in the env file," >&2
    echo "  export AWS_DEFAULT_REGION, or run 'aws configure set region <region>'." >&2
    exit 1
fi

# If AWS_ACCOUNT_ID is empty, fetch it
if [ -z "$AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    export AWS_ACCOUNT_ID
fi

echo "Repository: $ECR_REPOSITORY_NAME"
echo "Tag: $IMAGE_TAG"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo ""

# Check if repository exists
echo "Checking if ECR repository exists..."
if aws ecr describe-repositories --repository-names "$ECR_REPOSITORY_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "✓ Repository '$ECR_REPOSITORY_NAME' exists"
else
    echo "✗ Repository '$ECR_REPOSITORY_NAME' does not exist"
    echo "Creating repository..."
    aws ecr create-repository \
        --repository-name "$ECR_REPOSITORY_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "✓ Repository created successfully"
fi

# Get ECR login token and authenticate Docker
echo ""
echo "Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "✓ Authentication successful"

# Tag the image for ECR
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG"
echo ""
echo "Tagging image for ECR..."
docker tag "$ECR_REPOSITORY_NAME:$IMAGE_TAG" "$ECR_URI"
echo "✓ Image tagged: $ECR_URI"

# Push to ECR
echo ""
echo "Pushing image to ECR..."
docker push "$ECR_URI"
echo "✓ Image pushed successfully"

echo ""
echo "Image URI: $ECR_URI"