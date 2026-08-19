# Bring Your Own Model with SageMaker AI: Script Mode in SDK v3

This project demonstrates **script mode** in the SageMaker Python SDK v3 through two end-to-end examples:

1. **Scikit-learn Random Forest** — Train a tabular ML model and deploy to a real-time endpoint
2. **Stable Diffusion 3.5 LoRA** — Fine-tune a generative AI model with multi-GPU distributed training

Both examples use `ModelTrainer` for training and `ModelBuilder` for deployment. The key pattern is `SourceCode` — your training and inference code is injected into minimal containers at runtime, so you never need to rebuild images when changing your algorithm code.

## Getting Started

### 1. Build and Push Docker Images

Before running the notebook, build and push the training containers to Amazon ECR. From a terminal in the project root:

```bash
# Scikit-learn container
bash build.sh --env .env.docker.sklearn
bash push.sh --env .env.docker.sklearn

# Stable Diffusion container
bash build.sh --env .env.docker.stablediffusion
bash push.sh --env .env.docker.stablediffusion
```

These scripts handle ECR authentication, repository creation, and image pushing automatically.

### 2. Set Up Secrets (Stable Diffusion example)

The Stable Diffusion example needs one secret at runtime: a **Hugging Face token** (`HF_TOKEN`) with access to the gated Stable Diffusion 3.5 Medium repo, used to download the base model. (The secret also has an optional `WANDB_API_KEY` slot in case you prefer Weights & Biases, but it isn't needed for the default flow, which tracks experiments with the SageMaker managed MLflow server — see below.)

Rather than hardcoding this token in the notebook — where it would end up in the training job definition and CloudTrail — store it in **AWS Secrets Manager** and pass only the secret's **ARN** to the training job. At runtime, `base.sh` fetches the secret and exports each key as an environment variable inside the container, so the raw token never leaves Secrets Manager.

Deploy the included CloudFormation template (`cloudformation/training-secrets.yaml`) with your real Hugging Face token:

```bash
aws cloudformation deploy \
  --template-file cloudformation/training-secrets.yaml \
  --stack-name script-mode-blog-secrets \
  --parameter-overrides \
      HuggingFaceToken=hf_your_real_token_here \
      WandbApiKey=dummy_not_used_with_mlflow \
  --capabilities CAPABILITY_IAM
```

> These commands run in your AWS CLI's default region. Add `--region <region>` to
> target a specific one — it must match the region your notebook runs in, since the
> training job looks up the secret by ARN in that region.

The stack creates the secret (a JSON object `{"HF_TOKEN": "...", "WANDB_API_KEY": "..."}`) and an IAM managed policy that grants read access to it. Fetch the outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name script-mode-blog-secrets \
  --query "Stacks[0].Outputs" --output table
```

Then attach the `SecretReadPolicyArn` managed policy to your SageMaker execution role (so the training job can read the secret), and copy the `SecretArn` value into the `SECRETS_ARN` variable in the notebook's Stable Diffusion configuration cell.

> **Note:** If your execution role uses `AmazonSageMakerFullAccess`, it can already read secrets whose names begin with `AmazonSageMaker-` (the template's default name does), so attaching the managed policy is optional in that case.

### 3. Open the Notebook

Open `script_mode_sdkv3_blog.ipynb` and follow along. The notebook covers:

- Configuring training jobs with YAML recipe files
- Preparing the Stable Diffusion training data (a notebook cell pulls a sample dataset from Hugging Face and uploads it to S3 — no separate script to run)
- Launching training with `ModelTrainer` and `SourceCode`
- Deploying models to real-time endpoints with `ModelBuilder`
- Experiment tracking with the SageMaker managed MLflow server

## How It Works

### Training with `ModelTrainer`

`SourceCode` points at a local directory and a command to run. SageMaker syncs the directory into the container and executes your command — no image rebuild needed when you change your script.

**Example 1 — Scikit-learn:**

```python
source_code = SourceCode(
    source_dir="./train/random_forest",
    command="python random_forest.py --n_jobs 4 --max_depth 10 --n_estimators 120",
)

model_trainer = ModelTrainer(
    training_image=TRAINING_IMAGE_URI,
    source_code=source_code,
    compute=Compute(instance_type="ml.m5.2xlarge", instance_count=1),
    role=SAGEMAKER_EXECUTION_ROLE,
)

model_trainer.train(input_data_config=input_config, wait=True)
```

**Example 2 — Stable Diffusion LoRA (multi-GPU):**

```python
sd_source_code = SourceCode(
    source_dir="./train/stable_diffusion",
    command=(
        "/bin/bash base.sh"
        " --config recipes/default-medium-g5_12x.yaml"
        " --training-script train_text_to_image_lora.py"
        " --accelerate-config accelerate_configs/ddp.yaml"
        f" --mlflow-arn {SD_MLFLOW_ARN}"  # log to the SageMaker managed MLflow server
        f" --mlflow-experiment-name {SD_MLFLOW_EXPERIMENT_NAME}"
    ),
)

sd_model_trainer = ModelTrainer(
    training_image=SD_TRAINING_IMAGE_URI,
    source_code=sd_source_code,
    compute=Compute(instance_type="ml.g5.12xlarge", instance_count=1),  # 4x A10G GPUs
    environment={"SECRETS_ARN": SECRETS_ARN},  # base.sh fetches HF_TOKEN from Secrets Manager at runtime
    role=SAGEMAKER_EXECUTION_ROLE,
)

sd_model_trainer.train(input_data_config=sd_input_config, wait=True)
```

### Deployment with `ModelBuilder`

The same `SourceCode` pattern is used for inference. Point at a directory with your handler script, and `ModelBuilder` packages it with the model artifact and deploys to an endpoint.

```python
inference_source_code = SourceCode(
    source_dir="./deploy/random_forest",
    entry_script="inference.py",
)

model_builder = ModelBuilder(
    image_uri=INFERENCE_IMAGE_URI,
    model_server=ModelServer.DJL_SERVING,
    source_code=inference_source_code,
    s3_model_data_url=model_artifact_s3_uri,
    role_arn=SAGEMAKER_EXECUTION_ROLE,
)

model = model_builder.build()
predictor = model.deploy(instance_type="ml.m5.xlarge", initial_instance_count=1)
```

## Project Structure

```
├── docker/                  # Dockerfiles and requirements for training containers
│   ├── sklearn/
│   └── stable_diffusion/
├── train/                   # Training scripts and configs
│   ├── random_forest/       # Scikit-learn training script
│   └── stable_diffusion/    # LoRA fine-tuning script + accelerate configs
├── deploy/                  # Inference handlers
│   ├── random_forest/
│   └── stable_diffusion/
├── build.sh                 # Build Docker image locally
└── push.sh                  # Push Docker image to ECR
```

## Prerequisites

- AWS account with SageMaker AI access
- IAM execution role with SageMaker and S3 permissions
- SageMaker Python SDK v3
- Docker (for building containers)
- S3 bucket for training data and model artifacts
- (For Stable Diffusion) A HuggingFace token with access to the model
