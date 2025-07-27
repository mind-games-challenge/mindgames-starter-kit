# Running UnstableBaselines with Modal Labs

This tutorial walks you through deploying UnstableBaselines for MindGames competition training jobs on Modal Labs
## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **API Keys**:
   - Weights & Biases (W&B) API key from [wandb.ai/authorize](https://wandb.ai/authorize)
   - OpenRouter API key from [openrouter.ai/keys](https://openrouter.ai/keys)
3. **Local Setup**:
   - Python 3.10+
   - Git

## Step 1: Install and Configure Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup
```

## Step 2: Create API Secrets

Modal securely stores your API keys as secrets that are injected as environment variables at runtime.

```bash
# Create W&B secret
modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key_here

# Create OpenRouter secret
modal secret create openrouter-secret OPENROUTER_API_KEY=your_openrouter_api_key_here

# Verify secrets were created
modal secret list
```

**Tip**: Use `--interactive` flag for secure input:
```bash
modal secret create wandb-secret --interactive
# Then type: WANDB_API_KEY=your_key when prompted
```

## Step 3: Convert example_standard.py to Modal

Create a new file `example_standard_modal.py` with the following modifications:

### Key Changes from Original Script

1. **Import Modal and Create App**:
```python
import modal
import time

# Create persistent volume for checkpoints
VOLUME_NAME = "unstable-baselines-training-volume"
MOUNT_PATH = "/PATH/TO/STORE/LOCALLY"
training_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "ray[default]",
        "vllm",
        "wandb",
        "unstable-rl",  # or your package name
    )
)

app = modal.App("unstable-baselines-training", image=image)
```

2. **Wrap Training Logic in Modal Function**:
```python
@app.function(
    image=image,
    gpu="A100:3",  # Request 3x A100 GPUs
    timeout=60 * 60,  # 1 hour timeout
    volumes={MOUNT_PATH: training_vol},  # Mount persistent storage
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("openrouter-secret")
    ],
)
def train():
    # Your original training code here
    # All imports must be inside or use with image.imports():
```

3. **Adjust Resource Allocation**:
   - Reduce worker counts for Modal's environment:
   ```python
   # Original: collector.collect.remote(num_train_workers=384, num_eval_workers=16)
   collector.collect.remote(num_train_workers=32, num_eval_workers=4)
   ```
   - Reduce batch size if needed:
   ```python
   batch_size=64,  # Reduced from 384
   ```

4. **Modify Ray Initialization**:
```python
# Remove hardcoded namespace if using Modal's isolated environment
ray.init(
    num_gpus=3,  # Match your GPU request
)
```

5. **Add Entry Point**:
```python
@app.local_entrypoint()
def main():
    result = train.remote()
    print(result)
```

### Complete Example Structure

```python
import modal
import time

VOLUME_NAME = "unstable-baselines-training-volume"
MOUNT_PATH = "/mnt/training"
training_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "ray[default]",
        "vllm",
        "wandb",
        "unstable-rl",
    )
)

app = modal.App("unstable-baselines-training", image=image)

with image.imports():
    import ray
    import unstable
    import unstable.reward_transformations as retra

@app.function(
    image=image,
    gpu="A100:3",
    timeout=60 * 60,
    volumes={MOUNT_PATH: training_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("openrouter-secret")
    ],
)
def train():
    # Your training configuration
    MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
    MAX_TRAIN_SEQ_LEN = None
    MAX_GENERATION_LENGTH = 4096
    
    # ... rest of your training code ...
    
    return "Training completed successfully"

@app.local_entrypoint()
def main():
    result = train.remote()
    print(result)
```

## Step 4: Launch Training

```bash
# Navigate to your project directory
cd /path/to/UnstableBaselines

# Run the Modal script
modal run example_standard_modal.py
```
Note: The first run may take a few minutes to set up the environment, but all future runs will launch quickly.
## Step 5: Monitor Progress

1. **Terminal Output**: Modal streams logs directly to your terminal
2. **Modal Dashboard**: View job status at [modal.com/home](https://modal.com/home)
3. **W&B Dashboard**: Track metrics at [wandb.ai](https://wandb.ai)

## Resource Configuration Options

### GPU Types
```python
gpu="A100"        # Single A100
gpu="A100:2"      # Two A100s
gpu="A10G:4"      # Four A10Gs (cheaper option)
gpu="T4:2"        # Two T4s (budget option)
```

### Timeout Settings
```python
timeout=60 * 60      # 1 hour
timeout=60 * 60 * 6  # 6 hours
timeout=60 * 60 * 24 # 24 hours
```

### Container Resources, usually the default is fine
```python
@app.function(
    cpu=8,           # CPU cores
    memory=32768,    # Memory in MB (32GB)
    ephemeral_disk=50 * 1024,  # Disk in MB (50GB)
)
```
