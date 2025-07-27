# Modal Labs Setup Guide

Run your Mind Games Challenge agents in the cloud with free GPU credits.

##  $500 Free Credits

1. **Sign up** at [modal.com](https://modal.com) → Get $5 starter credits

2. **Run the example** to qualify for bonus credits:
   ```bash
   cd mindgames-starter-kit/src
   modal run online_play_track1_with_modal_lab.py
   ```
   - First run: ~5 minutes to build (not charged)
   - Play for 10+ minutes and complete one game
   
3. **Check your email** within 1 hour for the promo code
   - No email? Contact: mindgameschallenge2025@gmail.com

4. **Redeem** at https://modal.com/credits
   - Enter promo code, get $500 credits
   - For "What are you building?": `Mind Games Challenge`
   
5. **What you get**:
   - 800+ hours on T4 GPUs
   - or 230+ hours on A100 GPUs

## Quick Start

1. **Install Modal**:
   ```bash
   pip install modal
   ```

2. **Authenticate**:
   ```bash
   modal setup
   ```

3. **Run your agent**:
   ```bash
   cd mindgames-starter-kit/src
   modal run ./online_play_track1_with_modal_lab.py
   ```

## Configuration

### GPU Selection
```python
@app.function(
    gpu="T4",  # Options: "T4", "A10G", "A100", "H100"
)
```

### API Keys
Add secrets at https://modal.com/secrets, then reference in code:
```python
secrets=[
    modal.Secret.from_name("huggingface-secret"),
]
```

## Best Practices
- **T4 GPU**: Best for inference (0.59$/h)
- **A100 GPU**: For large models (2.10$/h)
- **Monitor usage**: https://modal.com/usage
- **Set timeouts** to prevent runaway costs
