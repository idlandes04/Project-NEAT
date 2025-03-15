#!/bin/bash
# Script to prepare the NEAT model for training on the Windows PC with 3080ti GPU

set -e  # Exit on error

# Create necessary directories
mkdir -p data/byte_training data/byte_eval data/visual_training data/neat_training outputs

# Step 1: Run mock model creation for testing on Windows
echo "Step 1: Creating mock BLT entropy estimator and MVoT visual codebook for testing..."
python3 scripts/create_mock_models.py --output_dir outputs --create_training_data

# Step 2: Generate synthetic math dataset
echo "Step 2: Generating synthetic math dataset for training and testing..."
python3 scripts/generate_synthetic_data.py --output_dir data/synthetic --train_size 10000 --eval_size 2000

# Step 3: Create script for training on Windows
echo "Step 3: Creating training script for Windows PC with 3080ti..."

cat > scripts/train_neat_model_windows.bat << 'EOF'
@echo off
REM Windows batch script to train NEAT model on 3080ti

echo Training NEAT model on Windows with 3080ti GPU...

python main.py --mode train ^
    --use_titans_memory ^
    --use_transformer2_adaptation ^
    --use_mvot_processor ^
    --use_blt_processor ^
    --blt_checkpoint_path ./outputs/blt/mock_byte_lm.pt ^
    --mvot_codebook_path ./outputs/mvot/mock_codebook.pt ^
    --hidden_size 768 ^
    --num_layers 12 ^
    --num_attention_heads 12 ^
    --batch_size 16 ^
    --learning_rate 5e-5 ^
    --max_steps 10000 ^
    --gradient_accumulation_steps 1 ^
    --mixed_precision ^
    --gradient_checkpointing ^
    --entropy_threshold 0.5 ^
    --output_dir ./outputs/neat_model_full

echo Training complete!
EOF

# Step 4: Create script for downloading MMR1-Math-RL-Data dataset
echo "Step 4: Creating script for downloading MMR1-Math-RL-Data dataset..."

cat > scripts/download_mmr1_math_dataset.py << 'EOF'
#!/usr/bin/env python3
"""
Script to download the MMR1-Math-RL-Data dataset.
This script should be run on the Windows PC with 3080ti before training.
"""

import os
from datasets import load_dataset

def main():
    """Download the MMR1-Math-RL-Data dataset."""
    print("Downloading MMR1-Math-RL-Data-v0 dataset...")
    
    # Create output directory
    output_dir = os.path.join("data", "mmr1_math")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training dataset
    print("Loading training split...")
    train_dataset = load_dataset("MMR1/MMR1-Math-RL-Data-v0", split="train")
    print(f"Downloaded {len(train_dataset)} training examples")
    
    # Load test dataset
    print("Loading test split...")
    test_dataset = load_dataset("MMR1/MMR1-Math-RL-Data-v0", split="test")
    print(f"Downloaded {len(test_dataset)} test examples")
    
    # Save dataset info
    print("Saving dataset info...")
    with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
        f.write(f"Training examples: {len(train_dataset)}\n")
        f.write(f"Test examples: {len(test_dataset)}\n")
        f.write(f"Example features: {train_dataset.features}\n")
        f.write(f"Example row: {train_dataset[0]}\n")
    
    print(f"MMR1-Math-RL-Data-v0 dataset downloaded and info saved to {os.path.join(output_dir, 'dataset_info.txt')}")
    print("The dataset is cached by the datasets library and can now be used for training.")

if __name__ == "__main__":
    main()
EOF

# Step 5: Create script for resuming training if needed
echo "Step 5: Creating script for resuming training if needed..."

cat > scripts/resume_training_windows.bat << 'EOF'
@echo off
REM Windows batch script to resume NEAT model training

echo Resuming NEAT model training on Windows with 3080ti GPU...

python main.py --mode train ^
    --use_titans_memory ^
    --use_transformer2_adaptation ^
    --use_mvot_processor ^
    --use_blt_processor ^
    --blt_checkpoint_path ./outputs/blt/mock_byte_lm.pt ^
    --mvot_codebook_path ./outputs/mvot/mock_codebook.pt ^
    --hidden_size 768 ^
    --num_layers 12 ^
    --num_attention_heads 12 ^
    --batch_size 16 ^
    --learning_rate 5e-5 ^
    --max_steps 10000 ^
    --gradient_accumulation_steps 1 ^
    --mixed_precision ^
    --gradient_checkpointing ^
    --entropy_threshold 0.5 ^
    --output_dir ./outputs/neat_model_full ^
    --model_path ./outputs/neat_model_full/checkpoint-latest

echo Training resumed and complete!
EOF

# Make scripts executable
chmod +x scripts/download_mmr1_math_dataset.py

echo "Preparation complete!"
echo ""
echo "To train on the Windows PC with 3080ti:"
echo "1. Copy the entire project directory to the Windows PC"
echo "2. Run 'scripts/download_mmr1_math_dataset.py' to download the MMR1-Math-RL-Data dataset"
echo "3. Run 'scripts/train_neat_model_windows.bat' to start training"
echo "4. If training is interrupted, use 'scripts/resume_training_windows.bat' to resume"