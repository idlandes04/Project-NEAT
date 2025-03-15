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
