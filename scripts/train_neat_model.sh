#!/bin/bash
# Script to prepare data and train the NEAT 100M parameter model

set -e  # Exit on error

# Create necessary directories
mkdir -p data/byte_training data/byte_eval data/visual_training data/neat_training outputs

# Step 1: Download training data for component models
echo "Step 1: Downloading training data for BLT and MVoT models..."
python scripts/download_training_data.py --output_dir data

# Step 2: Train the BLT entropy estimator
echo "Step 2: Training the BLT entropy estimator..."
python main.py --mode train_byte_lm \
    --train_data_dir ./data/byte_training \
    --eval_data_dir ./data/byte_eval \
    --batch_size 32 \
    --max_steps 5000 \
    --byte_lm_hidden_size 128 \
    --byte_lm_num_layers 2 \
    --byte_lm_dropout 0.1 \
    --output_dir ./outputs/byte_lm

# Step 3: Prepare the mock visual codebook for MVoT
echo "Step 3: Preparing mock visual codebook for MVoT..."
python data/visual_training/create_mock_codebook.py

# Step 4: Test the advanced problem types
echo "Step 4: Testing advanced problem types..."
python scripts/test_advanced_problems.py --test_standard --test_progressive

# Step 5: Generate training dataset for NEAT model
echo "Step 5: Generating training dataset for the NEAT model..."
python scripts/prepare_training_dataset.py \
    --output_dir ./data/neat_training \
    --general_size 50000 \
    --component_size 10000 \
    --eval_size 10000 \
    --vocab_size 1000 \
    --max_length 128

# Step 6: Train the NEAT model with all components enabled
echo "Step 6: Training the NEAT model..."
python main.py --mode train \
    --use_titans_memory \
    --use_transformer2_adaptation \
    --use_mvot_processor \
    --use_blt_processor \
    --use_two_pass_inference \
    --hidden_size 768 \
    --num_layers 12 \
    --num_attention_heads 12 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --max_steps 10000 \
    --gradient_accumulation_steps 1 \
    --mixed_precision \
    --gradient_checkpointing \
    --output_dir ./outputs/neat_model

echo "Training complete! Model saved to outputs/neat_model"