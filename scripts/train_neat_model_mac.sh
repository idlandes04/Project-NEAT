#!/bin/bash
# Modified script to prepare data and train the NEAT model on macOS

set -e  # Exit on error

# Create necessary directories
mkdir -p data/byte_training data/byte_eval data/visual_training data/neat_training outputs

# Step 1: Download training data for component models (using python3 explicitly)
echo "Step 1: Downloading training data for BLT and MVoT models..."
python3 scripts/download_training_data.py --output_dir data

# Step 2: Skip actual training and create mock models directly
echo "Step 2: Creating mock BLT entropy estimator and MVoT visual codebook..."
python3 scripts/create_mock_models.py --output_dir outputs

# Step 3: Test the advanced problem types
echo "Step 3: Testing advanced problem types..."
python3 scripts/test_advanced_problems.py --test_standard --test_progressive

# Step 4: Fix any test failures
echo "Step 4: Running synthetic data tests to ensure problem types are working..."
python3 -m pytest tests/test_synthetic_data.py -v

# Step 5: Generate a small training dataset for testing the model
echo "Step 5: Generating a small training dataset for testing..."
mkdir -p data/neat_training
# Using a very small dataset for quick testing
if [ -f "scripts/prepare_training_dataset.py" ]; then
    python3 scripts/prepare_training_dataset.py \
        --output_dir ./data/neat_training \
        --general_size 1000 \
        --component_size 500 \
        --eval_size 500
else
    echo "Warning: prepare_training_dataset.py script not found. Skipping dataset generation."
    # Create a placeholder file to indicate dataset generation was skipped
    touch data/neat_training/dataset_generation_skipped.txt
fi

echo "Mock models and test data preparation complete!"
echo "To train the NEAT model with the mock components, run:"
echo "python3 main.py --mode train \\
    --use_titans_memory \\
    --use_transformer2_adaptation \\
    --use_mvot_processor \\
    --use_blt_processor \\
    --blt_checkpoint_path ./outputs/blt/mock_byte_lm.pt \\
    --mvot_codebook_path ./outputs/mvot/mock_codebook.pt \\
    --hidden_size 768 \\
    --num_layers 12 \\
    --num_attention_heads 12 \\
    --batch_size 8 \\
    --learning_rate 5e-5 \\
    --max_steps 100 \\
    --output_dir ./outputs/neat_model_test"