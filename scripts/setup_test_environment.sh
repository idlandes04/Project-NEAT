#!/bin/bash
# Script to set up a test environment with mock models for rapid development

set -e  # Exit on error

echo "=========================================="
echo "Setting up NEAT test environment"
echo "=========================================="

# Create necessary directories
mkdir -p outputs/blt outputs/mvot data/byte_training data/byte_eval data/visual_training data/neat_training

# Create mock models
echo "Creating mock BLT and MVoT models..."
python3 scripts/create_mock_models.py --output_dir outputs

# Create some mock training data files
echo "Creating mock byte training data..."
echo "This is some mock text data for testing the BLT entropy estimator." > data/byte_training/mock_data1.txt
echo "Additional mock text with different patterns and content." > data/byte_training/mock_data2.txt
echo "A third mock file with random text for testing purposes." > data/byte_eval/mock_eval1.txt

# Create a mock visual codebook
echo "Creating mock visual codebook data..."
python3 -c "
import os
import torch

os.makedirs('data/visual_training', exist_ok=True)
# Create a small tensor and save it as mock image features
mock_features = torch.randn(10, 512)
torch.save(mock_features, 'data/visual_training/mock_image_features.pt')
print('Created mock image features for testing')
"

# Test the synthetic data generator
echo "Testing advanced problem types..."
python3 scripts/test_advanced_problems.py

# Run tests to verify everything is working
echo "Running unit tests..."
python3 -m pytest tests/test_synthetic_data.py -v

echo "=========================================="
echo "Test environment setup complete!"
echo "=========================================="
echo "To train a small test model, run:"
echo "python3 main.py --mode train \\"
echo "    --use_titans_memory \\"
echo "    --use_transformer2_adaptation \\"
echo "    --use_mvot_processor \\"
echo "    --use_blt_processor \\"
echo "    --blt_checkpoint_path ./outputs/blt/mock_byte_lm.pt \\"
echo "    --mvot_codebook_path ./outputs/mvot/mock_codebook.pt \\"
echo "    --hidden_size 128 \\"  # Small model for quick testing
echo "    --num_layers 2 \\"
echo "    --num_attention_heads 4 \\"
echo "    --batch_size 8 \\"
echo "    --learning_rate 5e-5 \\"
echo "    --max_steps 50 \\"
echo "    --output_dir ./outputs/neat_model_test"