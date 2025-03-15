# NEAT Model Training Instructions

## Overview

This document provides instructions for training the NEAT (Neural Adaptive Transformers) model, which integrates:

1. **Titans memory system** - Long-term memory with adaptive decay
2. **Transformer² adaptation** - SVD-based weight adaptation
3. **MVoT processor** - Multimodal visualization capabilities
4. **BLT processor** - Byte-level entropy-based processing

## Mac Testing (Completed)

We've successfully tested the model on Mac with:
- Small parameter count (64 hidden size, 2 layers, 4 attention heads)
- Mock BLT entropy estimator and MVoT visual codebook
- Synthetic math data for training

## Windows PC Training (Next Step)

### Preparation

1. Copy the entire project directory to the Windows PC with 3080ti GPU
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Download Datasets

1. Download the MMR1-Math-RL-Data dataset:
   ```
   python scripts/download_mmr1_math_dataset.py
   ```

2. Verify the synthetic data is properly generated:
   ```
   python scripts/generate_synthetic_data.py --output_dir data/synthetic --train_size 50000 --eval_size 10000
   ```

### Start Training

1. For full training (100M parameter model), run:
   ```
   scripts/train_neat_model_windows.bat
   ```

2. Monitor training progress in the console or logs.

3. If training is interrupted, resume with:
   ```
   scripts/resume_training_windows.bat
   ```

### Training Parameters

The model is configured for:
- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Batch size: 16
- Learning rate: 5e-5
- Maximum steps: 10,000
- Gradient accumulation: Enabled
- Mixed precision: Enabled
- Gradient checkpointing: Enabled

### Component Distribution

The 100M parameter model distributes parameters approximately as:
- Core transformer: ~40M parameters
- Titans memory system: ~20M parameters
- Transformer² adaptation: ~20M parameters
- BLT processor: ~10M parameters
- MVoT processor: ~10M parameters

## Evaluating Results

After training, evaluate the model with:
```
python main.py --mode eval --model_path ./outputs/neat_model_full/checkpoint-latest
```

## Troubleshooting

1. **Memory issues**: Try reducing batch size or enabling more aggressive gradient checkpointing
2. **Mock models**: Replace mock models with real pre-trained models when available
3. **Data loading**: Ensure the datasets are properly loaded by checking the log output

## Next Steps

1. Finish Phase 3.1.1: Complete Synthetic Data Generator Integration
2. Move to Phase 3.1.2: Baseline Transformer Implementation
3. Implement Phase 3.1.3: Component-Wise Ablation Testing
4. Continue with memory and learning evaluation in Phase 3.2