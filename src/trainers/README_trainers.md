# Project NEAT Trainers

This directory contains training modules for various components of the NEAT architecture.

## Structure

### Main Components (Consolidated)

- `__init__.py`: Exports all trainer modules and their public interfaces
- `main_env_prepare.py`: Environment preparation script for setting up data and output directories
- `main_trainer.py`: Unified training script for all NEAT components (includes hardware-aware training, training monitoring, and data preparation)
- `main_eval.py`: Unified evaluation script for all NEAT components (includes BLT interactive testing)

## Usage

The main trainer scripts provide a clean, streamlined interface. These scripts consolidate functionality from previously separate files:

- `main_trainer.py`: Includes functionality from `hardware_aware_trainer.py` (HardwareAwareTrainer, PerformanceProfiler) and `training_monitor.py` (TrainingMonitor, GPUStats) as well as data preparation utilities from `data_preparation.py`
- `main_eval.py`: Includes interactive testing functionality from `blt_interactive.py` (BLTInteractiveTester)

This consolidation reduces file count while maintaining all functionality.

```bash
# Prepare the environment
python -m src.trainers.main_env_prepare [--clean_all] [--preserve_models]

# Train a BLT entropy estimator
python -m src.trainers.main_trainer --model_type blt \
  --train_data_dir ./data/pile_subset/train \
  --eval_data_dir ./data/pile_subset/eval \
  --hidden_size 128 --num_layers 2 --num_heads 8 \
  --batch_size 32 --max_steps 10000

# Evaluate a trained BLT model
python -m src.trainers.main_eval --model_type blt \
  --model_path ./outputs/byte_lm/best_model.pt \
  --eval_mode interactive

# Train the full NEAT model
python -m src.trainers.main_trainer --model_type full \
  --hidden_size 768 --num_layers 12 --num_attention_heads 12 \
  --use_titans_memory --use_transformer2_adaptation \
  --use_mvot_processor --use_blt_processor \
  --blt_checkpoint_path ./outputs/byte_lm/best_model.pt \
  --batch_size 16 --max_steps 10000
```

## Configuration Files

All new consolidated scripts support loading parameters from JSON configuration files:

```bash
# Train using a configuration file
python -m src.trainers.main_trainer --model_type blt --config_file ./configs/blt_train.json
```

Example configuration file:
```json
{
  "train_data_dir": "./data/pile_subset/train",
  "eval_data_dir": "./data/pile_subset/eval",
  "hidden_size": 128,
  "num_layers": 2,
  "num_heads": 4,
  "block_size": 128,
  "batch_size": 32,
  "max_steps": 10000,
  "output_dir": "./outputs/byte_lm",
  "mixed_precision": true
}
```

## Component Support

The consolidated scripts support all Project NEAT components:

1. **BLT (Byte-Level Transformer)**: Entropy estimator for dynamic patching
2. **MVoT (Multimodal Vision-or-Text)**: Visual codebook for multimodal processing
3. **Full NEAT Model**: Complete neural architecture with all components
4. **Baseline Model**: Standard transformer for performance comparison

## Development

When adding new training functionality:

1. Decide which consolidated file is appropriate:
   - `main_env_prepare.py`: For environment setup and preparation functions
   - `main_trainer.py`: For model training and data preparation functions
   - `main_eval.py`: For model evaluation and testing functions
2. Add your new functionality to the appropriate file
3. Add public functions/classes to `__init__.py` if needed
4. Update this README.md with documentation of your additions

All trainers should be designed with flexibility and hardware awareness in mind, using the interfaces provided within main_trainer.py.