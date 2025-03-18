# Project NEAT Trainers

This directory contains training modules for various components of the NEAT architecture.

## Structure

### Core Components

- `__init__.py`: Exports all trainer modules and their public interfaces
- `blt_trainer.py`: BLT (Byte-Level Transformer) entropy estimator training
- `blt_interactive.py`: Interactive testing for trained BLT models
- `data_preparation.py`: Data preparation utilities, including downloading datasets and creating mock models
- `training_monitor.py`: Real-time monitoring of training progress
- `hardware_aware_trainer.py`: Hardware-aware training with adaptive resource allocation

### Consolidated Scripts (New!)

- `main_env_prepare.py`: Environment preparation script for setting up data and output directories
- `main_trainer.py`: Unified training script for all NEAT components
- `main_eval.py`: Unified evaluation script for all NEAT components

## Usage

### Original Approach

These modules can be used directly from `main.py` via command-line arguments:

```bash
# Train a BLT entropy estimator
python main.py train --training_type blt_entropy \
  --train_data_dir ./data/pile_subset/train \
  --eval_data_dir ./data/pile_subset/eval \
  --byte_lm_hidden_size 128 \
  --byte_lm_num_layers 2 \
  --batch_size 64 \
  --max_steps 10000

# Test a trained BLT model interactively
python main.py test --test_type blt_interactive \
  --blt_model_path ./outputs/byte_lm/best_model.pt
```

### Consolidated Approach (Recommended)

The new consolidated scripts provide a cleaner, more streamlined interface:

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

When adding new trainers or enhancing existing ones:

1. Create a new file with a descriptive name (e.g., `mycomponent_trainer.py`)
2. Add public functions/classes to `__init__.py`
3. Update the consolidated scripts (`main_trainer.py` and `main_eval.py`) to use your new component
4. Add documentation in this README.md

All trainers should be designed with flexibility and hardware awareness in mind, using the interfaces provided by `hardware_aware_trainer.py` when appropriate.