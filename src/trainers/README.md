# Project NEAT Trainers

This directory contains training modules for various components of the NEAT architecture.

## Structure

- `__init__.py`: Exports all trainer modules and their public interfaces
- `blt_trainer.py`: BLT (Byte-Level Transformer) entropy estimator training
- `blt_interactive.py`: Interactive testing for trained BLT models
- `data_preparation.py`: Data preparation utilities, including:
  - Downloading Pile dataset subsets
  - Creating mock models for testing
- `training_monitor.py`: Real-time monitoring of training progress
- `hardware_aware_trainer.py`: Hardware-aware training with adaptive resource allocation

## Usage

These modules are designed to be used directly from `main.py` via command-line arguments:

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

# Prepare a subset of the Pile dataset
python main.py prepare_data --data_type pile_subset \
  --pile_output_dir ./data/pile_subset \
  --pile_warc_count 5

# Monitor ongoing training
python main.py test --test_type blt_monitor \
  --output_dir ./outputs/byte_lm \
  --pid <process_id> \
  --max_steps 10000
```

## Development

When adding new trainers or enhancing existing ones:

1. Create a new file with a descriptive name (e.g., `mycomponent_trainer.py`)
2. Add public functions/classes to `__init__.py`
3. Update `main.py` to use your new trainer via the command-line interface
4. Add documentation in this README.md

All trainers should be designed with flexibility and hardware awareness in mind, using the interfaces provided by `hardware_aware_trainer.py` when appropriate.