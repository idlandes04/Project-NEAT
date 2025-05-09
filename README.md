# --- START OF FILE README.md ---

# Project NEAT V2 - Refactored Implementation

This repository contains a refactored implementation of Project NEAT, aiming to integrate concepts from four key research papers into a unified, efficient, and extensible architecture:

1.  **Byte Latent Transformer (BLT):** Dynamic byte patching for efficient input processing.
2.  **Titans:** Test-time memory adaptation with surprise mechanisms.
3.  **Transformer²:** Real-time task adaptation via Singular Value Fine-tuning (SVF).
4.  **Multimodal Visualization-of-Thought (MVoT):** Interleaved text/image reasoning.

This refactored version focuses on a cleaner codebase, reduced complexity compared to the original, and a modular design.

## Architecture Overview

The core architecture (`src/model/architecture.py:UnifiedModel`) is based on a standard Transformer decoder. Optional components from the papers above can be enabled via configuration:

-   **Input:** Can use standard tokenization or the BLT component (`src/components/blt.py`) for dynamic byte patching.
-   **Memory:** The Titans-inspired memory system (`src/components/memory.py`) can be integrated into the Transformer layers, providing short-term (windowed attention), long-term (surprise-based), and persistent memory capabilities.
-   **Adaptation:** The Transformer² component (`src/components/adaptation.py`) allows for adapting model weights (specifically singular values) based on task identification *before* a forward pass, enabling real-time specialization.
-   **Output:** Can use a standard LM head or the MVoT component (`src/components/multimodal.py`) for predicting both text and visual codebook tokens, including the token discrepancy loss for improved visual fidelity.

## Directory Structure
Use code with caution.
Markdown
project-neat/
├── src/ # Core source code
│ ├── components/ # BLT, Memory, Adaptation, Multimodal components
│ ├── model/ # Transformer blocks and UnifiedModel architecture
│ ├── training/ # Dataset, Trainer, collate function
│ └── utils/ # Config, hardware, tokenization, logging, SVD utils
├── data/ # Raw and processed data (needs population)
│ ├── raw/
│ └── processed/
├── output/ # Generated files (models, logs, results)
├── configs/ # Configuration files (YAML/JSON)
├── scripts/ # Helper scripts (e.g., training launch)
├── tests/ # Unit and integration tests (TODO)
├── README.md # This file
└── requirements.txt # Project dependencies

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd project-neat
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* Ensure you have a compatible PyTorch version installed (>= 2.0 recommended for potential FlashAttention support) that matches your hardware (CPU, CUDA, MPS). See [pytorch.org](https://pytorch.org/) for instructions.

## Basic Usage

### Training

1.  **Prepare your data:** Place your training and evaluation data (e.g., text files, pre-processed byte chunks) into directories specified in your configuration file (e.g., `data/processed/train`, `data/processed/eval`).
2.  **Configure your run:** Create or modify a configuration file in the `configs/` directory (e.g., `configs/config_base_500m.yaml`). Adjust parameters like model size, component activation, data paths, batch size, learning rate, etc.
3.  **Run the training script:**
    ```bash
    python scripts/train.py --config configs/your_config_file.yaml
    ```
    Training logs will be saved to the `output_dir` specified in the config (under a `logs/` subdirectory), and checkpoints will be saved under `checkpoints/`.

### Evaluation (TODO)

Evaluation scripts are not yet implemented but would typically involve loading a trained checkpoint and running inference on a test dataset.

### Inference (TODO)

Inference scripts are not yet implemented.

## Current Status & Next Steps

-   The core codebase structure is implemented based on the refactoring plan.
-   All major components (BLT, Memory, Adaptation, Multimodal) and the UnifiedModel architecture are defined with complete interfaces and internal logic.
-   Utilities for config, hardware, logging, tokenization, and SVD are implemented.
-   A basic training script (`scripts/train.py`) is provided.
-   **Next Steps:**
    -   Thorough testing (unit and integration tests).
    -   Data preparation scripts.
    -   Hyperparameter tuning for the target model size (~500M).
    -   Implementation of evaluation and inference scripts.
    -   Refinement of complex interactions (e.g., state passing for surprise calculation, T2 two-pass inference flow).
    -   Implementation of advanced features (e.g., MVoT interleaved generation, memory consolidation).

## Contributing (TODO)

Guidelines for contributing will be added later.

# --- END OF FILE README.md ---