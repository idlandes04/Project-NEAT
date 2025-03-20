#!/bin/bash
# Script to run optimized BLT training with enhanced data and configurations

# Set paths
CONFIG_FILE="./scripts/main_cli_configs/blt_entropy_final.json"
OUTPUT_DIR="./outputs/byte_lm_optimized"
DATA_DIR="./data/blt_training_data"
BINARY_DIR="./data/binary_samples"
LOG_DIR="./logs"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$BINARY_DIR"
mkdir -p "$LOG_DIR"

# Define log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/blt_training_$TIMESTAMP.log"

echo "Starting optimized BLT training at $(date)" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"

# Step 1: Generate training data
echo "Generating enhanced training data..." | tee -a "$LOG_FILE"
python3 ./scripts/generate_blt_training_data.py \
    --config "$CONFIG_FILE" \
    --output_dir "$DATA_DIR" \
    --num_samples 2000 2>&1 | tee -a "$LOG_FILE"

# Check if data generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# Step 2: Run BLT training with optimized configuration
echo "Starting BLT training with optimized parameters..." | tee -a "$LOG_FILE"
python3 -m src.trainers.main_trainer \
    --model_type blt \
    --config_file "$CONFIG_FILE" \
    --train_data_dir "$DATA_DIR/train" \
    --eval_data_dir "$DATA_DIR/eval" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 20000 2>&1 | tee -a "$LOG_FILE"

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
fi

# Step 3: Evaluate the trained model
echo "Evaluating the trained BLT model..." | tee -a "$LOG_FILE"
python3 -m src.trainers.main_eval \
    --model_type blt \
    --model_path "$OUTPUT_DIR/best_model.pt" \
    --eval_mode analyze 2>&1 | tee -a "$LOG_FILE"

echo "=====================================================" | tee -a "$LOG_FILE"
echo "Optimized BLT training completed at $(date)" | tee -a "$LOG_FILE"
echo "Model output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"