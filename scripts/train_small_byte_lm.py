"""
Simple script to train a small byte-level language model for BLT entropy estimation.

This script bypasses the config system to directly train the ByteLM model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from pathlib import Path
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.blt.byte_processor import SmallByteLM
from src.trainers.blt_trainer import ByteDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def train_byte_lm(args):
    """
    Train a small byte-level language model for entropy estimation.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Find training and evaluation files
    train_files = []
    for root, _, files in os.walk(args.train_data_dir):
        for file in files:
            train_files.append(os.path.join(root, file))
    
    eval_files = []
    for root, _, files in os.walk(args.eval_data_dir):
        for file in files:
            eval_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(train_files)} training files")
    logger.info(f"Found {len(eval_files)} evaluation files")
    
    # Create datasets
    train_dataset = ByteDataset(
        file_paths=train_files,
        block_size=args.block_size,
        cache_dir=args.cache_dir
    )
    
    eval_dataset = ByteDataset(
        file_paths=eval_files,
        block_size=args.block_size,
        cache_dir=args.cache_dir
    )
    
    logger.info(f"Created datasets with {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model with mock config object
    class MockConfig:
        def __init__(self):
            self.byte_lm_dropout = args.dropout
            self.byte_lm_max_position = args.block_size
    
    model = SmallByteLM(MockConfig())
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set scheduler
    warmup_pct = min(0.3, max(0.1, args.warmup_steps / max(1, args.max_steps)))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.max_steps,
        pct_start=warmup_pct,
        anneal_strategy="linear"
    )
    
    # Enable mixed precision if available
    if args.mixed_precision and hasattr(torch.cuda, 'amp'):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    logger.info("Starting training")
    model.train()
    
    try:
        while global_step < args.max_steps:
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss, _ = model(batch["input_ids"], batch["labels"])
                    
                # Backward pass with gradient scaling
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Update global step
                global_step += 1
                
                # Log progress
                if global_step % args.log_steps == 0:
                    logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
                # Evaluation
                if global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader, device, scaler)
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        save_model(model, os.path.join(args.output_dir, "best_model.pt"))
                        logger.info(f"New best model saved (loss: {eval_loss:.4f})")
                    
                    # Back to training mode
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_model(model, os.path.join(args.output_dir, f"checkpoint-{global_step}.pt"))
                    save_model(model, os.path.join(args.output_dir, "checkpoint-latest.pt"))
                    logger.info(f"Checkpoint saved at step {global_step}")
                
                # Check if we've reached max steps
                if global_step >= args.max_steps:
                    break
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        # Try to save a checkpoint in case of error
        try:
            save_model(model, os.path.join(args.output_dir, "checkpoint-error.pt"))
            logger.info("Saved checkpoint at error point")
        except Exception as save_error:
            logger.error(f"Failed to save error checkpoint: {save_error}")
    
    # Save final model
    save_model(model, os.path.join(args.output_dir, "final_model.pt"))
    logger.info("Training complete. Final model saved.")
    
    # Final evaluation
    eval_loss = evaluate(model, eval_dataloader, device, scaler)
    logger.info(f"Final evaluation loss: {eval_loss:.4f}")

def evaluate(model, dataloader, device, scaler):
    """
    Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader with evaluation data
        device: Device to use for evaluation
        scaler: Optional gradient scaler for mixed precision
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss, _ = model(batch["input_ids"], batch["labels"])
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / max(1, num_batches)
    
    return avg_loss

def save_model(model, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), path)
            
        logger.info(f"Model saved to {path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train BLT entropy estimator")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden size of the byte LM")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the byte LM")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads in byte LM")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability in the byte LM")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for byte LM training")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Number of steps between model saves")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Number of steps between logging")
    
    # Data parameters
    parser.add_argument("--train_data_dir", type=str, default="./data/pile_subset/train",
                        help="Directory containing training data files")
    parser.add_argument("--eval_data_dir", type=str, default="./data/pile_subset/eval",
                        help="Directory containing evaluation data files")
    parser.add_argument("--cache_dir", type=str, default="./data/cache/byte_lm",
                        help="Directory to cache processed data")
    parser.add_argument("--output_dir", type=str, default="./outputs/byte_lm",
                        help="Directory to save outputs")
    
    # Misc options
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Whether to use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    train_byte_lm(args)

if __name__ == "__main__":
    main()