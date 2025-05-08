# --- START OF FILE src/training/trainer.py ---

"""
Core Trainer class for Project NEAT.

Handles the training loop, evaluation, checkpointing, device management,
mixed precision, and integration of model components during training
(e.g., triggering adaptation).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import os
import time
import math
import logging
import shutil
from typing import Dict, Any, Optional, Tuple, Callable, Union
from contextlib import nullcontext # Use nullcontext for conditional autocast

# Assume model, config, dataset, collate_fn classes are available via imports
# from ..model.architecture import UnifiedModel
# from ..training.data import UnifiedDataset, collate_fn
# from ..utils.config import ModelConfig # Or your specific config class
# from ..utils.tokenization import TokenizerBase
# from ..utils.logging_utils import setup_logging # If using external setup
# from ..utils.hardware import detect_device, get_optimal_precision # If using hardware utils

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the training and evaluation loop for the UnifiedModel.

    Orchestrates data loading, optimization, mixed precision, adaptation calls,
    logging, evaluation, and checkpointing based on the provided configuration.
    """
    def __init__(
        self,
        model: nn.Module, # Expects an instance of UnifiedModel or similar interface
        config: Any, # Configuration object (e.g., ModelConfig)
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None, # Pass tokenizer if needed by model/collate
        collate_fn_func: Optional[Callable] = None # Pass the actual collate_fn function
    ):
        """
        Initializes the Trainer.

        Args:
            model: The model instance to train.
            config: Configuration object containing hyperparameters and settings.
            train_dataset: The training dataset instance.
            eval_dataset: Optional evaluation dataset instance.
            tokenizer: Optional tokenizer instance (might be needed for logging/debugging).
            collate_fn_func: The collate function to use for DataLoader (e.g., the one from data.py).
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer # Store for potential use, e.g., decoding examples
        self.collate_fn_func = collate_fn_func
        if self.collate_fn_func is None:
             logger.error("collate_fn_func must be provided to the Trainer.")
             raise ValueError("collate_fn_func is required.")

        # --- Logging Setup (ensure configured externally or here) ---
        if not logging.getLogger().hasHandlers():
             logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
             logger.info("Basic logging configured by Trainer.")
        # Example: setup_logging(log_dir=config.training.output_dir) # If using custom setup

        # --- Device Setup ---
        self._setup_device()
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # --- DataLoader Setup ---
        self.train_dataloader = self._create_dataloader(self.train_dataset, shuffle=True)
        self.eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=False) if self.eval_dataset else None

        # --- Optimization Setup ---
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer_scheduler()

        # --- Mixed Precision Setup ---
        self._setup_mixed_precision()

        # --- Checkpointing Setup ---
        self.output_dir = self.config.training.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.save_steps = self.config.training.save_steps

        # --- Load Checkpoint ---
        resume_path = self.config.training.resume_from
        if resume_path:
            self.load_checkpoint(resume_path)
        else:
             logger.info("No checkpoint provided, starting training from scratch.")

        # --- Adaptation Component SVD Precomputation ---
        if self.config.use_transformer2_adaptation and hasattr(self.model, 'adapt_comp') and self.model.adapt_comp is not None:
             if not self.model.adapt_comp.svd_cache: # Only compute if cache is empty
                  logger.info("Running SVD precomputation for Adaptation Component...")
                  try:
                       self.model.adapt_comp.precompute_svd(self.model)
                  except Exception as e:
                       logger.error(f"SVD precomputation failed: {e}. Adaptation might not work.", exc_info=True)


    def _setup_device(self):
        """Sets up the compute device based on availability and config."""
        if self.config.hardware.force_cpu:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             # Add functional check for MPS if needed based on hardware utils
             self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Trainer using device: {self.device}")

    def _setup_mixed_precision(self):
        """Sets up Automatic Mixed Precision (AMP) based on config and device."""
        hw_config = self.config.hardware
        self.use_mixed_precision = False
        self.compute_dtype = torch.float32

        if hw_config.mixed_precision:
            if self.device.type == 'cuda':
                # Check CUDA capability for bf16 support
                cap_major, _ = torch.cuda.get_device_capability()
                if cap_major >= 8: # Ampere+ supports bfloat16
                    self.compute_dtype = torch.bfloat16
                    self.use_mixed_precision = True
                    logger.info("Mixed precision enabled (CUDA, bfloat16).")
                else: # Older GPUs support float16
                    self.compute_dtype = torch.float16
                    self.use_mixed_precision = True
                    logger.info("Mixed precision enabled (CUDA, float16).")
            elif self.device.type == 'mps':
                 # MPS supports float16, bfloat16 support varies
                 self.compute_dtype = torch.float16 # Default to fp16 for MPS AMP
                 self.use_mixed_precision = True
                 logger.info("Mixed precision enabled (MPS, float16).")
            else: # CPU
                 logger.warning("Mixed precision requested but device is CPU. Disabled.")
        else:
            logger.info("Mixed precision disabled by configuration.")

        self.scaler = GradScaler(enabled=self.use_mixed_precision)


    def _create_dataloader(self, dataset: Optional[Dataset], shuffle: bool) -> Optional[DataLoader]:
        """Creates a DataLoader for a given dataset."""
        if dataset is None:
            return None
        if self.collate_fn_func is None:
             raise ValueError("Cannot create DataLoader without a collate_fn_func.")

        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.hardware.num_workers,
            collate_fn=self.collate_fn_func, # Pass the function directly
            pin_memory=(self.device.type == 'cuda'), # Pin memory only for CUDA
            prefetch_factor=2 if self.config.hardware.num_workers > 0 else None, # Modest prefetch
            persistent_workers=True if self.config.hardware.num_workers > 0 else False
        )

    def _setup_optimizer_scheduler(self):
        """Creates the AdamW optimizer and learning rate scheduler."""
        train_config = self.config.training
        # Filter out parameters that don't require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
             logger.warning("Model has no trainable parameters. Optimizer not created.")
             self.optimizer = None
             self.scheduler = None
             return

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=train_config.learning_rate,
            betas=(train_config.adam_beta1, train_config.adam_beta2),
            eps=train_config.adam_epsilon,
            weight_decay=train_config.weight_decay
        )
        logger.info(f"Optimizer: AdamW (LR={train_config.learning_rate}, WD={train_config.weight_decay})")

        # Scheduler: Linear warmup then cosine decay
        max_steps = train_config.max_steps
        warmup_steps = train_config.warmup_steps # Use direct warmup steps

        if warmup_steps >= max_steps:
             logger.warning(f"warmup_steps ({warmup_steps}) >= max_steps ({max_steps}). Using linear warmup only.")
             def lr_lambda_warmup_only(current_step: int):
                  return float(current_step) / float(max(1, warmup_steps))
             lr_lambda = lr_lambda_warmup_only
        else:
             def lr_lambda_cosine(current_step: int):
                  if current_step < warmup_steps:
                       return float(current_step) / float(max(1, warmup_steps))
                  # Cosine decay phase
                  progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                  # Cosine schedule from 1 down to 0
                  return 0.5 * (1.0 + math.cos(math.pi * progress))
             lr_lambda = lr_lambda_cosine

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Scheduler: LambdaLR (Linear Warmup: {warmup_steps} steps, Cosine Decay to 0 over {max_steps} steps)")


    def train(self):
        """Runs the main training loop."""
        if self.optimizer is None or self.train_dataloader is None:
             logger.error("Optimizer or Training DataLoader not initialized. Cannot train.")
             return

        train_config = self.config.training
        max_steps = train_config.max_steps
        accumulation_steps = train_config.gradient_accumulation_steps
        log_steps = train_config.logging_steps
        eval_steps = train_config.eval_steps

        logger.info(f"***** Starting Training *****")
        logger.info(f"  Total Steps = {max_steps}")
        logger.info(f"  Instantaneous Batch Size per Device = {train_config.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {accumulation_steps}")
        logger.info(f"  Effective Batch size = {train_config.batch_size * accumulation_steps}")
        logger.info(f"  Mixed Precision = {self.use_mixed_precision} ({self.compute_dtype})")
        logger.info(f"  Logging every {log_steps} steps")
        logger.info(f"  Evaluating every {eval_steps} steps")
        logger.info(f"  Saving every {self.save_steps} steps")

        self.model.train()
        log_loss_accum = 0.0
        steps_processed_in_log_period = 0
        log_period_start_time = time.time()

        # Use nullcontext if not using mixed precision for cleaner code
        autocast_context = autocast(dtype=self.compute_dtype) if self.use_mixed_precision else nullcontext()

        # --- Training Loop ---
        while self.global_step < max_steps:
            for step, batch in enumerate(self.train_dataloader):
                if self.global_step >= max_steps: break
                is_accumulation_step = (step + 1) % accumulation_steps == 0

                # --- Prepare Batch ---
                try:
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                except Exception as e:
                     logger.error(f"Step {self.global_step}: Error moving batch to device: {e}. Skipping batch.", exc_info=True)
                     continue

                # --- Adaptation Step (Before Forward) ---
                if self.config.use_transformer2_adaptation and hasattr(self.model, 'adapt_comp') and self.model.adapt_comp is not None:
                    try:
                        with torch.no_grad(): # Task identification shouldn't require gradients here
                            # 1. Get input embeddings (needed for task identifier)
                            input_embeds = self.model.token_embedding(batch['input_ids'])
                            # 2. Get task weights from the adaptation component's forward method
                            task_weights = self.model.adapt_comp(input_embeds) # [batch, num_tasks]
                        # 3. Apply adaptation using the average task weight across the batch
                        self.model.adapt_weights(task_weights.mean(dim=0, keepdim=True))
                    except Exception as e:
                         logger.error(f"Step {self.global_step}: Error during weight adaptation: {e}. Continuing without adaptation.", exc_info=True)


                # --- Forward Pass ---
                with autocast_context:
                    try:
                        # Pass only necessary inputs to model forward
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch.get('attention_mask'), # Use .get for optional keys
                            token_type_ids=batch.get('token_type_ids') if self.config.use_mvot_processor else None
                            # Note: labels/image_targets are used in calculate_loss, not forward
                        )
                        # Calculate loss using model's method
                        loss = self.model.calculate_loss(
                            outputs,
                            labels=batch.get('labels'),
                            image_targets=batch.get('image_targets') # Pass if exists
                        )

                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                             logger.warning(f"Step {self.global_step}: Invalid loss ({loss}). Skipping backward/step.")
                             self.optimizer.zero_grad() # Zero grads even if step is skipped
                             continue

                        # Scale loss for accumulation
                        loss = loss / accumulation_steps

                    except Exception as e:
                         logger.error(f"Step {self.global_step}: Error during forward/loss calculation: {e}", exc_info=True)
                         self.optimizer.zero_grad() # Clear potentially corrupted grads
                         continue # Skip this step

                # --- Backward Pass ---
                try:
                    self.scaler.scale(loss).backward()
                except Exception as e:
                     logger.error(f"Step {self.global_step}: Error during backward pass: {e}", exc_info=True)
                     self.optimizer.zero_grad() # Clear potentially corrupted grads
                     continue # Skip this step

                log_loss_accum += loss.item() * accumulation_steps # Log unscaled loss
                steps_processed_in_log_period += 1

                # --- Optimizer Step (Only update weights every accumulation_steps) ---
                if is_accumulation_step:
                    try:
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), train_config.max_grad_norm
                        )
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        # Update scaler for next iteration
                        self.scaler.update()
                        # Scheduler step (after optimizer step)
                        self.scheduler.step()
                        # Zero gradients for the next accumulation cycle
                        self.optimizer.zero_grad()

                        self.global_step += 1

                        # --- Logging ---
                        if self.global_step % log_steps == 0:
                            avg_loss = log_loss_accum / steps_processed_in_log_period
                            current_lr = self.scheduler.get_last_lr()[0]
                            elapsed_time = time.time() - log_period_start_time
                            steps_per_sec = steps_processed_in_log_period * accumulation_steps / elapsed_time
                            logger.info(
                                f"Step: {self.global_step}/{max_steps} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {current_lr:.3e} | "
                                f"Steps/sec: {steps_per_sec:.2f}"
                            )
                            # Reset log accumulators
                            log_loss_accum = 0.0
                            steps_processed_in_log_period = 0
                            log_period_start_time = time.time()

                        # --- Evaluation & Checkpointing ---
                        if self.global_step % eval_steps == 0 and self.eval_dataloader is not None:
                            eval_metrics = self.evaluate()
                            eval_loss = eval_metrics.get('eval_loss', float('inf'))
                            logger.info(f"Step {self.global_step}: Eval Loss = {eval_loss:.4f}, Perplexity = {eval_metrics.get('perplexity', 'N/A'):.2f}")
                            self.model.train() # Set model back to train mode
                            # Save best model checkpoint based on eval loss
                            if eval_loss < self.best_eval_loss:
                                logger.info(f"New best eval loss: {eval_loss:.4f} (previous: {self.best_eval_loss:.4f}). Saving 'best' checkpoint.")
                                self.best_eval_loss = eval_loss
                                self.save_checkpoint(step="best")

                        if self.global_step % self.save_steps == 0:
                            self.save_checkpoint(step=self.global_step)
                            self.save_checkpoint(step="latest") # Overwrite latest

                    except Exception as e:
                         logger.error(f"Step {self.global_step}: Error during optimizer step/logging/eval/saving: {e}", exc_info=True)
                         self.optimizer.zero_grad() # Ensure grads are zeroed if error occurs mid-step


        logger.info(f"Maximum steps ({max_steps}) reached. Training finished.")
        # Save final model state
        self.save_checkpoint(step="final")


    def evaluate(self) -> Dict[str, float]:
        """Runs evaluation on the evaluation dataset."""
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided, skipping evaluation.")
            return {}

        self.model.eval() # Set model to evaluation mode
        total_eval_loss = 0.0
        total_tokens_evaluated = 0 # For perplexity calculation

        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(self.eval_dataset)}")
        logger.info(f"  Batch size = {self.config.training.batch_size}")

        autocast_context = autocast(dtype=self.compute_dtype) if self.use_mixed_precision else nullcontext()
        eval_start_time = time.time()

        with torch.no_grad(): # Disable gradient calculations
            for batch in self.eval_dataloader:
                try:
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                except Exception as e:
                     logger.error(f"Eval: Error moving batch to device: {e}. Skipping batch.", exc_info=True)
                     continue

                with autocast_context:
                     try:
                         outputs = self.model(
                              input_ids=batch['input_ids'],
                              attention_mask=batch.get('attention_mask'),
                              token_type_ids=batch.get('token_type_ids') if self.config.use_mvot_processor else None
                         )
                         loss = self.model.calculate_loss(
                             outputs,
                             labels=batch.get('labels'),
                             image_targets=batch.get('image_targets')
                         )

                         if loss is not None and not torch.isnan(loss):
                              # Loss is typically averaged over tokens in the batch by CrossEntropyLoss
                              # Multiply by batch size (or number of non-ignored tokens) if loss needs aggregation
                              # Assuming loss is already mean loss over batch tokens
                              total_eval_loss += loss.item() * batch['input_ids'].size(0) # Weight by batch size
                              # Count non-ignored tokens for perplexity
                              if 'labels' in batch:
                                   valid_labels = batch['labels'] != -100
                                   total_tokens_evaluated += valid_labels.sum().item()
                         else:
                              logger.warning(f"Eval: Invalid loss ({loss}) encountered.")

                     except Exception as e:
                          logger.error(f"Eval: Error during forward/loss calculation: {e}", exc_info=True)

        eval_duration = time.time() - eval_start_time
        num_eval_samples = len(self.eval_dataset)
        avg_eval_loss = total_eval_loss / num_eval_samples if num_eval_samples > 0 else float('inf')

        perplexity = math.exp(total_eval_loss / total_tokens_evaluated) if total_tokens_evaluated > 0 else float('inf')

        metrics = {"eval_loss": avg_eval_loss, "perplexity": perplexity}
        logger.info(f"Evaluation finished in {eval_duration:.2f}s. Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")
        self.model.train() # Set model back to training mode
        return metrics

    def save_checkpoint(self, step: Union[int, str]):
        """Saves a training checkpoint."""
        if self.optimizer is None or self.scheduler is None:
             logger.warning("Cannot save checkpoint, optimizer/scheduler not initialized.")
             return

        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        state = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_mixed_precision else None,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.to_dict() # Save config as dict
        }
        try:
            # Save checkpoint
            torch.save(state, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path} at step {self.global_step}")

            # Handle 'latest' symlink or copy for convenience
            if step == "latest":
                 # No special handling needed, just overwrites
                 pass
            elif step == "best":
                 # Optionally copy best checkpoint to a fixed name like 'model_best.pt'
                 best_path = os.path.join(self.output_dir, "model_best.pt")
                 shutil.copyfile(checkpoint_path, best_path)
                 logger.info(f"Copied best checkpoint to {best_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint '{checkpoint_name}' at step {self.global_step}: {e}", exc_info=True)


    def load_checkpoint(self, path: str):
        """Loads training state from a checkpoint."""
        if not os.path.exists(path):
            logger.error(f"Checkpoint path not found: {path}. Cannot resume.")
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        logger.info(f"Attempting to load checkpoint from: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device) # Load directly to target device

            # --- Load Model State ---
            # Use strict=False to handle cases where model architecture might have changed slightly
            # (e.g., adding/removing heads during development) or if saving only part of the model.
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                 logger.warning(f"Checkpoint loading: Missing keys in model state_dict: {missing_keys}")
            if unexpected_keys:
                 logger.warning(f"Checkpoint loading: Unexpected keys in model state_dict: {unexpected_keys}")

            # --- Load Optimizer and Scheduler State ---
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                 try:
                      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                      logger.info("Optimizer state loaded.")
                 except Exception as e:
                      logger.warning(f"Could not load optimizer state: {e}. Optimizer state will be reset.")
            elif self.optimizer:
                 logger.warning("Optimizer state not found in checkpoint. Optimizer state will be reset.")

            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                 try:
                      self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                      logger.info("Scheduler state loaded.")
                 except Exception as e:
                      logger.warning(f"Could not load scheduler state: {e}. Scheduler state will be reset.")
            elif self.scheduler:
                 logger.warning("Scheduler state not found in checkpoint. Scheduler state will be reset.")

            # --- Load AMP Scaler State ---
            if self.use_mixed_precision and self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                 try:
                      self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                      logger.info("AMP GradScaler state loaded.")
                 except Exception as e:
                      logger.warning(f"Could not load GradScaler state: {e}. Scaler state will be reset.")
            elif self.use_mixed_precision and self.scaler:
                 logger.warning("Scaler state not found in checkpoint or was None. Scaler state will be reset.")


            # --- Load Training Progress ---
            self.global_step = checkpoint.get('step', 0)
            self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))

            # Optionally validate loaded config against current config
            if 'config' in checkpoint:
                 logger.info("Checkpoint includes configuration (not automatically applied, for reference only).")
                 # config_from_ckpt = checkpoint['config']
                 # TODO: Add comparison logic if needed

            logger.info(f"Successfully resumed training from checkpoint: {path} at step {self.global_step}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}. Training will start from scratch.", exc_info=True)
            # Reset state variables if loading fails
            self.global_step = 0
            self.best_eval_loss = float('inf')
            # Optionally reset optimizer/scheduler? Depends on desired behavior.
            # self._setup_optimizer_scheduler() # Re-initialize optimizer/scheduler


# --- END OF FILE src/training/trainer.py ---