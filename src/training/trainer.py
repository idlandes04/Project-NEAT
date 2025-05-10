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
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.cuda.amp import GradScaler as CudaGradScaler
from torch.amp import autocast
import os
import time
import math
import logging
import shutil
import pickle
from typing import Dict, Any, Optional, Tuple, Callable, Union
from contextlib import nullcontext

# Use absolute imports
from src.utils.config import ModelConfig
from src.model.architecture import UnifiedModel

logger = logging.getLogger(__name__)

# Conditionally import MpsGradScaler
MpsGradScaler = None
try:
    from torch.mps import GradScaler as MpsGradScalerImport
    MpsGradScaler = MpsGradScalerImport
except (ImportError, AttributeError):
    logger.info("torch.mps.GradScaler not available in this PyTorch version or environment.")


class Trainer:
    """
    Handles the training and evaluation loop for the UnifiedModel.

    Orchestrates data loading, optimization, mixed precision, adaptation calls,
    logging, evaluation, and checkpointing based on the provided configuration.
    """
    def __init__(
        self,
        model: UnifiedModel,
        config: ModelConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
        collate_fn_func: Optional[Callable] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.collate_fn_func = collate_fn_func
        if self.collate_fn_func is None:
             logger.error("collate_fn_func must be provided to the Trainer.")
             raise ValueError("collate_fn_func is required.")

        if not logging.getLogger().hasHandlers():
             logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
             logger.info("Basic logging configured by Trainer as fallback.")

        self._setup_device()
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Use config.training.eval_batch_size if specified, else default to config.training.batch_size
        eval_batch_size_config = getattr(self.config.training, "eval_batch_size", None)
        eval_batch_size = eval_batch_size_config if eval_batch_size_config is not None else self.config.training.batch_size


        train_shuffle = not isinstance(self.train_dataset, IterableDataset)
        if not train_shuffle and isinstance(self.train_dataset, IterableDataset):
            logger.info("Training dataset is IterableDataset, shuffle=False for DataLoader. Dataset's __iter__ controls order.")
        
        self.train_dataloader = self._create_dataloader(self.train_dataset, shuffle=train_shuffle, batch_size=self.config.training.batch_size)
        
        eval_shuffle = False # Evaluation dataloader should not shuffle
        self.eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=eval_shuffle, batch_size=eval_batch_size) if self.eval_dataset else None


        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self._setup_optimizer_scheduler()

        self._setup_mixed_precision()

        self.output_dir = self.config.training.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.save_steps = self.config.training.save_steps

        resume_path = self.config.training.resume_from
        if resume_path:
            try:
                self.load_checkpoint(resume_path)
            except FileNotFoundError:
                 logger.error(f"Resume checkpoint not found at {resume_path}. Starting from scratch.")
            except Exception as e:
                 logger.error(f"Failed to load resume checkpoint {resume_path}: {e}. Starting from scratch.", exc_info=True)
        else:
             logger.info("No checkpoint provided, starting training from scratch.")

        # SVD Precomputation for Transformer2 Adaptation
        if self.config.use_transformer2_adaptation and hasattr(self.model, 'adapt_comp') and self.model.adapt_comp is not None:
             needs_precompute = True
             # Check if cache already exists from a previous run or if it's populated
             if hasattr(self.model.adapt_comp, 'svd_cache') and self.model.adapt_comp.svd_cache:
                 # Further check if cache dir has files if disk caching is on
                 if self.config.transformer2.enable_svd_caching and \
                    os.path.exists(self.config.transformer2.svd_cache_dir) and \
                    len(os.listdir(self.config.transformer2.svd_cache_dir)) > 0:
                      logger.info("SVD cache directory exists and is not empty. Assuming precomputation done or will be loaded by svd_utils.")
                      needs_precompute = False # Rely on svd_utils to load from disk if needed
                 elif self.model.adapt_comp.svd_cache: # In-memory cache might be populated
                      logger.info("SVD in-memory cache is populated. Skipping explicit precomputation.")
                      needs_precompute = False


             if needs_precompute:
                  logger.info("Running SVD precomputation for Adaptation Component...")
                  try:
                       self.model.adapt_comp.precompute_svd(self.model)
                  except Exception as e:
                       logger.error(f"SVD precomputation failed: {e}. Adaptation might not work.", exc_info=True)

    def _setup_device(self):
        if self.config.hardware.force_cpu:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             try:
                  # Perform a functional check for MPS
                  _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps")
                  self.device = torch.device("mps")
             except RuntimeError: # Catch specific error if MPS is not truly available
                  logger.warning("MPS available but functional check failed. Using CPU.")
                  self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Trainer using device: {self.device}")

    def _setup_mixed_precision(self):
        hw_config = self.config.hardware
        self.use_mixed_precision = False
        self.compute_dtype = torch.float32 # Default
        self.scaler: Optional[Union[CudaGradScaler, Any]] = None # Union for MpsGradScaler type

        if hw_config.mixed_precision:
            if self.device.type == 'cuda':
                try:
                    # Check CUDA capability for bfloat16 support
                    cap_major, _ = torch.cuda.get_device_capability()
                    if cap_major >= 8: # Ampere and newer support bfloat16
                        self.compute_dtype = torch.bfloat16
                        self.use_mixed_precision = True
                        logger.info("Mixed precision enabled (CUDA, bfloat16).")
                    elif cap_major >=7: # Volta supports float16
                        self.compute_dtype = torch.float16
                        self.use_mixed_precision = True
                        logger.info("Mixed precision enabled (CUDA, float16).")
                    else:
                        logger.warning("CUDA device capability < 7.0. Mixed precision (float16/bfloat16) not fully supported or optimal. Disabling.")
                except RuntimeError as e:
                     logger.warning(f"Could not get CUDA device capability: {e}. Disabling mixed precision for CUDA.", exc_info=True)

            elif self.device.type == 'mps':
                 # MPS typically uses float16 for mixed precision
                 self.compute_dtype = torch.float16
                 self.use_mixed_precision = True
                 logger.info("Mixed precision enabled (MPS, float16).")
            else: # CPU
                 logger.warning("Mixed precision requested but device is CPU. Disabled.")
        else:
            logger.info("Mixed precision disabled by configuration.")

        if self.use_mixed_precision:
            if self.device.type == 'cuda':
                self.scaler = CudaGradScaler(enabled=True)
            elif self.device.type == 'mps':
                if MpsGradScaler is not None:
                    self.scaler = MpsGradScaler(enabled=True)
                    logger.info("Using MPS GradScaler.")
                else:
                    logger.warning("torch.mps.GradScaler not available. Disabling mixed precision for MPS.")
                    self.use_mixed_precision = False # Disable if scaler not found
                    self.scaler = None # Ensure scaler is None
        else:
            self.scaler = None # Ensure scaler is None if mixed precision is off


    def _create_dataloader(self, dataset: Optional[Dataset], shuffle: bool, batch_size: int) -> Optional[DataLoader]:
        if dataset is None:
            return None
        if self.collate_fn_func is None:
             # This should have been caught in __init__
             raise ValueError("Cannot create DataLoader without a collate_fn_func.")

        num_workers = self.config.hardware.num_workers
        # pin_memory should be True if using CUDA and num_workers > 0 generally
        pin_memory_flag = (self.device.type == 'cuda' and getattr(self.config.hardware, 'pin_memory', False))
        # persistent_workers only if num_workers > 0
        persistent_workers_flag = True if num_workers > 0 else False
        # prefetch_factor only if num_workers > 0
        prefetch_factor_val = 2 if num_workers > 0 else None


        effective_shuffle = shuffle
        if isinstance(dataset, IterableDataset) and shuffle:
            # This warning is good, DataLoader handles it by not shuffling.
            logger.warning("Shuffle=True provided to DataLoader for an IterableDataset. DataLoader will ignore this. Shuffling must be handled within the dataset's __iter__ method.")
            effective_shuffle = False # Explicitly set to False for clarity

        logger.debug(f"Creating DataLoader with: shuffle={effective_shuffle}, num_workers={num_workers}, pin_memory={pin_memory_flag}, persistent_workers={persistent_workers_flag}, prefetch_factor={prefetch_factor_val}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle, # Will be False for IterableDataset if shuffle was True
            num_workers=num_workers,
            collate_fn=self.collate_fn_func,
            pin_memory=pin_memory_flag,
            prefetch_factor=prefetch_factor_val, # Only relevant if num_workers > 0
            persistent_workers=persistent_workers_flag # Only relevant if num_workers > 0
        )

    def _setup_optimizer_scheduler(self):
        train_config = self.config.training
        # Filter parameters that require gradients
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

        max_steps = train_config.max_steps
        warmup_steps = train_config.warmup_steps

        # Ensure warmup_steps is not greater than max_steps
        if max_steps > 0 and warmup_steps > max_steps:
            logger.warning(f"Warmup steps ({warmup_steps}) is greater than max steps ({max_steps}). Setting warmup_steps to {max_steps}.")
            warmup_steps = max_steps
        
        if max_steps == 0: # Handle case for evaluation-only or pre-trained model loading
            logger.info("max_steps is 0, scheduler will not change learning rate (constant).")
            # Lambda function that always returns 1.0 (no change to LR)
            lr_lambda = lambda current_step: 1.0
        elif warmup_steps == max_steps and max_steps > 0: # All steps are warmup
             logger.info(f"All {max_steps} steps are warmup steps. LR will increase linearly.")
             def lr_lambda_warmup_only(current_step: int):
                  # current_step is 0-indexed. Add 1 for 1-based step counting in LR calculation.
                  return float(current_step + 1) / float(max(1, warmup_steps))
             lr_lambda = lr_lambda_warmup_only
        else: # Standard warmup then cosine decay
             def lr_lambda_cosine(current_step: int):
                  if current_step < warmup_steps:
                       return float(current_step + 1) / float(max(1, warmup_steps))
                  # Calculate progress for cosine decay part
                  # Ensure decay_steps is at least 1 to avoid division by zero if max_steps == warmup_steps
                  decay_steps = max(1, max_steps - warmup_steps)
                  progress = float(current_step - warmup_steps) / decay_steps
                  # Cosine decay from 1 down to 0
                  return 0.5 * (1.0 + math.cos(math.pi * progress))
             lr_lambda = lr_lambda_cosine

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"Scheduler: LambdaLR (Linear Warmup: {warmup_steps} steps, Cosine Decay to 0 over {max_steps} steps)")


    def train(self):
        if self.optimizer is None or self.train_dataloader is None:
             logger.error("Optimizer or Training DataLoader not initialized. Cannot train.")
             return

        train_config = self.config.training
        max_steps = train_config.max_steps
        if max_steps == 0: # If max_steps is 0, just evaluate if possible and exit.
            logger.info("max_steps is 0. Skipping training loop.")
            if self.eval_dataloader is not None:
                logger.info("Running initial evaluation as max_steps is 0.")
                self.evaluate()
            return

        accumulation_steps = train_config.gradient_accumulation_steps
        log_steps = train_config.logging_steps
        eval_steps = train_config.eval_steps

        logger.info(f"***** Starting Training *****")
        logger.info(f"  Total Steps = {max_steps}")
        logger.info(f"  Instantaneous Batch Size per Device = {train_config.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {accumulation_steps}")
        logger.info(f"  Effective Batch size = {train_config.batch_size * accumulation_steps}")
        logger.info(f"  Mixed Precision = {self.use_mixed_precision} ({self.compute_dtype if self.use_mixed_precision else 'N/A'})")
        logger.info(f"  Logging every {log_steps} steps")
        logger.info(f"  Evaluating every {eval_steps} steps")
        logger.info(f"  Saving every {self.save_steps} steps")

        self.model.train()
        log_loss_accum = 0.0
        steps_processed_in_log_period = 0 # Counts micro-batches for averaging loss
        log_period_start_time = time.time()

        # autocast context manager
        autocast_context = autocast(self.device.type, dtype=self.compute_dtype) if self.use_mixed_precision else nullcontext()
        
        # For IterableDataset, we need to manually handle epochs or step limits
        train_iter = iter(self.train_dataloader)
        
        # Loop for micro-batches
        for current_micro_batch_idx in range(max_steps * accumulation_steps):
            if self.global_step >= max_steps:
                break # Reached max global steps
            
            try:
                batch = next(train_iter)
            except StopIteration:
                logger.info("Training dataloader exhausted. Re-initializing for new epoch.")
                train_iter = iter(self.train_dataloader) # Re-initialize iterator
                try:
                    batch = next(train_iter)
                except StopIteration:
                    logger.error("Training dataloader exhausted immediately after re-initialization. This might indicate an empty dataset or an issue with __iter__.")
                    break # Exit training if dataset is truly empty

            # Determine if it's time for an optimizer step
            is_optimizer_step = (current_micro_batch_idx + 1) % accumulation_steps == 0

            # Move batch to device
            try:
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            except RuntimeError as e:
                 logger.error(f"Step {self.global_step}: Error moving batch to device: {e}. Skipping micro-batch.", exc_info=True)
                 continue # Skip this micro-batch

            # --- Optional: Transformer2 Adaptation ---
            if self.config.use_transformer2_adaptation and hasattr(self.model, 'adapt_comp') and self.model.adapt_comp is not None:
                try:
                    with torch.no_grad(): # Adaptation logic should not track gradients itself
                        # Task identification should happen on the input to the main model
                        input_for_task_id = batch['input_ids']
                        if self.model.blt_comp is not None:
                             # TODO: How to get a good task embedding from raw bytes for BLT?
                             # This is a research question. For now, skip or use a placeholder.
                             # For a placeholder, one might use a fixed task ID or try to embed a prefix.
                             logger.debug("Transformer2 adaptation with BLT: Task ID from raw bytes needs specific design. Skipping adaptation for now in BLT mode during training.")
                        else:
                             # Assume input_ids are token IDs for standard tokenizer
                             input_embeds = self.model.token_embedding(input_for_task_id)
                             task_weights = self.model.adapt_comp(input_embeds) # Get task weights [B, num_tasks]
                             # Adapt model weights based on the average task weights for the batch
                             self.model.adapt_weights(task_weights.mean(dim=0, keepdim=True))
                except RuntimeError as e:
                     logger.error(f"Step {self.global_step}: Error during weight adaptation: {e}. Continuing without adaptation for this step.", exc_info=True)


            # Forward pass
            with autocast_context:
                try:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        token_type_ids=batch.get('token_type_ids') if self.config.use_mvot_processor else None
                        # Pass other necessary inputs if your model expects them
                    )
                    loss = self.model.calculate_loss(
                        outputs,
                        labels=batch.get('labels'),
                        image_targets=batch.get('image_targets') # For MVoT
                    )
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                         logger.warning(f"Step {self.global_step}: Invalid loss ({loss}). Skipping backward for this micro-batch.")
                         if self.optimizer and is_optimizer_step: self.optimizer.zero_grad(set_to_none=True) # Clear grads if optimizer step was due
                         continue
                    loss = loss / accumulation_steps # Normalize loss for accumulation
                except RuntimeError as e:
                     logger.error(f"Step {self.global_step}: Error during forward/loss calculation: {e}", exc_info=True)
                     if self.optimizer and is_optimizer_step: self.optimizer.zero_grad(set_to_none=True)
                     continue # Skip this micro-batch

            # Backward pass
            try:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                 logger.error(f"Step {self.global_step}: Error during backward pass: {e}", exc_info=True)
                 if self.optimizer and is_optimizer_step: self.optimizer.zero_grad(set_to_none=True)
                 continue # Skip this micro-batch

            log_loss_accum += loss.item() * accumulation_steps # Accumulate un-normalized loss for logging
            steps_processed_in_log_period += 1 # Count micro-batches processed

            # Optimizer step (if accumulation is done)
            if is_optimizer_step:
                try:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), train_config.max_grad_norm
                    )
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.optimizer: self.optimizer.step() # Optimizer step
                    
                    if self.scheduler: self.scheduler.step() # Scheduler step after optimizer
                    if self.optimizer: self.optimizer.zero_grad(set_to_none=True) # Clear gradients

                    self.global_step += 1

                    # Logging
                    if self.global_step % log_steps == 0:
                        avg_loss = log_loss_accum / steps_processed_in_log_period if steps_processed_in_log_period > 0 else 0.0
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else train_config.learning_rate
                        elapsed_time = time.time() - log_period_start_time
                        # steps_per_sec is global steps per second
                        steps_per_sec = (log_steps) / elapsed_time if elapsed_time > 0 else 0

                        logger.info(
                            f"Step: {self.global_step}/{max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.3e} | "
                            f"Steps/sec: {steps_per_sec:.2f}"
                        )
                        log_loss_accum = 0.0
                        steps_processed_in_log_period = 0
                        log_period_start_time = time.time()

                    # Evaluation
                    if self.global_step % eval_steps == 0 and self.eval_dataloader is not None:
                        eval_metrics = self.evaluate()
                        eval_loss_per_token = eval_metrics.get('eval_loss_per_token', float('inf'))
                        if eval_loss_per_token < self.best_eval_loss:
                            logger.info(f"New best eval loss (per token): {eval_loss_per_token:.4f} (previous: {self.best_eval_loss:.4f}). Saving 'best' checkpoint.")
                            self.best_eval_loss = eval_loss_per_token
                            self.save_checkpoint(step="best")
                        self.model.train() # Set model back to train mode

                    # Checkpointing
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(step=self.global_step)
                        self.save_checkpoint(step="latest") # Always save latest

                except RuntimeError as e:
                     logger.error(f"Step {self.global_step}: Error during optimizer step/logging/eval/saving: {e}", exc_info=True)
                     if self.optimizer: self.optimizer.zero_grad(set_to_none=True) # Ensure grads are cleared

        logger.info(f"Maximum steps ({max_steps}) reached. Training finished.")
        self.save_checkpoint(step="final") # Save final checkpoint

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided, skipping evaluation.")
            return {}

        self.model.eval() # Set model to evaluation mode
        sum_eval_loss_weighted_by_tokens = 0.0
        total_tokens_evaluated = 0 # Count of non-padding tokens
        
        # Get eval_max_batches from config
        eval_max_batches_config = getattr(self.config.training, 'eval_max_batches', None)
        effective_eval_max_batches: Optional[int] = None
        if isinstance(eval_max_batches_config, int) and eval_max_batches_config > 0:
            effective_eval_max_batches = eval_max_batches_config
        elif eval_max_batches_config is not None: # If it's set but not valid
             logger.warning(f"config.training.eval_max_batches is '{eval_max_batches_config}', expected positive integer or null. Full evaluation will be run.")


        logger.info(f"***** Running Evaluation *****")
        if self.eval_dataset: logger.info(f"  Num examples (estimated for IterableDataset) = {len(self.eval_dataset)}")
        
        # Determine actual batch size used by eval_dataloader
        eval_dl_batch_size = self.eval_dataloader.batch_size if self.eval_dataloader.batch_size is not None else self.config.training.batch_size
        logger.info(f"  Batch size = {eval_dl_batch_size}")
        if effective_eval_max_batches is not None:
            logger.info(f"  Evaluating on max {effective_eval_max_batches} batches.")


        autocast_context = autocast(self.device.type, dtype=self.compute_dtype) if self.use_mixed_precision else nullcontext()
        eval_start_time = time.time()
        batches_processed = 0

        with torch.no_grad(): # Ensure no gradients are computed
            for i, batch in enumerate(self.eval_dataloader):
                if effective_eval_max_batches is not None and i >= effective_eval_max_batches:
                    logger.info(f"Reached eval_max_batches ({effective_eval_max_batches}). Stopping evaluation early.")
                    break
                
                batches_processed += 1

                try:
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                except RuntimeError as e:
                     logger.error(f"Eval: Error moving batch {i} to device: {e}. Skipping batch.", exc_info=True)
                     continue

                with autocast_context:
                     try:
                         outputs = self.model(
                              input_ids=batch['input_ids'],
                              attention_mask=batch.get('attention_mask'),
                              token_type_ids=batch.get('token_type_ids') if self.config.use_mvot_processor else None
                         )
                         # Calculate loss using the model's method
                         loss = self.model.calculate_loss(
                             outputs,
                             labels=batch.get('labels'),
                             image_targets=batch.get('image_targets')
                         )

                         if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                              # For per-token loss, we need to count non-padding tokens
                              # The loss from model.calculate_loss should already be per-token (mean over non-ignored tokens)
                              # If it's total batch loss, we need to divide by num_valid_tokens_in_batch.
                              # Assuming model.calculate_loss returns mean loss over valid tokens.
                              
                              # Count valid (non-padding) tokens in the current batch's labels
                              # This is crucial for accurate perplexity if loss is sum or if batches have variable padding.
                              # If loss is already a mean over non-ignored tokens, this is for weighting.
                              labels_for_count = batch.get('labels')
                              num_valid_tokens_in_batch = 0
                              if labels_for_count is not None:
                                   # Shift labels to match logits for counting
                                   shift_labels = labels_for_count[..., 1:].contiguous()
                                   valid_mask = (shift_labels != -100) # -100 is ignore_index
                                   num_valid_tokens_in_batch = valid_mask.sum().item()

                              if num_valid_tokens_in_batch > 0:
                                   sum_eval_loss_weighted_by_tokens += loss.item() * num_valid_tokens_in_batch
                                   total_tokens_evaluated += num_valid_tokens_in_batch
                              elif loss.item() != 0: # If loss is non-zero but no valid tokens, log warning
                                   logger.warning(f"Eval: Batch {i} has non-zero loss ({loss.item()}) but 0 valid tokens for averaging. Loss for this batch not included in average.")
                         elif loss is not None: # Loss is NaN or Inf
                              logger.warning(f"Eval: Invalid loss ({loss}) encountered for batch {i}.")
                     except RuntimeError as e:
                          logger.error(f"Eval: Error during forward/loss calculation for batch {i}: {e}", exc_info=True)

        eval_duration = time.time() - eval_start_time
        
        avg_loss_per_token = sum_eval_loss_weighted_by_tokens / total_tokens_evaluated if total_tokens_evaluated > 0 else float('inf')
        perplexity = math.exp(avg_loss_per_token) if total_tokens_evaluated > 0 and avg_loss_per_token != float('inf') and avg_loss_per_token >= 0 else float('inf')
        
        # Calculate average loss per sequence (or per batch if sequences vary)
        num_eval_sequences_processed = batches_processed * eval_dl_batch_size # Total sequences seen
        avg_loss_per_sequence = sum_eval_loss_weighted_by_tokens / num_eval_sequences_processed if num_eval_sequences_processed > 0 and total_tokens_evaluated > 0 else float('inf')


        metrics = {
            "eval_loss_per_token": avg_loss_per_token,
            "eval_loss_per_sequence": avg_loss_per_sequence, # This might be less meaningful than per-token
            "perplexity": perplexity,
            "eval_batches_processed": batches_processed # Log how many batches were actually processed
        }
        logger.info(f"Evaluation finished in {eval_duration:.2f}s. Batches processed: {batches_processed}. Eval Loss (per token): {avg_loss_per_token:.4f}, Perplexity: {perplexity:.2f}")
        self.model.train() # Set model back to training mode
        return metrics

    def save_checkpoint(self, step: Union[int, str]):
        """Saves model, optimizer, scheduler, and scaler states."""
        if self.optimizer is None or self.scheduler is None:
             logger.warning("Cannot save full checkpoint, optimizer/scheduler not properly initialized. Saving model state only.")
             model_state_path = os.path.join(self.checkpoint_dir, f"checkpoint-model_only-{step}.pt")
             try:
                  # Save config as dict for easier loading without full class definition if needed elsewhere
                  torch.save({'model_state_dict': self.model.state_dict(), 'config': self.config.to_dict()}, model_state_path)
                  logger.info(f"Saved model-only state to {model_state_path} at global_step {self.global_step}")
             except (OSError, IOError, pickle.PicklingError) as e: # More specific exceptions
                  logger.error(f"Failed to save model-only state to {model_state_path}: {e}", exc_info=True)
             return

        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        state = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.to_dict() # Save config for reproducibility
        }
        try:
            torch.save(state, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path} at step {self.global_step}")
            # If this is the best model, also save it to a fixed name
            if step == "best":
                 best_path = os.path.join(self.output_dir, "model_best.pt")
                 shutil.copyfile(checkpoint_path, best_path)
                 logger.info(f"Copied best checkpoint to {best_path}")
        except (OSError, IOError, pickle.PicklingError) as e: # More specific exceptions
            logger.error(f"Failed to save checkpoint '{checkpoint_name}' at step {self.global_step}: {e}", exc_info=True)

    def load_checkpoint(self, path: str):
        """Loads model, optimizer, scheduler, and scaler states from a checkpoint."""
        if not os.path.exists(path):
            logger.error(f"Checkpoint path not found: {path}. Cannot resume.")
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        logger.info(f"Attempting to load checkpoint from: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            # Use strict=False to allow loading partial models or models with different heads
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys: logger.warning(f"Checkpoint loading: Missing keys in model state_dict: {missing_keys}")
            if unexpected_keys: logger.warning(f"Checkpoint loading: Unexpected keys in model state_dict: {unexpected_keys}")

            # Load optimizer state if available and optimizer exists
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                 try: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); logger.info("Optimizer state loaded.")
                 except Exception as e: logger.warning(f"Could not load optimizer state: {e}. Optimizer state will be reset.", exc_info=True)
            elif self.optimizer: logger.warning("Optimizer state not found in checkpoint. Optimizer state will be reset.")

            # Load scheduler state if available and scheduler exists
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                 try: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']); logger.info("Scheduler state loaded.")
                 except Exception as e: logger.warning(f"Could not load scheduler state: {e}. Scheduler state will be reset.", exc_info=True)
            elif self.scheduler: logger.warning("Scheduler state not found in checkpoint. Scheduler state will be reset.")

            # Load scaler state if available and scaler exists
            if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                 try: self.scaler.load_state_dict(checkpoint['scaler_state_dict']); logger.info("AMP GradScaler state loaded.")
                 except Exception as e: logger.warning(f"Could not load GradScaler state: {e}. Scaler state will be reset.", exc_info=True)
            elif self.scaler : logger.warning("Scaler state not found in checkpoint or was None. Scaler state will be reset.")

            self.global_step = checkpoint.get('step', 0)
            self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
            
            # Optionally, could verify or use the config from checkpoint if needed,
            # but current design assumes config is passed at Trainer init.
            if 'config' in checkpoint: logger.info("Checkpoint includes configuration (for reference).")
            
            logger.info(f"Successfully resumed training from checkpoint: {path} at step {self.global_step}")

        except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
            logger.error(f"Checkpoint file not found at {path}. Training will start from scratch.")
            self.global_step = 0 # Reset state
            self.best_eval_loss = float('inf')
        except (pickle.UnpicklingError, KeyError, TypeError, RuntimeError) as e: # More specific exceptions
            logger.error(f"Failed to load checkpoint from {path} due to error: {e}. Training will start from scratch.", exc_info=True)
            self.global_step = 0 # Reset state
            self.best_eval_loss = float('inf')
# --- END OF FILE src/training/trainer.py ---