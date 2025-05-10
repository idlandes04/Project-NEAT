# --- START OF CORRECTED src/training/trainer.py (v2) ---
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
# Corrected AMP imports
from torch.cuda.amp import GradScaler as CudaGradScaler # For explicit cuda scaler
# MpsGradScaler will be imported conditionally
from torch.amp import autocast # General autocast
import os
import time
import math
import logging
import shutil
from typing import Dict, Any, Optional, Tuple, Callable, Union
from contextlib import nullcontext # Use nullcontext for conditional autocast

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
        model: nn.Module,
        config: Any,
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
             logger.info("Basic logging configured by Trainer.")

        self._setup_device()
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        eval_batch_size = getattr(self.config.training, "eval_batch_size", self.config.training.batch_size)

        self.train_dataloader = self._create_dataloader(self.train_dataset, shuffle=True, batch_size=self.config.training.batch_size)
        self.eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=False, batch_size=eval_batch_size) if self.eval_dataset else None


        self.optimizer = None
        self.scheduler = None
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
            self.load_checkpoint(resume_path)
        else:
             logger.info("No checkpoint provided, starting training from scratch.")

        if self.config.use_transformer2_adaptation and hasattr(self.model, 'adapt_comp') and self.model.adapt_comp is not None:
             needs_precompute = True
             if hasattr(self.model.adapt_comp, 'svd_cache') and self.model.adapt_comp.svd_cache:
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
                  _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps")
                  self.device = torch.device("mps")
             except Exception:
                  logger.warning("MPS available but functional check failed. Using CPU.")
                  self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Trainer using device: {self.device}")

    def _setup_mixed_precision(self):
        hw_config = self.config.hardware
        self.use_mixed_precision = False
        self.compute_dtype = torch.float32
        self.scaler: Optional[Union[CudaGradScaler, Any]] = None # Use Any for MpsGradScaler initially

        if hw_config.mixed_precision:
            if self.device.type == 'cuda':
                cap_major, _ = torch.cuda.get_device_capability()
                if cap_major >= 8:
                    self.compute_dtype = torch.bfloat16
                    self.use_mixed_precision = True
                    logger.info("Mixed precision enabled (CUDA, bfloat16).")
                elif cap_major >=7:
                    self.compute_dtype = torch.float16
                    self.use_mixed_precision = True
                    logger.info("Mixed precision enabled (CUDA, float16).")
                else:
                    logger.warning("CUDA device capability < 7.0. Mixed precision (float16/bfloat16) not fully supported or optimal. Disabling.")
            elif self.device.type == 'mps':
                 self.compute_dtype = torch.float16 # MPS typically uses float16 for AMP
                 self.use_mixed_precision = True
                 logger.info("Mixed precision enabled (MPS, float16).")
            else:
                 logger.warning("Mixed precision requested but device is CPU. Disabled.")
        else:
            logger.info("Mixed precision disabled by configuration.")

        if self.use_mixed_precision:
            if self.device.type == 'cuda':
                self.scaler = CudaGradScaler(enabled=True)
            elif self.device.type == 'mps':
                if MpsGradScaler is not None: # Check if import was successful
                    self.scaler = MpsGradScaler(enabled=True)
                    logger.info("Using MPS GradScaler.")
                else:
                    logger.warning("torch.mps.GradScaler not available. Disabling mixed precision for MPS.")
                    self.use_mixed_precision = False # Disable AMP if scaler is missing
                    self.scaler = None
        else:
            self.scaler = None


    def _create_dataloader(self, dataset: Optional[Dataset], shuffle: bool, batch_size: int) -> Optional[DataLoader]:
        if dataset is None:
            return None
        if self.collate_fn_func is None:
             raise ValueError("Cannot create DataLoader without a collate_fn_func.")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.hardware.num_workers,
            collate_fn=self.collate_fn_func,
            pin_memory=(self.device.type == 'cuda'),
            prefetch_factor=2 if self.config.hardware.num_workers > 0 else None,
            persistent_workers=True if self.config.hardware.num_workers > 0 else False
        )

    def _setup_optimizer_scheduler(self):
        train_config = self.config.training
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

        if warmup_steps >= max_steps:
             logger.warning(f"warmup_steps ({warmup_steps}) >= max_steps ({max_steps}). Using linear warmup only.")
             def lr_lambda_warmup_only(current_step: int):
                  return float(current_step) / float(max(1, warmup_steps))
             lr_lambda = lr_lambda_warmup_only
        else:
             def lr_lambda_cosine(current_step: int):
                  if current_step < warmup_steps:
                       return float(current_step) / float(max(1, warmup_steps))
                  progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
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
        steps_processed_in_log_period = 0
        log_period_start_time = time.time()

        autocast_context = autocast(self.device.type, dtype=self.compute_dtype) if self.use_mixed_precision else nullcontext()

        while self.global_step < max_steps:
            for step_in_epoch, batch in enumerate(self.train_dataloader):
                if self.global_step >= max_steps: break
                
                is_optimizer_step = (step_in_epoch + 1) % accumulation_steps == 0 or (step_in_epoch + 1 == len(self.train_dataloader))

                try:
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                except Exception as e:
                     logger.error(f"Step {self.global_step}: Error moving batch to device: {e}. Skipping batch.", exc_info=True)
                     continue

                if self.config.use_transformer2_adaptation and hasattr(self.model, 'adapt_comp') and self.model.adapt_comp is not None:
                    try:
                        with torch.no_grad():
                            input_embeds = self.model.token_embedding(batch['input_ids'])
                            task_weights = self.model.adapt_comp(input_embeds)
                        self.model.adapt_weights(task_weights.mean(dim=0, keepdim=True))
                    except Exception as e:
                         logger.error(f"Step {self.global_step}: Error during weight adaptation: {e}. Continuing without adaptation.", exc_info=True)

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
                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                             logger.warning(f"Step {self.global_step}: Invalid loss ({loss}). Skipping backward/step.")
                             if self.optimizer: self.optimizer.zero_grad()
                             continue
                        loss = loss / accumulation_steps
                    except Exception as e:
                         logger.error(f"Step {self.global_step}: Error during forward/loss calculation: {e}", exc_info=True)
                         if self.optimizer: self.optimizer.zero_grad()
                         continue

                try:
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except Exception as e:
                     logger.error(f"Step {self.global_step}: Error during backward pass: {e}", exc_info=True)
                     if self.optimizer: self.optimizer.zero_grad()
                     continue

                log_loss_accum += loss.item() * accumulation_steps
                steps_processed_in_log_period += 1

                if is_optimizer_step:
                    try:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), train_config.max_grad_norm
                        )
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            if self.optimizer: self.optimizer.step() # Check if optimizer exists
                        
                        if self.scheduler: self.scheduler.step()
                        if self.optimizer: self.optimizer.zero_grad()

                        self.global_step += 1

                        if self.global_step % log_steps == 0:
                            avg_loss = log_loss_accum / steps_processed_in_log_period if steps_processed_in_log_period > 0 else 0.0
                            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else train_config.learning_rate
                            elapsed_time = time.time() - log_period_start_time
                            steps_per_sec = steps_processed_in_log_period / elapsed_time if elapsed_time > 0 else 0

                            logger.info(
                                f"Step: {self.global_step}/{max_steps} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {current_lr:.3e} | "
                                f"Steps/sec: {steps_per_sec:.2f}"
                            )
                            log_loss_accum = 0.0
                            steps_processed_in_log_period = 0
                            log_period_start_time = time.time()

                        if self.global_step % eval_steps == 0 and self.eval_dataloader is not None:
                            eval_metrics = self.evaluate()
                            eval_loss_per_token = eval_metrics.get('eval_loss_per_token', float('inf'))
                            if eval_loss_per_token < self.best_eval_loss:
                                logger.info(f"New best eval loss (per token): {eval_loss_per_token:.4f} (previous: {self.best_eval_loss:.4f}). Saving 'best' checkpoint.")
                                self.best_eval_loss = eval_loss_per_token
                                self.save_checkpoint(step="best")
                            self.model.train()

                        if self.global_step % self.save_steps == 0:
                            self.save_checkpoint(step=self.global_step)
                            self.save_checkpoint(step="latest")

                    except Exception as e:
                         logger.error(f"Step {self.global_step}: Error during optimizer step/logging/eval/saving: {e}", exc_info=True)
                         if self.optimizer: self.optimizer.zero_grad()

        logger.info(f"Maximum steps ({max_steps}) reached. Training finished.")
        self.save_checkpoint(step="final")

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided, skipping evaluation.")
            return {}

        self.model.eval()
        sum_eval_loss_weighted_by_tokens = 0.0
        total_tokens_evaluated = 0

        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(self.eval_dataset)}")
        logger.info(f"  Batch size = {self.eval_dataloader.batch_size}")

        autocast_context = autocast(self.device.type, dtype=self.compute_dtype) if self.use_mixed_precision else nullcontext()
        eval_start_time = time.time()

        with torch.no_grad():
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
                              num_valid_tokens_in_batch = (batch.get('labels') != -100).sum().item() if 'labels' in batch else (batch['input_ids'].size(0) * batch['input_ids'].size(1))
                              if num_valid_tokens_in_batch > 0:
                                   sum_eval_loss_weighted_by_tokens += loss.item() * num_valid_tokens_in_batch
                                   total_tokens_evaluated += num_valid_tokens_in_batch
                         else:
                              logger.warning(f"Eval: Invalid loss ({loss}) encountered.")
                     except Exception as e:
                          logger.error(f"Eval: Error during forward/loss calculation: {e}", exc_info=True)

        eval_duration = time.time() - eval_start_time
        
        avg_loss_per_token = sum_eval_loss_weighted_by_tokens / total_tokens_evaluated if total_tokens_evaluated > 0 else float('inf')
        perplexity = math.exp(avg_loss_per_token) if total_tokens_evaluated > 0 and avg_loss_per_token != float('inf') else float('inf')
        
        num_eval_sequences = len(self.eval_dataset)
        avg_loss_per_sequence = sum_eval_loss_weighted_by_tokens / num_eval_sequences if num_eval_sequences > 0 and total_tokens_evaluated > 0 else float('inf')


        metrics = {
            "eval_loss_per_token": avg_loss_per_token,
            "eval_loss_per_sequence": avg_loss_per_sequence,
            "perplexity": perplexity
        }
        logger.info(f"Evaluation finished in {eval_duration:.2f}s. Eval Loss (per token): {avg_loss_per_token:.4f}, Perplexity: {perplexity:.2f}")
        self.model.train()
        return metrics

    def save_checkpoint(self, step: Union[int, str]):
        if self.optimizer is None or self.scheduler is None:
             logger.warning("Cannot save checkpoint, optimizer/scheduler not properly initialized.")
             if self.model:
                  model_state_path = os.path.join(self.checkpoint_dir, f"checkpoint-model_only-{step}.pt")
                  torch.save({'model_state_dict': self.model.state_dict(), 'config': self.config.to_dict()}, model_state_path)
                  logger.info(f"Saved model-only state to {model_state_path} at step {self.global_step}")
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
            'config': self.config.to_dict()
        }
        try:
            torch.save(state, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path} at step {self.global_step}")
            if step == "best":
                 best_path = os.path.join(self.output_dir, "model_best.pt")
                 shutil.copyfile(checkpoint_path, best_path)
                 logger.info(f"Copied best checkpoint to {best_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint '{checkpoint_name}' at step {self.global_step}: {e}", exc_info=True)

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.error(f"Checkpoint path not found: {path}. Cannot resume.")
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        logger.info(f"Attempting to load checkpoint from: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys: logger.warning(f"Checkpoint loading: Missing keys in model state_dict: {missing_keys}")
            if unexpected_keys: logger.warning(f"Checkpoint loading: Unexpected keys in model state_dict: {unexpected_keys}")

            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                 try: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']); logger.info("Optimizer state loaded.")
                 except Exception as e: logger.warning(f"Could not load optimizer state: {e}. Optimizer state will be reset.")
            elif self.optimizer: logger.warning("Optimizer state not found in checkpoint. Optimizer state will be reset.")

            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                 try: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']); logger.info("Scheduler state loaded.")
                 except Exception as e: logger.warning(f"Could not load scheduler state: {e}. Scheduler state will be reset.")
            elif self.scheduler: logger.warning("Scheduler state not found in checkpoint. Scheduler state will be reset.")

            if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                 try: self.scaler.load_state_dict(checkpoint['scaler_state_dict']); logger.info("AMP GradScaler state loaded.")
                 except Exception as e: logger.warning(f"Could not load GradScaler state: {e}. Scaler state will be reset.")
            elif self.scaler : logger.warning("Scaler state not found in checkpoint or was None. Scaler state will be reset.")

            self.global_step = checkpoint.get('step', 0)
            self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
            if 'config' in checkpoint: logger.info("Checkpoint includes configuration (not automatically applied, for reference only).")
            logger.info(f"Successfully resumed training from checkpoint: {path} at step {self.global_step}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}. Training will start from scratch.", exc_info=True)
            self.global_step = 0
            self.best_eval_loss = float('inf')
# --- END OF CORRECTED src/training/trainer.py (v2) ---