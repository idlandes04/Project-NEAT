# --- START OF FILE src/utils/config.py ---

"""
Configuration system for Project NEAT using dataclasses.

Defines hierarchical configuration structures for the model, components,
data processing, training, and hardware settings. Ensures dependencies
are resolved and provides loading/saving capabilities.
"""

import dataclasses
from typing import Dict, List, Optional, Union, Any, Type
import json
import yaml # Requires PyYAML installation: pip install pyyaml
import logging
import os
import math

logger = logging.getLogger(__name__)

# Helper function to recursively initialize dataclasses from dicts
def _init_from_dict(cls: Type[Any], data: Dict[str, Any]) -> Any:
    """
    Initializes a dataclass instance from a dictionary.
    Recursively initializes nested dataclasses.
    Ignores extra keys in the dictionary.
    Uses default values for missing keys in the dictionary.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass type.")

    field_defaults = {f.name: f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
                      for f in dataclasses.fields(cls)}
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    init_args = {}

    for name, default_value in field_defaults.items():
        field_type = field_types[name]
        # Check if the field type itself is a dataclass for nested initialization
        origin_type = getattr(field_type, "__origin__", None)
        is_nested_dataclass = False
        nested_type = None

        if dataclasses.is_dataclass(field_type):
            is_nested_dataclass = True
            nested_type = field_type
        # Handle Optional[Dataclass] or Union[Dataclass, None]
        elif origin_type is Union:
             union_args = getattr(field_type, "__args__", ())
             for arg in union_args:
                  if dataclasses.is_dataclass(arg):
                       is_nested_dataclass = True
                       nested_type = arg
                       break

        if name in data:
            value = data[name]
            if is_nested_dataclass and isinstance(value, dict) and nested_type is not None:
                # Recursively initialize nested dataclass
                init_args[name] = _init_from_dict(nested_type, value)
            elif isinstance(value, dict) and not is_nested_dataclass and default_value != dataclasses.MISSING:
                 # If field is not a dataclass but data provides a dict, maybe warn or use default?
                 logger.warning(f"Received dict for non-dataclass field '{name}' of type {field_type}. Using default value: {default_value}")
                 init_args[name] = default_value
            elif type(value) == field_type or isinstance(value, field_type):
                 # Basic type match or instance match
                 init_args[name] = value
            else:
                 # Attempt type coercion or use default if coercion fails/inappropriate
                 try:
                     # Handle Optional[type] case where value might be None
                     if origin_type is Union and type(None) in getattr(field_type, "__args__", ()) and value is None:
                          init_args[name] = None
                     # Basic type coercion
                     elif not isinstance(value, field_type):
                          init_args[name] = field_type(value)
                     else: # Should not happen based on earlier check, but fallback
                          init_args[name] = value
                 except (TypeError, ValueError):
                      logger.warning(f"Type mismatch for field '{name}'. Expected {field_type}, got {type(value)}. Using default value: {default_value}")
                      init_args[name] = default_value

        else:
            # Key missing in data dict, use default value
            init_args[name] = default_value

    # Filter out args that are MISSING (should only happen if no default was defined)
    final_args = {k: v for k, v in init_args.items() if v is not dataclasses.MISSING}

    try:
        return cls(**final_args)
    except TypeError as e:
         logger.error(f"Error initializing {cls.__name__} with args {final_args}: {e}")
         raise


@dataclasses.dataclass
class TitansConfig:
    """Configuration for the Titans memory system."""
    use_window_attention: bool = True
    use_surprise_based: bool = True
    use_persistent: bool = True
    window_size: int = 512
    memory_size: int = 4096 # Increased default
    surprise_threshold: float = 0.5
    # max_memory_updates_per_step: int = 10 # Maybe handle in Trainer logic?
    num_persistent_vectors: int = 64
    persistent_init_scale: float = 0.02
    base_decay_rate: float = 0.99 # Decay factor per step/call
    importance_half_life: int = 1000 # Steps for importance to halve via decay (alternative way to think about rate)
    memory_prune_threshold: float = 0.01 # Importance below which inactive entries might be pruned
    surprise_method: str = "gradient_norm" # Options: 'gradient_norm', 'prediction_error', 'hidden_norm'

@dataclasses.dataclass
class Transformer2Config:
    """Configuration for the Transformer² adaptation."""
    # use_task_dispatcher: bool = True # Controlled by use_transformer2_adaptation
    # use_svd_adaptation: bool = True # Controlled by use_transformer2_adaptation
    num_tasks: int = 8 # Number of expert SV vectors
    task_embedding_dim: int = 64 # Dimension for internal task representation/classifier
    num_singular_values: int = 128 # Max number of SVs to adapt (k)
    expert_init_scale: float = 0.01 # Scale for initializing expert SV offsets
    adapt_attention: bool = True # Adapt attention matrices (q,k,v,o)
    adapt_ffn: bool = True # Adapt FFN matrices (fc1, fc2)
    adapt_embeddings: bool = False # Adapt input/position embeddings
    adapt_lm_head: bool = False # Adapt output projection
    layer_specific: bool = False # Use shared or layer-specific expert SVs
    use_randomized_svd: bool = True # Use randomized SVD if faster
    svd_precision: str = "fixed" # 'fixed' (use num_singular_values), 'adaptive' (future: estimate k)
    svd_n_oversamples: int = 10
    svd_n_iter: int = 5
    enable_svd_caching: bool = True
    svd_cache_dir: str = ".svd_cache"

@dataclasses.dataclass
class MVoTConfig:
    """Configuration for the MVoT token processor."""
    # is_multimodal: bool = True # Controlled by use_mvot_processor
    codebook_size: int = 8192 # Size of the visual codebook
    embedding_dim: int = 768 # Dimension of codebook embeddings (adjust to match pretrained)
    discrepancy_loss_weight: float = 0.1 # Weight for the token discrepancy loss
    codebook_model_type: str = "vqvae" # Type hint for loading ('vqvae', 'vqgan', 'dalle')
    codebook_path: Optional[str] = None # Path to pretrained codebook state_dict
    use_pretrained_codebook: bool = False # Whether to load pretrained weights
    # decision_strategy: str = "heuristic" # Future: 'neural', 'hybrid'

@dataclasses.dataclass
class ByteLMConfig:
    """Configuration specific to the SmallByteLM used by BLT."""
    hidden_size: int = 256 # Smaller hidden size for the byte LM
    num_layers: int = 4 # More layers might capture byte patterns better
    num_attention_heads: int = 4
    intermediate_size: int = 1024 # 4 * hidden_size
    byte_lm_dropout: float = 0.1
    byte_lm_model_type: str = "transformer" # 'transformer' or 'gru'
    # byte_lm_max_position: int = 512 # Max sequence length for the small LM (maybe link to main block_size?)

@dataclasses.dataclass
class BLTConfig:
    """Configuration for the BLT byte processor."""
    entropy_threshold: float = 0.5 # Threshold for creating new patches
    min_patch_size: int = 8 # Minimum bytes per patch
    max_patch_size: int = 128 # Maximum bytes per patch
    num_local_layers: int = 1 # Layers in LocalEncoder/Decoder
    num_latent_layers: int = 2 # Layers in LatentTransformer
    # Byte LM config is nested
    byte_lm: ByteLMConfig = dataclasses.field(default_factory=ByteLMConfig)
    blt_checkpoint_path: Optional[str] = None # Path to pretrained byte LM if needed

@dataclasses.dataclass
class DataConfig:
    """Configuration for the data pipeline."""
    train_data_dir: Optional[str] = None # Directory containing training files
    eval_data_dir: Optional[str] = None # Directory containing evaluation files
    train_file_pattern: str = "*.txt" # Pattern to match training files
    eval_file_pattern: str = "*.txt" # Pattern to match eval files
    block_size: int = 4096 # Sequence length / chunk size for datasets

@dataclasses.dataclass
class HardwareConfig:
    """Configuration for hardware-specific optimizations."""
    mixed_precision: bool = True # Enable AMP (Automatic Mixed Precision)
    # compute_dtype: str = 'bf16' # Deduced automatically based on hardware
    gradient_checkpointing: bool = False # Enable gradient checkpointing
    force_cpu: bool = False # Force CPU usage
    num_workers: int = 0 # Dataloader workers
    use_flash_attention: bool = True # Use FlashAttention if available

@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer
    learning_rate: float = 3e-4 # Common LR for ~1B models
    weight_decay: float = 0.1 # Weight decay (AdamW)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95 # Often 0.95 for large models
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0 # Gradient clipping norm

    # Schedule
    max_steps: int = 200000 # Example total steps (adjust based on dataset size)
    warmup_steps: int = 2000 # Number of linear warmup steps (alternative to ratio)
    # lr_scheduler_type: str = "cosine_with_warmup" # Handled by LambdaLR in Trainer for now

    # Batching
    batch_size: int = 8 # Per-device batch size (adjust based on GPU memory)
    gradient_accumulation_steps: int = 4 # Accumulate gradients over N steps

    # Logging & Checkpointing
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 1000
    output_dir: str = "./output/project_neat_model"
    resume_from: Optional[str] = None # Path to checkpoint to resume


@dataclasses.dataclass
class ModelConfig:
    """Main configuration aggregating all settings."""
    # Core Model Parameters (~500M target)
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16 # hidden_size / num_heads = 128
    intermediate_size: int = 8192 # 4 * hidden_size
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 4096
    vocab_size: int = 32000 # Example size, adjust to match tokenizer

    # Component Activation Flags
    use_blt_processor: bool = False # Default to standard tokenizer
    use_titans_memory: bool = True
    use_transformer2_adaptation: bool = True
    use_mvot_processor: bool = False # Default to text-only

    # Nested Component Configurations
    blt: BLTConfig = dataclasses.field(default_factory=BLTConfig)
    titans: TitansConfig = dataclasses.field(default_factory=TitansConfig)
    transformer2: Transformer2Config = dataclasses.field(default_factory=Transformer2Config)
    mvot: MVoTConfig = dataclasses.field(default_factory=MVoTConfig)

    # Data, Training, Hardware Configs
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    hardware: HardwareConfig = dataclasses.field(default_factory=HardwareConfig)

    # --- Methods for handling config ---

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls: Type['ModelConfig'], config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Creates a ModelConfig instance from a dictionary, handling nested structures."""
        instance = _init_from_dict(cls, config_dict)
        # Resolve dependencies after loading
        instance = resolve_config(instance)
        return instance

    def save(self, path: str):
        """Saves the configuration to a JSON or YAML file."""
        config_dict = self.to_dict()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith(".yaml") or path.endswith(".yml"):
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                logger.info(f"Configuration saved to YAML: {path}")
            else: # Default to JSON
                if not path.endswith(".json"):
                     # Ensure .json extension if not yaml
                     base, ext = os.path.splitext(path)
                     path = base + ".json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Configuration saved to JSON: {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")

# --- Helper Functions ---

def get_default_config() -> ModelConfig:
    """Returns a ModelConfig instance with default values, resolved."""
    return resolve_config(ModelConfig())

def load_config(path: str) -> Optional[ModelConfig]:
    """Loads configuration from a JSON or YAML file."""
    if not os.path.exists(path):
        logger.error(f"Configuration file not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            elif path.endswith(".json"):
                config_dict = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {path}. Use .json or .yaml")
                return None

        if config_dict is None: # Handle empty file case
             logger.error(f"Configuration file is empty or invalid: {path}")
             return None

        config = ModelConfig.from_dict(config_dict)
        logger.info(f"Configuration loaded from: {path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}", exc_info=True)
        return None

def resolve_config(config: ModelConfig) -> ModelConfig:
    """
    Resolves potential inconsistencies or dependencies in the configuration.
    Modifies the config object in-place and returns it.
    """
    logger.debug("Resolving configuration dependencies...")

    # --- Basic Model Checks ---
    if config.hidden_size % config.num_attention_heads != 0:
        logger.error(f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads}).")
        # Attempt to fix or raise error? Raising is safer.
        raise ValueError("Incompatible hidden_size and num_attention_heads.")

    if config.data.block_size > config.max_position_embeddings:
        logger.warning(f"data.block_size ({config.data.block_size}) > max_position_embeddings ({config.max_position_embeddings}). Adjusting block_size.")
        config.data.block_size = config.max_position_embeddings

    # --- Component-Specific Checks ---
    # Titans
    if config.use_titans_memory:
        if config.titans.window_size > config.max_position_embeddings:
            logger.warning(f"Titans window_size ({config.titans.window_size}) > max_position_embeddings ({config.max_position_embeddings}). Adjusting window_size.")
            config.titans.window_size = config.max_position_embeddings

    # Transformer2
    if config.use_transformer2_adaptation:
        # Ensure k <= relevant dimension
        k = config.transformer2.num_singular_values
        if config.transformer2.adapt_attention or config.transformer2.adapt_lm_head or config.transformer2.adapt_embeddings:
             max_k_attn = min(config.hidden_size, config.hidden_size) # M=N for self-attn, maybe different for emb/head
             if k > max_k_attn:
                  logger.warning(f"T2 k ({k}) > hidden_size ({config.hidden_size}). Clamping k.")
                  config.transformer2.num_singular_values = max_k_attn
        if config.transformer2.adapt_ffn:
             max_k_ffn1 = min(config.hidden_size, config.intermediate_size)
             max_k_ffn2 = min(config.intermediate_size, config.hidden_size)
             if k > min(max_k_ffn1, max_k_ffn2):
                  logger.warning(f"T2 k ({k}) > FFN dimensions. Clamping k.")
                  config.transformer2.num_singular_values = min(k, max_k_ffn1, max_k_ffn2)

    # MVoT
    if config.use_mvot_processor:
        if config.mvot.embedding_dim != config.hidden_size:
            logger.info("MVoT embedding_dim differs from hidden_size. Ensure projection layers are used.")
        if not config.mvot.use_pretrained_codebook:
             logger.warning("MVoT enabled but use_pretrained_codebook is False. VisualCodebook will use random init.")
        elif not config.mvot.codebook_path:
             logger.error("MVoT use_pretrained_codebook is True, but codebook_path is not set.")
             raise ValueError("MVoT codebook_path must be specified when use_pretrained_codebook=True")

    # BLT
    if config.use_blt_processor:
        if config.vocab_size != 260: # SimpleByteTokenizer vocab size
             logger.warning(f"BLT processor is enabled, but vocab_size ({config.vocab_size}) is not 260. Ensure tokenizer/model handles this.")
             # We don't force vocab_size=260 here, as the main model might still need a text vocab if BLT is disabled later.
        if config.blt.max_patch_size > config.data.block_size:
             logger.warning(f"BLT max_patch_size ({config.blt.max_patch_size}) > data.block_size ({config.data.block_size}). This might lead to issues.")

    # --- Hardware Checks ---
    try:
        import torch
        can_use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        can_use_cuda = torch.cuda.is_available()

        if config.hardware.force_cpu:
            effective_device = 'cpu'
        elif can_use_cuda:
            effective_device = 'cuda'
        elif can_use_mps:
            effective_device = 'mps'
        else:
            effective_device = 'cpu'

        # Disable mixed precision if not on CUDA/MPS or forced CPU
        if config.hardware.mixed_precision and effective_device == 'cpu':
            logger.info("Disabling mixed precision because device is CPU.")
            config.hardware.mixed_precision = False

        # Disable FlashAttention if not on CUDA or explicitly disabled
        if config.hardware.use_flash_attention:
             if effective_device != 'cuda':
                  logger.info("Disabling FlashAttention because device is not CUDA.")
                  config.hardware.use_flash_attention = False
             elif not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                  logger.info("Disabling FlashAttention because it's not available in this PyTorch version.")
                  config.hardware.use_flash_attention = False

    except ImportError:
        logger.warning("PyTorch not found during config resolution. Hardware checks skipped.")
        config.hardware.mixed_precision = False
        config.hardware.use_flash_attention = False

    logger.debug("Configuration resolution complete.")
    return config

# --- END OF FILE src/utils/config.py ---