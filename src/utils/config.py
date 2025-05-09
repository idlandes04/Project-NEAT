# --- START OF CORRECTED src/utils/config.py ---

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
        origin_type = getattr(field_type, "__origin__", None)
        is_nested_dataclass = False
        nested_type = None

        if dataclasses.is_dataclass(field_type):
            is_nested_dataclass = True
            nested_type = field_type
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
                init_args[name] = _init_from_dict(nested_type, value)
            elif isinstance(value, dict) and not is_nested_dataclass and default_value != dataclasses.MISSING:
                 logger.warning(f"Received dict for non-dataclass field '{name}' of type {field_type}. Using default value: {default_value}")
                 init_args[name] = default_value
            # Check for exact type match or if value is an instance of field_type
            # This handles cases like int for float, or subclasses.
            elif isinstance(value, field_type) or type(value) == field_type:
                 init_args[name] = value
            # Handle Optional[type] where value might be None
            elif origin_type is Union and type(None) in getattr(field_type, "__args__", ()) and value is None:
                init_args[name] = None
            else:
                 try: # Attempt basic type coercion for simple types
                      if not isinstance(value, field_type): # Check if coercion is needed
                           # Only coerce if field_type is a basic type that supports it
                           if field_type in [int, float, str, bool]:
                                init_args[name] = field_type(value)
                           else: # Cannot coerce complex types, use default
                                raise TypeError("Cannot coerce") # Trigger warning and default
                      else: # Already correct type
                           init_args[name] = value
                 except (TypeError, ValueError):
                      logger.warning(f"Type mismatch or coercion failed for field '{name}'. Expected {field_type}, got {type(value)}. Using default value: {default_value}")
                      init_args[name] = default_value
        else:
            init_args[name] = default_value

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
    memory_size: int = 4096
    surprise_threshold: float = 0.5
    num_persistent_vectors: int = 64
    persistent_init_scale: float = 0.02
    base_decay_rate: float = 0.99
    importance_half_life: int = 1000
    memory_prune_threshold: float = 0.01
    surprise_method: str = "gradient_norm"
    integration_layers: List[int] = dataclasses.field(default_factory=lambda: [0, 12, 23]) # Example for 24 layers

@dataclasses.dataclass
class Transformer2Config:
    """Configuration for the Transformer² adaptation."""
    num_tasks: int = 8
    task_embedding_dim: int = 64
    num_singular_values: int = 128
    expert_init_scale: float = 0.01
    adapt_attention: bool = True
    adapt_ffn: bool = True
    adapt_embeddings: bool = False
    adapt_lm_head: bool = False
    layer_specific: bool = False
    use_randomized_svd: bool = True
    svd_precision: str = "fixed"
    svd_n_oversamples: int = 10
    svd_n_iter: int = 5
    enable_svd_caching: bool = True
    svd_cache_dir: str = ".svd_cache"

@dataclasses.dataclass
class MVoTConfig:
    """Configuration for the MVoT token processor."""
    codebook_size: int = 8192
    embedding_dim: int = 768
    discrepancy_loss_weight: float = 0.1
    codebook_model_type: str = "vqvae"
    codebook_path: Optional[str] = None
    use_pretrained_codebook: bool = False

@dataclasses.dataclass
class ByteLMConfig:
    """Configuration specific to the SmallByteLM used by BLT."""
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    byte_lm_dropout: float = 0.1
    byte_lm_model_type: str = "transformer"
    byte_lm_max_position: int = 512

@dataclasses.dataclass
class BLTConfig:
    """Configuration for the BLT byte processor."""
    entropy_threshold: float = 0.5
    min_patch_size: int = 8
    max_patch_size: int = 128
    num_local_layers: int = 1
    num_latent_layers: int = 2
    byte_lm: ByteLMConfig = dataclasses.field(default_factory=ByteLMConfig)
    blt_checkpoint_path: Optional[str] = None

@dataclasses.dataclass
class DataConfig:
    """Configuration for the data pipeline."""
    train_data_dir: Optional[str] = None
    eval_data_dir: Optional[str] = None
    train_file_pattern: str = "*.txt"
    eval_file_pattern: str = "*.txt"
    block_size: int = 4096

@dataclasses.dataclass
class HardwareConfig:
    """Configuration for hardware-specific optimizations."""
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    force_cpu: bool = False
    num_workers: int = 0
    use_flash_attention: bool = True

@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    max_steps: int = 200000
    warmup_steps: int = 2000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 1000
    output_dir: str = "./output/project_neat_model"
    resume_from: Optional[str] = None

@dataclasses.dataclass
class ModelConfig:
    """Main configuration aggregating all settings."""
    # Core Model Parameters
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 4096
    vocab_size: int = 32000
    layer_norm_eps: float = 1.0e-5 # Added default

    # Tokenizer name (used if BLT is off)
    tokenizer_name: Optional[str] = None # <<<< ----- ADDED THIS FIELD ----- >>>>

    # Component Activation Flags
    use_blt_processor: bool = False
    use_titans_memory: bool = True
    use_transformer2_adaptation: bool = True
    use_mvot_processor: bool = False

    # Nested Component Configurations
    blt: BLTConfig = dataclasses.field(default_factory=BLTConfig)
    titans: TitansConfig = dataclasses.field(default_factory=TitansConfig)
    transformer2: Transformer2Config = dataclasses.field(default_factory=Transformer2Config)
    mvot: MVoTConfig = dataclasses.field(default_factory=MVoTConfig)

    # Data, Training, Hardware Configs
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    hardware: HardwareConfig = dataclasses.field(default_factory=HardwareConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls: Type['ModelConfig'], config_dict: Dict[str, Any]) -> 'ModelConfig':
        instance = _init_from_dict(cls, config_dict)
        instance = resolve_config(instance)
        return instance

    def save(self, path: str):
        config_dict = self.to_dict()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith((".yaml", ".yml")):
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                logger.info(f"Configuration saved to YAML: {path}")
            else:
                if not path.endswith(".json"):
                     base, _ = os.path.splitext(path)
                     path = base + ".json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Configuration saved to JSON: {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")

def get_default_config() -> ModelConfig:
    return resolve_config(ModelConfig())

def load_config(path: str) -> Optional[ModelConfig]:
    if not os.path.exists(path):
        logger.error(f"Configuration file not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith((".yaml", ".yml")):
                config_dict = yaml.safe_load(f)
            elif path.endswith(".json"):
                config_dict = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {path}. Use .json or .yaml")
                return None
        if config_dict is None:
             logger.error(f"Configuration file is empty or invalid: {path}")
             return None
        config = ModelConfig.from_dict(config_dict)
        logger.info(f"Configuration loaded from: {path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}", exc_info=True)
        return None

def resolve_config(config: ModelConfig) -> ModelConfig:
    logger.debug("Resolving configuration dependencies...")
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads}).")
    if config.data.block_size > config.max_position_embeddings:
        logger.warning(f"data.block_size ({config.data.block_size}) > max_position_embeddings ({config.max_position_embeddings}). Adjusting block_size.")
        config.data.block_size = config.max_position_embeddings
    if config.use_titans_memory:
        if config.titans.window_size > config.max_position_embeddings:
            logger.warning(f"Titans window_size ({config.titans.window_size}) > max_position_embeddings. Adjusting window_size.")
            config.titans.window_size = config.max_position_embeddings
        # Update Titans integration layers if num_layers changed
        default_integration_layers = [0, config.num_layers // 2, config.num_layers -1 if config.num_layers > 0 else 0]
        if config.titans.integration_layers == [0, 12, 23] and config.num_layers != 24 : # if default was used and num_layers isn't 24
             config.titans.integration_layers = default_integration_layers
             logger.info(f"Adjusted Titans integration_layers to {config.titans.integration_layers} for num_layers={config.num_layers}")


    if config.use_transformer2_adaptation:
        k = config.transformer2.num_singular_values
        max_k_attn = config.hidden_size
        if k > max_k_attn and (config.transformer2.adapt_attention or config.transformer2.adapt_lm_head or config.transformer2.adapt_embeddings):
            logger.warning(f"T2 k ({k}) > hidden_size ({max_k_attn}). Clamping k.")
            config.transformer2.num_singular_values = max_k_attn
        max_k_ffn = min(config.hidden_size, config.intermediate_size)
        if k > max_k_ffn and config.transformer2.adapt_ffn:
            logger.warning(f"T2 k ({k}) > FFN dimensions ({max_k_ffn}). Clamping k.")
            config.transformer2.num_singular_values = min(k, max_k_ffn)
    if config.use_mvot_processor:
        if config.mvot.embedding_dim != config.hidden_size:
             logger.info(f"MVoT embedding_dim ({config.mvot.embedding_dim}) differs from model hidden_size ({config.hidden_size}). Ensure VisualCodebook projection layers handle this.")
        if config.mvot.use_pretrained_codebook and not config.mvot.codebook_path:
            raise ValueError("MVoT use_pretrained_codebook is True, but codebook_path is not set.")
    if config.use_blt_processor:
        if config.vocab_size != 260:
             logger.warning(f"BLT processor is enabled. Overriding vocab_size from {config.vocab_size} to 260.")
             config.vocab_size = 260
        if config.blt.max_patch_size > config.data.block_size:
             logger.warning(f"BLT max_patch_size ({config.blt.max_patch_size}) > data.block_size ({config.data.block_size}). This might lead to issues if block_size refers to bytes.")
    else: # Not using BLT
         if not config.tokenizer_name:
              logger.warning("BLT is disabled, but 'tokenizer_name' is not set in ModelConfig. Defaulting to 'gpt2'. Training script might override.")
              config.tokenizer_name = "gpt2" # Provide a default if none specified

    try:
        import torch
        can_use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        can_use_cuda = torch.cuda.is_available()
        effective_device = 'cpu'
        if config.hardware.force_cpu: effective_device = 'cpu'
        elif can_use_cuda: effective_device = 'cuda'
        elif can_use_mps: effective_device = 'mps'
        if config.hardware.mixed_precision and effective_device == 'cpu':
            logger.info("Disabling mixed precision because device is CPU.")
            config.hardware.mixed_precision = False
        if config.hardware.use_flash_attention:
            if effective_device != 'cuda' or not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                logger.info(f"Disabling FlashAttention (device: {effective_device}, PyTorch supports: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}).")
                config.hardware.use_flash_attention = False
    except ImportError:
        logger.warning("PyTorch not found during config resolution. Hardware checks skipped.")
        config.hardware.mixed_precision = False
        config.hardware.use_flash_attention = False
    logger.debug("Configuration resolution complete.")
    return config

# --- END OF CORRECTED src/utils/config.py ---