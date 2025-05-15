# --- START OF FILE src/utils/config.py ---
"""
Configuration system for Project NEAT using dataclasses.

Defines hierarchical configuration structures for the model, components,
data processing, training, and hardware settings. Ensures dependencies
are resolved and provides loading/saving capabilities.
"""

import dataclasses
from typing import Dict, List, Optional, Union, Any, Type, get_origin, get_args
import json
import yaml # Requires PyYAML installation: pip install pyyaml
import logging
import os
import math

# Ensure logger is defined before use, especially in helper functions
logger = logging.getLogger(__name__)

# Helper function to recursively initialize dataclasses from dicts
def _init_from_dict(cls: Type[Any], data: Dict[str, Any]) -> Any:
    """
    Initializes a dataclass instance from a dictionary.
    Recursively initializes nested dataclasses.
    Ignores extra keys in the dictionary if not defined in dataclass.
    Uses default values for missing keys in the dictionary.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass type.")

    fields_info = {f.name: f for f in dataclasses.fields(cls)}
    init_args = {}

    for name, field_info in fields_info.items():
        field_type_hint = field_info.type
        default_value = field_info.default_factory() if field_info.default_factory is not dataclasses.MISSING else field_info.default
        
        origin_type_hint = get_origin(field_type_hint) # e.g., list, dict, Union, Optional
        type_args_hint = get_args(field_type_hint)   # e.g., (int,) for List[int], (int, str) for Union[int, str]

        is_nested_dataclass = False
        nested_type = None

        # Determine if the field is a dataclass or Optional[dataclass]
        if dataclasses.is_dataclass(field_type_hint):
            is_nested_dataclass = True
            nested_type = field_type_hint
        elif origin_type_hint is Union: # Handles Optional[DataclassType] and other Unions
             for arg in type_args_hint:
                  if dataclasses.is_dataclass(arg):
                       is_nested_dataclass = True
                       nested_type = arg
                       break
        
        if name in data:
            value_from_dict = data[name]
            if is_nested_dataclass and isinstance(value_from_dict, dict) and nested_type is not None:
                init_args[name] = _init_from_dict(nested_type, value_from_dict)
            else:
                # --- More robust type checking ---
                type_matches = False
                is_optional_and_value_is_none = (origin_type_hint is Union and type(None) in type_args_hint and value_from_dict is None)

                if is_optional_and_value_is_none:
                    type_matches = True
                elif origin_type_hint is Union:
                    # For Union, check if value_from_dict is an instance of ANY of the types in the Union
                    for union_arg_type in type_args_hint:
                        if union_arg_type is type(None): continue # Already handled by is_optional_and_value_is_none
                        actual_type_to_check = get_origin(union_arg_type) or union_arg_type
                        if actual_type_to_check is List: actual_type_to_check = list
                        elif actual_type_to_check is Dict: actual_type_to_check = dict
                        # Add other common generics if needed
                        if isinstance(value_from_dict, actual_type_to_check):
                            type_matches = True
                            break
                else:
                    # For non-Union types (or simple Optional[T] where T is not None)
                    actual_type_to_check = origin_type_hint or field_type_hint
                    if actual_type_to_check is List: actual_type_to_check = list
                    elif actual_type_to_check is Dict: actual_type_to_check = dict
                    # Add other common generics
                    if isinstance(value_from_dict, actual_type_to_check):
                        type_matches = True
                
                if type_matches:
                    init_args[name] = value_from_dict
                else:
                    # Attempt coercion for basic types if direct isinstance fails
                    coerced = False
                    primary_type_for_coercion = field_type_hint
                    if origin_type_hint is Union:
                        non_none_types = [t for t in type_args_hint if t is not type(None)]
                        if len(non_none_types) == 1: # If it's Optional[basic_type]
                            primary_type_for_coercion = non_none_types[0]
                        # If it's Union[typeA, typeB], coercion is ambiguous, prefer not to coerce
                        # unless we add more sophisticated logic or expect specific coercion rules.

                    if primary_type_for_coercion in [int, float, str, bool]:
                        try:
                            init_args[name] = primary_type_for_coercion(value_from_dict)
                            coerced = True
                            logger.debug(f"Coerced field '{name}' from {type(value_from_dict)} to {primary_type_for_coercion} with value '{init_args[name]}'.")
                        except (ValueError, TypeError):
                            pass # Coercion failed
                    
                    if not coerced:
                        logger.warning(
                            f"Type mismatch for field '{name}'. Expected {field_type_hint}, "
                            f"got {type(value_from_dict)} with value '{str(value_from_dict)[:100]}'. "
                            f"Using default value: {default_value}"
                        )
                        if default_value is not dataclasses.MISSING:
                            init_args[name] = default_value
                        # If no default, dataclass constructor will raise error if required.
        elif default_value is not dataclasses.MISSING:
            init_args[name] = default_value

    try:
        return cls(**init_args)
    except TypeError as e:
         logger.error(f"Error initializing {cls.__name__} with args {init_args}: {e}")
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
    surprise_method: str = "associative_loss_grad" 

    integration_layers: List[int] = dataclasses.field(default_factory=lambda: [0, 12, 23])

    memory_mlp_num_layers: int = 2 
    mem_mlp_intermediate_size: int = 1024 
    memory_learning_rate: float = 0.01    
    memory_momentum: float = 0.9          
    memory_weight_decay: float = 0.001    
    active_update_during_eval: bool = True 

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
    pin_memory: bool = False

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
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 4
    logging_steps: int = 100
    eval_steps: int = 1000
    eval_max_batches: Optional[int] = None
    save_steps: int = 1000
    output_dir: str = "./output/project_neat_model"
    resume_from: Optional[str] = None


@dataclasses.dataclass
class ModelConfig:
    """Main configuration aggregating all settings."""
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192 
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 4096
    vocab_size: int = 32000 
    layer_norm_eps: float = 1.0e-5

    tokenizer_name: Optional[str] = None 

    use_blt_processor: bool = False
    use_titans_memory: bool = True
    use_transformer2_adaptation: bool = True
    use_mvot_processor: bool = False

    blt: BLTConfig = dataclasses.field(default_factory=BLTConfig)
    titans: TitansConfig = dataclasses.field(default_factory=TitansConfig)
    transformer2: Transformer2Config = dataclasses.field(default_factory=Transformer2Config)
    mvot: MVoTConfig = dataclasses.field(default_factory=MVoTConfig)

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
            dir_name = os.path.dirname(path)
            if dir_name: 
                os.makedirs(dir_name, exist_ok=True)

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
            logger.error(f"Failed to save configuration to {path}: {e}", exc_info=True)

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
        logger.warning(f"data.block_size ({config.data.block_size}) > max_position_embeddings ({config.max_position_embeddings}). Adjusting block_size to {config.max_position_embeddings}.")
        config.data.block_size = config.max_position_embeddings
    
    if config.use_titans_memory:
        if config.titans.window_size > config.max_position_embeddings: 
            logger.warning(f"Titans window_size ({config.titans.window_size}) > max_position_embeddings. Adjusting window_size to {config.max_position_embeddings}.")
            config.titans.window_size = config.max_position_embeddings
        
        old_default_num_layers_for_integration = 24 
        if config.titans.integration_layers == [0, old_default_num_layers_for_integration // 2, old_default_num_layers_for_integration - 1] and \
           config.num_layers != old_default_num_layers_for_integration:
            new_default_integration_layers = [0, config.num_layers // 2, config.num_layers -1 if config.num_layers > 0 else 0]
            config.titans.integration_layers = new_default_integration_layers
            logger.info(f"Adjusted Titans integration_layers to {config.titans.integration_layers} for num_layers={config.num_layers}")
        
        if config.titans.memory_mlp_num_layers <= 0:
             logger.warning(f"Titans memory_mlp_num_layers ({config.titans.memory_mlp_num_layers}) is not positive. Setting to 1 (Linear layer).")
             config.titans.memory_mlp_num_layers = 1
        if config.titans.memory_mlp_num_layers > 1 and config.titans.mem_mlp_intermediate_size <= 0:
             default_intermediate = config.hidden_size // 2 if config.hidden_size > 0 else 256 
             logger.warning(f"Titans memory_mlp_num_layers > 1 but mem_mlp_intermediate_size ({config.titans.mem_mlp_intermediate_size}) is not positive. Setting to {default_intermediate}.")
             config.titans.mem_mlp_intermediate_size = default_intermediate


    if config.use_transformer2_adaptation:
        k = config.transformer2.num_singular_values
        max_k_attn_like = config.hidden_size
        if k > max_k_attn_like and (config.transformer2.adapt_attention or config.transformer2.adapt_lm_head or config.transformer2.adapt_embeddings):
            logger.warning(f"Transformer2 num_singular_values ({k}) > hidden_size ({max_k_attn_like}). Effective k will be clamped by SVD computation.")
        
        max_k_ffn_fc1 = min(config.hidden_size, config.intermediate_size)
        max_k_ffn_fc2 = min(config.intermediate_size, config.hidden_size)
        if k > max_k_ffn_fc1 and config.transformer2.adapt_ffn:
            logger.warning(f"Transformer2 num_singular_values ({k}) > FFN fc1 dimensions ({max_k_ffn_fc1}). Effective k will be clamped by SVD computation.")
        if k > max_k_ffn_fc2 and config.transformer2.adapt_ffn:
            logger.warning(f"Transformer2 num_singular_values ({k}) > FFN fc2 dimensions ({max_k_ffn_fc2}). Effective k will be clamped by SVD computation.")
    
    if config.use_mvot_processor:
        if config.mvot.use_pretrained_codebook and not config.mvot.codebook_path:
            raise ValueError("MVoT use_pretrained_codebook is True, but mvot.codebook_path is not set.")
    
    if config.use_blt_processor:
        if config.vocab_size != 260: 
             logger.warning(f"BLT processor is enabled. Overriding ModelConfig.vocab_size from {config.vocab_size} to 260 (byte vocab + special).")
             config.vocab_size = 260
        if config.blt.max_patch_size > config.data.block_size:
             logger.warning(f"BLT max_patch_size ({config.blt.max_patch_size}) > data.block_size ({config.data.block_size}). Ensure block_size is sufficient for max_patch_size.")
    else: 
         if not config.tokenizer_name:
              default_tokenizer = "gpt2"
              logger.warning(f"BLT is disabled, but ModelConfig.tokenizer_name is not set. Defaulting to '{default_tokenizer}'. Training script might override if it loads a tokenizer.")
              config.tokenizer_name = default_tokenizer

    try:
        import torch
        can_use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        can_use_cuda = torch.cuda.is_available()
        effective_device = 'cpu'
        if config.hardware.force_cpu: effective_device = 'cpu'
        elif can_use_cuda: effective_device = 'cuda'
        elif can_use_mps: effective_device = 'mps'

        if config.hardware.mixed_precision and effective_device == 'cpu':
            logger.info("Disabling mixed precision because effective device is CPU.")
            config.hardware.mixed_precision = False
        
        if config.hardware.use_flash_attention:
            if effective_device != 'cuda' or not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                logger.info(f"Disabling FlashAttention (device: {effective_device}, PyTorch supports scaled_dot_product_attention: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}).")
                config.hardware.use_flash_attention = False
    except ImportError:
        logger.warning("PyTorch not found during config resolution. Hardware checks skipped. mixed_precision and use_flash_attention defaulted to False.")
        config.hardware.mixed_precision = False
        config.hardware.use_flash_attention = False

    logger.debug("Configuration resolution complete.")
    return config

# --- END OF FILE src/utils/config.py ---