"""
Configuration system for the neural architecture integration.

This module provides a hierarchical configuration system for the neural
architecture integration project. It includes configuration classes for
each component and a main configuration class that combines them.
"""
import dataclasses
from typing import Dict, List, Optional, Union, Any


@dataclasses.dataclass
class TitansConfig:
    """Configuration for the Titans memory system."""
    # Memory types
    use_window_attention: bool = True
    use_surprise_based: bool = True
    use_persistent: bool = True
    
    # Window attention parameters
    window_size: int = 512
    
    # Surprise-based memory parameters
    memory_size: int = 1024
    surprise_threshold: float = 0.5
    max_memory_updates_per_step: int = 10
    
    # Persistent memory parameters
    num_persistent_vectors: int = 64
    persistent_init_scale: float = 0.02


@dataclasses.dataclass
class Transformer2Config:
    """Configuration for the Transformer² adaptation."""
    # Component activation
    use_task_dispatcher: bool = True
    use_svd_adaptation: bool = True
    use_two_pass_inference: bool = True
    
    # Task dispatcher parameters
    num_tasks: int = 8
    task_embedding_dim: int = 64
    
    # SVD adaptation parameters
    num_singular_values: int = 768  # Should match hidden_size
    expert_init_scale: float = 0.1
    
    # Adaptation control
    adapt_embeddings: bool = False  # Whether to adapt embedding matrices
    adapt_lm_head: bool = False  # Whether to adapt LM head
    layer_specific: bool = True  # Whether to use layer-specific adaptations
    
    # SVD computation parameters
    use_randomized_svd: bool = True  # Whether to use randomized SVD for large matrices
    svd_precision: str = "adaptive"  # "full", "adaptive", or "fixed"
    svd_n_oversamples: int = 10  # Number of extra samples in randomized SVD
    svd_n_iter: int = 5  # Number of power iterations in randomized SVD
    
    # SVD caching parameters
    enable_svd_caching: bool = True  # Whether to cache SVD results
    svd_cache_dir: str = ".svd_cache"  # Directory for persistent SVD cache
    
    # Two-pass inference parameters
    cache_first_pass: bool = True
    reuse_threshold: float = 0.9
    
    # Task embedding cache parameters
    max_task_cache_size: int = 50  # Maximum number of task embeddings to store
    task_similarity_threshold: float = 0.85  # Minimum similarity for reusing cached embeddings


@dataclasses.dataclass
class MVoTConfig:
    """Configuration for the MVoT token processor."""
    # Component activation
    is_multimodal: bool = True
    
    # Token processing parameters
    token_type_vocab_size: int = 2  # 0 for text, 1 for image
    
    # Token discrepancy loss parameters
    codebook_size: int = 8192
    discrepancy_loss_weight: float = 0.1


@dataclasses.dataclass
class ByteLMConfig:
    """Configuration for the byte-level language model (entropy estimator)."""
    # Model parameters
    hidden_size: int = 128
    num_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 512
    byte_lm_dropout: float = 0.1
    byte_lm_max_position: int = 512
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 32
    block_size: int = 128
    warmup_steps: int = 1000
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 500
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    
    # Data parameters
    train_files: List[str] = dataclasses.field(default_factory=list)
    eval_files: List[str] = dataclasses.field(default_factory=list)
    cache_dir: str = "./cache"
    output_dir: str = "./outputs/byte_lm"
    checkpoint_path: Optional[str] = None


@dataclasses.dataclass
class BLTConfig:
    """Configuration for the BLT byte processor."""
    # Component activation
    use_dynamic_patching: bool = True
    
    # Patching parameters
    entropy_threshold: float = 0.5
    min_patch_size: int = 8
    max_patch_size: int = 128
    
    # Architecture parameters
    num_local_layers: int = 2
    num_latent_layers: int = 4
    
    # Byte LM configuration
    byte_lm: ByteLMConfig = dataclasses.field(default_factory=ByteLMConfig)


@dataclasses.dataclass
class HardwareConfig:
    """Configuration for hardware-specific optimizations."""
    # Memory optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    
    # Resource allocation
    gpu_memory_threshold: float = 0.8
    cpu_memory_threshold: float = 0.7
    
    # Parallelism
    num_workers: Optional[int] = None  # None means use all available cores
    
    # Batch sizing
    dynamic_batch_sizing: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 64
    
    # Kernel optimization
    use_flash_attention: bool = True
    use_custom_kernels: bool = False


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training schedule
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    
    # Logging and checkpointing
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500


@dataclasses.dataclass
class ModelConfig:
    """Main configuration for the model."""
    # Core model parameters
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    vocab_size: int = 30522
    
    # Component activation
    use_titans_memory: bool = True
    use_transformer2_adaptation: bool = True
    use_mvot_processor: bool = True
    use_blt_processor: bool = True
    
    # Component activation thresholds
    titans_activation_threshold: float = 0.5
    transformer2_activation_threshold: float = 0.5
    mvot_activation_threshold: float = 0.5
    blt_activation_threshold: float = 0.5
    two_pass_activation_threshold: float = 0.8
    
    # Component configurations
    titans: TitansConfig = dataclasses.field(default_factory=TitansConfig)
    transformer2: Transformer2Config = dataclasses.field(default_factory=Transformer2Config)
    mvot: MVoTConfig = dataclasses.field(default_factory=MVoTConfig)
    blt: BLTConfig = dataclasses.field(default_factory=BLTConfig)
    
    # Hardware and training configurations
    hardware: HardwareConfig = dataclasses.field(default_factory=HardwareConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create a configuration from a dictionary."""
        # Extract component configurations
        titans_dict = config_dict.pop('titans', {})
        transformer2_dict = config_dict.pop('transformer2', {})
        mvot_dict = config_dict.pop('mvot', {})
        blt_dict = config_dict.pop('blt', {})
        hardware_dict = config_dict.pop('hardware', {})
        training_dict = config_dict.pop('training', {})
        
        # Extract nested ByteLM configuration
        byte_lm_dict = {}
        if 'byte_lm' in blt_dict:
            byte_lm_dict = blt_dict.pop('byte_lm', {})
        
        # Create component configurations
        titans_config = TitansConfig(**titans_dict)
        transformer2_config = Transformer2Config(**transformer2_dict)
        mvot_config = MVoTConfig(**mvot_dict)
        
        # Create ByteLM configuration
        byte_lm_config = ByteLMConfig(**byte_lm_dict)
        
        # Create BLT configuration with ByteLM
        blt_config = BLTConfig(**blt_dict)
        blt_config.byte_lm = byte_lm_config
        
        hardware_config = HardwareConfig(**hardware_dict)
        training_config = TrainingConfig(**training_dict)
        
        # Create main configuration
        config = cls(**config_dict)
        config.titans = titans_config
        config.transformer2 = transformer2_config
        config.mvot = mvot_config
        config.blt = blt_config
        config.hardware = hardware_config
        config.training = training_config
        
        return config


class ConfigurationManager:
    """
    Manager for loading, saving, and validating configurations.
    
    This class provides methods for loading configurations from files,
    validating configurations, and resolving dependencies between
    components.
    """
    
    def __init__(self):
        self.config = get_default_config()
    
    def load_from_yaml(self, yaml_path: str) -> ModelConfig:
        """Load configuration from a YAML file."""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self.config = ModelConfig.from_dict(config_dict)
            self.validate()
            return self.config
        except ImportError:
            print("PyYAML is required for loading YAML configurations.")
            return self.config
        except Exception as e:
            print(f"Error loading configuration from {yaml_path}: {e}")
            return self.config
    
    def save_to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        try:
            import yaml
            with open(yaml_path, 'w') as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        except ImportError:
            print("PyYAML is required for saving YAML configurations.")
        except Exception as e:
            print(f"Error saving configuration to {yaml_path}: {e}")
    
    def update_from_args(self, args: Any) -> ModelConfig:
        """Update configuration from command-line arguments."""
        config_dict = self.config.to_dict()
        
        # Update configuration from arguments
        for key, value in vars(args).items():
            if value is not None:
                # Handle nested configurations
                if '.' in key:
                    # e.g., "titans.window_size" -> config.titans.window_size
                    parts = key.split('.')
                    if len(parts) == 2 and parts[0] in config_dict:
                        if isinstance(config_dict[parts[0]], dict):
                            config_dict[parts[0]][parts[1]] = value
                else:
                    config_dict[key] = value
        
        self.config = ModelConfig.from_dict(config_dict)
        self.validate()
        return self.config
    
    def validate(self) -> bool:
        """Validate the configuration."""
        # Check for basic consistency
        if self.config.num_attention_heads <= 0:
            print("Error: num_attention_heads must be positive.")
            return False
        
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            print("Error: hidden_size must be divisible by num_attention_heads.")
            return False
        
        # Check component-specific configurations
        if self.config.use_titans_memory:
            if self.config.titans.memory_size <= 0:
                print("Error: titans.memory_size must be positive.")
                return False
        
        if self.config.use_transformer2_adaptation:
            if self.config.transformer2.num_tasks <= 0:
                print("Error: transformer2.num_tasks must be positive.")
                return False
        
        # Resolve dependencies
        self._resolve_dependencies()
        
        return True
    
    def _resolve_dependencies(self) -> None:
        """Resolve dependencies between components."""
        # Ensure transformer2.num_singular_values matches hidden_size
        self.config.transformer2.num_singular_values = self.config.hidden_size
        
        # Adjust window size based on max_position_embeddings
        if self.config.titans.window_size > self.config.max_position_embeddings:
            self.config.titans.window_size = self.config.max_position_embeddings
        
        # Ensure hardware settings are compatible
        if self.config.hardware.mixed_precision and not torch_is_available():
            print("Warning: mixed_precision requires PyTorch with CUDA. Disabling.")
            self.config.hardware.mixed_precision = False


def get_default_config() -> ModelConfig:
    """Get the default configuration."""
    return ModelConfig()


def torch_is_available() -> bool:
    """Check if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
