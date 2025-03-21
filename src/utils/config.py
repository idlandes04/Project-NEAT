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
    embedding_dim: int = 512  # Dimension of the codebook embeddings
    discrepancy_loss_weight: float = 0.1
    
    # Visual codebook parameters
    codebook_model_type: str = "vqvae"  # "vqvae", "vqgan", "dalle"
    codebook_path: Optional[str] = None
    use_pretrained_codebook: bool = False
    
    # Decision mechanism parameters
    decision_strategy: str = "hybrid"  # "neural", "heuristic", "hybrid"
    heuristic_weight: float = 0.5
    neural_weight: float = 0.5
    max_images: int = 5
    min_tokens_between_images: int = 20
    image_threshold: float = 0.7
    
    # Visualization benefit thresholds
    spatial_threshold: float = 0.15
    visual_threshold: float = 0.15
    complexity_threshold: float = 0.10
    reasoning_threshold: float = 0.20
    specificity_threshold: float = 0.10
    
    # Visualization benefit weights
    spatial_weight: float = 1.0
    visual_weight: float = 1.0
    complexity_weight: float = 0.7
    reasoning_weight: float = 0.5
    specificity_weight: float = 0.8


@dataclasses.dataclass
class TextProcessorConfig:
    """Configuration for text data processing."""
    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 32
    min_chunk_size: int = 64
    
    # Encoding parameters
    encoding: str = "utf-8"
    errors: str = "replace"
    
    # Filtering parameters
    min_entropy: float = 0.0
    max_entropy: float = 9.0
    require_complete_sentences: bool = False
    
    # Preprocessing
    normalize_whitespace: bool = True
    strip_html: bool = False
    lowercase: bool = False


@dataclasses.dataclass
class BinaryProcessorConfig:
    """Configuration for binary data processing."""
    # Chunking parameters
    chunk_size: int = 1024
    chunk_overlap: int = 0
    min_chunk_size: int = 128
    
    # Format detection
    enable_format_detection: bool = True
    format_specific_chunking: bool = True
    
    # Filtering parameters
    min_entropy: float = 0.0
    max_entropy: float = 9.0
    
    # Processing parameters
    compute_byte_frequency: bool = True
    skip_zero_blocks: bool = True


@dataclasses.dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    # General parameters
    num_samples: int = 2000
    use_cache: bool = True
    
    # Problem parameters
    problem_types: List[str] = dataclasses.field(
        default_factory=lambda: ["arithmetic", "algebra", "logic"]
    )
    difficulty_levels: List[str] = dataclasses.field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    include_metadata: bool = True
    
    # Entropy patterns
    generate_entropy_patterns: bool = True
    pattern_complexity: str = "medium"  # "simple", "medium", "complex"
    
    # Specific problem parameters
    max_operands: int = 5
    max_nested_operations: int = 3
    include_fractions: bool = True
    include_negative_numbers: bool = True


@dataclasses.dataclass
class DataMixerConfig:
    """Configuration for data mixing strategies."""
    # Mixing strategy
    strategy: str = "balanced"  # "balanced", "weighted", "sequential"
    
    # Weights for different sources (when using weighted strategy)
    text_weight: float = 0.4
    binary_weight: float = 0.3
    synthetic_weight: float = 0.3
    
    # Batch construction
    ensure_source_diversity: bool = True
    min_sources_per_batch: int = 2
    
    # Sampling parameters
    sampling_seed: Optional[int] = None
    replacement_sampling: bool = False


@dataclasses.dataclass
class CacheConfig:
    """Configuration for data caching."""
    # General cache parameters
    use_cache: bool = True
    cache_dir: str = "./cache"
    version_tracking: bool = True
    
    # Cache management
    max_cache_size_gb: float = 10.0
    auto_clean: bool = True
    clean_on_start: bool = False
    
    # Expiration and invalidation
    expiration_days: int = 30
    invalidate_on_config_change: bool = True
    
    # Performance options
    compress_cache: bool = False
    compression_level: int = 6


@dataclasses.dataclass
class DataConfig:
    """Configuration for the data pipeline."""
    # Directory paths
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    metadata_dir: str = "./data/metadata"
    
    # Data sources
    use_text_data: bool = True
    use_binary_data: bool = True
    use_synthetic_data: bool = True
    
    # Data scanning parameters
    text_file_extensions: List[str] = dataclasses.field(
        default_factory=lambda: [".txt", ".md", ".log", ".json", ".xml", ".csv"]
    )
    binary_file_extensions: List[str] = dataclasses.field(
        default_factory=lambda: [".bin", ".dat", ".exe", ".dll", ".so"]
    )
    recursive_scan: bool = True
    follow_symlinks: bool = False
    
    # File limits
    max_files_per_source: int = 1000
    max_chunks_per_file: int = 100
    
    # Processor configurations
    text_processor: TextProcessorConfig = dataclasses.field(default_factory=TextProcessorConfig)
    binary_processor: BinaryProcessorConfig = dataclasses.field(default_factory=BinaryProcessorConfig)
    synthetic_data: SyntheticDataConfig = dataclasses.field(default_factory=SyntheticDataConfig)
    data_mixer: DataMixerConfig = dataclasses.field(default_factory=DataMixerConfig)
    cache: CacheConfig = dataclasses.field(default_factory=CacheConfig)


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
    latent_hidden_size: int = 768  # Hidden size for latent transformer
    
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
    
    # Data pipeline configuration
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    
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
        data_dict = config_dict.pop('data', {})
        hardware_dict = config_dict.pop('hardware', {})
        training_dict = config_dict.pop('training', {})
        
        # Extract nested ByteLM configuration
        byte_lm_dict = {}
        if 'byte_lm' in blt_dict:
            byte_lm_dict = blt_dict.pop('byte_lm', {})
        
        # Extract nested data configurations
        text_processor_dict = data_dict.pop('text_processor', {})
        binary_processor_dict = data_dict.pop('binary_processor', {})
        synthetic_data_dict = data_dict.pop('synthetic_data', {})
        data_mixer_dict = data_dict.pop('data_mixer', {})
        cache_dict = data_dict.pop('cache', {})
        
        # Create component configurations
        titans_config = TitansConfig(**titans_dict)
        transformer2_config = Transformer2Config(**transformer2_dict)
        mvot_config = MVoTConfig(**mvot_dict)
        
        # Create ByteLM configuration
        byte_lm_config = ByteLMConfig(**byte_lm_dict)
        
        # Create BLT configuration with ByteLM
        blt_config = BLTConfig(**blt_dict)
        blt_config.byte_lm = byte_lm_config
        
        # Create data processor configurations
        text_processor_config = TextProcessorConfig(**text_processor_dict)
        binary_processor_config = BinaryProcessorConfig(**binary_processor_dict)
        synthetic_data_config = SyntheticDataConfig(**synthetic_data_dict)
        data_mixer_config = DataMixerConfig(**data_mixer_dict)
        cache_config = CacheConfig(**cache_dict)
        
        # Create data configuration with processors
        data_config = DataConfig(**data_dict)
        data_config.text_processor = text_processor_config
        data_config.binary_processor = binary_processor_config
        data_config.synthetic_data = synthetic_data_config
        data_config.data_mixer = data_mixer_config
        data_config.cache = cache_config
        
        hardware_config = HardwareConfig(**hardware_dict)
        training_config = TrainingConfig(**training_dict)
        
        # Create main configuration
        config = cls(**config_dict)
        config.titans = titans_config
        config.transformer2 = transformer2_config
        config.mvot = mvot_config
        config.blt = blt_config
        config.data = data_config
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


def convert_cli_config_to_byte_lm_config(cli_config: Dict[str, Any]) -> ByteLMConfig:
    """
    Convert a CLI-style configuration to ByteLMConfig.
    
    Args:
        cli_config: Dictionary containing CLI configuration parameters
        
    Returns:
        ByteLMConfig object initialized with parameters from CLI config
    """
    # Map CLI parameter names to ByteLMConfig parameter names
    param_mapping = {
        "byte_lm_hidden_size": "hidden_size",
        "byte_lm_num_layers": "num_layers",
        "byte_lm_num_heads": "num_attention_heads",
        "byte_lm_dropout": "byte_lm_dropout",
        "block_size": "byte_lm_max_position",  # Also used for block_size
    }
    
    # Extract parameters for ByteLMConfig
    config_params = {}
    
    # Process each parameter from CLI config
    for cli_param, value in cli_config.items():
        # Map parameter name if needed
        if cli_param in param_mapping:
            config_params[param_mapping[cli_param]] = value
        elif cli_param in ["hidden_size", "num_layers", "num_attention_heads", 
                          "byte_lm_dropout", "byte_lm_max_position",
                          "learning_rate", "batch_size", "warmup_steps", 
                          "max_steps", "eval_steps", "save_steps",
                          "gradient_accumulation_steps", "weight_decay",
                          "cache_dir", "output_dir", "checkpoint_path",
                          "block_size"]:
            # Direct parameters that match ByteLMConfig fields
            config_params[cli_param] = value
    
    # Ensure block_size is set for both block_size and byte_lm_max_position
    if "block_size" in cli_config:
        config_params["block_size"] = cli_config["block_size"]
        if "byte_lm_max_position" not in config_params:
            config_params["byte_lm_max_position"] = cli_config["block_size"]
    
    # Special handling for train_data_dir and eval_data_dir (not direct fields)
    # These will be handled by the caller to set train_files and eval_files
    
    # Create ByteLMConfig with extracted parameters
    return ByteLMConfig(**config_params)
