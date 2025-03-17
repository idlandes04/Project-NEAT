# Project NEAT: Phase 3.1.1.5 - Optimized Test Run with Sleep-Wake Integration

## Overview

This master plan reconciles the two approaches for Phase 3.1.1.5, creating a comprehensive strategy that focuses on an optimized test run while integrating the critical sleep-wake memory consolidation mechanism into the Titans memory system. This plan leverages parallel development tracks to maximize productivity - enabling implementation of the sleep-wake mechanism and component improvements while models are training.

The goals of this phase are to:
1. Establish component stability and verify convergence
2. Implement and evaluate the sleep-wake memory consolidation system
3. Create a testing environment that quickly reveals integration issues
4. Begin refining the NEAT architecture through component-specific improvements
5. Prepare for comparative evaluation against baseline models

## Hardware Environment & Optimization

### Current Specifications
- **CPU**: AMD Ryzen 5950x (16 cores/32 threads, 5GHz)
- **GPU**: NVIDIA RTX 3080Ti (12GB VRAM)
- **RAM**: 64GB allocated to WSL out of 128GB system total
- **Environment**: Windows 11 WSL 2

### Hardware Optimization Steps

1. **WSL Memory Allocation**
   - Increase WSL memory allocation from 64GB to 96GB
   ```bash
   # In Windows PowerShell (Admin):
   wsl --shutdown
   notepad "$env:USERPROFILE/.wslconfig"
   # Add/modify:
   [wsl2]
   memory=96GB
   processors=28  # Leave 4 cores for Windows
   wsl --start
   ```

2. **CUDA Configuration**
   - Optimize CUDA settings for 3080Ti
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export CUDA_AUTO_BOOST=1
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2:True"
   # Set PyTorch thread settings for Ryzen 5950x architecture
   export OMP_NUM_THREADS=16
   export MKL_NUM_THREADS=16
   ```

3. **Filesystem Optimization**
   - Set up RAM disk for temporary files and dataset caching
   ```bash
   sudo mkdir -p /mnt/ramdisk
   sudo mount -t tmpfs -o size=20g tmpfs /mnt/ramdisk
   export TMPDIR=/mnt/ramdisk
   export TORCH_HOME=/mnt/ramdisk/torch
   ```

4. **Thread Management**
   - Set worker threads to 24 (leaving cores for system operations)
   - Configure NUMA-aware threading for the 5950x

5. **Storage Optimization**
   - Set up training checkpoints on SSD storage
   - Configure dataset caching to minimize I/O bottlenecks

## Development Parallelization Strategy

To maximize productivity, we'll run multiple development tracks in parallel:

| Track | Focus | Timeline |
|-------|-------|----------|
| **Track A** | Component Pre-training & Main Model Training | Follows main training schedule |
| **Track B** | Sleep-Wake Implementation & Testing | Run during Track A training periods |
| **Track C** | Component-specific improvements | Continuous refinement |

This strategy allows us to utilize training downtime for implementing new features and optimizations. While pre-training the BLT entropy estimator (6 hours) and MVoT visual codebook (14 hours), we'll implement the sleep-wake mechanism and component improvements.

## Track A: Pre-training & Main Model Training

### Phase A.1: BLT Entropy Estimator Pre-Training

```bash
mkdir -p data/{byte_training,byte_eval,cache/byte_lm}
mkdir -p outputs/blt_pretrain

# Download and preprocess data
wget -O data/byte_training/pile_subset.jsonl https://the-eye.eu/public/AI/pile/train/00.jsonl
python src/data/preprocessing/prepare_byte_data.py \
    --input data/byte_training/pile_subset.jsonl \
    --output data/byte_training/processed/ \
    --eval_split 0.05

# Launch BLT pre-training
python main.py --mode train_byte_lm \
    --train_data_dir data/byte_training/processed/ \
    --eval_data_dir data/byte_eval/ \
    --byte_lm_hidden_size 128 \
    --byte_lm_num_layers 2 \
    --num_attention_heads 4 \
    --batch_size 128 \
    --max_steps 10000 \
    --block_size 256 \
    --learning_rate 5e-5 \
    --output_dir outputs/blt_pretrain \
    --mixed_precision \
    --cache_dir data/cache/byte_lm
```

**Configuration Details:**
- Small model: ~5M parameters
- Hidden size: 128
- Layers: 2
- Training time: ~6 hours on RTX 3080Ti
- Block size: 256 bytes
- Batch size: 128

### Phase A.2: MVoT Visual Codebook Training

```bash
mkdir -p data/visual_codebook
mkdir -p outputs/mvot_pretrain

# Download MS-COCO dataset
wget -O data/visual_codebook/coco_subset.zip "http://images.cocodataset.org/zips/val2017.zip"
unzip data/visual_codebook/coco_subset.zip -d data/visual_codebook/

# Launch MVoT pre-training
python tools/train_visual_codebook.py \
    --data_dir data/visual_codebook/ \
    --output_dir outputs/mvot_pretrain \
    --batch_size 64 \
    --num_epochs 20 \
    --codebook_size 8192 \
    --embedding_dim 512 \
    --model_type vqvae \
    --mixed_precision
```

**Configuration Details:**
- Codebook size: 8192
- Embedding dimension: 512
- Training time: ~14 hours on RTX 3080Ti
- Model type: VQVAE
- Dataset: MS-COCO (smaller than LAION-400M for faster training)

### Phase A.3: Synthetic Data Generation

Generate synthetic mathematical problems with varying difficulty for testing model reasoning capabilities, including special sleep-wake evaluation tasks:

```bash
# Generate synthetic math data
python src/data/synthetic/generate_math_dataset.py \
    --output_dir data/synthetic_math \
    --num_problems 500000 \
    --num_test_problems 50000 \
    --max_difficulty 3 \
    --problem_types "arithmetic,algebra,sequences,word_problems" \
    --include_step_by_step True \
    --include_sleep_wake_tasks True  # New parameter for memory consolidation tasks
```

**Dataset Configuration:**
- 500,000 training problems
- 50,000 test problems
- Difficulty levels: Basic, Medium, Advanced
- Problem types: Arithmetic, Algebra, Sequences, Word Problems
- Sleep-wake specific problem sets:
  - Recall problems (testing short-term → long-term transfer)
  - Temporal gap problems (testing memory retention over time)
  - Surprise-based memory tests (high-entropy recall tasks)

### Phase A.4: Main Model Configuration (100M Parameters)

Create a configuration file for the 100M parameter model with sleep-wake capabilities:

```bash
cat > configs/neat_100m_sleep_wake.json << 'EOL'
{
  "hidden_size": 768,
  "num_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "max_position_embeddings": 2048,
  "vocab_size": 32000,
  
  "use_titans_memory": true,
  "use_transformer2_adaptation": true,
  "use_mvot_processor": true,
  "use_blt_processor": true,
  
  "titans": {
    "window_size": 512,
    "memory_size": 2048,
    "surprise_threshold": 0.5,
    "max_memory_updates_per_step": 10,
    "num_persistent_vectors": 128,
    "persistent_init_scale": 0.02,
    "enable_sleep_wake": true,
    "sleep_cycle_steps": 1000,
    "consolidation_threshold": 0.7,
    "sleep_phase_duration": 100,
    "importance_decay_rate": 0.95,
    "memory_to_weight_lr": 1e-5
  },
  
  "transformer2": {
    "num_tasks": 8,
    "task_embedding_dim": 128,
    "use_svd_adaptation": true,
    "layer_specific": true,
    "enable_svd_caching": true
  },
  
  "mvot": {
    "is_multimodal": true,
    "codebook_size": 8192,
    "embedding_dim": 512,
    "discrepancy_loss_weight": 0.1,
    "codebook_path": "outputs/mvot_pretrain/codebook.pt",
    "use_pretrained_codebook": true
  },
  
  "blt": {
    "use_dynamic_patching": true,
    "entropy_threshold": 0.5,
    "min_patch_size": 8,
    "max_patch_size": 128,
    "num_local_layers": 2,
    "num_latent_layers": 4,
    "byte_lm": {
      "checkpoint_path": "outputs/blt_pretrain/best_model"
    }
  },
  
  "hardware": {
    "mixed_precision": true,
    "gradient_checkpointing": true,
    "dynamic_batch_sizing": true,
    "gpu_memory_threshold": 0.85,
    "cpu_memory_threshold": 0.75,
    "num_workers": 14,
    "use_flash_attention": true
  },
  
  "training": {
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 2,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 250,
    "sleep_wake_enabled": true,
    "wake_phase_steps": 1000,
    "sleep_phase_steps": 100,
    "memory_consolidation_strength": 0.5
  }
}
EOL
```

### Parameter Distribution

| Component             | Parameters | Percentage | Configuration                     |
|-----------------------|------------|------------|-----------------------------------|
| Core Transformer      | 40M        | 40%        | 12 layers, hidden_size=768        |
| Titans Memory System  | 20M        | 20%        | memory_size=2048, window_size=512 |
| Transformer² Adaptation| 20M        | 20%        | SVD_adaptation=True, tasks=8      |
| BLT Processor         | 10M        | 10%        | num_local_layers=2, num_latent_layers=4 |
| MVoT Processor        | 10M        | 10%        | codebook_size=8192, is_multimodal=True |

### Phase A.5: Progressive Training Strategy

```bash
# Start the training run with sleep-wake cycles enabled
python main.py --mode train \
    --config configs/neat_100m_sleep_wake.json \
    --output_dir outputs/neat_100m_initial \
    --batch_size 16 \
    --max_steps 10000 \
    --dynamic_component_activation \
    --optimize_for_hardware \
    --cross_platform_compatibility \
    --detect_hardware \
    --enable_sleep_wake \
    --sleep_wake_logging
```

The progressive training approach will follow these phases:

1. **Warm-up Phase (Steps 1-1000)**
   - Start with a smaller batch size (8)
   - Gradually increase learning rate
   - Activate only core transformer and Titans memory
   - Initial "wake-only" phase to establish baseline

2. **First Sleep-Wake Cycle (Steps 1001-2100)**
   - First sleep phase at step 1001-1100
   - Activate Transformer² adaptation
   - Evaluate memory consolidation effects

3. **Component Integration Phase (Steps 2101-4000)**
   - Add BLT processor (step 2101)
   - Add MVoT processor (step 3001)
   - Regular sleep-wake cycles every 1000 steps

4. **Full Model Training (Steps 4001-10000)**
   - All components active
   - Full batch size (16)
   - Advanced sleep-wake patterns with varying cycle lengths

## Track B: Sleep-Wake Mechanism Implementation

While Track A training runs are happening, we'll implement the sleep-wake mechanism for the Titans memory system.

### Phase B.1: Sleep-Wake Base Framework

Create a memory consolidation system that mimics human sleep phases:

```python
class MemoryConsolidation:
    """Sleep-phase memory consolidation mechanism for Titans memory system."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.memory_system = model.memory_system
        
        # Configuration
        self.consolidation_lr = config.titans.memory_to_weight_lr
        self.importance_threshold = config.titans.consolidation_threshold
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.consolidation_lr)
        
    def consolidate_memories(self):
        """Consolidate important memories into model weights."""
        # Get memories and importance scores
        memories = self.memory_system.get_memories()
        importance_scores = self.memory_system.get_importance_scores()
        
        # Filter important memories
        important_indices = torch.where(importance_scores > self.importance_threshold)[0]
        important_memories = memories[important_indices]
        
        if len(important_memories) == 0:
            logging.info("No important memories to consolidate")
            return 0.0
            
        # Use memories to update model weights
        self.optimizer.zero_grad()
        
        # Create a target distribution from memories
        memory_representation = self._create_memory_representation(important_memories)
        
        # Generate outputs from current model
        model_outputs = self._generate_from_memories(important_memories)
        
        # Compute loss between model outputs and memory representation
        loss = F.mse_loss(model_outputs, memory_representation)
        
        # Backpropagate and update weights
        loss.backward()
        
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Reset short-term memory after consolidation
        self.memory_system.reset_short_term_memory()
        
        return loss.item()

    def _create_memory_representation(self, memories):
        """Create a target representation from memories."""
        # This is a simplified implementation
        return memories.mean(dim=0)
        
    def _generate_from_memories(self, memories):
        """Generate model outputs from memories."""
        with torch.no_grad():
            outputs = self.model(memories)
        return outputs
```

### Phase B.2: Sleep-Wake Integration with Titans

Extend the Titans memory system to support sleep-wake cycles:

```python
# Add to TitansMemorySystem class in src/components/titans/memory_system.py

def enable_sleep_wake_mechanism(self):
    """Enable sleep-wake mechanism for memory consolidation."""
    self.sleep_wake_enabled = True
    self.current_phase = "wake"
    self.steps_in_current_phase = 0
    self.consolidation_system = MemoryConsolidation(self._get_parent_model(), self.config)
    self.wake_memory_snapshots = []
    logging.info("Sleep-wake mechanism enabled for Titans memory system")

def enter_sleep_phase(self):
    """Enter sleep phase for memory consolidation."""
    if not hasattr(self, 'sleep_wake_enabled') or not self.sleep_wake_enabled:
        return
        
    self.current_phase = "sleep"
    self.steps_in_current_phase = 0
    
    # Take a snapshot of current memory state before consolidation
    self.pre_sleep_memory_state = {
        "memory": self.memory.clone(),
        "importance_scores": self.importance_scores.clone(),
        "memory_usage": self.memory_usage.clone()
    }
    
    # Run memory consolidation
    consolidation_loss = self.consolidation_system.consolidate_memories()
    
    logging.info(f"Entered sleep phase, consolidation loss: {consolidation_loss}")
    
def enter_wake_phase(self):
    """Enter wake phase after memory consolidation."""
    if not hasattr(self, 'sleep_wake_enabled') or not self.sleep_wake_enabled:
        return
        
    self.current_phase = "wake"
    self.steps_in_current_phase = 0
    
    # Reset short-term memory
    self.reset_short_term_memory()
    
    # Take a snapshot of memory after sleep for comparison
    self.post_sleep_memory_state = {
        "memory": self.memory.clone(),
        "importance_scores": self.importance_scores.clone(),
        "memory_usage": self.memory_usage.clone()
    }
    
    # Analyze sleep phase effects
    self._analyze_sleep_phase_effects()
    
    logging.info(f"Entered wake phase")

def _analyze_sleep_phase_effects(self):
    """Analyze the effects of the sleep phase on memory and weights."""
    if not hasattr(self, 'pre_sleep_memory_state') or not hasattr(self, 'post_sleep_memory_state'):
        return
        
    # Calculate memory changes
    memory_diff = torch.norm(
        self.post_sleep_memory_state["memory"] - self.pre_sleep_memory_state["memory"]
    ).item()
    
    importance_diff = torch.norm(
        self.post_sleep_memory_state["importance_scores"] - self.pre_sleep_memory_state["importance_scores"]
    ).item()
    
    logging.info(f"Sleep phase effects - Memory change: {memory_diff:.4f}, Importance change: {importance_diff:.4f}")
    
    # Store sleep phase metrics
    if not hasattr(self, 'sleep_phase_metrics'):
        self.sleep_phase_metrics = []
        
    self.sleep_phase_metrics.append({
        "step": self.global_step.item(),
        "memory_diff": memory_diff,
        "importance_diff": importance_diff,
    })
```

### Phase B.3: Sleep-Wake Cycle Management

Add a sleep-wake cycle manager to the UnifiedArchitecture:

```python
# Add to UnifiedArchitecture class in src/models/unified_architecture.py

def _init_sleep_wake_cycle(self):
    """Initialize sleep-wake cycle management."""
    self.sleep_wake_enabled = self.config.training.sleep_wake_enabled
    self.wake_steps = self.config.training.wake_phase_steps
    self.sleep_steps = self.config.training.sleep_phase_steps
    self.current_phase = "wake"
    self.steps_in_current_phase = 0
    
    # Enable sleep-wake in memory system
    if hasattr(self, 'memory_system') and self.sleep_wake_enabled:
        if hasattr(self.memory_system, 'enable_sleep_wake_mechanism'):
            self.memory_system.enable_sleep_wake_mechanism()
    
    logging.info(f"Sleep-wake cycle initialized: {self.wake_steps} wake steps, {self.sleep_steps} sleep steps")

def _manage_sleep_wake_cycle(self, step: int):
    """Manage sleep-wake cycle transitions."""
    if not self.sleep_wake_enabled:
        return
        
    self.steps_in_current_phase += 1
    
    # Check if it's time to transition
    if self.current_phase == "wake" and self.steps_in_current_phase >= self.wake_steps:
        # Transition to sleep phase
        self.current_phase = "sleep"
        self.steps_in_current_phase = 0
        
        # Notify memory system
        if hasattr(self, 'memory_system') and hasattr(self.memory_system, 'enter_sleep_phase'):
            self.memory_system.enter_sleep_phase()
            
        logging.info(f"Step {step}: Transitioned to SLEEP phase")
        
    elif self.current_phase == "sleep" and self.steps_in_current_phase >= self.sleep_steps:
        # Transition to wake phase
        self.current_phase = "wake"
        self.steps_in_current_phase = 0
        
        # Notify memory system
        if hasattr(self, 'memory_system') and hasattr(self.memory_system, 'enter_wake_phase'):
            self.memory_system.enter_wake_phase()
            
        logging.info(f"Step {step}: Transitioned to WAKE phase")
```

### Phase B.4: Forward Pass Modification

Modify the forward pass to integrate sleep-wake cycles:

```python
# Modify forward method in UnifiedArchitecture class

def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    process_feedback: bool = True,
    _skip_two_pass: bool = False,
    step: Optional[int] = None,  # Add step for sleep-wake management
    **kwargs,
):
    """Forward pass through the unified architecture with sleep-wake management."""
    # Handle sleep-wake cycle if step is provided
    if step is not None and hasattr(self, 'sleep_wake_enabled') and self.sleep_wake_enabled:
        self._manage_sleep_wake_cycle(step)
        
    # If in sleep phase, process differently
    if hasattr(self, 'current_phase') and self.current_phase == "sleep":
        # In sleep phase, we're consolidating memory, not processing new inputs
        # Return a placeholder output during sleep phase
        dummy_logits = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size), 
            device=input_ids.device
        )
        return {"logits": dummy_logits, "phase": "sleep"}
    
    # Normal forward pass for wake phase
    # [Original forward pass code here]
    
    # Add phase information to output
    outputs["phase"] = getattr(self, 'current_phase', "wake")
    
    return outputs
```

### Phase B.5: Memory Consolidation Visualization

Create tools to visualize the memory consolidation process:

```python
# Add to tools/visualize_sleep_wake.py

def plot_memory_consolidation(model_dir, checkpoint_steps):
    """Plot memory consolidation metrics across training steps."""
    import matplotlib.pyplot as plt
    
    # Load model checkpoints
    metrics = []
    for step in checkpoint_steps:
        model_path = f"{model_dir}/checkpoint-{step}"
        if not os.path.exists(model_path):
            continue
            
        model = torch.load(model_path, map_location="cpu")
        
        # Extract sleep phase metrics
        if hasattr(model, 'memory_system') and hasattr(model.memory_system, 'sleep_phase_metrics'):
            metrics.extend(model.memory_system.sleep_phase_metrics)
    
    if not metrics:
        print("No sleep phase metrics found")
        return
        
    # Sort by step
    metrics.sort(key=lambda x: x["step"])
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps = [m["step"] for m in metrics]
    memory_diffs = [m["memory_diff"] for m in metrics]
    importance_diffs = [m["importance_diff"] for m in metrics]
    
    ax1.plot(steps, memory_diffs, 'b-', label='Memory Change')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Memory Change')
    ax1.legend()
    
    ax2.plot(steps, importance_diffs, 'r-', label='Importance Change')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Importance Change')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_dir}/memory_consolidation.png")
    plt.close()
```

## Track C: Component-Specific Improvements

### Phase C.1: Titans Memory System Optimizations

Implement memory efficiency improvements:

```python
class MemoryPruningOptimizer:
    """Optimize memory usage through pruning low-importance slots."""
    
    def __init__(self, config):
        self.config = config
        self.prune_threshold = config.titans.memory_prune_threshold
        self.prune_interval = config.titans.prune_interval
        self.last_prune_step = 0
        
    def should_prune(self, current_step):
        """Determine if memory should be pruned at current step."""
        return current_step - self.last_prune_step >= self.prune_interval
        
    def prune_memory(self, memory, importance_scores, memory_usage, memory_age):
        """Prune low-importance memory slots."""
        # Find low-importance and old slots
        prune_candidates = torch.where(
            (importance_scores < self.prune_threshold) & 
            (memory_usage < 5) &
            (memory_age > 1000)
        )[0]
        
        if len(prune_candidates) == 0:
            return 0
        
        # Reset pruned slots
        for idx in prune_candidates:
            memory[0, idx] = 0
            importance_scores[0, idx] = 0
            memory_usage[0, idx] = 0
            memory_age[0, idx] = 0
            
        return len(prune_candidates)
```

### Phase C.2: MVoT Visualization Decision Improvements

Enhance the decision mechanism for when to visualize reasoning:

```python
class VisualizationDecisionMechanism:
    """Improved decision mechanism for visualization generation."""
    
    def __init__(self, config):
        self.config = config
        self.visualization_threshold = config.mvot.visualization_threshold
        self.max_visualizations = config.mvot.max_visualizations
        self.min_tokens_between = config.mvot.min_tokens_between_visuals
        self.tokens_since_last = 0
        self.visualizations_generated = 0
        
        # Decision criteria weights
        self.criteria_weights = {
            "entropy": 0.3,        # High text entropy suggests visualization benefit
            "complexity": 0.25,    # Complex concepts benefit from visualization
            "spatial": 0.2,        # Spatial descriptions benefit from visualization
            "reasoning": 0.15,     # Multi-step reasoning benefits from visualization
            "context": 0.1         # Context from previous visualizations
        }
        
        # Decision neural network (simple MLP)
        self.decision_network = nn.Sequential(
            nn.Linear(len(self.criteria_weights), 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def should_visualize(self, hidden_states, token_ids, text_content):
        """Determine if visualization would benefit understanding."""
        # Check basic conditions
        if self.visualizations_generated >= self.max_visualizations:
            return False, {"reason": "max_reached"}
            
        if self.tokens_since_last < self.min_tokens_between:
            return False, {"reason": "too_soon"}
        
        # Calculate criteria scores
        criteria_scores = {
            "entropy": self._calculate_entropy(hidden_states),
            "complexity": self._estimate_complexity(text_content),
            "spatial": self._detect_spatial_language(text_content),
            "reasoning": self._detect_reasoning_steps(text_content),
            "context": self._evaluate_context()
        }
        
        # Apply weights
        weighted_scores = [
            criteria_scores[c] * self.criteria_weights[c]
            for c in self.criteria_weights
        ]
        
        # Make decision using neural network
        decision_input = torch.tensor(weighted_scores, dtype=torch.float32)
        decision_score = self.decision_network(decision_input).item()
        
        # Update counters
        if decision_score > self.visualization_threshold:
            self.visualizations_generated += 1
            self.tokens_since_last = 0
            return True, {"score": decision_score, "criteria": criteria_scores}
        else:
            self.tokens_since_last += len(token_ids)
            return False, {"score": decision_score, "criteria": criteria_scores}
```

### Phase C.3: BLT Entropy Estimation Improvements

Enhance entropy estimation for better patching:

```python
class AdaptiveEntropyEstimator:
    """Adaptive entropy estimator for dynamic patching."""
    
    def __init__(self, config, byte_lm):
        self.config = config
        self.byte_lm = byte_lm
        self.base_threshold = config.blt.entropy_threshold
        self.min_threshold = config.blt.min_entropy_threshold
        self.max_threshold = config.blt.max_entropy_threshold
        
        # Adaptive threshold parameters
        self.adaptation_rate = 0.1
        self.target_patch_ratio = 0.1  # Aim for 10% of bytes to be patch boundaries
        self.current_threshold = self.base_threshold
        self.ema_patch_ratio = self.target_patch_ratio
        self.ema_alpha = 0.1
        
    def estimate_entropy(self, input_bytes):
        """Estimate entropy of byte sequence."""
        with torch.no_grad():
            # Get next-byte probabilities
            probs = self.byte_lm.generate_probs(input_bytes)
            
            # Calculate entropy
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-10),
                dim=-1
            )
            
            return entropy
            
    def update_threshold(self, actual_patch_ratio):
        """Update entropy threshold based on observed patch ratio."""
        # Update EMA
        self.ema_patch_ratio = (1 - self.ema_alpha) * self.ema_patch_ratio + self.ema_alpha * actual_patch_ratio
        
        # Adjust threshold
        if self.ema_patch_ratio > self.target_patch_ratio:
            # Too many patches, increase threshold
            adjustment = self.adaptation_rate * (self.ema_patch_ratio - self.target_patch_ratio) / self.target_patch_ratio
            self.current_threshold += adjustment
        else:
            # Too few patches, decrease threshold
            adjustment = self.adaptation_rate * (self.target_patch_ratio - self.ema_patch_ratio) / self.target_patch_ratio
            self.current_threshold -= adjustment
            
        # Clamp threshold
        self.current_threshold = max(self.min_threshold, min(self.max_threshold, self.current_threshold))
        
        return self.current_threshold
```

### Phase C.4: Transformer² Adaptation Optimizations

Optimize SVD computation for faster adaptation:

```python
class OptimizedSVDAdapter:
    """Optimized SVD adaptation with caching and randomized SVD."""
    
    def __init__(self, config):
        self.config = config
        self.svd_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # SVD computation settings
        self.use_randomized_svd = config.transformer2.use_randomized_svd
        self.n_oversamples = config.transformer2.svd_n_oversamples
        self.n_iter = config.transformer2.svd_n_iter
        
    def decompose_matrix(self, weight, n_components):
        """Decompose weight matrix using efficient SVD computation."""
        # Generate cache key
        matrix_shape = weight.shape
        matrix_sum = weight.sum().item()
        cache_key = f"{matrix_shape}_{matrix_sum:.4f}"
        
        # Check cache
        if cache_key in self.svd_cache:
            self.cache_hits += 1
            return self.svd_cache[cache_key]
            
        self.cache_misses += 1
        
        # Compute SVD
        if self.use_randomized_svd and min(weight.shape) > 128:
            try:
                # Convert to NumPy for randomized SVD
                weight_np = weight.detach().cpu().numpy()
                
                # Use randomized SVD from scikit-learn
                from sklearn.utils.extmath import randomized_svd
                U_np, S_np, Vh_np = randomized_svd(
                    weight_np,
                    n_components=min(n_components, min(weight.shape)),
                    n_oversamples=self.n_oversamples,
                    n_iter=self.n_iter,
                    random_state=42
                )
                
                # Convert back to PyTorch
                U = torch.from_numpy(U_np).to(weight.device)
                S = torch.from_numpy(S_np).to(weight.device)
                Vh = torch.from_numpy(Vh_np).to(weight.device)
                
            except Exception as e:
                # Fallback to standard SVD
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                U = U[:, :min(n_components, U.shape[1])]
                S = S[:min(n_components, S.shape[0])]
                Vh = Vh[:min(n_components, Vh.shape[0]), :]
        else:
            # Use standard PyTorch SVD
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            U = U[:, :min(n_components, U.shape[1])]
            S = S[:min(n_components, S.shape[0])]
            Vh = Vh[:min(n_components, Vh.shape[0]), :]
        
        # Cache result
        self.svd_cache[cache_key] = (U, S, Vh)
        
        # Manage cache size
        if len(self.svd_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(self.svd_cache))
            del self.svd_cache[oldest_key]
            
        return U, S, Vh
```

## Monitoring and Evaluation

### Sleep-Wake Specific Metrics

Create a special set of metrics for sleep-wake cycle evaluation:

```bash
python tools/sleep_wake_eval.py \
    --model_path outputs/neat_100m_initial/checkpoint-5000 \
    --test_set data/synthetic_math/memory_tests.json \
    --output_dir results/sleep_wake_eval
```

Key metrics to track:

1. **Memory Consolidation Efficiency**
   - Percentage of memories successfully transferred from short-term to long-term
   - Average importance score of consolidated memories

2. **Sleep Phase Impact**
   - Performance before vs. after sleep phases
   - Weight change magnitude during sleep phases

3. **Knowledge Retention**
   - Recall accuracy after N wake-sleep cycles
   - Performance on temporal gap problems

### Component-Specific Metrics

Create a dashboard to monitor component-specific metrics:

```bash
# Launch monitoring dashboard
python tools/monitor_training.py \
    --log_dir outputs/neat_100m_initial/logs \
    --port 8080 \
    --component_metrics
```

Key metrics to track:

- **Overall Model**
  - Training loss
  - Validation accuracy
  - Memory usage
  - Training speed (tokens/second)

- **Titans Memory System**
  - Surprise detection frequency
  - Memory update rate
  - Memory utilization
  - Sleep-wake cycle metrics

- **Transformer² Adaptation**
  - SVD computation time
  - Adaptation magnitude
  - Task embedding diversity
  - Cache hit/miss ratio

- **BLT Processor**
  - Entropy distribution
  - Patch size statistics
  - Processing efficiency

- **MVoT Processor**
  - Token discrepancy loss
  - Visualization decisions
  - Modality balance

### Advanced Test Scenarios

#### Memory Consolidation Specific Tests

```bash
python tools/sleep_wake_test.py \
    --model_path outputs/neat_100m_initial/final_model \
    --test_case memory_consolidation \
    --output_dir results/consolidation_test
```

Test scenarios:
1. **Temporal Memory Tests**: Present information, then test recall after N steps
2. **Interference Tests**: Present conflicting information and test memory robustness
3. **Consolidation Tests**: Verify if important memories from wake phase persist after sleep phase
4. **Progressive Learning Tests**: Test if model retains knowledge better with sleep-wake cycles than without

#### Component Interaction Analysis

```bash
python tools/component_interaction.py \
    --model_path outputs/neat_100m_initial/final_model \
    --components "titans,transformer2" \
    --analysis_type "memory_task_correlation" \
    --output_dir results/component_interaction
```

Interactions to analyze:
1. Surprise detection in Titans ↔ Task identification in Transformer²
2. Memory updates in Titans ↔ SVD adaptation in Transformer²
3. Sleep-wake cycles ↔ Task embedding cache refreshing

## Timeline and Milestones

By combining the parallel development tracks, we can optimize the timeline:

| Phase | Milestone | Duration | Tracks |
|-------|-----------|----------|--------|
| **A.1-B.1** | BLT Pre-training + Sleep-Wake Framework | 1 day | A+B |
| **A.2-B.2** | MVoT Pre-training + Titans Integration | 2 days | A+B |
| **A.3-B.3** | Data Generation + Cycle Management | 1 day | A+B |
| **A.4-B.4-C.1** | Configuration + Forward Pass + Titans Optimization | 1 day | A+B+C |
| **A.5-B.5-C.2** | Start Training + Visualization Tools | 5 days | A+B+C |
| **C.3-C.4** | BLT + Transformer² Optimizations | 2 days | C |
| **Testing** | Component and Integration Testing | 1 day | A+B+C |
| **Refinement** | Final Adjustments | 2 days | A+B+C |

**Total Duration: 15 days**

## Contingency Plans

### Training Stability Issues

If the model becomes unstable with sleep-wake integration:

1. **Gradual Integration**
   - Disable sleep-wake for first 3000 steps
   - Begin with shorter sleep phases (50 steps)
   - Gradually increase sleep duration as training stabilizes

2. **Memory Isolation**
   - Implement temporary memory isolation during early sleep phases
   - Prevent critical core model weights from being modified initially

3. **Adaptive Consolidation Rate**
   ```python
   # Add to MemoryConsolidation class
   def _adjust_consolidation_rate(self, loss_value):
       """Dynamically adjust consolidation learning rate based on loss stability."""
       if loss_value > self.max_safe_loss:
           # Reduce learning rate if loss is too high
           self.consolidation_lr *= 0.5
           self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.consolidation_lr)
           logging.warning(f"Reducing consolidation rate to {self.consolidation_lr} due to high loss")
       elif loss_value < self.min_effective_loss:
           # Increase learning rate if loss is too low
           self.consolidation_lr *= 1.2
           self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.consolidation_lr)
           logging.info(f"Increasing consolidation rate to {self.consolidation_lr}")
   ```

### Hardware Resource Constraints

If the 3080Ti's 12GB VRAM becomes a bottleneck:

1. **Dynamic Component Allocation**
   - Implement dynamic component activation based on memory pressure
   - Prioritize Titans memory system and core transformer during consolidation

2. **CPU Offloading for Sleep Phase**
   ```python
   # Add to enter_sleep_phase method
   def enter_sleep_phase(self):
       """Enter sleep phase with memory optimization."""
       # Move non-essential components to CPU during sleep phase
       if hasattr(self.model, 'token_processor'):
           self.model.token_processor.to('cpu')
       if hasattr(self.model, 'byte_processor'):
           self.model.byte_processor.to('cpu')
           
       # Continue with normal sleep phase...
   ```

3. **Gradient Accumulation Increase**
   - Increase gradient accumulation steps during sleep phase
   - Reduce batch size during sleep phase

## Next Steps After Successful Initial Run

After successfully implementing the sleep-wake mechanism in this phase, the next steps will be:

1. **Fine-grained Sleep-Wake Patterns**
   - Implement variable sleep-wake cycle lengths based on task complexity
   - Add REM-like vs. deep-sleep-like phases with different consolidation strategies

2. **Memory Hierarchies**
   - Implement proper memory hierarchies mimicking human memory types:
     - Sensory memory (very short-term)
     - Working memory (short-term)
     - Long-term memory (consolidated during sleep)

3. **Memory-Weight Integration**
   - Develop advanced methods for bidirectional transfer between model weights and memory
   - Implement context-aware memory consolidation

4. **Comprehensive Evaluation Framework**
   - Create test suite for evaluating memory and consolidation
   - Develop metrics to quantify "sleep quality" and its impact on performance

5. **Component Ablation Studies**
   - Perform detailed component ablation studies
   - Measure the contribution of each component to overall performance

This master plan represents a comprehensive approach to implementing and testing the sleep-wake memory consolidation mechanism within the NEAT architecture. By following this structured approach with parallel development tracks, we can efficiently identify and resolve integration issues, establish component stability, and begin tuning the model for optimal performance, all while advancing the architecture with this critical human-like memory consolidation capability.