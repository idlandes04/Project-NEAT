# CLAUDE.md - Project NEAT

## Always Do This
- Read @PLAN_MAIN and @TECHNICALd.md OR keep them in context.
- Review current Project NEAT status, determine where we left off if not told, and determine the best path forward to complete the given task and align with the theory in the @TECHNICALd.md
- Your main goal is to complete this project, so think holisticly and deeply about how exactly would be best to integrate all components and create a truly breakthrough algorithm. Be honest and accurate, do not ever sandbag tests or take the easy route. Make effecive and as simple as possible chages, but do not shy away from challges, simply adjsut the steps manually in @PLAN_MD and then continue with the further broken down task if needed. Don't get overwhelemed, stay calm, just code and think hard.
- Use pytest inteigently to verify all components and processes that feasibly can be, creating comprehesive platfrom agnostic robust and efficent tests that do not sandbag or mis use mocks. Solid, professional and smart tests.
- This project will be run on mac m3 apple silicon and windows 11 x86 (read metal_docs.md for details on how to use pytorch with apples metal framework)
- Update progress regulary and anything you learn about the project to this document that you deem very important to remember forever.


## Build & Test Commands
- Run all tests: `python -m pytest tests/`
- Run a single test: `python -m pytest tests/test_components.py::TestName::test_method_name`
- Train model: `python main.py --mode train --use_titans_memory --use_transformer2_adaptation --use_mvot_processor`
- Evaluate model: `python main.py --mode eval --model_path ./outputs/best_model`
- Profile components: `python main.py --mode profile`

## Code Style Guidelines
- Use type hints (from typing import Dict, List, Optional, Union, Any)
- Add docstrings for all modules, classes, and methods
- Imports order: standard library, third-party, local modules
- Use snake_case for variables/methods, CamelCase for classes
- Exception handling: catch specific exceptions
- Python 3.8+ features are allowed
- Class structure: __init__ first, then public methods, then private methods (_method_name)
- Follow PyTorch conventions for tensor operations
- Use absolute imports (from src.utils... not from ..utils...)
- Return dict for model outputs with keys like "logits", "hidden_states"

## Project Progress and Insights

### Titans Memory System Implementation (Completed 1.1.x)
- **Key Insight 1**: Test-time learning in Titans requires careful gradient management and safeguards to prevent destabilizing the model.
- **Key Insight 2**: Memory decay mechanisms need to be adaptive based on context length - longer contexts need slower decay rates.
- **Key Insight 3**: Platform-agnostic code is critical; Apple Silicon (Metal) may have different capabilities for some autograd operations than CUDA.
- **Key Insight 4**: Memory management requires balancing between recency, importance, and surprise factors.
- **Key Insight 5**: Gradient checkpointing significantly reduces memory overhead for long sequence processing.

### Transformer² Adaptation Implementation (Completed 1.2.x)
- **Key Insight 1**: SVD decomposition of weight matrices provides an efficient way to adapt models to specific tasks.
- **Key Insight 2**: Randomized SVD significantly speeds up adaptation for large matrices while maintaining accuracy.
- **Key Insight 3**: Adaptive precision in SVD (varying component count based on matrix properties) optimizes the computation-accuracy tradeoff.
- **Key Insight 4**: Caching decompositions and task embeddings yields dramatic performance improvements for similar inputs.
- **Key Insight 5**: Task embedding similarity matching enables efficient reuse of previous computations without sacrificing adaptation quality.

### BLT Core Implementation (Completed 1.3.x)
- **Key Insight 1**: The byte-level entropy estimator is critical for efficient patching - a small but well-trained model significantly improves patch quality.
- **Key Insight 2**: Position embeddings in the byte LM are essential for capturing contextual patterns in the entropy calculation.
- **Key Insight 3**: Caching processed data for the byte-level training significantly improves iteration speed during development.
- **Key Insight 4**: The entropy threshold is the most critical hyperparameter - it directly controls the computation-accuracy tradeoff.
- **Key Insight 5**: Adaptive loading of pretrained entropy models enables platform-agnostic deployment with graceful fallbacks.
- **Key Insight 6**: Variable-length patch handling with proper masking is essential for efficient batch processing and handling different sequence lengths.
- **Key Insight 7**: Computation budget management through adaptive entropy thresholds enables balancing between accuracy and performance.
- **Key Insight 8**: Comprehensive profiling tools are vital for understanding and optimizing patch-based processing systems.

### MVoT Visual Codebook Integration (Completed 1.4.1)
- **Key Insight 1**: Visual codebook abstractions should support multiple pretrained VQ-VAE model formats (VQVAE, VQGAN, DALL-E) through adapter patterns.
- **Key Insight 2**: Lazy initialization of the visual codebook prevents unnecessary memory usage when multimodal capabilities aren't required.
- **Key Insight 3**: Embedding space conversion between model hidden size and codebook embedding size requires careful handling to maintain quality.
- **Key Insight 4**: Token discrepancy loss calculation can be optimized with batched distance computation rather than element-wise operations.
- **Key Insight 5**: Different VQ-VAE models use inconsistent naming conventions for codebook embeddings, requiring robust key pattern matching.
- **Key Insight 6**: Dimension mismatch between codebook embeddings and model spaces can be handled with interpolation techniques when necessary.

### MVoT Decision Mechanism (Completed 1.4.2)
- **Key Insight 1**: Combining both heuristic and neural approaches for modality decisions creates a more robust system than either approach alone.
- **Key Insight 2**: Spatial and visual keywords are particularly strong indicators of when visualization would be beneficial to reasoning.
- **Key Insight 3**: Context-aware decision logic requires looking at the pattern of previous decisions, not just the current token.
- **Key Insight 4**: Limiting the number of visualizations prevents overuse and focuses them on truly beneficial reasoning steps.
- **Key Insight 5**: Converting between boolean logic and continuous scores requires careful handling to avoid loss of information.
- **Key Insight 6**: Explicit diagram requests should override other heuristics, as they represent direct user intent for visualization.

### BLT-MVoT Integration (Completed 1.4.3)
- **Key Insight 1**: Bidirectional conversion between byte-level and token-level representations requires careful alignment of embedding spaces.
- **Key Insight 2**: Vector normalization before computing similarity metrics drastically improves the quality of round-trip conversions.
- **Key Insight 3**: The conversion process should be modular and independent of the specific BLT or MVoT implementation details.
- **Key Insight 4**: Using an intermediate dimensional space larger than both source and target spaces improves conversion fidelity.
- **Key Insight 5**: Layer normalization is essential for stabilizing the training dynamics of conversion networks.
- **Key Insight 6**: Quality assessment of conversions through cosine similarity provides an effective measure for monitoring fidelity loss.

### Implementation Patterns
- Use torch.utils.checkpoint.checkpoint for memory-efficient gradient computation
- Implement fallback mechanisms when operations aren't supported on all platforms
- Use adaptive parameters that adjust based on training/inference mode
- Track usage statistics to inform memory management decisions
- Implement platform-specific device detection and handling
- Cache expensive computations (SVD, task embeddings) with appropriate invalidation strategies
- Use similarity metrics for content-based lookup in caches
- Apply pruning mechanisms based on recency and frequency for memory management
- Use randomized algorithms when appropriate to speed up processing of large matrices