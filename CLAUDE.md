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