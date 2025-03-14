# CLAUDE.md - Project NEAT

## Always Do This
- Read @PLAN_MAIN and @TECHNICALd.md OR keep them in context.
- Review current Project NEAT status, determine where we left off if not told, and determine the best path forward to complete the given task and align with the theory in the @TECHNICALd.md
- Your main goal is to complete this project, so think holisticly and deeply about how exactly would be best to integrate all components and create a truly breakthrough algorithm. Be honest and accurate, do not ever sandbag tests or take the easy route. Make effecive and as simple as possible chages, but do not shy away from challges, simply adjsut the steps manually in @PLAN_MD and then continue with the further broken down task if needed. Don't get overwhelemed, stay calm, just code and think hard.
- Use pytest inteigently to verify all components and processes that feasibly can be, creating comprehesive platfrom agnostic robust and efficent tests that do not sandbag or mis use mocks. Solid, professional and smart tests.
- This project will be run on mac m3 apple silicon and windows 11 x86 (read metal_docs.md for details on how to use pytorch with apples metal framework)
- Update progress regulary and anything you learn about the project to this document that you deem very important to remember forever.


## Build & Test Commands
- Run all tests: `python3 -m pytest tests/`
- Run a single test: `python3 -m pytest tests/test_components.py::TestName::test_method_name`
- Train model: `python3 main.py --mode train --use_titans_memory --use_transformer2_adaptation --use_mvot_processor`
- Evaluate model: `python3 main.py --mode eval --model_path ./outputs/best_model`
- Profile components: `python3 main.py --mode profile`

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

### Cross-Component Communication (Completed 2.1.x)
- **Key Insight 1**: A message-based pub-sub architecture provides flexibility for loosely coupled components to communicate.
- **Key Insight 2**: Priority-based message processing ensures critical information is processed first during heavy computational loads.
- **Key Insight 3**: State tracking with subscriptions enables components to react to relevant state changes without tight coupling.
- **Key Insight 4**: Task-memory correlation enables more efficient memory allocation by prioritizing regions associated with the current task.
- **Key Insight 5**: Surprise-driven adaptation creates a feedback loop that dynamically adjusts model parameters where uncertainty is high.
- **Key Insight 6**: Bidirectional modality feedback enables coordinated processing between text and visual representations.
- **Key Insight 7**: Separating message types from content allows extensibility while maintaining a structured communication protocol.
- **Key Insight 8**: Decentralized state management with centralized synchronization balances component autonomy with system coherence.

### Test-Time Learning Coordination (Completed 2.2.1)
- **Key Insight 1**: Coordinated gradient computation across components requires centralized management with component-specific customization.
- **Key Insight 2**: Context managers provide an elegant way to manage gradient computation lifecycles and resource cleanup.
- **Key Insight 3**: Gradient priority levels allow the system to focus computational resources on the most important learning tasks.
- **Key Insight 4**: Cross-component gradient isolation layers enable controlled information flow between components during backpropagation.
- **Key Insight 5**: Platform-agnostic gradient computation requires fallback mechanisms for operations not supported in all environments (e.g., Metal vs. CUDA).
- **Key Insight 6**: Memory pressure management through selective parameter offloading significantly reduces memory usage during gradient computation.
- **Key Insight 7**: Thread-safe gradient coordination is essential for preventing deadlocks and race conditions in concurrent learning scenarios.
- **Key Insight 8**: Component-specific gradient managers simplify integration by providing tailored interfaces to the shared coordination system.

### Adaptive Learning Rate Management (Completed 2.2.2)
- **Key Insight 1**: Different components have unique learning dynamics that require specialized learning rate schedules.
- **Key Insight 2**: Multiple stability metrics (loss trends, gradient norms, update ratios) together provide a comprehensive view of learning stability.
- **Key Insight 3**: Scheduled learning rates (cosine decay) work well for predictable tasks, while adaptive schedules are better for unpredictable tasks.
- **Key Insight 4**: Parameter backup and restoration enables recovery from catastrophic failures during test-time learning.
- **Key Insight 5**: Stability factors that dynamically adjust learning rates based on metrics prevent divergence while maintaining learning efficiency.
- **Key Insight 6**: Learning rate synchronization across components is essential to prevent oscillation and conflicting updates.
- **Key Insight 7**: Emergency recovery mechanisms need staged responses proportional to the severity of the stability issue.
- **Key Insight 8**: Thread-safe learning rate management is essential for concurrent optimization of different components.
- **Key Insight 9**: Forward/backward progress tracking provides an early warning system for divergent optimization trajectories.
- **Key Insight 10**: Special case handling for initialization phases prevents false stability warnings during early learning.

### Test-Time Optimization Monitoring (Completed 2.2.3)
- **Key Insight 1**: Optimization quality requires distinct metrics from stability - efficiency and effectiveness rather than just stability.
- **Key Insight 2**: Computing time and memory usage during optimization need to be balanced against quality improvements.
- **Key Insight 3**: The ratio of parameter update magnitude to gradient magnitude is a critical indicator of optimization health.
- **Key Insight 4**: Forward vs. backward progress counts provide a simple but effective measure of overall optimization trajectory.
- **Key Insight 5**: Component-specific correction mechanisms need to be tailored to the component's particular failure modes.
- **Key Insight 6**: Gradient angle history (cosine similarity between consecutive gradients) effectively detects oscillation during learning.
- **Key Insight 7**: Optimization quality assessment can be automated with carefully designed scoring functions.
- **Key Insight 8**: Cross-component correction requires coordinated, proportional responses to maintain balanced learning.
- **Key Insight 9**: Floating-point precision management is critical for consistent optimization quality assessment.
- **Key Insight 10**: Parameter reset capabilities provide an essential emergency mechanism when incremental corrections fail.

### Component-Specific Resource Allocation (Completed 2.3.1)
- **Key Insight 1**: Memory pressure-aware allocation prevents OOM errors by dynamically redistributing resources.
- **Key Insight 2**: Prioritizing critical components during resource constraints maintains core functionality.
- **Key Insight 3**: Component importance scoring requires both static design-time metrics and dynamic runtime feedback.
- **Key Insight 4**: Thread-safe resource management is essential for preventing deadlocks in component resource requests.
- **Key Insight 5**: Memory resource requests should be flexible with minimum viable allocations to handle pressure.
- **Key Insight 6**: Monitoring pressure trends enables proactive resource reallocation before critical situations occur.
- **Key Insight 7**: Unified resource APIs across CPU and GPU simplifies component implementations.
- **Key Insight 8**: Component-specific optimal precision varies by operation type, not just by component.
- **Key Insight 9**: Resource-aware architecture allows centralized decisions that individual components cannot make.
- **Key Insight 10**: Memory, compute, and precision resources have different allocation patterns and require different APIs.

### Hardware-Aware Integration (In Progress 2.3.x)
- **Key Insight 1**: Different components have distinct memory and computation requirements that vary dynamically during execution.
- **Key Insight 2**: Priority-based resource allocation enables focusing limited resources on the most important components.
- **Key Insight 3**: Hardware-specific optimizations with fallback paths are essential for consistent cross-platform performance.
- **Key Insight 4**: Adaptive precision selection enables balancing accuracy and performance based on hardware capabilities.
- **Key Insight 5**: Component dependency analysis enables identifying parallelization opportunities for efficient execution.
- **Key Insight 6**: Dynamic batch sizing based on component characteristics improves throughput on diverse hardware.
- **Key Insight 7**: Execution pipeline optimization minimizes idle time by coordinating component scheduling.
- **Key Insight 8**: Hardware profiling and fingerprinting enable selecting the most appropriate optimization strategies.

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
- Use message-based pub-sub architecture for cross-component communication
- Implement state management with subscriptions for loose coupling
- Employ priority-based message handling to focus on critical information first
- Use context managers for coordinated gradient computation across components
- Implement component-specific gradient management with shared infrastructure
- Apply gradient isolation layers for controlled cross-component gradient flow
- Use adaptive precision and priority for gradient computation to optimize memory usage
- Implement parameter backup and restoration mechanisms for emergency recovery
- Use multiple metrics (loss trends, gradient norms) to assess learning stability
- Apply stability-aware learning rate scheduling based on monitored metrics
- Implement different scheduler types (cosine, adaptive) for different learning scenarios
- Use thread-safe locks for all shared state in learning-related components
- Track optimization quality with comprehensive metrics and scoring functions
- Apply correction mechanisms proportional to the severity of detected issues
- Implement cross-component coordination for learning rate synchronization
- Use automatic quality assessment and correction for self-adaptive learning

## Project Reflections and Planning

### Current State Assessment (Updated: 2025-03-15)
We've successfully completed all phases through 2.3.2, implementing a comprehensive platform-agnostic architecture with robust test-time learning, cross-component communication, and hardware-aware resource management. All tests now pass across diverse environments, demonstrating the robustness of our implementation to handle varying hardware configurations from Apple Silicon to NVIDIA GPUs to systems without dedicated graphics acceleration.

Key achievements to date include:

1. **Core Components**:
   - **Titans Memory System**: Long-context memory with adaptive decay and test-time learning (1.1.x)
   - **Transformer² Adaptation**: SVD-based weight adaptation with task embedding matching (1.2.x)
   - **BLT Core**: Byte-level entropy estimation and variable-length patch handling (1.3.x)
   - **MVoT Processing**: Visual codebook integration, modality decisions, and byte-token mapping (1.4.x)

2. **Component Integration**:
   - **Robust Messaging System**: Priority-based pub/sub architecture for component communication (2.1.1)
   - **State Management**: Centralized state tracking with deadlock prevention (2.1.2-2.1.3)
   - **Feedback Loops**: Three essential feedback mechanisms (2.1.2):
     - Task-Memory correlation for efficient memory allocation
     - Surprise-driven adaptation for dynamic parameter adjustment
     - Multimodal coordination between text and visual representations
   - **Cross-Component Integration**: Verified coherent operation of all components (2.1.3)

3. **Test-Time Learning Framework**:
   - **Coordinated Gradient Computation**: Centralized system for test-time learning (2.2.1)
   - **Gradient Isolation**: Controlled gradient flow between components (2.2.1)
   - **Memory-Efficient Learning**: Optimized computation with checkpointing and parameter offloading (2.2.1)
   - **Adaptive Learning Rate Management**: Component-specific scheduling with stability monitoring (2.2.2)
   - **Emergency Stabilization**: Detection and recovery from learning instability (2.2.2)
   - **Optimization Quality Monitoring**: Comprehensive quality assessment system (2.2.3)
   - **Cross-Component Correction**: Coordinated correction mechanisms for harmonious learning (2.2.3)

4. **Hardware-Aware Integration**:
   - **Component Resource Management**: Dynamic resource allocation based on component importance (2.3.1)
   - **Platform-Agnostic Implementation**: Robust fallback paths ensuring compatibility across hardware (2.3.1)
   - **Memory Pressure Adaptation**: Automatic component deactivation under system constraints (2.3.2)
   - **Hardware Capability Detection**: Intelligent feature detection enabling optimal hardware utilization (2.3.2)
   - **End-to-End Integration Testing**: Comprehensive test suite validating the full system pipeline (2.3.2)
   - **Cross-Platform Testing**: Verified operation on both Apple Silicon and x86 architectures (2.3.2)

These achievements represent significant progress in our architectural vision: loose coupling with controlled interactions, efficient cross-component communication, coordinated behavior through feedback mechanisms, synchronized test-time learning, and robust hardware-aware resource management.

### Lessons Learned from Implementation
1. **Component Architecture**:
   - Loose coupling with well-defined interfaces enables independent development and testing
   - Message-based communication provides flexibility for evolving component needs
   - Component-specific optimizations with shared infrastructure balance specialization and code reuse
   - Graceful fallbacks and exception handling enable robust operation across diverse environments

2. **Test-Time Learning**:
   - Coordinated gradient computation requires careful threading and synchronization
   - Different components require tailored learning strategies for optimal performance
   - Stability monitoring with multiple metrics enables early detection of potential issues
   - Parameter backup and emergency recovery mechanisms are essential safeguards
   - Thread-safe resource management prevents deadlocks during concurrent optimization

3. **Integration Challenges**:
   - Thread synchronization requires timeout mechanisms to prevent deadlocks
   - Byte-level and token-level representations need careful dimension alignment
   - Cross-component feedback loops must be carefully designed to prevent oscillation
   - Platform-specific code requires robust fallback mechanisms for cross-platform compatibility
   - Recursive calls between components can create infinite loops if not carefully managed

4. **Testing and Validation**:
   - Comprehensive tests with appropriate timeouts are crucial for complex systems
   - Mock implementations enable testing components in isolation
   - Floating-point precision handling requires special attention in test assertions
   - Component tests must cover both normal operation and emergency recovery scenarios
   - End-to-end tests require carefully designed setup methods to ensure proper component initialization
   - Integration tests help identify interface mismatches and incompatibilities between components
   - Platform-specific tests should have robust fallbacks for environments without required hardware
   - Hardware-aware tests should be capable of running in diverse environments without skipping

5. **Hardware Adaptation Insights**:
   - Dynamic memory allocation based on system capabilities ensures optimal resource utilization
   - Auto-detection of hardware features enables intelligent scaling of component activation
   - Proper tensor shape management and SVD adaptation padding prevents runtime errors
   - Two-pass inference optimization requires careful parameter handling to prevent infinite recursion
   - Unified architecture model implementations must handle diverse access patterns to internal components
   - Resource management must be robust to both high-end hardware and resource-constrained environments
   - Priority-based component activation ensures critical functionality remains available under pressure

### Current Focus: Execution Scheduling Optimization (Phase 2.3.3)
Building on our successful hardware-aware integration framework, we've completed Tasks 2.3.1 and 2.3.2, establishing a robust foundation for efficient resource utilization. Our current focus is on the final phase of hardware-aware integration, targeting execution scheduling optimization:

1. **Component-Specific Resource Allocation** (Task 2.3.1) - ✅ COMPLETED:
   - Dynamic memory budgeting across components based on priority and needs
   - Computation distribution that allocates resources to the most critical components
   - Adaptive precision selection that balances accuracy and performance
   - Platform-specific optimizations for Metal (Apple Silicon) and CUDA (NVIDIA GPUs)
   - Comprehensive end-to-end integration tests validating resource management

2. **Hardware Capability Adaptation** (Task 2.3.2) - ✅ COMPLETED:
   - Automatic hardware feature detection for optimal component activation
   - Memory pressure monitoring with dynamic component deactivation
   - Graceful fallbacks for environments without advanced hardware features
   - Cross-platform compatibility with robust error handling
   - Tensor shape management preventing runtime errors during adaptation

3. **Execution Scheduling Optimization** (Task 2.3.3) - CURRENT FOCUS:
   - Priority-based execution scheduling minimizing waiting time
   - Parallelization opportunity identification for concurrent execution
   - Adaptive batching based on component characteristics and available hardware
   - Pipeline optimization to minimize idle time during execution
   - Performance benchmarking to measure optimization effectiveness

These efforts will enable Project NEAT to not only efficiently utilize available hardware resources but also intelligently schedule component execution for maximum throughput, ultimately improving both performance and scalability across diverse computing environments.

### Long-Term Vision and Next Steps
Our ultimate goal remains demonstrating that coordinated components offer superior efficiency and capability compared to monolithic scaling. We aim to prove that:

1. **Efficient Computation**: Dynamic component activation saves computation while maintaining quality
2. **Improved Generalization**: Test-time learning provides better generalization on long contexts and novel tasks
3. **Synergistic Interaction**: Cross-component communication improves overall system performance
4. **Versatility**: The combined system handles a wider array of tasks than any single component alone
5. **Hardware Adaptability**: Intelligent resource allocation enables deployment across diverse environments

After completing Phase 2.3.3, our next major focus will be comprehensive evaluation and benchmarking (Phase 3.x):

1. **Baseline Comparison** (Phase 3.1):
   - Establish monolithic baseline models for fair comparison
   - Implement common evaluation tasks spanning component capabilities
   - Create metrics for measuring computation efficiency and quality tradeoffs
   
2. **Comprehensive Benchmarking** (Phase 3.2):
   - Evaluate performance across diverse hardware environments
   - Measure quality-compute tradeoffs with various component configurations
   - Assess generalization to novel tasks and long-context scenarios
   
3. **Deployment Optimization** (Phase 3.3):
   - Fine-tune component activation thresholds for optimal deployment
   - Create environment-specific optimization profiles
   - Develop intelligent auto-configuration for new deployment targets

As we prepare to complete the hardware-aware integration phase, we're excited by the robustness and adaptability our system demonstrates, further validating our architectural approach to efficient, component-based neural processing.

project-neat/
├── src/
│   ├── components/
│   │   ├── titans/                   # Titans memory system
│   │   │   └── memory_system.py      # Three-tiered memory implementation
│   │   ├── transformer2/             # Transformer² adaptation
│   │   │   └── adaptation.py         # SVD-based weight adaptation
│   │   ├── mvot/                     # MVoT token processor
│   │   │   ├── visual_codebook.py    # VQ-VAE integration
│   │   │   ├── token_processor.py    # Multimodal token processing
│   │   │   ├── decision/             # Decision mechanisms
│   │   │   └── mapping/              # Byte-token mapping
│   │   ├── blt/                      # BLT byte processor
│   │   │   ├── byte_processor.py     # Entropy-based patching
│   │   │   └── entropy_estimator_trainer.py # Byte-level entropy estimation
│   │   ├── feedback/                 # Component feedback mechanisms
│   │   │   ├── task_memory_feedback.py # Task-memory correlation
│   │   │   ├── adaptation_feedback.py  # Surprise-driven adaptation
│   │   │   └── modality_feedback.py    # Modality feedback
│   │   ├── learning/                 # Learning components
│   │   │   ├── adaptive_learning_rate.py # Dynamic learning rate management
│   │   │   ├── gradient_coordination.py  # Cross-component gradient management
│   │   │   └── optimization_monitoring.py # Test-time optimization quality
│   │   └── messaging/                # Component communication
│   │       ├── message_protocol.py   # Message passing system
│   │       └── component_state.py    # Component state tracking
│   ├── models/                       # Unified architecture
│   │   ├── unified_architecture.py   # Main architecture integration
│   │   ├── unified_architecture_resource_adapter.py # Resource-aware architecture
│   │   └── transformer.py            # Base transformer implementation
│   ├── trainers/                     # Training infrastructure
│   │   └── hardware_aware_trainer.py # Platform-specific training
│   └── utils/                        # Utility functions
│       ├── config.py                 # Configuration handling
│       ├── memory_optimization.py    # Memory usage optimization
│       └── component_resource_management.py # Component-specific resource allocation
├── tests/                            # Test cases
│   ├── test_components.py            # Component-level tests
│   ├── test_integration.py           # Integration tests
│   ├── test_learning.py              # Learning system tests
│   ├── test_feedback.py              # Feedback mechanism tests
│   ├── test_messaging.py             # Messaging system tests
│   ├── test_component_resource_management.py # Resource management tests
│   └── test_resource_aware_architecture.py  # Resource-aware architecture tests
├── docs/                             # Documentation
│   ├── PLAN_MAIN.MD                  # Project planning document
│   ├── TECHNICALd.md                 # Technical details and theory
│   ├── phase2.2.2-3_plan.md          # Hardware-aware integration plan
│   └── metal_docs.md                 # Apple Metal framework integration
├── scripts/                          # Helper scripts
│   └── train_byte_lm.py              # BLT training script
└── main.py                           # Main script
```