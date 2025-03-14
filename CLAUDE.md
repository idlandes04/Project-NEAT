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

### Current State Assessment (Updated: 2025-03-13)
We've successfully completed all phases through 2.2.3, implementing a comprehensive test-time learning system with coordinated gradient computation, adaptive learning rate management, and optimization quality monitoring. All core architectural components are fully implemented and integrated through robust cross-component communication.

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

These achievements demonstrate our key architectural innovations: loose coupling with controlled interactions, efficient cross-component communication, coordinated behavior through feedback mechanisms, synchronized test-time learning, and robust stability management for test-time adaptation.

### Lessons Learned from Implementation
1. **Component Architecture**:
   - Loose coupling with well-defined interfaces enables independent development and testing
   - Message-based communication provides flexibility for evolving component needs
   - Component-specific optimizations with shared infrastructure balance specialization and code reuse

2. **Test-Time Learning**:
   - Coordinated gradient computation requires careful threading and synchronization
   - Different components require tailored learning strategies for optimal performance
   - Stability monitoring with multiple metrics enables early detection of potential issues
   - Parameter backup and emergency recovery mechanisms are essential safeguards

3. **Integration Challenges**:
   - Thread synchronization requires timeout mechanisms to prevent deadlocks
   - Byte-level and token-level representations need careful dimension alignment
   - Cross-component feedback loops must be carefully designed to prevent oscillation
   - Platform-specific code requires robust fallback mechanisms for cross-platform compatibility

4. **Testing and Validation**:
   - Comprehensive tests with appropriate timeouts are crucial for complex systems
   - Mock implementations enable testing components in isolation
   - Floating-point precision handling requires special attention in test assertions
   - Component tests must cover both normal operation and emergency recovery scenarios

### Current Focus: Hardware-Aware Integration (Phase 2.3.x)
Building on our successful test-time learning framework, we've begun Phase 2.3.x focusing on hardware-aware integration to optimize performance across different platforms. This phase consists of three main components:

1. **Component-Specific Resource Allocation** (Task 2.3.1):
   - Dynamic memory budgeting across components based on priority and needs
   - Computation distribution that allocates resources to the most critical components
   - Adaptive precision selection that balances accuracy and performance
   - Platform-specific optimizations for Metal (Apple Silicon) and CUDA (NVIDIA GPUs)

2. **Latency-Aware Component Scheduling** (Task 2.3.2):
   - Priority-based execution scheduling that minimizes waiting time
   - Parallelization opportunity identification for concurrent execution
   - Adaptive batching based on component characteristics and hardware capabilities
   - Execution pipeline optimization to minimize idle time

3. **Target Hardware Optimization Profiles** (Task 2.3.3):
   - Hardware-specific profiles for different target platforms
   - Dynamic feature detection to identify hardware capabilities
   - Fallback mechanisms for graceful degradation on limited hardware
   - Performance benchmarking to measure optimization effectiveness

These efforts will enable Project NEAT to efficiently utilize available hardware resources while maintaining consistent behavior across platforms, ultimately improving both performance and scalability.

### Long-Term Vision
Our ultimate goal remains demonstrating that coordinated components offer superior efficiency and capability compared to monolithic scaling. We aim to prove that:

1. **Efficient Computation**: Dynamic component activation saves computation while maintaining quality
2. **Improved Generalization**: Test-time learning provides better generalization on long contexts and novel tasks
3. **Synergistic Interaction**: Cross-component communication improves overall system performance
4. **Versatility**: The combined system handles a wider array of tasks than any single component alone

By completing Phase 2.1 and moving into Phase 2.2, we're making significant progress toward realizing this vision. The successful integration of all four components and their communication systems creates a solid foundation for the more sophisticated test-time learning coordination we'll implement next.

As we continue development, we'll maintain our focus on both theoretical soundness and practical implementation, ensuring that Project NEAT demonstrates clear advantages over conventional approaches while remaining deployable on diverse hardware platforms.