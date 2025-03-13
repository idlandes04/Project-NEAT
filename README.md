# Project NEAT: Neural Enhancement Architecture Toolkit

This project implements a unified neural architecture that integrates four cutting-edge techniques:

1. **Titans**: Learning to Memorize at Test Time
2. **Transformer²**: Self-Adaptive LLMs
3. **MVoT**: Multimodal Visualization-of-Thought
4. **BLT**: Byte Latent Transformer

## Current Status

### Completed Components:
- ✅ **Titans Memory System** (1.1.x): Test-time learning with surprise-based memory updates
- ✅ **Transformer² Adaptation** (1.2.x): SVD-based weight adaptation with efficient caching
- ✅ **BLT Core Implementation** (1.3.x): Entropy-based byte patching with variable-length handling
- ✅ **MVoT Visual Codebook** (1.4.1): Framework for loading pretrained VQ-VAE models

### In Progress:
- 🔄 **MVoT Decision Mechanism** (1.4.2): Building the text/image generation decision system
- 🔄 **MVoT Token Mapping** (1.4.3): Creating byte-to-token mappings for BLT compatibility

### Coming Soon:
- 📅 **Cross-Component Integration** (2.x): Connecting all components together
- 📅 **Test-Time Learning Synchronization** (2.2.x): Coordinating learning across components
- 📅 **Testing Infrastructure** (3.x): Comprehensive evaluation framework

The implementation is optimized for both Apple Silicon (M-series) and NVIDIA GPUs, with platform-specific optimizations and fallback mechanisms.

## Architecture Overview

The unified architecture combines these components in a modular way:

```
[Input] → [Byte Processor (BLT)] → [Memory System (Titans)] → [Token Processor (MVoT)] → [Output]
                                      ↑
                          [Task Adaptation (Transformer²)]
```

Each component can be selectively enabled or disabled, and the system includes dynamic resource allocation to optimize performance based on available hardware.

## Key Features

- **Modular Component Design**: Each architecture is implemented as a swappable module
- **Hardware-Aware Optimization**: Dynamic tensor offloading, mixed precision training, gradient checkpointing
- **Dynamic Component Activation**: Components are activated based on input complexity and available resources
- **Comprehensive Profiling**: Performance metrics for each component and combination

## Project Structure

```
neural_architecture_integration/
├── src/
│   ├── components/
│   │   ├── titans/         # Titans memory system
│   │   ├── transformer2/   # Transformer² adaptation
│   │   ├── mvot/           # MVoT token processor
│   │   └── blt/            # BLT byte processor
│   ├── models/             # Unified architecture
│   ├── trainers/           # Hardware-aware trainer
│   └── utils/              # Utility functions
├── tests/                  # Test cases
└── main.py                 # Main script
└── PLAN_MAIN.md            # The plan for Cline and Roo to follow
└── TECHNICALd.md           # The back-bone technical details and basal theory for Cline and Roo to follow
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-architecture-integration.git
cd neural-architecture-integration

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train a model with all components enabled:

```bash
python main.py --mode train \
    --use_titans_memory \
    --use_transformer2_adaptation \
    --use_mvot_processor \
    --use_blt_processor \
    --mixed_precision \
    --gradient_checkpointing \
    --dynamic_component_activation \
    --batch_size 8 \
    --max_steps 1000 \
    --output_dir ./outputs
```

### Evaluation

To evaluate a trained model:

```bash
python main.py --mode eval \
    --model_path ./outputs/best_model \
    --batch_size 16
```

### Component Profiling

To profile the performance of individual components:

```bash
python main.py --mode profile \
    --use_titans_memory \
    --use_transformer2_adaptation \
    --use_mvot_processor \
    --use_blt_processor
```

## Component Details

### Titans Memory System

The Titans memory system implements three types of memory:

1. **Short-term Memory**: Window-based attention over recent tokens
2. **Long-term Memory**: Surprise-based memory updates
3. **Persistent Memory**: Task-agnostic knowledge

Key implementation details:
- Gradient-based surprise measurement
- Selective memory updates based on surprise threshold
- Memory decay mechanism

### Transformer² Adaptation

The Transformer² adaptation implements:

1. **Two-pass Inference**: First pass for task identification, second pass for adaptation
2. **Singular Value Fine-tuning (SVF)**: Adapting only singular values of weight matrices

Key implementation details:
- SVD-based weight adaptation
- Task dispatcher for identifying tasks
- Expert composition for adapting to specific tasks

### MVoT Token Processor

The MVoT token processor implements:

1. **Multimodal Token Processing**: Processing text and image tokens
2. **Token Discrepancy Loss**: Measuring discrepancy between predicted token distributions and codebook embeddings

Key implementation details:
- Interleaved text and image token generation
- Codebook-based visual token representation
- Token discrepancy loss for training

### BLT Byte Processor

The BLT byte processor implements:

1. **Entropy-based Patching**: Creating patches based on byte entropy
2. **Local-global-local Architecture**: Local encoder, latent transformer, and local decoder

Key implementation details:
- Entropy calculation for dynamic patching
- Local encoder for processing individual patches
- Latent transformer for global processing
- Local decoder for generating bytes

## Hardware Optimization

The implementation includes several hardware optimization techniques:

1. **Mixed Precision Training**: Using FP16/BF16 computation with FP32 master weights
2. **Gradient Checkpointing**: Selective activation storage to reduce memory footprint
3. **CPU Offloading**: Moving large tensors to CPU when not in use
4. **Dynamic Batch Sizing**: Adjusting batch size based on available memory
5. **Parallel Data Processing**: Utilizing all CPU cores for data preprocessing

## Parameter Tuning

The system supports comprehensive parameter tuning through command-line arguments or configuration files. Key parameters include:

- **Model Size**: Hidden size, number of layers, number of attention heads
- **Component Activation**: Which components to enable/disable
- **Memory Parameters**: Window size, memory size, surprise threshold
- **Adaptation Parameters**: Number of tasks, number of singular values
- **Hardware Optimization**: Mixed precision, gradient checkpointing, CPU offloading

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project implements ideas from the following papers:

- "Titans: Learning to Memorize at Test Time"
- "TRANSFORMER2: SELF-ADAPTIVE LLMS"
- "Imagine while Reasoning in Space: Multimodal Visualization-of-Thought"
- "Byte Latent Transformer: Patches Scale Better Than Tokens"

Below is a consolidated reference of *exact (or near-exact)* core formulas, definitions, and pseudo-code from the four papers (Titans, Transformer2, MVoT, and BLT) as gleaned from their text. Where the authors only summarized or hinted at ideas, I’ve provided those summaries. When they provided explicit math, I’ve listed their equations closely to the original. If you see any small formatting differences, that’s just to fit them in this response.

**Master Overview** that merges (1) *Titans: Learning to Memorize at Test Time*, (2) *Transformer2: Self-Adaptive LLMs*, (3) *Multimodal Visualization-of-Thought (MVoT)*, and (4) *Byte Latent Transformer (BLT)* into **one cohesive reference**. It highlights all the essential equations, methods, and pseudo-code details each paper provides (based on the publicly available text), so you can reconstruct them or combine them into a single model.

---

## 1. **Titans**  
**Paper:** “*Titans: Learning to Memorize at Test Time*”  
**Core Contributions:**  
- A **long-term neural memory** module to complement short-term attention-based memory.  
- **Surprise-driven** updates—measured by gradient magnitudes w.r.t. inputs—determine when to store or decay new information.  
- **Fast parallelizable training** approach for memory updates.

### 1.1 Memory in RNN or Transformer Format

They unify an RNN-style memory perspective:
\[
M_{t} = f(M_{t-1}, x_{t}), \quad y_{t} = g(M_{t}, x_{t}),
\]
- \(M_{t}\) is the memory or hidden state at time \(t\).  
- \(x_{t}\) is the input; \(y_{t}\) is the output.

For Transformers, they equate “growing memory” with appending Key/Value pairs \((K_{t}, V_{t})\) in the context window. However, they propose a new **neural memory** that can store beyond the immediate window.

### 1.2 Surprise-Driven Memorization & Decay

- **Associative Memory Loss**:  
  They mention measuring “surprise” by gradient magnitude \(\bigl\|\tfrac{\partial \mathcal{L}_{\text{assoc}}}{\partial x}\bigr\|\). High surprise triggers stronger memorization of the data.  

- **Decay Mechanism**:  
  A forgetting or decay step to keep memory from overgrowing:
  \[
    M \leftarrow (1 - \alpha)\,M,
  \]
  with \(\alpha\) possibly determined by how large the memory already is, or how surprising the new data is.

### 1.3 The Titans Architecture

They define **three “hyper-heads”**:

1. **Core** (Short-Term Memory): Standard Transformer attention, limited by context length.  
2. **Long-Term Memory**: A deep module that can store historical or surprising info beyond the context window.  
3. **Persistent Memory**: A set of learnable, task-agnostic parameters (like universal knowledge weights).

They experiment with different ways to incorporate the long-term module (e.g., treat it as extra context, insert it as a special layer, or gate it). All else is conceptual: partial updates at test-time if a surprising event is detected, otherwise memory decays.

---

## 2. **Transformer2**  
**Paper:** “*TRANSFORMER2: SELF-ADAPTIVE LLMS*”  
**Core Contributions:**  
- A **two-pass** inference pipeline.  
- **Singular Value Fine-Tuning (SVF)** to adapt an LLM to new tasks in real time using minimal extra parameters.

### 2.1 Two-Pass Inference

1. **First Pass** (“Dispatch”):
   - The model sees the prompt, partially processes it, and infers task properties.
   - A “dispatch system” decides which “expert vectors” or singular values might be relevant for the second pass.

2. **Second Pass** (“Expert Composition”):
   - The model merges specialized “expert vectors” into the base model via singular value offsets, refining the final forward pass for that specific domain or task.

### 2.2 Singular Value Fine-Tuning (SVF)

- For each weight matrix \(W\), do an SVD: \(\;W = U\,\mathrm{diag}(\sigma)\,V^\top.\)  
- **Only** \(\sigma\) is updated or replaced with a specialized vector \(\sigma_{\text{expert}}\).  
- The model can mix multiple sets of singular values if it identifies multiple relevant tasks.  
- No explicit formula for the RL policy is given, but conceptually:
  \[
    W_{\text{adapted}} = U\,\mathrm{diag}(\sigma_{\text{base}} + \Delta\sigma)\,V^\top,
  \]
  where \(\Delta\sigma\) is the learned offset or the new singular values for a specific skill domain.

---

## 3. **Multimodal Visualization-of-Thought (MVoT)**  
**Paper:** “*Imagine while Reasoning in Space: Multimodal Visualization-of-Thought*”  
**Core Contributions:**  
- Extends chain-of-thought to **interleave image and text** as intermediate steps.  
- Adds a **token discrepancy loss** to improve the fidelity of generated images.

### 3.1 Interleaved Text–Image Reasoning

They factorize the generation:

\[
v_{i} \sim P_{\theta}\bigl(v_{i} \mid z_{1},v_{1}, \dots, z_{i}\bigr),\quad
z_{i+1} \sim P_{\theta}\bigl(z_{i+1} \mid x,\; z_{1},v_{1}, \dots, z_{i},v_{i}\bigr),
\]

So step \(i\) might produce an image \(v_i\), next step might produce text \(z_{i+1}\), and so on.

### 3.2 Token Discrepancy Loss

- Suppose the model uses a discrete image tokenizer with a codebook \(\{e^k_{\mathrm{vis}}\}_{k=1}^N\).  
- Let \(S_{t_{\mathrm{vis}}^i}\) measure MSE distance between the ground-truth image embedding and every codebook embedding:

  \[
    S_{t_{\mathrm{vis}}^i} = \Bigl[\mathrm{MSE}\bigl(e_{\mathrm{vis}}^i, e_{\mathrm{vis}}^1\bigr), \dots, \mathrm{MSE}\bigl(e_{\mathrm{vis}}^i, e_{\mathrm{vis}}^N\bigr)\Bigr].
  \]

- If \(P(t_i)\) is the predicted distribution over those image codebook tokens, the discrepancy loss is:

  \[
    L_{D} = \sum_{i=1}^n S_{t_{\mathrm{vis}}^i} \cdot P(t_i).
  \]

- Finally, they add \(L_{D}\) to the usual cross-entropy to keep the generated image tokens closer to the correct codebook entries:

  \[
    L = L_{C} + L_{D}.
  \]

---

## 4. **Byte Latent Transformer (BLT)**  
**Paper:** “*Byte Latent Transformer: Patches Scale Better Than Tokens*”  
**Core Contributions:**  
- Eliminates subword tokenization. Operates on **raw bytes** but groups them adaptively into “patches” based on **entropy**.  
- Has a local–global–local pipeline: (1) local encoder for each patch, (2) a big “latent transformer” across patches, and (3) a local decoder if needed to predict the bytes.

### 4.1 Entropy-Based Patching

- Train a small byte LM to get \(p_e(x_i \mid x_{<i})\).  
- Compute the next-byte entropy:

  \[
    H(x_i) = -\sum_{v\in V} p_{e}(x_i = v \mid x_{<i})\;\log\,p_{e}(x_i = v \mid x_{<i}).
  \]

- If \(H(x_i)\) exceeds a threshold \(\theta_g\), start a new patch. That way, **high-entropy** (“hard” or uncertain) regions get frequent patches, **low-entropy** (“easy”) regions get fewer patches.

### 4.2 Local + Latent + Local

- Each patch \(p_j\) is encoded by a small local module (often a small Transformer or CNN) into some embedding \(E_j\).  
- The main “Latent Transformer” processes \(\{E_1, E_2, \dots, E_j\}\) to produce a hidden \(H_j\).  
- A local decoder uses \((H_j, E_j)\) to finalize the byte-level predictions for patch \(p_j\).  
  \[
    H_j = \mathrm{LatentTransformer}\bigl(E_1, E_2, \dots, E_j\bigr), \quad
    \widehat{p_j} = \mathrm{LocalDecoder}\bigl(H_j, E_j\bigr).
  \]

---

## Putting It **All** Together

1. **Titans** gives you **modular memory** (short-term, long-term, persistent) plus a method to incorporate *test-time memory updates* triggered by “surprise.”  
2. **Transformer2** adds a **two-pass** approach with a “dispatch system” for *real-time adaptation* using **Singular Value Fine-Tuning**.  
3. **MVoT** extends chain-of-thought to produce **multimodal** outputs (text + images) with a special **token discrepancy loss** to keep the images consistent with the ground-truth codebook.  
4. **BLT** removes subword tokenization in favor of **entropy-based patching** at the byte level, drastically saving compute in easy contexts.

### Master Pseudocode (Hypothetical Combined)

```python
# Step 1: Byte-level input to patches (BLT)
patches = BLT_local_encoder(raw_bytes)  # Use small LM for entropy, define patch boundaries

# Step 2: Titans memory management (short-term + long-term)
surprise = measure_surprise(gradient_wrt_input)  # Titans eqn, not explicitly given
if surprise > threshold:
    # Update long-term memory
    M_long_term = update_memory(M_long_term, some_decay_rule, patches)

# Step 3: Transformer2 - Two-Pass for adaptation
#   First pass: get dispatch signals
dispatch_info = run_first_pass_dispatch(patches, M_long_term)

#   Second pass: apply SVF to refine weights
sigma_expert = choose_svf_experts(dispatch_info)
W_adapted = recompose_weights_SVD(U, sigma_base + sigma_expert, V)

# Step 4: MVoT - Interleaved text + image generation
for step in range(num_steps):
    # Possibly produce an image token v_i:
    if condition_for_image:
        image_token = sample_image_token(MVoT_image_head, W_adapted)

    # Then produce next text token z_{i+1}:
    text_token = sample_text_token(MVoT_text_head, W_adapted)

    # Add token discrepancy loss if generating an image
    if generating_image:
        L_D = compute_token_discrepancy_loss(image_token, codebook_embeddings)
        L = L + L_D
```

---

## Final Observations

- **Exact** numeric or code-level details are missing in the original texts: none of them provides an all-inclusive reference implementation.  
- Many references to “reinforcement learning,” “momentum-based decays,” or “gradient-based surprise” are concept-level.  
- **Nevertheless**, the key formulas for each approach—memory updates, SVD-based fine-tuning, text–image factorization, and entropy-based patch segmentation—appear above. 

If you **literally** wanted to re-implement each paper’s method, you’d need to:

1. **Titans**:  
   - Incorporate an extra memory “tensor” with RNN-like or short+long memory states.  
   - Implement the gradient-based “surprise” gating & memory decay.  

2. **Transformer2**:  
   - Modify your Transformer’s forward pass to run *twice.*  
   - On pass one, produce “dispatch signals” for *which singular values to tweak.*  
   - On pass two, shift the \(\sigma\) in the SVD of each weight.  

3. **MVoT**:  
   - Ensure your model can produce image tokens (via a discrete VAE codebook or similar).  
   - Factor in the token discrepancy loss for image tokens.  
   - Alternate text steps \(z_i\) and visual steps \(v_i\).  

4. **BLT**:  
   - A small LM to measure next-byte entropy.  
   - A local encoder + big latent Transformer + local decoder pipeline, building patches dynamically.  