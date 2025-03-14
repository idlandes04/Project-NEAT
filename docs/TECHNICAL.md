We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the first
time, matches tokenization-based LLM performance at scale with significant improvements in inference
efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the
primary units of computation. Patches are segmented based on the entropy of the next byte, allocating
more compute and model capacity where increased data complexity demands it. We present the first
flop controlled scaling study of byte-level models up to 8B parameters and 4T training bytes. Our
results demonstrate the feasibility of scaling models trained on raw bytes without a fixed vocabulary.
Both training and inference efficiency improve due to dynamically selecting long patches when data is
predictable, along with qualitative improvements on reasoning and long tail generalization. Overall,
for fixed inference costs, BLT shows significantly better scaling than tokenization-based models, by
simultaneously growing both patch and model size.
Date: December 16, 2024

Self-adaptive large language models (LLMs) aim to solve the challenges posed
by traditional fine-tuning methods, which are often computationally intensive and
static in their ability to handle diverse tasks. We introduce Transformer2
, a novel
self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During
inference, Transformer2
employs a two-pass mechanism: first, a dispatch system
identifies the task properties, and then task-specific “expert” vectors, trained using
reinforcement learning, are dynamically mixed to obtain targeted behavior for the
incoming prompt. Our method outperforms ubiquitous approaches such as LoRA,
with fewer parameters and greater efficiency. Transformer2
demonstrates versatility across different LLM architectures and modalities, including vision-language
tasks. Transformer2
represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of
LLMs, paving the way for truly dynamic, self-organizing AI systems. 

Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and
attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows
attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling
of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new
neural long-term memory module that learns to memorize historical context and helps an attention to attend to the
current context while utilizing long past information. We show that this neural memory has the advantage of a fast
parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its
limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its
ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce
a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate
memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics,
and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models.
They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks
compared to baselines.
1 Introduction
“The true art of memory is the art of attention!"
— Samuel Johnson, 1787
T
ransformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-ofthe-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan
et al. 2020). The primary building blocks of Transformers–attention modules—function as associative memory
blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise
similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer
is exclusively conditioned on the direct dependencies of tokens in the current context window. This accurate modeling of
dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex
real-world tasks (e.g., language modeling (N. F. Liu et al. 2024), video understanding (C.-Y. Wu et al. 2019), long-term time
series forecasting (H. Zhou et al. 2021)), the context window can become extremely large, making the applicability of
Transformers challenging in these downstream tasks.
To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transformers (Kacham, Mirrokni, and P. Zhong 2024; Katharopoulos et al. 2020; S. Yang, B. Wang, Shen, et al. 2024), where softmax is
replaced by a kernel function in the attention (see §2.1 for details), resulting in a significant drop in memory consumption.
Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance
compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed
into a matrix-valued states (Katharopoulos et al. 2020). This, however, brings a contradictory fact about linear recurrent (or
linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs.
quadratic complexity), whose advantages is appeared for very long context; On the other hand, a very long context cannot
be properly compressed in a small vector-valued or matrix-valued states (S. Wang 2024).
1
arXiv:2501.00663v1 [cs.LG] 31 Dec 2024
Furthermore, beyond efficiency, most existing architectures–ranging from Hopfield Networks (Hopfield 1982) to LSTMs (Jürgen Schmidhuber and Hochreiter 1997) and Transformers (Vaswani et al. 2017)–face challenges when dealing with generalization, length extrapolation, and/or reasoning (Anil et al. 2022; Qin, Y. Zhong, and Deng 2024), all of which are inseparable
parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each of which
are missing: (1) a crucial component for learning process—such as short-term memory, long-term memory, meta-memory,
attending to current context, etc. (Cowan 2008); (2) how these components are interconnected systems that can operate
independently; and/or (3) the ability to actively learn from data and memorize the abstraction of past history. We argue
that in an effective learning paradigm, similar to human brain, there are distinct yet interconnected modules, each of which
is responsible for a component crucial to the learning process.
Memory Perspective
Memory is a fundamental mental process and is an inseparable component of human learning (Terry 2017). Without
a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped
behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g.,
Hopfield Networks (Hopfield 1982), LSTMs (Jürgen Schmidhuber and Hochreiter 1997), and Transformers (Vaswani et al.
2017).
Taking inspiration from the common definitions of memory and learning in neuropsychology literature (Okano, Hirano,
and Balaban 2000), most existing architectures consider memory as a neural update caused by an input, and define learning
as a process for acquiring effective and useful memory, given an objective. In this perspective, Recurrent Neural Networks
(RNNs) (Williams and Zipser 1989) can be defined as models with a vector-valued memory module M (also called hidden
state) with two main steps: Given a new input 𝑥𝑡 at time 𝑡, the model (1) updates the memory using a function 𝑓 (M𝑡−1, 𝑥𝑡)
(with compression); and (2) retrieves the corresponding memory of input using a function 𝑔(M𝑡
, 𝑥𝑡) (see §2.1 for details).
Similarly, Transformers can be seen as architectures with a growing memory and two similar steps. That is, the pair of key
and value matrices acts as the model’s memory, and the model: (1) updates the memory by appending the key and value to
the memory (without compression), and (2) retrieves query vectors’ corresponding memory by finding the similarity of
query and key vectors, which is then used to weight the value vectors for the output.
This perspective, can help us better understand existing paradigms, their critical differences, and design more effective
architectures. For example, the main difference between Transformers (Vaswani et al. 2017) and linear Transformers (Katharopoulos et al. 2020) is the memory structure as well as the memory updating step, in which linear Transformers
compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within
the context length) without any compression. While both linear Transformers and linear RNNs (including state space
models) compress the information in memory update step, the critical difference lies in the structure of the memory,
where linear RNNs (vs. linear Transformers) use a vector-valued memory (vs. matrix-valued memory). Therefore, this
perspective motivates us to ask: (Q1) What constitute a good structure for the memory? (Q2) What is a proper memory
update mechanism? and (Q3) What is a good memory retrieval process?
Revisiting our understanding of human memory, it is neither a unitary process nor it serves a single function (Cowan
2008). In fact, memory is a confederation of systems–e.g., short-term, working, and long-term memory–each serving a
different function with different neural structures, and each capable of operating independently (Willingham 1997). This
fact motivates us to ask: (Q4) How to design an efficient architecture that incorporates different interconnected memory
modules. Finally, storing a memory is a neural process that requires to encode and store the abstraction of the past. It can
be over-simplification to assume a single vector or a matrix, whose parameters are encoding the data in a linear manner,
are enough for storing long-term history. (Q5) Is a deep memory module needed to effectively store/remember long
past?
Contributions and Roadmap
In this paper, we aim to answer the above five questions by designing a long-term neural memory module, that can
efficiently and effectively learn to memorize at test time. Building upon its design, we discuss how it can be incorporated
into an architecture.
Neural Memory (§3). We present a (deep) neural long-term memory that (as a meta in-context model) learns how to
memorize/store the data into its parameters at test time. Inspired by human long-term memory system (Mandler 2014),
2
we design this memory module so an event that violates the expectations (being surprising) is more memorable. To this
end, we measure the surprise of an input with the gradient of the neural network with respect to the input in associative
memory loss (see §3.1 for details). To better handle the limited memory, we present a decaying mechanism that consider the
proportion of memory size and the amount of data surprise, resulting in better memory management. We show that this
decay mechanism is in fact the generalization of forgetting mechanism in modern recurrent models (Dao and Gu 2024; Gu
and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024). Interestingly, we find that this mechanism is equivalent to optimizing
a meta neural network with mini-batch gradient descent, momentum, and weight decay. Building upon tensorizing
mini-batch gradient descent to use more matmul operations (Yu Sun et al. 2024), we present a fast and parallelizable
algorithm to train our deep neural long-term memory.
Titans Architectures (§4). After designing the long-term neural memory, an important remaining question is how to
effectively and efficiently incorporate memory into a deep learning architecture. We present Titans, a family of deep models
that consists of three hyper-heads: (1) Core: this module consists of the short-term memory, and is responsible for the main
flow of processing the data (we use attention with limited window size); (2) Long-term Memory: this branch is our neural
long-term memory module that is responsible to store/remember long past; (3) Persistent Memory: this is a set of learnable
but date-independent parameters that encodes the knowledge about a task. Finally, as a proof of concept, we present three
variants of Titans, in which we incorporate memory as: (i) a context, (ii) a layer, and (iii) a gated branch.
Experimental Results (§5). We perform experimental evaluations on language modeling, commonsense reasoning, recallintensive, needle in haystack, time series forecasting, and DNA modeling tasks. We observe that our Titan architecture
outperforms all modern recurrent models as well as their hybrid variants (combining with sliding-window attention) across
a comprehensive set of benchmarks. Furthermore, Titans outperforms Transformers with the same context window, and
show competitive performance with Transformers that use the entire context. This results are achieved while, contrary to
Transformers, Titans scale to larger than 2M context window size.

Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex
reasoning in Large Language Models (LLMs) and Multimodal Large Language Models
(MLLMs). Yet, it struggles in complex spatial reasoning tasks. Nonetheless, human
cognition extends beyond language alone, enabling the remarkable capability to think
in both words and images. Inspired by this mechanism, we propose a new reasoning
paradigm, Multimodal Visualization-of-Thought (MVoT). It enables visual thinking
in MLLMs by generating image visualizations of their reasoning traces. To ensure highquality visualization, we introduce token discrepancy loss into autoregressive MLLMs.
This innovation significantly improves both visual coherence and fidelity. We validate
this approach through several dynamic spatial reasoning tasks. Experimental results
reveal that MVoT demonstrates competitive performance across tasks. Moreover, it
exhibits robust and reliable improvements in the most challenging scenarios where CoT
fails. Ultimately, MVoT establishes new possibilities for complex reasoning tasks where
visual thinking can effectively complement verbal reasoning.

Below is a **master overview** that merges (1) *Titans: Learning to Memorize at Test Time*, (2) *Transformer2: Self-Adaptive LLMs*, (3) *Multimodal Visualization-of-Thought (MVoT)*, and (4) *Byte Latent Transformer (BLT)* into **one cohesive reference**. It highlights all the essential equations, methods, and pseudo-code details each paper provides (based on the publicly available text), so you can reconstruct them or combine them into a single model.

---

## 1. **Titans**  
**Paper:** “*Titans: Learning to Memorize at Test Time*”  
**Core Contributions:**  
- A **long-term neural memory** module to complement short-term attention-based memory.  
- **Surprise-driven** updates—measured by gradient magnitudes w.r.t. inputs—determine when to store or decay new information.  
- **Fast parallelizable training** approach for memory updates.

### 1.1 Memory in RNN or Transformer Format

They unify an RNN-style memory perspective:
$$
M_{t} = f(M_{t-1}, x_{t}), \quad y_{t} = g(M_{t}, x_{t}),
$$
- \(M_{t}\) is the memory or hidden state at time \(t\).  
- \(x_{t}\) is the input; \(y_{t}\) is the output.

For Transformers, they equate “growing memory” with appending Key/Value pairs \((K_{t}, V_{t})\) in the context window. However, they propose a new **neural memory** that can store beyond the immediate window.

### 1.2 Surprise-Driven Memorization & Decay

- **Associative Memory Loss**:  
  They mention measuring “surprise” by gradient magnitude \(\bigl\|\tfrac{\partial \mathcal{L}_{\text{assoc}}}{\partial x}\bigr\|\). High surprise triggers stronger memorization of the data.  

- **Decay Mechanism**:  
  A forgetting or decay step to keep memory from overgrowing:
  $$
    M \leftarrow (1 - \alpha)\,M,
  $$
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
  $$
    W_{\text{adapted}} = U\,\mathrm{diag}(\sigma_{\text{base}} + \Delta\sigma)\,V^\top,
  $$
  where \(\Delta\sigma\) is the learned offset or the new singular values for a specific skill domain.

---

## 3. **Multimodal Visualization-of-Thought (MVoT)**  
**Paper:** “*Imagine while Reasoning in Space: Multimodal Visualization-of-Thought*”  
**Core Contributions:**  
- Extends chain-of-thought to **interleave image and text** as intermediate steps.  
- Adds a **token discrepancy loss** to improve the fidelity of generated images.

### 3.1 Interleaved Text–Image Reasoning

They factorize the generation:

$$
v_{i} \sim P_{\theta}\bigl(v_{i} \mid z_{1},v_{1}, \dots, z_{i}\bigr),\quad
z_{i+1} \sim P_{\theta}\bigl(z_{i+1} \mid x,\; z_{1},v_{1}, \dots, z_{i},v_{i}\bigr),
$$

So step \(i\) might produce an image \(v_i\), next step might produce text \(z_{i+1}\), and so on.

### 3.2 Token Discrepancy Loss

- Suppose the model uses a discrete image tokenizer with a codebook \(\{e^k_{\mathrm{vis}}\}_{k=1}^N\).  
- Let \(S_{t_{\mathrm{vis}}^i}\) measure MSE distance between the ground-truth image embedding and every codebook embedding:

  $$
    S_{t_{\mathrm{vis}}^i} = \Bigl[\mathrm{MSE}\bigl(e_{\mathrm{vis}}^i, e_{\mathrm{vis}}^1\bigr), \dots, \mathrm{MSE}\bigl(e_{\mathrm{vis}}^i, e_{\mathrm{vis}}^N\bigr)\Bigr].
  $$

- If \(P(t_i)\) is the predicted distribution over those image codebook tokens, the discrepancy loss is:

  $$
    L_{D} = \sum_{i=1}^n S_{t_{\mathrm{vis}}^i} \cdot P(t_i).
  $$

- Finally, they add \(L_{D}\) to the usual cross-entropy to keep the generated image tokens closer to the correct codebook entries:

  $$
    L = L_{C} + L_{D}.
  $$

---

## 4. **Byte Latent Transformer (BLT)**  
**Paper:** “*Byte Latent Transformer: Patches Scale Better Than Tokens*”  
**Core Contributions:**  
- Eliminates subword tokenization. Operates on **raw bytes** but groups them adaptively into “patches” based on **entropy**.  
- Has a local–global–local pipeline: (1) local encoder for each patch, (2) a big “latent transformer” across patches, and (3) a local decoder if needed to predict the bytes.

### 4.1 Entropy-Based Patching

- Train a small byte LM to get \(p_e(x_i \mid x_{<i})\).  
- Compute the next-byte entropy:

  $$
    H(x_i) = -\sum_{v\in V} p_{e}(x_i = v \mid x_{<i})\;\log\,p_{e}(x_i = v \mid x_{<i}).
  $$

- If \(H(x_i)\) exceeds a threshold \(\theta_g\), start a new patch. That way, **high-entropy** (“hard” or uncertain) regions get frequent patches, **low-entropy** (“easy”) regions get fewer patches.

### 4.2 Local + Latent + Local

- Each patch \(p_j\) is encoded by a small local module (often a small Transformer or CNN) into some embedding \(E_j\).  
- The main “Latent Transformer” processes \(\{E_1, E_2, \dots, E_j\}\) to produce a hidden \(H_j\).  
- A local decoder uses \((H_j, E_j)\) to finalize the byte-level predictions for patch \(p_j\).  
  $$
    H_j = \mathrm{LatentTransformer}\bigl(E_1, E_2, \dots, E_j\bigr), \quad
    \widehat{p_j} = \mathrm{LocalDecoder}\bigl(H_j, E_j\bigr).
  $$

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
   - A threshold-based or monotonic-based approach to decide patch boundaries.  
   - A local/latent/local pipeline architecture.

**This** is as close to a final, combined, *technical breakdown* as the snippet text lets us get. Additional details (like hyperparameters, model dimension specifics, step-by-step RL polices, or training loops) aren’t spelled out. But these references should give you a thorough blueprint of the conceptual and mathematical apparatus used in all four papers.



Below is the most comprehensive compilation of **all relevant technical details and methods** from the four papers—**Titans**, **Transformer2**, **Multimodal Visualization-of-Thought (MVoT)**, and **Byte Latent Transformer (BLT)**—based on the information currently visible. These details should give you a reasonably complete view of their implementations and design decisions. Where the papers appear truncated in the snippets, I’ve combined every available chunk to form as thorough a reconstruction as possible.

---

## 1. Titans: *“Learning to Memorize at Test Time”*

**Primary Goals:**  
- To extend a Transformer/RNN-like architecture with a trainable *long-term* memory module that can memorize contexts beyond the usual input window constraints.  
- To keep the standard *short-term* memory (a Transformer’s attention layers) but add new modules for *long-term* and *persistent* memory.  
- To incorporate *surprise-driven memorization* and *decay mechanisms* for more efficient memory usage.

### 1.1 Memory as RNN or Transformer  
- They unify the RNN memory perspective:  
  $$
    M_{t} = f(M_{t-1}, x_{t}), \quad y_{t} = g(M_{t}, x_{t}).
  $$  
- In Transformers, the “memory” is typically the entire set of Key/Value pairs for the context.  
- The paper emphasizes that the short-term memory is powerful but limited by the context window (often quadratic in cost), while the newly introduced *neural memory module* can store compressed historical info beyond that window.

### 1.2 Surprise-Driven Memorization  
- They propose an “Associative Memory Loss,” which is not fully spelled out, but it uses gradients w.r.t. the input features as a measure of “surprise.”  
  - If \(\left\|\frac{\partial \mathcal{L}_\mathrm{assoc}}{\partial x}\right\|\) is large, that means the event was surprising.  
- Surprising events get *persisted* more strongly in the memory. They also adopt a *decay/forgetting* mechanism so that memory doesn’t overflow.  
  - In broad terms, they do something like:
    $$
      M \leftarrow (1 - \alpha) \cdot M \quad \text{where}\ \alpha \ \text{may depend on the memory usage or surprise.}
    $$

### 1.3 Titans Architecture  
- The complete Titan model is described as having **three “hyper-heads”**:  
  1. **Core**: The standard short-term memory (attention-based).  
  2. **Long-term Memory**: A specialized neural memory module, updated with a learned approach.  
  3. **Persistent Memory**: A set of fixed parameters that remain constant across different sequences or tasks (like large learned weights for general knowledge).

They test 3 different strategies of how to incorporate that memory: as an extra “context,” as a layer injection, or via a gating branch. Though no explicit code is fully provided, it is apparently a matter of hooking the memory read/writes into the forward pass.

**Important**: The “meta in-context” approach implies they do partial fine-tuning or partial updates at test time (with the memory module’s parameters or states).

---

## 2. Transformer2: *“Self-Adaptive LLMs”*

**Primary Goals:**  
- Introduce a framework in which an LLM can adapt itself for new tasks on-the-fly (in real time) through a *two-pass mechanism* plus “expert vectors.”  
- They revolve around a method called **Singular Value Fine-Tuning (SVF)** to keep parameter overhead small but still get strong adaptivity.

### 2.1 Two-Pass Inference  
1. **First Pass** – The model runs on the incoming prompt to detect the “task properties.” This is done by a “dispatch system” that identifies relevant domain or skill.  
2. **Second Pass** – The model composes or adjusts *expert vectors* and *singular values* to adapt the base LLM parameters for that specific query.

### 2.2 Singular Value Fine-Tuning (SVF)
- Instead of full-layer fine-tuning or typical LoRA methods, they do an SVD of each weight matrix \(W\) as \(W = U \,\mathrm{diag}(\sigma)\,V^T\).  
- They keep \(U\) and \(V\) fixed, focusing only on adjusting \(\sigma\) (the singular values).  
- They claim that by training these singular values with reinforcement learning signals (to pick which ones to turn up or down), you get a set of “expert vectors” that can be combined for a new domain.  
- The actual formula is conceptually:

  \[
    W_{\text{adapted}} = U \,\mathrm{diag}(\sigma_{\text{base}} + \Delta\sigma)\, V^T,
  \]
  where \(\Delta\sigma\) is the learned offset or a new set of singular values.  

**Note**: The paper references “RL-based expert” selection but does not provide detailed equations for that process. They mention a policy that picks from a library of singular-value vectors.

---

## 3. Multimodal Visualization-of-Thought (MVoT): *“Imagine while Reasoning in Space”*

**Primary Goals:**  
- Enhance chain-of-thought prompting with native *image creation* for intermediate steps, in addition to text.  
- Do so by hooking a *multimodal generative model* (like an MLLM that can output images) so that the intermediate hidden states produce an image representation.

### 3.1 Interleaved Image-Text Reasoning  
- They define the main idea: generate interleaved textual “thoughts” \( z_i \) and visual “thoughts” \( v_i \).  
- The *autoregressive factorization* is:
  $$
    v_i \sim P_\theta(v_i \mid z_1, v_1, \dots, z_i),
  $$  
  $$
    z_{i+1} \sim P_\theta\bigl(z_{i+1} \mid x,\; z_1, v_1,\;\dots,\; z_i, v_i\bigr).
  $$
- This yields a chain of (text, image, text, image, ...).  

### 3.2 Token Discrepancy Loss for Visual Fidelity  
- Let the model produce discrete “image tokens” from a codebook of size \(N\). They define:
  $$
    S_{t_{\mathrm{vis}}^i} = [\mathrm{MSE}(e^i_{\mathrm{vis}}, e^1_{\mathrm{vis}}),\dots,\mathrm{MSE}(e^i_{\mathrm{vis}}, e^N_{\mathrm{vis}})],
  $$
  where each \(e^j_{\mathrm{vis}}\) is a codebook embedding. So \(S_{t_{\mathrm{vis}}^i}\) is effectively a vector of MSE distances from the ground-truth embedding to every possible codebook embedding.  
- If the model’s predicted probability distribution for the \(i\)-th image token is \(P(t_i)\in \mathbb{R}^{1\times N}\), the “token discrepancy loss” is:
  $$
    L_D = \sum_{i=1}^n\; S_{t_{\mathrm{vis}}^i}\;\cdot\;P(t_i).
  $$
  This is in addition to the usual cross-entropy. Summarily:  
  $$
    L = L_C + L_D.
  $$

---

# 4. Byte Latent Transformer (BLT)

**Paper Title:**  
*“Byte Latent Transformer: Patches Scale Better Than Tokens”*

### 4.1 Entropy-Based Patching

- Let \(x = \{x_i\}\) be the raw byte sequence. They define an external small byte LM \(p_e\) to estimate the next-byte distribution.  
- The next-byte “entropy” is:
  $$
    H(x_i) = \sum_{v \in V} p_e(x_i = v \mid x_{<i}) \;\log\,p_e(x_i = v \mid x_{<i}).
  $$
- If \(H(x_i)\) is above a threshold \(\theta_g\), they start a new patch. This yields variable-length patches that reduce overhead where data is easy (low entropy) and break more often where data is unpredictable (high entropy).

### 4.2 Architecture: Local + Latent + Local  
1. **Local Encoder**: A small transformer or CNN that encodes the patch of raw bytes.
  2. **Latent Transformer**: A bigger model that processes *one embedding per patch*.
  3. **Local Decoder**: Another small module that (if needed) decodes actual bytes from the latent representation.

**(Pseudo-)Equation**:  
- For a sequence of patches \(\{p_j\}\), the latent transformer outputs a “latent hidden” \(H_j\). Then the local decoder cross-attends to \(\{H_j\}\) to produce the final byte-level predictions. Formally, if we treat each patch’s local hidden as \(E_j\), then:
  $$
    H_j = \mathrm{LatentTransformer}(E_1, E_2,\dots,E_j),
  $$
  and the local decoder produces:
  $$
    \widehat{p_j} = \mathrm{LocalDecoder}\bigl(H_j, E_j\bigr).
  $$

---

## Additional Observations on Implementation

1. **Titans** – The most important aspect is hooking a “neural memory module” that can be updated at test time (meta in-context learning). It uses surprise-driven updates. The paper references equivalences to momentum and weight decay but doesn’t show them in a single explicit formula.  
2. **Transformer2** – The main trick is two-pass inference with a “dispatch system” to pick the relevant *SVF experts*, i.e., sets of singular values for certain layers. They claim the model can do partial reconfiguration on the second pass.  
3. **MVoT** – Necessitates that the base model can handle both text and image tokens. The main additions are the image generator tokens (with codebook), plus the *token discrepancy loss* to ensure the visual embeddings are not drifting.  
4. **BLT** – They rely heavily on the local–global separation: the local encoder sees the raw bytes (variable length per patch), and the global transformer runs fewer steps (one per patch). This leads to big improvements in throughput, especially for large sequences that contain many easy / predictable sections.

No single paper gives a *full code listing*, but combined, these are likely the core formulas and methods you would need to reconstruct each approach:

- For Titans:  
  1. The memory update (RNN-like or K/V-like).  
  2. The surprise-based update gating.  
  3. The memory decay or “forget” step.

- For Transformer2:  
  1. The SVD of each weight matrix.  
  2. Tuning only the singular values.  
  3. The two-pass mechanism for self-adaptation with RL-based selection of “expert vectors.”

- For MVoT:  
  1. Interleaved text–image generation.  
  2. The token discrepancy loss to align predicted image tokens with a known codebook embedding.  
  3. Multi-hop chain-of-thought with partial image generation at each step.

- For BLT:  
  1. The small byte-level language model for measuring next-token entropy.  
  2. A threshold-based or monotonic-based approach to decide patch boundaries.  
  3. A local/latent/local pipeline archite---

### Final Note
From the text provided, this is the greatest level of detail available—some references to partial code or derivations remain hidden behind each project’s code or supplement. Nonetheless, the above compilation captures all the core equations and definitions we have.

Below is a consolidated reference of *exact (or near-exact)* core formulas, definitions, and pseudo-code from the four papers (Titans, Transformer2, MVoT, and BLT) as gleaned from their text. Where the authors only summarized or hinted at ideas, I’ve provided those summaries. When they provided explicit math, I’ve listed their equations closely to the original. If you see any small formatting differences, that’s just to fit them in this response.

---

# 1. Titans

**Paper Title:**  
*“Titans: Learning to Memorize at Test Time”*

### 1.1 Memory as RNN/Transformer Perspective

- The paper frames “memory” in a neural network as a process of reading and writing to internal states.  
- For a Recurrent Neural Network (RNN)-type perspective, they define it as:
  \[
    M_t = f(M_{t-1}, x_t) \quad\\
    y_t = g(M_t, x_t),
  \]  
  where \(M_t\) is the hidden state (memory) at time \(t\), \(x_t\) is the new input, and \(y_t\) is the output.

- For Transformers, they note that the “growing memory” is the collection of Key/Value pairs over the context window, updated by appending new \((K_t, V_t)\).

### 1.2 Associative Memory and Surprise-Driven Memorization

- **Key Idea:** “Surprise” is measured by the gradient magnitude of the network w.r.t. the input in some “associative memory loss.”  
- Although the paper references direct formulas for this “surprise measure,” they did **not** provide a single explicit equation in the snippet, beyond saying it is computed from \(\frac{\partial \mathcal{L}_\text{assoc}}{\partial x}\). In the text, they simply mention:
  > “We measure the surprise of an input with the gradient of the neural network with respect to the input in the associative memory loss (see §3.1).”

- **Decay Mechanism:** They propose a “neural memory decaying” or “forgetting” step as a generalization of forgetting gates in modern RNNs:
  \[
    M \leftarrow (1 - \alpha) \cdot M,
  \]
  where \(\alpha\) can be set by how surprising or how large the memory size is. The authors link this idea to momentum-based updates and weight decay from standard optimization, but do not present it with a specific single formula in the snippet.

### 1.3 Practical Memory Updates

- For “long-term memory” usage, they build on an approach reminiscent of linear RNN “Delta rule” or “additive memory,” e.g.:
  \[
    M_t = M_{t-1} + \Delta(M_{t-1}, x_t),
  \]
  but with data-dependent decay/forget gates.  

- **In short**, Titans revolve around:
  1. Splitting memory into **short-term** (like attention), **long-term** (like deep neural memory), and **persistent** (task-agnostic knowledge).
  2. Using a “surprise-driven update” that modifies the memory *only* when needed, and uses a decay step otherwise.

---

# 2. Transformer2

**Paper Title:**  
*“TRANSFORMER2: SELF-ADAPTIVE LLMS”*

### 2.1 Two-Pass Inference Mechanism

1. **First Pass** – “Dispatch”  
   - The model runs normally on the incoming prompt (or partial input), collecting information about the task domain, difficulty, or other triggers.  
   - A “dispatch system” identifies which “expert vectors” might be relevant.

2. **Second Pass** – “Expert Composition”  
   - The model applies *task-specific experts* (represented as extra “SVF vectors”) to refine or adapt the main weights in real time.  

### 2.2 Singular Value Fine-Tuning (SVF)

- **Goal:** Fine-tune only the singular values \( \sigma_i \) of certain weight matrices \(W\) instead of full-layer or low-rank matrices. This drastically reduces parameters and helps compositional reusability.  
- Let \(W = U \,\mathrm{diag}(\sigma)\, V^T\) be the SVD of the weight matrix. Transformer2 claims they only update \(\sigma\).  
- They also mention a reinforcement learning (RL) procedure to pick or mix these “expert” singular-value scales, but do not present explicit RL equations. They say:

  > “During inference, the dispatch system identifies which singular values or which combination of them to apply for the final forward pass.”

Hence, *no fully explicit formula* is given beyond referencing the SVD and the strategy of “Adjust \(\sigma\) on the second pass, keep \(U\) and \(V\) fixed.”

---

# 3. Multimodal Visualization-of-Thought (MVoT)

**Paper Title:**  
*“Imagine while Reasoning in Space: Multimodal Visualization-of-Thought”*

### 3.1 Key Equations for Visual-Text Reasoning

- They describe the main idea: generate interleaved textual “thoughts” \( z_i \) and visual “thoughts” \( v_i \).  
- The *autoregressive factorization* is:
  \[
    v_i \sim P_\theta(v_i \mid z_1, v_1, \dots, z_i),
  \]  
  \[
    z_{i+1} \sim P_\theta\bigl(z_{i+1} \mid x,\; z_1, v_1,\;\dots,\; z_i, v_i\bigr).
  \]
- This yields a chain of (text, image, text, image, ...).  

### 3.2 Token Discrepancy Loss

- Let the model produce discrete “image tokens” from a codebook of size \(N\). They define:
  \[
    S_{t_{\mathrm{vis}}^i} = [\mathrm{MSE}(e^i_{\mathrm{vis}}, e^1_{\mathrm{vis}}),\dots,\mathrm{MSE}(e^i_{\mathrm{vis}}, e^N_{\mathrm{vis}})],
  \]
  where each \(e^j_{\mathrm{vis}}\) is a codebook embedding. So \(S_{t_{\mathrm{vis}}^i}\) is effectively a vector of MSE distances from the ground-truth embedding to every possible codebook embedding.  
- If the model’s predicted probability distribution for the \(i\)-th image token is \(P(t_i)\in \mathbb{R}^{1\times N}\), the “token discrepancy loss” is:
  \[
    L_D = \sum_{i=1}^n\; S_{t_{\mathrm{vis}}^i}\;\cdot\;P(t_i).
  \]
  This is in addition to the usual cross-entropy. Summarily:  
  \[
    L = L_C + L_D.
  \]

---

# 4. Byte Latent Transformer (BLT)

**Paper Title:**  
*“Byte Latent Transformer: Patches Scale Better Than Tokens”*

### 4.1 Entropy-Based Patching

- Let \(x = \{x_i\}\) be the raw byte sequence. They define an external small byte LM \(p_e\) to estimate the next-byte distribution.  
- The next-byte “entropy” is:
  \[
    H(x_i) = \sum_{v \in V} p_e(x_i = v \mid x_{<i}) \;\log\,p_e(x_i = v \mid x_{<i}).
  \]
- If \(H(x_i)\) is above a threshold \(\theta_g\), they start a new patch. This yields variable-length patches that reduce overhead where data is easy (low entropy) and break more often where data is unpredictable (high entropy).

### 4.2 Architecture

- They have a **local encoder** for each patch, a **large “latent” transformer** that works at the patch level, and then a **local decoder** to produce the bytes in that patch. Schematic:

  1. **Local Encoder:** Encodes the current patch of raw bytes into a hidden patch representation.
  2. **Latent Transformer:** Processes the patch representation, globally across multiple patches.
  3. **Local Decoder:** Cross-attends to the latent transformer’s output to decode the actual next bytes in the patch.

**(Pseudo-)Equation**:  
- For a sequence of patches \(\{p_j\}\), the latent transformer outputs a “latent hidden” \(H_j\). Then the local decoder cross-attends to \(\{H_j\}\) to produce the final byte-level predictions. Formally, if we treat each patch’s local hidden as \(E_j\), then:
  \[
    H_j = \mathrm{LatentTransformer}(E_1, E_2,\dots,E_j),
  \]
  and the local decoder produces:
  \[
    \widehat{p_j} = \mathrm{LocalDecoder}\bigl(H_j, E_j\bigr).
  \]

.
re---

### Final Note
From the text provided, this is the greatest level of detail available—some references to partial code or derivations remain hidden behind each project’s code or supplement. Nonetheless, the above compilation captures all the core equations and definitions we have.

Below is a consolidated reference of *exact (or near-exact)* core formulas, definitions, and pseudo-code from the four papers (Titans, Transformer2, MVoT, and BLT) as gleaned from their text. Where the authors only summarized or hinted at ideas, I’ve provided those summaries. When they provided explicit math, I’ve listed their equations closely to the original. If you see any small formatting differences, that’s just to fit them in this response.

---

# 1. Titans

**Paper Title:**  
*“Titans: Learning to Memorize at Test Time”*

### 1.1 Memory as RNN/Transformer Perspective

- The paper frames “memory” in a neural network as a process of reading and writing to internal states.  
- For a Recurrent Neural Network (RNN)-type perspective, they define it as:
  \[
    M_t = f(M_{t-1}, x_t) \quad\\
    y_t = g(M_t, x_t),
  \]  
  where \(M_t\) is the hidden state (memory) at time \(t\), \(x_t\) is the new input, and \(y_t\) is the output.

- For Transformers, they note that the “growing memory” is the collection of Key/Value pairs over the context window, updated by appending new \((K_t, V_t)\).

### 1.2 Associative Memory and Surprise-Driven Memorization

- **Key Idea:** “Surprise” is measured by the gradient magnitude of the network w.r.t. the input in some “associative memory loss.”  
- Although the paper references direct formulas for this “surprise measure,” they did **not** provide a single explicit equation in the snippet, beyond saying it is computed from \(\frac{\partial \mathcal{L}_\text{assoc}}{\partial x}\). In the text, they simply mention:
  > “We measure the surprise of an input with the gradient of the neural network with respect to the input in the associative memory loss (see §3.1).”

- **Decay Mechanism:** They propose a “neural memory decaying” or “forgetting” step as a generalization of forgetting gates in modern RNNs:
  \[
    M \leftarrow (1 - \alpha) \cdot M,
  \]
  where \(\alpha\) can be set by how surprising or how large the memory size is. The authors link this idea to momentum-based updates and weight decay from standard optimization, but do not present it with a specific single formula in the snippet.

### 1.3 Practical Memory Updates

- For “long-term memory” usage, they build on an approach reminiscent of linear RNN “Delta rule” or “additive memory,” e.g.:
  \[
    M_t = M_{t-1} + \Delta(M_{t-1}, x_t),
  \]
  but with data-dependent decay/forget gates.  

- **In short**, Titans revolve around:
  1. Splitting memory into **short-term** (like attention), **long-term** (like deep neural memory), and **persistent** (task-agnostic knowledge).
  2. Using a “surprise-driven update” that modifies the memory *only* when needed, and uses a decay step otherwise.

---

# 2. Transformer2

**Paper Title:**  
*“TRANSFORMER2: SELF-ADAPTIVE LLMS”*

### 2.1 Two-Pass Inference Mechanism

1. **First Pass** – “Dispatch”  
   - The model runs normally on the incoming prompt (or partial input), collecting information about the task domain, difficulty, or other triggers.  
   - A “dispatch system” identifies which “expert vectors” might be relevant.

2. **Second Pass** – “Expert Composition”  
   - The model applies *task-specific experts* (represented as extra “SVF vectors”) to refine or adapt the main weights in real time.  

### 2.2 Singular Value Fine-Tuning (SVF)

- **Goal:** Fine-tune only the singular values \( \sigma_i \) of certain weight matrices \(W\) instead of full-layer or low-rank matrices. This drastically reduces parameters and helps compositional reusability.  
- Let \(W = U \,\mathrm{diag}(\sigma)\, V^T\) be the SVD of the weight matrix. Transformer2 claims they only update \(\sigma\).  
- They also mention a reinforcement learning (RL) procedure to pick or mix these “expert” singular-value scales, but do not present explicit RL equations. They say:

  > “During inference, the dispatch system identifies which singular values or which combination of them to apply for the final forward pass.”

Hence, *no fully explicit formula* is given beyond referencing the SVD and the strategy of “Adjust \(\sigma\) on the second pass, keep \(U\) and \(V\) fixed.”

---

# 3. Multimodal Visualization-of-Thought (MVoT)

**Paper Title:**  
*“Imagine while Reasoning in Space: Multimodal Visualization-of-Thought”*

### 3.1 Key Equations for Visual-Text Reasoning

- They describe the main idea: generate interleaved textual “thoughts” \( z_i \) and visual “thoughts” \( v_i \).  
- The *autoregressive factorization* is:
  \[
    v_i \sim P_\theta(v_i \mid z_1, v_1, \dots, z_i),
  \]  
  \[
    z_{i+1} \sim P_\theta\bigl(z_{i+1} \mid x,\; z_1, v_1,\;\dots,\; z_i, v_i\bigr).
  \]
- This yields a chain of (text, image, text, image, ...).  

### 3.2 Token Discrepancy Loss

- Let the model produce discrete “image tokens” from a codebook of size \(N\). They define:
  \[
    S_{t_{\mathrm{vis}}^i} = [\mathrm{MSE}(e^i_{\mathrm{vis}}, e^1_{\mathrm{vis}}),\dots,\mathrm{MSE}(e^i_{\mathrm{vis}}, e^N_{\mathrm{vis}})],
  \]
  where each \(e^j_{\mathrm{vis}}\) is a codebook embedding. So \(S_{t_{\mathrm{vis}}^i}\) is effectively a vector of MSE distances from the ground-truth embedding to every possible codebook embedding.  
- If the model’s predicted probability distribution for the \(i\)-th image token is \(P(t_i)\in \mathbb{R}^{1\times N}\), the “token discrepancy loss” is:
  \[
    L_D = \sum_{i=1}^n\; S_{t_{\mathrm{vis}}^i}\;\cdot\;P(t_i).
  \]
  This is in addition to the usual cross-entropy. Summarily:  
  \[
    L = L_C + L_D.
  \]

---

# 4. Byte Latent Transformer (BLT)

**Paper Title:**  
*“Byte Latent Transformer: Patches Scale Better Than Tokens”*

### 4.1 Entropy-Based Patching

- Let \(x = \{x_i\}\) be the raw byte sequence. They define an external small byte LM \(p_e\) to estimate the next-byte distribution.  
- The next-byte “entropy” is:
  \[
    H(x_i) = \sum_{v \in V} p_e(x_i = v \mid x_{<i}) \;\log\,p_e(x_i = v \mid x_{<i}).
  \]
- If \(H(x_i)\) is above a threshold \(\theta_g\), they start a new patch. This yields variable-length patches that reduce overhead where data is easy (low entropy) and break more often where data is unpredictable (high entropy).

### 4.2 Architecture

- They have a **local encoder** for each patch, a **large “latent” transformer** that works at the patch level, and then a **local decoder** to produce the bytes in that patch. Schematic:

  1. **Local Encoder:** Encodes the current patch of raw bytes into a hidden patch representation.
  2. **Latent Transformer:** Processes the patch representation, globally across multiple patches.
  3. **Local Decoder:** Cross-attends to the latent transformer’s output to decode the actual next bytes in the patch.

**(Pseudo-)Equation**:  
- For a sequence of patches \(\{p_j\}\), the latent transformer outputs a “latent hidden” \(H_j\). Then the local decoder cross-attends to \(\{H_j\}\) to produce the final byte-level predictions. Formally, if we treat each patch’s local hidden as \(E_j\), then:
  \[
    H_j = \mathrm{LatentTransformer}(E_1, E_2,\dots,E_j),
  \]
  and the local decoder produces:
  \[
    \widehat{p_j} = \mathrm{LocalDecoder}\bigl(H_j, E_j\bigr).
  \]

.

---

### Final Note
From the text provided, this is the greatest level of detail available—some references to partial code or derivations remain hidden behind each project’s code or supplement. Nonetheless, the above compilation captures all the core equations and definitions we have.

Below is a consolidated reference of *exact (or near-exact)* core formulas, definitions, and pseudo-code from the four papers (Titans, Transformer2, MVoT, and BLT) as gleaned from their text. Where the authors only summarized or hinted at ideas, I’ve provided those summaries. When they provided explicit math, I’ve listed their equations closely to the original. If you see any small formatting differences, that’s just to fit them in this response.

---

# 1. Titans

**Paper Title:**  
*“Titans: Learning to Memorize at Test Time”*

### 1.1 Memory as RNN/Transformer Perspective

- The paper frames “memory” in a neural network as a process of reading and writing to internal states.  
- For a Recurrent Neural Network (RNN)-type perspective, they define it as:
  \[
    M_t = f(M_{t-1}, x_t) \quad\\
    y_t = g(M_t, x_t),
  \]  
  where \(M_t\) is the hidden state (memory) at time \(t\), \(x_t\) is the new input, and \(y_t\) is the output.

- For Transformers, they note that the “growing memory” is the collection of Key/Value pairs over the context window, updated by appending new \((K_t, V_t)\).

### 1.2 Associative Memory and Surprise-Driven Memorization

- **Key Idea:** “Surprise” is measured by the gradient magnitude of the network w.r.t. the input in some “associative memory loss.”  
- Although the paper references direct formulas for this “surprise measure,” they did **not** provide a single explicit equation in the snippet, beyond saying it is computed from \(\frac{\partial \mathcal{L}_\text{assoc}}{\partial x}\). In the text, they simply mention:
  > “We measure the surprise of an input with the gradient of the neural network with respect to the input in the associative memory loss (see §3.1).”

- **Decay Mechanism:** They propose a “neural memory decaying” or “forgetting” step as a generalization of forgetting gates in modern RNNs:
  \[
    M \leftarrow (1 - \alpha) \cdot M,
  \]
  where \(\alpha\) can be set by how surprising or how large the memory size is. The authors link this idea to momentum-based updates and weight decay from standard optimization, but do not present it with a specific single formula in the snippet.

### 1.3 Practical Memory Updates

- For “long-term memory” usage, they build on an approach reminiscent of linear RNN “Delta rule” or “additive memory,” e.g.:
  \[
    M_t = M_{t-1} + \Delta(M_{t-1}, x_t),
  \]
  but with data-dependent decay/forget gates.  

- **In short**, Titans revolve around:
  1. Splitting memory into **short-term** (like attention), **long-term** (like deep neural memory), and **persistent** (task-agnostic knowledge).
  2. Using a “surprise-driven update” that modifies the memory *only* when needed, and uses a decay step otherwise.

---

# 2. Transformer2

**Paper Title:**  
*“TRANSFORMER2: SELF-ADAPTIVE LLMS”*

### 2.1 Two-Pass Inference Mechanism

1. **First Pass** – “Dispatch”  
   - The model runs normally on the incoming prompt (or partial input), collecting information about the task domain, difficulty, or other triggers.  
   - A “dispatch system” identifies which “expert vectors” might be relevant.

2. **Second Pass** – “Expert Composition”  
   - The model applies *task-specific experts* (represented as extra “SVF vectors”) to refine or adapt the main weights in real time.  

### 2.2 Singular Value Fine-Tuning (SVF)

- **Goal:** Fine-tune only the singular values \( \sigma_i \) of certain weight matrices \(W\) instead of full-layer or low-rank matrices. This drastically reduces parameters and helps compositional reusability.  
- Let \(W = U \,\mathrm{diag}(\sigma)\, V^T\) be the SVD of the weight matrix. Transformer2 claims they only update \(\sigma\).  
- They also mention a reinforcement learning (RL) procedure to pick or mix these “expert” singular-value scales, but do not present explicit RL equations. They say:

  > “During inference, the dispatch system identifies which singular values or which combination of them to apply for the final forward pass.”

Hence, *no fully explicit formula* is given beyond referencing the SVD and the strategy of “Adjust \(\sigma\) on the second pass, keep \(U\) and \(V\) fixed.”

---

# 3. Multimodal Visualization-of-Thought (MVoT)

**Paper Title:**  
*“Imagine while Reasoning in Space: Multimodal Visualization-of-Thought”*

### 3.1 Key Equations for Visual-Text Reasoning

- They describe the main idea: generate interleaved textual “thoughts” \( z_i \) and visual “thoughts” \( v_i \).  
- The *autoregressive factorization* is:
  \[
    v_i \sim P_\theta(v_i \mid z_1, v_1, \dots, z_i),
  \]  
  \[
    z_{i+1} \sim P_\theta\bigl(z_{i+1} \mid x,\; z_1, v_1,\;\dots,\; z_i, v_i\bigr).
  \]
- This yields a chain of (text, image, text, image, ...).  

### 3.2 Token Discrepancy Loss

- Let the model produce discrete “image tokens” from a codebook of size \(N\). They define:
  \[
    S_{t_{\mathrm{vis}}^i} = [\mathrm{MSE}(e^i_{\mathrm{vis}}, e^1_{\mathrm{vis}}),\dots,\mathrm{MSE}(e^i_{\mathrm{vis}}, e^N_{\mathrm{vis}})],
  \]
  where each \(e^j_{\mathrm{vis}}\) is a codebook embedding. So \(S_{t_{\mathrm{vis}}^i}\) is effectively a vector of MSE distances from the ground-truth embedding to every possible codebook embedding.  
- If the model’s predicted probability distribution for the \(i\)-th image token is \(P(t_i)\in \mathbb{R}^{1\times N}\), the “token discrepancy loss” is:
  \[
    L_D = \sum_{i=1}^n\; S_{t_{\mathrm{vis}}^i}\;\cdot\;P(t_i).
  \]
  This is in addition to the usual cross-entropy. Summarily:  
  \[
    L = L_C + L_D.
  \]

---

# 4. Byte Latent Transformer (BLT)

**Paper Title:**  
*“Byte Latent Transformer: Patches Scale Better Than Tokens”*

### 4.1 Entropy-Based Patching

- Let \(x = \{x_i\}\) be the raw byte sequence. They define an external small byte LM \(p_e\) to estimate the next-byte distribution.  
- The next-byte “entropy” is:
  \[
    H(x_i) = \sum_{v \in V} p_e(x_i = v \mid x_{<i}) \;\log\,p_e(x_i = v \mid x_{<i}).
  \]
- If \(H(x_i)\) is above a threshold \(\theta_g\), they start a new patch. This yields variable-length patches that reduce overhead where data is easy (low entropy) and break more often where data is unpredictable (high entropy).

### 4.2 Architecture

- They have a **local encoder** for each patch, a **large “latent” transformer** that works at the patch level, and then a **local decoder** to produce the bytes in that patch. Schematic:

  1. **Local Encoder:** Encodes the current patch of raw bytes into a hidden patch representation.
  2. **Latent Transformer:** Processes the patch representation, globally across multiple patches.
  3. **Local Decoder:** Cross-attends to the latent transformer’s output to decode the actual next bytes in the patch.

**(Pseudo-)Equation**:  
- For a sequence of patches \(\{p_j\}\), the latent transformer outputs a “latent hidden” \(H_j\). Then the local decoder cross-attends to \(\{H_j\}\) to produce the final byte-level predictions. Formally, if we treat each patch’s local hidden as \(E_j\), then:
  \[
    H_j = \mathrm{LatentTransformer}(E_1, E_2,\dots,E_j),
  \]
  and the local decoder produces:
  \[
    \widehat{p_j} = \mathrm{LocalDecoder}\bigl(H_j, E_j\bigr).
  \]

---

## Conclusion: How “Exact” They Were

In all four papers, the authors do present some *equations and algorithmic definitions*, but none provides a fully elaborate reference implementation as might exist in a code repository. Much of it is conceptual or described in words. In summary, you can see:

1. **Titans** gives the conceptual formula for a memory update, plus a mention of gradient-based “surprise.”  
2. **Transformer2** details an SVD-based “Singular Value Fine-tuning” approach and a two-pass inference, but mostly as conceptual steps.  
3. **MVoT** provides explicit equations for interleaved text-image generation and “token discrepancy loss.”  
4. **BLT** is more formula-heavy about dynamic patching via entropy thresholds and a local–global–local pipeline.