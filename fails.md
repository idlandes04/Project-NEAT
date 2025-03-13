isaac@Isaacs-MacBook-Air Project-NEAT % pytest -v
================================================= test session starts =================================================
platform darwin -- Python 3.13.2, pytest-8.3.5, pluggy-1.5.0 -- /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
cachedir: .pytest_cache
rootdir: /Users/isaac/Documents/GitHub/Project-NEAT
collected 20 items                                                                                                    

tests/test_components.py::TestTitansMemorySystem::test_adaptive_decay_mechanism PASSED                          [  5%]
tests/test_components.py::TestTitansMemorySystem::test_efficient_gradient_computation FAILED                    [ 10%]
tests/test_components.py::TestTitansMemorySystem::test_persistent_memory PASSED                                 [ 15%]
tests/test_components.py::TestTitansMemorySystem::test_surprise_based_memory FAILED                             [ 20%]
tests/test_components.py::TestTitansMemorySystem::test_titans_memory_system PASSED                              [ 25%]
tests/test_components.py::TestTitansMemorySystem::test_window_attention_memory PASSED                           [ 30%]
tests/test_components.py::TestTransformer2Adaptation::test_svd_adaptation PASSED                                [ 35%]
tests/test_components.py::TestTransformer2Adaptation::test_task_dispatcher PASSED                               [ 40%]
tests/test_components.py::TestTransformer2Adaptation::test_transformer2_adaptation PASSED                       [ 45%]
tests/test_components.py::TestMVoTTokenProcessor::test_mvot_token_processor PASSED                              [ 50%]
tests/test_components.py::TestMVoTTokenProcessor::test_token_discrepancy_loss PASSED                            [ 55%]
tests/test_components.py::TestBLTByteProcessor::test_blt_byte_processor PASSED                                  [ 60%]
tests/test_components.py::TestBLTByteProcessor::test_entropy_calculator PASSED                                  [ 65%]
tests/test_components.py::TestBLTByteProcessor::test_small_byte_lm PASSED                                       [ 70%]
tests/test_components.py::TestMemoryEfficientTransformer::test_flash_attention PASSED                           [ 75%]
tests/test_components.py::TestMemoryEfficientTransformer::test_memory_efficient_transformer PASSED              [ 80%]
tests/test_components.py::TestMemoryEfficientTransformer::test_transformer_layer PASSED                         [ 85%]
tests/test_components.py::TestUnifiedArchitecture::test_component_activation PASSED                             [ 90%]
tests/test_components.py::TestUnifiedArchitecture::test_dynamic_component_controller PASSED                     [ 95%]
tests/test_components.py::TestUnifiedArchitecture::test_unified_architecture PASSED                             [100%]

====================================================== FAILURES =======================================================
_____________________________ TestTitansMemorySystem.test_efficient_gradient_computation ______________________________

self = <tests.test_components.TestTitansMemorySystem testMethod=test_efficient_gradient_computation>

    def test_efficient_gradient_computation(self):
        """Test efficient gradient computation with checkpointing."""
        memory = SurpriseBasedMemory(self.config)
    
        # Create a longer sequence to test checkpointing
        long_seq_length = 64
        long_hidden_states = torch.randn(self.batch_size, long_seq_length, self.config.hidden_size)
    
        # Enable efficient gradient computation with checkpointing
        memory.use_efficient_grad = True
        memory.grad_checkpoint_segments = 4
    
        # Ensure we can compute gradients for both short and long sequences
        short_surprise = memory._compute_efficient_gradient(self.hidden_states)
>       long_surprise = memory._compute_efficient_gradient(long_hidden_states)

tests/test_components.py:114: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = SurpriseBasedMemory(
  (layer_norm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
  (query_proj): Linear(in_fe...=True)
  (output_proj): Linear(in_features=64, out_features=64, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
hidden_states = tensor([[[ 1.0007, -0.7520,  0.0960,  ..., -0.5049,  0.5428,  0.0358],
         [ 0.3620,  1.5572, -0.7959,  ..., -1.4... -0.8900,  1.5163],
         [-0.7006,  0.4409, -0.7814,  ..., -1.0854, -1.8139, -0.1487]]],
       requires_grad=True)

    def _compute_efficient_gradient(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient gradient computation for surprise measurement.
    
        This method implements several optimizations for efficient gradient computation:
        1. Gradient checkpointing to reduce memory usage
        2. Gradient clipping for numerical stability
        3. Chunked processing for large sequences
        4. Platform-specific optimizations
    
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
    
        Returns:
            Surprise measure of shape [batch_size, seq_len, 1]
        """
        # Enable gradient computation
        hidden_states.requires_grad_(True)
        batch_size, seq_length, _ = hidden_states.shape
    
        # Use gradient checkpointing for memory efficiency when sequences are long
        if seq_length > 32 and self.use_efficient_grad:
            # Split the sequence into chunks for checkpointed processing
            chunk_size = (seq_length + self.grad_checkpoint_segments - 1) // self.grad_checkpoint_segments
            chunks = []
    
            for i in range(0, seq_length, chunk_size):
                end_idx = min(i + chunk_size, seq_length)
                chunk = hidden_states[:, i:end_idx, :]
                chunks.append(chunk)
    
            # Process each chunk with gradient checkpointing
            surprise_chunks = []
            for chunk in chunks:
                # Use torch.utils.checkpoint to save memory
>               chunk_surprise = torch.utils.checkpoint.checkpoint(
                    self._compute_chunk_gradient,
                    chunk
                )
E               AttributeError: module 'torch.utils' has no attribute 'checkpoint'

src/components/titans/memory_system.py:501: AttributeError
__________________________________ TestTitansMemorySystem.test_surprise_based_memory __________________________________

self = <tests.test_components.TestTitansMemorySystem testMethod=test_surprise_based_memory>

    def test_surprise_based_memory(self):
        """Test surprise-based memory."""
        memory = SurpriseBasedMemory(self.config)
    
        # Test in training mode
        memory.train()
        output_train = memory(self.hidden_states)
    
        # Check output shape in training mode
        self.assertEqual(output_train.shape, self.hidden_states.shape)
    
        # Test in evaluation mode
        memory.eval()
        output_eval = memory(self.hidden_states)
    
        # Check output shape in evaluation mode
        self.assertEqual(output_eval.shape, self.hidden_states.shape)
    
        # Test that memory is being updated in both modes
        # This verifies that we've removed the training-only condition
        memory_before = memory.memory.clone()
    
        # Run forward pass with high surprise inputs to trigger memory updates
        random_input = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size) * 5.0
        memory(random_input)
    
        # Check that memory has been updated
>       self.assertFalse(torch.allclose(memory_before, memory.memory))
E       AssertionError: True is not false

tests/test_components.py:74: AssertionError
=============================================== short test summary info ===============================================
FAILED tests/test_components.py::TestTitansMemorySystem::test_efficient_gradient_computation - AttributeError: module 'torch.utils' has no attribute 'checkpoint'
FAILED tests/test_components.py::TestTitansMemorySystem::test_surprise_based_memory - AssertionError: True is not false
============================================ 2 failed, 18 passed in 1.08s =============================================
isaac@Isaacs-MacBook-Air Project-NEAT % 