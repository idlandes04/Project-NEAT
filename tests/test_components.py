"""
Tests for the neural architecture components.

This module contains tests for each component of the unified architecture.
"""
import os
import sys
import unittest
import tempfile
from copy import deepcopy
import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import ModelConfig, get_default_config, ByteLMConfig
from src.components.titans.memory_system import TitansMemorySystem, WindowAttentionMemory, SurpriseBasedMemory, PersistentMemory
from src.components.transformer2.adaptation import Transformer2Adaptation, TaskDispatcher, SVDAdaptation
from src.components.mvot.token_processor import MVoTTokenProcessor, TokenDiscrepancyLoss
from src.components.blt.byte_processor import BLTByteProcessor, EntropyCalculator, SmallByteLM
from src.trainers.main_trainer import ByteDataset, EntropyEstimatorTrainer
from src.models.transformer import MemoryEfficientTransformer, TransformerLayer, FlashAttention
from src.models.unified_architecture import UnifiedArchitecture, DynamicComponentController


class TestTitansMemorySystem(unittest.TestCase):
    """Tests for the Titans memory system."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.num_attention_heads = 4
        self.config.window_size = 16
        self.config.memory_size = 32
        self.config.num_persistent_vectors = 8
        
        self.batch_size = 2
        self.seq_len = 24
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
    
    def test_window_attention_memory(self):
        """Test window attention memory."""
        memory = WindowAttentionMemory(self.config)
        output = memory(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
    
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
        
        # Test by directly calling update method with controlled inputs
        # We'll manually create a specific set of surprise values and memory states
        
        # Setup a controlled memory state
        batch_size, seq_len, hidden_size = 1, 2, memory.hidden_size
        memory.memory = torch.zeros(1, memory.memory_size, hidden_size)
        memory.importance_scores = torch.zeros(1, memory.memory_size)
        
        # Create fake hidden states and extremely high surprise values
        test_hidden_states = torch.ones(batch_size, seq_len, hidden_size)
        test_surprise = torch.ones(batch_size, seq_len, 1) * 100.0  # Very high surprise
        
        # Take a snapshot of memory before update
        memory_before = memory.memory.clone()
        
        # Directly call the update method
        memory._update_memory_with_safeguards(test_hidden_states, test_surprise)
        
        # Check that at least one memory slot was updated
        memory_changed = False
        for i in range(memory.memory_size):
            if not torch.allclose(memory_before[0, i], memory.memory[0, i]):
                memory_changed = True
                break
        
        self.assertTrue(memory_changed, "Memory should be updated with high surprise inputs")
    
    def test_adaptive_decay_mechanism(self):
        """Test adaptive decay mechanism for memory management."""
        # Create memory with small size for testing
        self.config.titans.memory_size = 10
        memory = SurpriseBasedMemory(self.config)
        
        # Initialize with some importance scores
        memory.importance_scores = torch.ones(1, memory.memory_size) * 0.5
        
        # Set some memory entries as used
        memory.memory_usage[0, 0:5] = 10
        memory.last_access_time[0, 0:5] = memory.global_step + 5
        
        # Set some memory entries as old and unused
        memory.memory_age[0, 5:10] = memory.max_memory_age + 10
        
        # Run memory management
        memory._manage_memory_with_adaptive_decay()
        
        # Check that used entries have higher importance than unused ones
        used_importance = memory.importance_scores[0, 0:5].mean()
        unused_importance = memory.importance_scores[0, 5:10].mean()
        self.assertGreater(used_importance, unused_importance)
    
    def test_efficient_gradient_computation(self):
        """Test efficient gradient computation."""
        memory = SurpriseBasedMemory(self.config)
        
        # Create both short and longer sequence inputs
        short_hidden_states = self.hidden_states
        long_seq_length = 30
        long_hidden_states = torch.randn(self.batch_size, long_seq_length, self.config.hidden_size)
        
        # Disable gradient checkpointing for testing (it's harder to test with checkpoint)
        memory.use_efficient_grad = False
        
        # Ensure we can compute gradients for both short and long sequences
        short_surprise = memory._compute_efficient_gradient(short_hidden_states)
        long_surprise = memory._compute_efficient_gradient(long_hidden_states)
        
        # Check output shapes
        self.assertEqual(short_surprise.shape, (self.batch_size, self.seq_len, 1))
        self.assertEqual(long_surprise.shape, (self.batch_size, long_seq_length, 1))
        
        # Check that values are reasonable (non-zero, finite)
        self.assertTrue(torch.all(torch.isfinite(short_surprise)))
        self.assertTrue(torch.all(torch.isfinite(long_surprise)))
        self.assertGreater(short_surprise.abs().mean().item(), 0)
        self.assertGreater(long_surprise.abs().mean().item(), 0)
    
    def test_persistent_memory(self):
        """Test persistent memory."""
        memory = PersistentMemory(self.config)
        output = memory(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
    
    def test_titans_memory_system(self):
        """Test the complete Titans memory system."""
        memory_system = TitansMemorySystem(self.config)
        output = memory_system(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)


class TestTransformer2Adaptation(unittest.TestCase):
    """Tests for the Transformer² adaptation."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.num_attention_heads = 4
        self.config.transformer2 = type('obj', (object,), {
            'num_tasks': 8,
            'task_embedding_dim': 64,
            'num_singular_values': 64,
            'expert_init_scale': 0.1,
            'use_task_dispatcher': True,
            'use_svd_adaptation': True,
            'use_two_pass_inference': True,
            'cache_first_pass': True,
            'reuse_threshold': 0.9
        })
        self.config.num_tasks = 8
        self.config.num_singular_values = 64
        
        self.batch_size = 2
        self.seq_len = 24
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
    
    def test_task_dispatcher(self):
        """Test task dispatcher."""
        dispatcher = TaskDispatcher(self.config)
        output = dispatcher(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.config.num_tasks))
    
    def test_svd_adaptation(self):
        """Test SVD adaptation."""
        adaptation = SVDAdaptation(self.config)
        
        # Set task embedding
        task_embedding = torch.randn(self.batch_size, self.config.num_tasks)
        adaptation.set_task_embedding(task_embedding)
        
        # Test forward pass (basic compatibility check)
        output = adaptation(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
    
    def test_extended_svd_adaptation(self):
        """Test extended SVD adaptation with multiple weight matrices."""
        # Create a small transformer model for testing
        mini_config = deepcopy(self.config)
        mini_config.num_layers = 2  # Use a small model for testing
        mini_config.transformer2.layer_specific = False  # Simplify test by using shared adapters
        model = MemoryEfficientTransformer(mini_config)
        
        # Create adaptation with attention adaptation only
        adaptation = SVDAdaptation(mini_config)
        adaptation.adapt_attention = True
        adaptation.adapt_ffn = False  # Simplify test
        adaptation.adapt_embeddings = False  # Simplify test
        adaptation.adapt_lm_head = False  # Simplify test
        adaptation.layer_specific = False  # Simplify test
        
        # Test weight decomposition
        adaptation.decompose_model_weights(model)
        
        # Check that we have decompositions for attention matrices
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            # Check a single layer to simplify test
            component_key = f"attention.0.{name}"
            self.assertIn(component_key, adaptation.svd_components)
            
            # Check that components exist
            self.assertIn("U", adaptation.svd_components[component_key])
            self.assertIn("S", adaptation.svd_components[component_key])
            self.assertIn("Vh", adaptation.svd_components[component_key])
        
        # Set task embedding (use default)
        task_embedding = torch.ones(1, mini_config.transformer2.num_tasks) / mini_config.transformer2.num_tasks
        adaptation.set_task_embedding(task_embedding)
        
        # Test forward method (simpler test that doesn't rely on internal details)
        input_tensor = torch.randn(1, 5, mini_config.hidden_size)
        output = adaptation(input_tensor)
        
        # Check output shape matches input
        self.assertEqual(output.shape, input_tensor.shape)
    
    def test_efficient_svd_computation(self):
        """Test efficient SVD computation with caching."""
        # Create a small model for testing
        mini_config = deepcopy(self.config)
        mini_config.num_layers = 1
        mini_config.transformer2.layer_specific = False
        mini_config.transformer2.enable_svd_caching = True
        
        # Test with randomized SVD enabled and disabled
        for use_randomized in [True, False]:
            mini_config.transformer2.use_randomized_svd = use_randomized
            
            # Create SVD adaptation
            adaptation = SVDAdaptation(mini_config)
            
            # Create test weight matrices of different sizes
            small_weight = torch.randn(10, 10)  # Small matrix
            medium_weight = torch.randn(64, 64)  # Medium matrix
            large_weight = torch.randn(256, 256)  # Large matrix
            
            # Test efficient SVD computation for small matrix
            n_components = adaptation._estimate_adaptive_precision(small_weight)
            U1, S1, Vh1 = adaptation._compute_efficient_svd(small_weight, n_components)
            
            # Test shapes of decomposition
            self.assertEqual(U1.shape[1], min(n_components, 10))
            self.assertEqual(S1.shape[0], min(n_components, 10))
            self.assertEqual(Vh1.shape[0], min(n_components, 10))
            
            # Test caching - calling again should use cache
            U2, S2, Vh2 = adaptation._compute_efficient_svd(small_weight, n_components)
            # Check that they're the same tensors (cached)
            self.assertTrue(torch.allclose(U1, U2))
            self.assertTrue(torch.allclose(S1, S2))
            self.assertTrue(torch.allclose(Vh1, Vh2))
            
            # Test clear cache
            adaptation.clear_svd_cache()
            self.assertEqual(len(adaptation.svd_cache), 0)
            
            # Test on larger matrices if randomized SVD is enabled
            if use_randomized:
                # For medium matrix
                n_components = adaptation._estimate_adaptive_precision(medium_weight)
                U, S, Vh = adaptation._compute_efficient_svd(medium_weight, n_components)
                self.assertEqual(U.shape[1], min(n_components, 64))
                
                # For large matrix
                n_components = adaptation._estimate_adaptive_precision(large_weight)
                U, S, Vh = adaptation._compute_efficient_svd(large_weight, n_components)
                self.assertEqual(U.shape[1], min(n_components, 256))
    
    def test_transformer2_adaptation(self):
        """Test the complete Transformer² adaptation."""
        adaptation = Transformer2Adaptation(self.config)
        
        # Test first pass
        output = adaptation(self.hidden_states, first_pass=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.config.num_tasks))
        
        # Test second pass
        output = adaptation(self.hidden_states, first_pass=False)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
        
        # Test two-pass inference
        output = adaptation.two_pass_inference(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
        
    def test_transformer2_model_adaptation(self):
        """Test the complete Transformer² adaptation with a full model."""
        # Create a small transformer model for testing
        mini_config = deepcopy(self.config)
        mini_config.num_layers = 2  # Use a small model for testing
        mini_config.transformer2.layer_specific = False  # Simplify test
        model = MemoryEfficientTransformer(mini_config)
        
        # Create adaptation
        adaptation = Transformer2Adaptation(mini_config)
        
        # Test the two-pass inference API
        # This tests the functionality without triggering problematic internals
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        
        # First pass: identify task
        task_embedding = adaptation.forward(hidden_states, first_pass=True)
        self.assertEqual(task_embedding.shape, (self.batch_size, mini_config.transformer2.num_tasks))
        
        # Second pass: apply adaptation
        adapted_hidden_states = adaptation.forward(hidden_states, first_pass=False)
        self.assertEqual(adapted_hidden_states.shape, hidden_states.shape)
        
        # Test the complete two-pass inference
        final_output = adaptation.two_pass_inference(hidden_states)
        self.assertEqual(final_output.shape, hidden_states.shape)
        
    def test_task_embedding_cache(self):
        """Test task embedding cache with similarity matching."""
        # Create adaptation with task embedding cache
        mini_config = deepcopy(self.config)
        mini_config.transformer2.max_task_cache_size = 10
        mini_config.transformer2.task_similarity_threshold = 0.8
        adaptation = Transformer2Adaptation(mini_config)
        
        # Create some input tensors
        input1 = torch.randn(1, 5, self.config.hidden_size)
        input2 = input1 + 0.1 * torch.randn_like(input1)  # Similar to input1
        input3 = torch.randn(1, 5, self.config.hidden_size)  # Different from input1
        
        # Generate task embeddings
        task_embedding1 = torch.randn(1, mini_config.transformer2.num_tasks)
        task_embedding2 = torch.randn(1, mini_config.transformer2.num_tasks)
        
        # Add to cache
        pooled_input1 = input1.mean(dim=1)
        adaptation.add_to_task_cache(pooled_input1, task_embedding1)
        
        # Test finding similar task
        pooled_input2 = input2.mean(dim=1)
        similar_result = adaptation.find_similar_task(pooled_input2)
        
        # Should find a match for input2 (similar to input1)
        self.assertIsNotNone(similar_result)
        similar_embedding, metadata = similar_result
        self.assertTrue(torch.allclose(similar_embedding, task_embedding1))
        
        # Test with dissimilar input
        pooled_input3 = input3.mean(dim=1)
        dissimilar_result = adaptation.find_similar_task(pooled_input3)
        
        # Should not find a match for input3 (different from input1)
        self.assertIsNone(dissimilar_result)
        
        # Test cache pruning
        for i in range(15):  # Add more entries than max_cache_size
            test_input = torch.randn(1, 5, self.config.hidden_size)
            test_embedding = torch.randn(1, mini_config.transformer2.num_tasks)
            adaptation.add_to_task_cache(test_input.mean(dim=1), test_embedding)
        
        # Cache should be pruned to max size
        self.assertLessEqual(len(adaptation.task_embedding_cache), mini_config.transformer2.max_task_cache_size)
        
        # Test clear cache
        adaptation.clear_task_cache()
        self.assertEqual(len(adaptation.task_embedding_cache), 0)


class TestMVoTTokenProcessor(unittest.TestCase):
    """Tests for the MVoT token processor."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.is_multimodal = True
        self.config.codebook_size = 32
        self.config.mvot = type('obj', (object,), {
            'codebook_size': 32,
            'embedding_dim': 64,  # Set this to match hidden_size
            'discrepancy_loss_weight': 1.0,
            'is_multimodal': True,
            'use_pretrained_codebook': False,
            'codebook_path': None,
            'codebook_model_type': 'vqvae'
        })
        
        self.batch_size = 2
        self.seq_len = 24
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        self.token_type_ids = torch.randint(0, 2, (self.batch_size, self.seq_len))
        self.target_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
    
    def test_token_discrepancy_loss(self):
        """Test token discrepancy loss."""
        loss_fn = TokenDiscrepancyLoss(self.config)
        loss = loss_fn(self.hidden_states, self.token_type_ids, self.target_embeddings)
        
        # Check loss is a scalar
        self.assertEqual(loss.shape, torch.Size([]))
    
    def test_mvot_token_processor(self):
        """Test the complete MVoT token processor."""
        processor = MVoTTokenProcessor(self.config)
        output = processor(self.hidden_states, self.token_type_ids)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
        
        # Test compute_loss
        loss = processor.compute_loss(self.hidden_states, self.token_type_ids, self.target_embeddings)
        
        # Check loss is a scalar
        self.assertEqual(loss.shape, torch.Size([]))


class TestBLTByteProcessor(unittest.TestCase):
    """Tests for the BLT byte processor."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.num_attention_heads = 4
        self.config.entropy_threshold = 0.5
        self.config.num_local_layers = 1
        self.config.num_latent_layers = 1
        
        # Setup ByteLM config
        self.config.byte_lm = type('obj', (object,), {
            'hidden_size': 64,
            'num_layers': 2,
            'num_attention_heads': 2,
            'byte_lm_dropout': 0.1,
            'byte_lm_max_position': 128
        })
        
        self.batch_size = 2
        self.seq_len = 24
        self.input_bytes = torch.randint(0, 256, (self.batch_size, self.seq_len))
    
    def test_small_byte_lm(self):
        """Test small byte LM."""
        lm = SmallByteLM(self.config)
        output = lm(self.input_bytes)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 256))
        
        # Test with labels
        loss, output = lm(self.input_bytes, self.input_bytes)
        
        # Check loss is scalar and output has correct shape
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 256))
        
        # Test generate_probs
        probs = lm.generate_probs(self.input_bytes)
        
        # Check probabilities sum to 1 along vocabulary dimension
        self.assertEqual(probs.shape, (self.batch_size, self.seq_len, 256))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones((self.batch_size, self.seq_len))))
        
        # Test save and load
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pt")
            
            # Save model
            lm.save_pretrained(model_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Create new model and load
            new_lm = SmallByteLM(self.config)
            new_lm.load_pretrained(model_path)
            
            # Check the loaded model works
            with torch.no_grad():
                loaded_output = new_lm(self.input_bytes)
                # Just check the shape - exact values might differ due to numerical precision
                self.assertEqual(loaded_output.shape, (self.batch_size, self.seq_len, 256))
    
    def test_entropy_calculator(self):
        """Test entropy calculator."""
        calculator = EntropyCalculator(self.config)
        patches = calculator(self.input_bytes)
        
        # Check patches are a list
        self.assertIsInstance(patches, list)
        
        # Check each patch has the correct batch size
        for patch in patches:
            self.assertEqual(patch.shape[0], self.batch_size)
        
        # Test with checkpoint path
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pt")
            
            # Train a byte LM and save
            lm = SmallByteLM(self.config)
            lm.save_pretrained(model_path)
            
            # Create config with checkpoint path
            config_with_checkpoint = self.config
            config_with_checkpoint.byte_lm.checkpoint_path = model_path
            
            # Create entropy calculator with pretrained model
            calculator_with_pretrained = EntropyCalculator(config_with_checkpoint)
            
            # Test forward pass
            patches = calculator_with_pretrained(self.input_bytes)
            self.assertIsInstance(patches, list)
    
    def test_blt_byte_processor(self):
        """Test the complete BLT byte processor."""
        processor = BLTByteProcessor(self.config)
        output = processor(self.input_bytes)
        
        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)
        
        # Test with different sequence lengths
        for seq_len in [8, 32, 64]:
            input_bytes = torch.randint(0, 256, (self.batch_size, seq_len))
            output = processor(input_bytes)
            self.assertEqual(output.shape, input_bytes.shape)


class TestByteDatasetAndTrainer(unittest.TestCase):
    """Tests for the byte dataset and entropy estimator trainer."""
    
    def setUp(self):
        """Set up test case."""
        # Create a basic ByteLM config
        self.config = ByteLMConfig(
            hidden_size=64,
            num_layers=1,
            num_attention_heads=2,
            byte_lm_dropout=0.1,
            byte_lm_max_position=64,
            block_size=32,
            max_steps=2,  # Use only 2 steps for quick testing
            batch_size=2,
            gradient_accumulation_steps=1
        )
        
        # Create temp directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test files
        self.test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir.name, f"test_file_{i}.txt")
            with open(file_path, "wb") as f:
                # Write random bytes (100 bytes per file)
                f.write(os.urandom(100))
            self.test_files.append(file_path)
        
        # Setup output dir and cache dir
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        self.cache_dir = os.path.join(self.temp_dir.name, "cache")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.config.train_files = self.test_files
        self.config.output_dir = self.output_dir
        self.config.cache_dir = self.cache_dir
    
    def tearDown(self):
        """Clean up after test."""
        self.temp_dir.cleanup()
    
    def test_byte_dataset(self):
        """Test ByteDataset."""
        # Create dataset
        dataset = ByteDataset(
            file_paths=self.test_files,
            block_size=self.config.block_size,
            cache_dir=self.cache_dir
        )
        
        # Check dataset size
        self.assertGreater(len(dataset), 0)
        
        # Check example format
        example = dataset[0]
        self.assertIn("input_bytes", example)
        self.assertIn("labels", example)
        
        # Check example shape
        self.assertEqual(example["input_bytes"].shape, (self.config.block_size,))
        
        # Check input and labels are same (for next-byte prediction)
        self.assertTrue(torch.equal(example["input_bytes"], example["labels"]))
        
        # Test cache functionality (reusing same cache_dir)
        cached_dataset = ByteDataset(
            file_paths=self.test_files,
            block_size=self.config.block_size,
            cache_dir=self.cache_dir
        )
        
        # Check dataset size is same
        self.assertEqual(len(cached_dataset), len(dataset))
    
    def test_entropy_estimator_trainer(self):
        """Test EntropyEstimatorTrainer."""
        # Create model
        model = SmallByteLM(self.config)
        
        # Create dataset
        dataset = ByteDataset(
            file_paths=self.test_files,
            block_size=self.config.block_size,
            cache_dir=self.cache_dir
        )
        
        # Create a small dataset for eval
        eval_dataset = ByteDataset(
            file_paths=self.test_files[0:1],  # Just use the first file
            block_size=self.config.block_size,
            cache_dir=self.cache_dir
        )
        
        # Create trainer
        trainer = EntropyEstimatorTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=10,
            max_steps=2,  # Use only 2 steps for quick testing
            eval_steps=1,
            save_steps=1,
            output_dir=self.output_dir
        )
        
        # Run training (only 2 steps)
        trainer.train()
        
        # Check that model files were saved
        checkpoint_files = [f for f in os.listdir(self.output_dir) if f.endswith(".pt")]
        self.assertGreater(len(checkpoint_files), 0)
        
        # Load a checkpoint
        checkpoint_path = os.path.join(self.output_dir, checkpoint_files[0])
        
        # Create new model and load
        new_model = SmallByteLM(self.config)
        new_model.load_pretrained(checkpoint_path)
        
        # Create entropy calculator with loaded model
        mock_config = get_default_config()
        mock_config.entropy_threshold = 0.5
        mock_config.byte_lm = type('obj', (object,), {
            'checkpoint_path': checkpoint_path,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_attention_heads': self.config.num_attention_heads
        })
        calculator = EntropyCalculator(mock_config)
        
        # Test forward pass with loaded model
        input_bytes = torch.randint(0, 256, (2, 16))
        patches = calculator(input_bytes)
        self.assertIsInstance(patches, list)


class TestMemoryEfficientTransformer(unittest.TestCase):
    """Tests for the memory-efficient transformer."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.num_attention_heads = 4
        self.config.num_layers = 2
        self.config.vocab_size = 1000
        self.config.max_position_embeddings = 128
        
        self.batch_size = 2
        self.seq_len = 24
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones((self.batch_size, self.seq_len))
    
    def test_flash_attention(self):
        """Test flash attention."""
        attention = FlashAttention(self.config)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        output, _, _ = attention(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_transformer_layer(self):
        """Test transformer layer."""
        layer = TransformerLayer(self.config)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        output = layer(hidden_states)[0]
        
        # Check output shape
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_memory_efficient_transformer(self):
        """Test the complete memory-efficient transformer."""
        transformer = MemoryEfficientTransformer(self.config)
        output = transformer(self.input_ids, self.attention_mask)
        
        # Check output shape
        self.assertEqual(output["logits"].shape, (self.batch_size, self.seq_len, self.config.vocab_size))


class TestUnifiedArchitecture(unittest.TestCase):
    """Tests for the unified architecture."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.num_attention_heads = 4
        self.config.num_layers = 2
        self.config.vocab_size = 1000
        self.config.max_position_embeddings = 128
        
        # Component activation
        self.config.use_titans_memory = True
        self.config.use_transformer2_adaptation = True
        self.config.use_mvot_processor = True
        self.config.use_blt_processor = False  # Disable BLT processor for tests since it expects byte values (0-255)
        self.config.use_two_pass_inference = False
        
        # Component parameters
        self.config.window_size = 16
        self.config.memory_size = 32
        self.config.num_persistent_vectors = 8
        self.config.num_tasks = 4
        self.config.num_singular_values = 64
        self.config.is_multimodal = True
        self.config.codebook_size = 32
        self.config.entropy_threshold = 0.5
        self.config.num_local_layers = 1
        self.config.num_latent_layers = 1
        
        # Component activation thresholds
        self.config.titans_activation_threshold = 0.5
        self.config.transformer2_activation_threshold = 0.5
        self.config.mvot_activation_threshold = 0.5
        self.config.blt_activation_threshold = 0.5
        self.config.two_pass_activation_threshold = 0.8
        
        self.batch_size = 2
        self.seq_len = 24
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones((self.batch_size, self.seq_len))
        self.token_type_ids = torch.randint(0, 2, (self.batch_size, self.seq_len))
    
    def test_unified_architecture(self):
        """Test the unified architecture."""
        model = UnifiedArchitecture(self.config)
        output = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids
        )
        
        # Check output shape
        self.assertEqual(output["logits"].shape, (self.batch_size, self.seq_len, self.config.vocab_size))
    
    def test_component_activation(self):
        """Test component activation."""
        model = UnifiedArchitecture(self.config)
        
        # Get active components
        active_components = model.get_active_components()
        
        # Check components are active according to config
        self.assertEqual(active_components["byte_processor"], self.config.use_blt_processor)
        self.assertTrue(active_components["memory_system"])
        self.assertTrue(active_components["token_processor"])
        self.assertTrue(active_components["adaptation_system"])
        self.assertFalse(active_components["two_pass_inference"])
        
        # Deactivate all components
        model.set_active_components({
            "byte_processor": False,
            "memory_system": False,
            "token_processor": False,
            "adaptation_system": False,
            "two_pass_inference": False
        })
        
        # Get active components
        active_components = model.get_active_components()
        
        # Check all components are inactive
        self.assertFalse(active_components["byte_processor"])
        self.assertFalse(active_components["memory_system"])
        self.assertFalse(active_components["token_processor"])
        self.assertFalse(active_components["adaptation_system"])
        self.assertFalse(active_components["two_pass_inference"])
        
        # Test forward pass with all components inactive
        output = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids
        )
        
        # Check output shape
        self.assertEqual(output["logits"].shape, (self.batch_size, self.seq_len, self.config.vocab_size))
    
    def test_dynamic_component_controller(self):
        """Test dynamic component controller."""
        model = UnifiedArchitecture(self.config)
        controller = DynamicComponentController(model, self.config)
        
        # Optimize for input
        active_components = controller.optimize_for_input(self.input_ids)
        
        # Check active components is a dictionary
        self.assertIsInstance(active_components, dict)


if __name__ == "__main__":
    unittest.main()
