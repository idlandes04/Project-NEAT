"""
Tests for the neural architecture components.

This module contains tests for each component of the unified architecture.
"""
import os
import sys
import unittest
import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import ModelConfig, get_default_config
from src.components.titans.memory_system import TitansMemorySystem, WindowAttentionMemory, SurpriseBasedMemory, PersistentMemory
from src.components.transformer2.adaptation import Transformer2Adaptation, TaskDispatcher, SVDAdaptation
from src.components.mvot.token_processor import MVoTTokenProcessor, TokenDiscrepancyLoss
from src.components.blt.byte_processor import BLTByteProcessor, EntropyCalculator, SmallByteLM
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
        output = memory(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
    
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
        
        # Test forward pass
        output = adaptation(self.hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, self.hidden_states.shape)
    
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


class TestMVoTTokenProcessor(unittest.TestCase):
    """Tests for the MVoT token processor."""
    
    def setUp(self):
        """Set up test case."""
        self.config = get_default_config()
        self.config.hidden_size = 64
        self.config.is_multimodal = True
        self.config.codebook_size = 32
        
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
        
        self.batch_size = 2
        self.seq_len = 24
        self.input_bytes = torch.randint(0, 256, (self.batch_size, self.seq_len))
    
    def test_small_byte_lm(self):
        """Test small byte LM."""
        lm = SmallByteLM(self.config)
        output = lm(self.input_bytes)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 256))
    
    def test_entropy_calculator(self):
        """Test entropy calculator."""
        calculator = EntropyCalculator(self.config)
        patches = calculator(self.input_bytes)
        
        # Check patches are a list
        self.assertIsInstance(patches, list)
        
        # Check each patch has the correct batch size
        for patch in patches:
            self.assertEqual(patch.shape[0], self.batch_size)
    
    def test_blt_byte_processor(self):
        """Test the complete BLT byte processor."""
        processor = BLTByteProcessor(self.config)
        output = processor(self.input_bytes)
        
        # Check output shape
        self.assertEqual(output.shape[0], self.batch_size)


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
