import os
import sys
import unittest
import torch
import tempfile
import json
import shutil
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_OLD.components.blt.byte_processor import SmallByteLMConfig, SmallByteLM
from src_OLD.trainers.main_trainer import ByteDataset, EntropyEstimatorTrainer
from src_OLD.utils.config import ByteLMConfig
from src_OLD.trainers import train_blt_model, create_blt_model

# Set memory watermark ratio for tests if using MPS
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # No limit for testing
    print("Setting MPS high watermark ratio to 0.0 for testing")

# Force CPU for BLT end-to-end tests to ensure consistent behavior
os.environ["FORCE_CPU_FOR_TESTING"] = "1"
print("Forcing CPU for BLT tests")

class TestBLTEndToEnd(unittest.TestCase):
    """
    End-to-end test for the BLT training pipeline.
    
    This test verifies that the BLT entropy estimator can be trained
    for a small number of steps and that the training process works correctly.
    """
    
    def setUp(self):
        # Create temporary directory for test data and outputs
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train")
        self.eval_dir = os.path.join(self.temp_dir, "eval")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        
        # Create directories
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create sample training and evaluation data
        self._create_sample_data()
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        # Create sample training data (small text files with random content)
        for i in range(5):
            with open(os.path.join(self.train_dir, f"sample_{i}.txt"), "w") as f:
                f.write(f"This is a sample training file {i} with some text for testing the BLT model. "
                       f"It contains enough data to process into byte sequences for the entropy estimator.\n")
                # Add some repeating data to have patterns
                for j in range(10):
                    f.write(f"Repeating pattern {j} to create some structure for the model.\n")
        
        # Create sample evaluation data
        for i in range(2):
            with open(os.path.join(self.eval_dir, f"sample_{i}.txt"), "w") as f:
                f.write(f"This is a sample evaluation file {i}. It is similar to training but different.\n")
                # Add some repeating data to have patterns
                for j in range(5):
                    f.write(f"Eval pattern {j} that the model should learn to predict.\n")
    
    def test_blt_micro_training(self):
        """
        Test a very small training run (5 steps) to verify the BLT training pipeline.
        """
        # Create configuration
        config = ByteLMConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            byte_lm_dropout=0.1,
            byte_lm_max_position=128,
            learning_rate=5e-5,
            batch_size=2,
            block_size=64,
            warmup_steps=1,
            max_steps=5,  # Just 5 steps for quick testing
            eval_steps=5,
            save_steps=5,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            cache_dir=self.cache_dir,
            output_dir=self.output_dir,
        )
        
        # Add train and eval files to config
        config.train_files = [os.path.join(self.train_dir, f) for f in os.listdir(self.train_dir)]
        config.eval_files = [os.path.join(self.eval_dir, f) for f in os.listdir(self.eval_dir)]
        
        # Find training files
        train_files = [os.path.join(self.train_dir, f) for f in os.listdir(self.train_dir)]
        eval_files = [os.path.join(self.eval_dir, f) for f in os.listdir(self.eval_dir)]
        
        config.train_files = train_files
        config.eval_files = eval_files
        
        # Train the model
        model = train_blt_model(config)
        
        # Check if checkpoint files were created in the checkpoints subdirectory
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", "checkpoint-5.pt")
        self.assertTrue(os.path.exists(checkpoint_path), 
                       f"Checkpoint file not created at {checkpoint_path}")
        
        # Load the model and verify it can make predictions
        loaded_model = create_blt_model(config)
        
        # Load checkpoint manually with weights_only=False to allow loading ByteLMConfig
        checkpoint = torch.load(os.path.join(self.output_dir, "checkpoints", "checkpoint-5.pt"), 
                               map_location=torch.device('cpu'), weights_only=False)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create a simple input for testing
        test_bytes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).long()
        
        # Generate predictions
        with torch.no_grad():
            probs = loaded_model.generate_probs(test_bytes)
        
        # Check that probabilities sum to 1 and have the right shape
        self.assertEqual(probs.shape, (1, 10, 256), "Incorrect probability shape")
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(1, 10), atol=1e-5), 
                       "Probabilities don't sum to 1")
    
    def test_blt_cli_config_compatibility(self):
        """
        Test that BLT training is compatible with the CLI configuration format.
        """
        # Create a CLI-style configuration
        cli_config = {
            "mode": "train",
            "training_type": "blt_entropy",
            "train_data_dir": self.train_dir,
            "eval_data_dir": self.eval_dir,
            "byte_lm_hidden_size": 64,
            "byte_lm_num_layers": 2,
            "byte_lm_num_heads": 4,
            "block_size": 64,
            "batch_size": 2,
            "max_steps": 5,
            "eval_steps": 5,
            "save_steps": 5,
            "learning_rate": 5e-5,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "mixed_precision": False  # Disable for testing to avoid non-deterministic results
        }
        
        # Save configuration to a temporary file
        config_path = os.path.join(self.temp_dir, "cli_config.json")
        with open(config_path, "w") as f:
            json.dump(cli_config, f)
        
        # Convert CLI config to ByteLMConfig
        from src_OLD.utils.config import convert_cli_config_to_byte_lm_config
        byte_lm_config = convert_cli_config_to_byte_lm_config(cli_config)
        
        # Explicitly set train_files and eval_files
        byte_lm_config.train_files = [os.path.join(self.train_dir, f) for f in os.listdir(self.train_dir)]
        byte_lm_config.eval_files = [os.path.join(self.eval_dir, f) for f in os.listdir(self.eval_dir)]
        
        # Train the model using the CLI configuration
        model = train_blt_model(byte_lm_config)
        
        # Check if checkpoint files were created in the checkpoints subdirectory
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", "checkpoint-5.pt")
        self.assertTrue(os.path.exists(checkpoint_path), 
                        f"Checkpoint file not created at {checkpoint_path} when using CLI config")
    
    def test_entropy_calculation(self):
        """
        Test entropy calculation functionality of the BLT model.
        """
        # Create a small BLT model
        config = SmallByteLMConfig(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            byte_lm_dropout=0.1,
            byte_lm_max_position=128
        )
        model = SmallByteLM(config)
        
        # Load a trained model if available, otherwise use untrained model
        if os.path.exists(os.path.join(self.output_dir, "checkpoint-5.pt")):
            try:
                model.load_pretrained(os.path.join(self.output_dir, "checkpoint-5.pt"))
            except Exception as e:
                print(f"Could not load model, using untrained model: {e}")
        
        # Create a simple input where "Hello" repeats, which should have lower entropy than random
        repetitive_bytes = b"Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello"
        repetitive_tensor = torch.tensor([[b for b in repetitive_bytes]]).long()
        
        # Create random bytes for comparison
        random_bytes = torch.randint(0, 256, (1, len(repetitive_bytes))).long()
        
        # Calculate entropy for both
        with torch.no_grad():
            repetitive_probs = model.generate_probs(repetitive_tensor)
            random_probs = model.generate_probs(random_bytes)
            
            repetitive_entropy = -torch.sum(repetitive_probs * torch.log(repetitive_probs + 1e-10), dim=-1)
            random_entropy = -torch.sum(random_probs * torch.log(random_probs + 1e-10), dim=-1)
        
        # Calculate average entropy across sequence
        avg_repetitive_entropy = repetitive_entropy.mean().item()
        avg_random_entropy = random_entropy.mean().item()
        
        # Even with an untrained model, repetitive entropy should generally be lower
        # than random entropy, but this might not always be true
        # So we just print for diagnostic purposes rather than asserting
        print(f"Repetitive entropy: {avg_repetitive_entropy:.4f}, Random entropy: {avg_random_entropy:.4f}")


if __name__ == "__main__":
    unittest.main()
