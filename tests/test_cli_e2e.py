import os
import sys
import unittest
import tempfile
import subprocess
import shutil
import json
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCLIEndToEnd(unittest.TestCase):
    """
    End-to-end test for the CLI interface.
    
    This test verifies that the CLI interface can be launched and used
    to run BLT training with a small test configuration.
    """
    
    def setUp(self):
        # Create temporary directory for test data and outputs
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train")
        self.eval_dir = os.path.join(self.temp_dir, "eval")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.config_dir = os.path.join(self.temp_dir, "configs")
        
        # Create directories
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create sample data
        self._create_sample_data()
        
        # Create test configuration
        self.config_path = self._create_test_config()
        
        # Path to python executable
        self.python_path = sys.executable
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample training and evaluation data for testing."""
        # Create sample training data (small text files with random content)
        for i in range(3):
            with open(os.path.join(self.train_dir, f"sample_{i}.txt"), "w") as f:
                f.write(f"This is a sample training file {i} for testing the CLI interface.\n")
                f.write(f"It contains enough text to be processed by the BLT model.\n")
        
        # Create sample evaluation data
        for i in range(2):
            with open(os.path.join(self.eval_dir, f"sample_{i}.txt"), "w") as f:
                f.write(f"This is a sample evaluation file {i} for testing the CLI interface.\n")
    
    def _create_test_config(self):
        """Create a test configuration file."""
        config = {
            "mode": "train",
            "training_type": "blt_entropy",
            "train_data_dir": self.train_dir,
            "eval_data_dir": self.eval_dir,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "block_size": 64,
            "batch_size": 2,
            "max_steps": 5,
            "eval_steps": 5,
            "save_steps": 5,
            "learning_rate": 5e-5,
            "warmup_steps": 1,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "mixed_precision": False  # Disable for testing
        }
        
        # Save configuration to file
        config_path = os.path.join(self.config_dir, "cli_test_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def test_main_cli_direct(self):
        """Test running main.py with direct arguments."""
        # Command to run main.py with arguments
        cmd = [
            self.python_path,
            "main.py",
            "--output_dir", self.output_dir,
            "--force_cpu",  # Use CPU for testing to avoid MPS issues - must be before subcommand
            "train",  # Subcommand must come after global arguments
            "--training_type", "blt_entropy",
            "--train_data_dir", self.train_dir,
            "--eval_data_dir", self.eval_dir,
            "--byte_lm_hidden_size", "32",  # Smaller model for faster tests
            "--byte_lm_num_layers", "1",
            "--byte_lm_num_heads", "2",
            "--block_size", "32",  # Smaller block size for faster tests
            "--batch_size", "2",
            "--max_steps", "3",  # Fewer steps for faster tests
            "--eval_steps", "3",
            "--save_steps", "3",
            "--learning_rate", "5e-5",
            "--cache_dir", self.cache_dir
            # No mixed precision in CPU mode
        ]
        
        # Run command with a timeout of 60 seconds (this should be enough for a small test)
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=120,  # 2 minutes should be plenty for this small test
            )
            
            # Check command output
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
            
            # Print output for debugging
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
            # List output directory contents for debugging
            print(f"Output directory contents:")
            for root, dirs, files in os.walk(self.output_dir):
                print(f"  {root}:")
                for d in dirs:
                    print(f"    dir: {d}")
                for f in files:
                    print(f"    file: {f}")
            
            # Check if output directory contains expected files
            # First try the expected checkpoints subdirectory, then try root
            expected_checkpoint = os.path.join(self.output_dir, "checkpoints", "checkpoint-3.pt")
            alt_expected_checkpoint = os.path.join(self.output_dir, "checkpoint-3.pt")
            
            # Accept either location for backward compatibility
            self.assertTrue(
                os.path.exists(expected_checkpoint) or os.path.exists(alt_expected_checkpoint), 
                f"Checkpoint file not created at {expected_checkpoint} or {alt_expected_checkpoint}"
            )
            
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout.decode('utf-8')}")
            print(f"STDERR: {e.stderr.decode('utf-8')}")
            self.fail(f"Command failed with exit code {e.returncode}")
    
    def test_run_cli_with_config(self):
        """Test running with a configuration file directly via main.py."""
        # In this test, we'll use main.py directly with arguments from the config file
        # instead of using the run_cli.py script
        
        # Load the config file
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        # Print config for debugging
        print(f"Using config: {config_data}")
        
        # Build command using config data directly with correctly mapped parameter names
        cmd = [
            self.python_path,
            "main.py",
            "--output_dir", self.output_dir,
            "--force_cpu",  # Place force_cpu before subcommand (it's a global arg)
            "train",
            "--training_type", "blt_entropy",
            "--max_steps", "3",  # Make sure these are set early
            "--eval_steps", "3",
            "--save_steps", "3"
        ]
        
        # Parameter name mapping (from config keys to main.py parameter names)
        param_mapping = {
            "num_heads": "byte_lm_num_heads",
            "dropout": "byte_lm_dropout",
            "hidden_size": "byte_lm_hidden_size",
            "num_layers": "byte_lm_num_layers"
        }
        
        # Add parameters with correct mapping
        for key, value in config_data.items():
            if key not in ['mode', 'training_type', 'output_dir', 'max_steps', 'eval_steps', 'save_steps']:
                # Map parameter name if needed
                param_name = param_mapping.get(key, key)
                
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{param_name}")
                elif value is not None:
                    cmd.append(f"--{param_name}")
                    cmd.append(str(value))
        
        # No environment variables needed for this approach
        env = os.environ.copy()
        
        # Run command with a timeout of 60 seconds
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=120,  # 2 minutes should be plenty for this small test
                env=env
            )
            
            # Check command output
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
            
            # Print output for debugging
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
            # List output directory contents for debugging
            print(f"Output directory contents:")
            for root, dirs, files in os.walk(self.output_dir):
                print(f"  {root}:")
                for d in dirs:
                    print(f"    dir: {d}")
                for f in files:
                    print(f"    file: {f}")
            
            # Check if output directory contains expected files
            # First try the expected checkpoints subdirectory, then try root
            expected_checkpoint = os.path.join(self.output_dir, "checkpoints", "checkpoint-3.pt")
            alt_expected_checkpoint = os.path.join(self.output_dir, "checkpoint-3.pt")
            
            # Accept either location for backward compatibility
            self.assertTrue(
                os.path.exists(expected_checkpoint) or os.path.exists(alt_expected_checkpoint), 
                f"Checkpoint file not created at {expected_checkpoint} or {alt_expected_checkpoint}"
            )
            
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout.decode('utf-8')}")
            print(f"STDERR: {e.stderr.decode('utf-8')}")
            self.fail(f"Command failed with exit code {e.returncode}")
    
    def test_quick_test_function(self):
        """Test running the quick test function through run_cli.py."""
        # Command to run run_cli.py with quick test
        cmd = [
            self.python_path,
            "scripts/run_cli.py",
            "test",
            "--auto-confirm",
            "--auto-continue"
        ]
        
        # Set environment variable for output directory
        env = os.environ.copy()
        env["NEAT_OUTPUT_DIR"] = self.output_dir
        
        # Run command with a timeout of 60 seconds
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=120,  # 2 minutes should be plenty for this small test
                env=env
            )
            
            # Check command output
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
            
            # Print output for debugging
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
            # List output directory contents for debugging
            print(f"Output directory contents:")
            for root, dirs, files in os.walk(self.output_dir):
                print(f"  {root}:")
                for d in dirs:
                    print(f"    dir: {d}")
                for f in files:
                    print(f"    file: {f}")
            
            # For quick test, we can't easily check for specific output files
            # but just make sure the command completes successfully
            
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout.decode('utf-8')}")
            print(f"STDERR: {e.stderr.decode('utf-8')}")
            self.fail(f"Command failed with exit code {e.returncode}")

if __name__ == "__main__":
    unittest.main()