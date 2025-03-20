# Comprehensive Refactoring Plan for Project NEAT CLI and BLT Training Pipeline

## 1. Issues Identified:

1. Parameter Mismatch: The CLI is passing parameters that aren't recognized by main_trainer.py (--memory_reserve_pct, etc.).
2. Inconsistent Argument Handling: Argument names in CLI don't match what main_trainer.py expects.
3. Fragmented Implementation: Functionality is spread across multiple scripts (main.py, cli_interface.py, main_trainer.py, main_eval.py).
4. MPS-Specific Issues: Apple Silicon handling is incomplete, causing watermark ratio errors.
5. Pipeline Integration Gaps: Analysis and evaluation functionality from separate scripts isn't properly integrated.
6. Configuration File Mismatch: JSON config files don't align with expected parameter formats.

## 2. Detailed Implementation Plan:

### Phase 1: Standardize Parameter Handling

1. Create a unified parameter schema:
   - Define a single source of truth for all parameters
   - Establish parameter mappings between CLI and trainer modules
   - Document required vs. optional parameters for each component
2. Update CLI interface parameter handling:
   - Modify _train_blt_entropy() to use standardized parameter names
   - Update configuration handling to use the same parameter structure as main_trainer
   - Fix the handling of boolean flags and list arguments
3. Update JSON configuration files:
   - Reformat all existing config files to use consistent naming
   - Create a standard parameter structure with clear sections
   - Add documentation within the JSON files

### Phase 2: Fix MPS Support ✅

1. Implement proper hardware detection and configuration: ✅
   - Create dedicated Apple Silicon configuration profiles ✅
   - Fix the watermark ratio issue in main_trainer.py ✅
   - Implement auto-detection of MPS capabilities with safe defaults ✅
2. Update memory management system: ✅
   - Create a configurable but safe memory management system ✅
   - Add platform-specific error handling for MPS vs. CUDA ✅
   - Implement graceful fallbacks for tensor operations ✅

### Phase 3: Integrate Analysis & Evaluation

1. Merge functionality from analysis scripts:
   - Fully integrate analyze_blt_model.py into main_eval.py (partial progress already made)
   - Add command-line arguments to main_eval.py for all analysis options
   - Create unified reporting functions across evaluation modes
2. Create comprehensive CLI commands:
   - Add dedicated menu items for all BLT analysis functions
   - Implement proper parameter passing for interactive vs. batch mode
   - Ensure consistent error handling across all modes

### Phase 4: Streamline Pipeline Flow

1. Create a consistent entry point pattern:
   - Update main.py to properly dispatch to the right module
   - Standardize return codes and error handling
   - Implement proper logging across all components
2. Implement unified configuration handling:
   - Create a Config class that handles all parameter validation
   - Add config migration to handle older formats
   - Implement parameter inheritance and override rules
3. Create comprehensive documentation:
   - Add command reference to CLAUDE.md
   - Document parameter interactions and common configurations
   - Create quick-start examples for each component

## 3. Implementation Details

Let's start with the most immediate fixes:

### 3.1. Fix Configuration Files

The scripts/main_cli_configs/blt_entropy_mps.json file needs to be updated to match the expected parameters of main_trainer.py:

```json
{
  "model_type": "blt",
  "hidden_size": 128,
  "num_layers": 2,
  "num_heads": 4,
  "dropout": 0.1,
  "block_size": 128,
  "batch_size": 16,
  "max_steps": 5000,
  "eval_steps": 250,
  "save_steps": 500,
  "learning_rate": 5e-5,
  "warmup_steps": 250,
  "gradient_accumulation_steps": 2,
  "weight_decay": 0.01,
  "mixed_precision": false,
  "output_dir": "./outputs/byte_lm",
  "train_data_dir": "./data/pile_subset/train",
  "eval_data_dir": "./data/pile_subset/eval",
  "cache_dir": "./data/cache/byte_lm",
  "num_workers": 2,
  "log_steps": 50,
  "entropy_threshold": 0.5,
  "force_cpu": false
}
```

### 3.2. Fix Parameter Handling in CLI Interface

The _train_blt_entropy() method needs updating to match the expected parameter names:

```python
def _train_blt_entropy(self):
    """Initialize training for the BLT entropy estimator."""
    # ...existing code...

    # Update parameter mapping to match main_trainer.py
    param_mapping = {
        "byte_lm_hidden_size": "hidden_size",
        "byte_lm_num_layers": "num_layers",
        "byte_lm_num_heads": "num_heads",
        "byte_lm_dropout": "dropout"
    }

    # Build command with proper parameters
    cmd = [python_cmd, "-m", "src.trainers.main_trainer", "--model_type", "blt"]

    # Special handling for MPS on Apple Silicon
    if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
        self.console.print("[yellow]Detected Apple Silicon. Using MPS-optimized settings...[/yellow]")
        # Force mixed precision to False
        self.current_config["mixed_precision"] = False
```

### 3.3. Fix Memory Management in main_trainer.py ✅

Simplify the MPS memory management code to use safe fixed values regardless of memory_reserve_pct:

```python
# Configure memory limits for Apple Silicon MPS
if not force_cpu and torch.backends.mps.is_available():
    try:
        # Always use a safe fixed value for MPS on Apple Silicon
        # Using 0.8 (80% of memory) is generally safe for M1/M2/M3 devices
        # Set low watermark to 0.0 to avoid errors where it was set to 1.4 for some reason
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
        logger.info(f"Using MPS with safe memory settings (HIGH=0.8, LOW=0.0)")
        self.device = torch.device('mps')
    except Exception as e:
        logger.warning(f"Failed to setup MPS: {e}. Falling back to CPU.")
        self.device = torch.device('cpu')
```

This fix has been implemented and resolves the "invalid low watermark ratio 1.4" error that was occurring on Apple Silicon.

### 3.4. Integrate Analysis Features in CLI Interface

Add a comprehensive BLT model analysis option to the CLI menu:

```python
def _evaluation_menu(self):
    """Show the evaluation menu."""
    # ...existing code...

    menu_table.add_row("4", "Interactive Evaluation")
    menu_table.add_row("5", "BLT Model Analysis")
    menu_table.add_row("6", "Configure Evaluation Parameters")

    # ... Handle new options...
    elif choice == "5":
        self._blt_model_analysis()

def _blt_model_analysis(self):
    """Run comprehensive BLT model analysis."""
    self._clear_screen()
    self.console.print(Panel("BLT Model Analysis", style=self.main_color))

    # Get model path
    model_path = Prompt.ask("Enter path to BLT model checkpoint")

    # Execute command
    cmd = [
        python_cmd, "-m", "src.trainers.main_eval",
        "--model_type", "blt",
        "--model_path", model_path,
        "--eval_mode", "analyze"
    ]

    self._execute_command_with_progress(" ".join(cmd), "BLT Model Analysis")
```

## 4. Implementation Schedule

1. Immediate Fixes (Today):
   - [x] Update blt_entropy_mps.json to match required parameters
   - [x] Fix MPS memory management in main_trainer.py to use safe defaults
2. Short-term Improvements (1-2 days):
   - [x] Update CLI interface parameter handling
   - [x] Complete the integration of analyze_blt_model.py into main_eval.py
   - [x] Add BLT analysis menu items to the CLI interface
3. Medium-term Refactoring (3-7 days):
   - [x] Create a unified parameter schema
   - [x] Standardize configuration loading and saving
   - [x] Implement comprehensive error handling
4. Long-term Integration (1-2 weeks):
   - [ ] Create a unified pipeline architecture
   - [ ] Implement automatic hardware detection and optimization
   - [ ] Add comprehensive documentation and examples
   - [ ] Ask use to test cli interface and fix any issues remaining. 

## 5. Testing Strategy

1. Basic Functionality Tests:
   - [ ] Test each command through both CLI and direct invocation
   - [ ] Verify parameter handling with various input formats
   - [ ] Check error handling with invalid inputs
2. Platform-specific Tests:
   - [ ] Test on Mac M3 with MPS acceleration
   - [ ] Test with available CUDA devices
   - [ ] Test with CPU-only configuration
3. Integration Tests:
   - [ ] Test full training and evaluation pipeline
   - [ ] Verify results match expected outputs
   - [ ] Test with real and synthetic data
4. Performance Benchmarks:
   - [ ] Measure training speed across platforms
   - [ ] Compare memory usage with different configurations
   - [ ] Validate optimization effectiveness


Progress/ work done: 
⏺ Great! We've successfully updated the CLI interface with a comprehensive refactoring that
  includes:

  1. Created a unified parameter schema using a ConfigSchema dataclass
  2. Standardized configuration loading and saving
  3. Implemented intelligent platform-specific adaptations (especially for Apple Silicon)
  4. Added comprehensive error handling with categorized errors and troubleshooting suggestions
  5. Made the BLT training method compatible with the new schema
  6. Updated the quick test functionality to use the schema-based approach

  This implementation covers the medium-term refactoring tasks from the plan:
  - Created a unified parameter schema ✅
  - Standardized configuration loading and saving ✅
  - Implemented comprehensive error handling ✅

  Let me summarize what we've accomplished:

⏺ Bash(git diff --stat src/utils/cli_interface.py)…
  ⎿   src/utils/cli_interface.py | 820 ++++++++++++++++++++++++++++-----------------
      1 file changed, 512 insertions(+), 308 deletions(-)

⏺ Let me summarize all the changes we've made:

  1. ConfigSchema Class: Created a dataclass to define a unified parameter schema for all NEAT model
   training configurations with:
    - Default values for all parameters
    - Type hints for IDE and validation support
    - Parameter grouping by function
    - Conversion methods between dataclass and dictionary format
    - Platform-specific adaptations (e.g., disabling mixed precision on MPS)
  2. Configuration Management:
    - Updated _load_configuration_menu to convert JSON configs to ConfigSchema objects
    - Enhanced _save_configuration_menu to save ConfigSchema objects to JSON
    - Improved _display_config_summary with categorized parameter display
  3. BLT Training Improvements:
    - Fixed _train_blt_entropy to use the new ConfigSchema
    - Added automatic platform detection for MPS (Apple Silicon)
    - Removed the problematic memory_reserve_pct parameter on MPS
    - Added usage of get_cli_command to generate consistent commands
  4. Config Validation and Error Handling:
    - Added from_dict and to_dict methods to support validation and consistent parameters
    - Updated _ensure_config to use ConfigSchema and properly load typed configurations
    - Improved error handling in configuration loading
  5. Command Execution Enhancements:
    - Enhanced _execute_command_with_progress with comprehensive error detection and categorization
    - Added real-time error detection for critical issues (memory, file/path errors)
    - Added intelligent error categorization with specialized error messages
    - Added troubleshooting suggestions tailored to error categories
    - Improved error presentation with structured, categorized errors
  6. Quick Test Improvements:
    - Updated _quick_test to use ConfigSchema
    - Created a modified test configuration to ensure consistency
    - Improved error handling and output

  These changes significantly improve the robustness and maintainability of the CLI interface,
  particularly for handling different hardware platforms like Apple Silicon and adapting
  configurations appropriately. They also make error handling more user-friendly with categorized
  errors and troubleshooting suggestions.